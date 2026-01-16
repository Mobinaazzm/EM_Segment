
"""
MGAC tracker for binary object segmentation in 2D frame sequences.

This module implements a Morphological Geodesic Active Contour (MGAC) tracker
with intensity+edge fusion, ROI-limited evolution, leak guard, and robust
re-acquisition. It produces per-frame masks plus metrics (IoU, Dice, pixel
accuracy/error) and summary plots.

"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_float, morphology
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_local, threshold_otsu
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour
from skimage.io import imread
from skimage.transform import resize, warp, AffineTransform

from scipy.ndimage import distance_transform_edt, label as ndi_label, center_of_mass
import imageio.v2 as imageio
from skimage.morphology import disk  

import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MGAC:
    """
    MGAC tracker with intensity+edge fusion and ROI-limited evolution.

    High-level flow:
        1) Load frames and (optionally replicated) ground-truth masks.
        2) For each frame, build a seed (motion + optional intensity).
        3) Evolve an MGAC level set inside a padded ROI.
        4) Guard against leaks and re-acquire if tracking fails.
        5) Save predicted masks and track per-frame metrics.

    Attributes:
        syn_dir (Path): Directory of input frames.
        msk_dir (Path): Directory of GT masks (used for metrics and seeding).
        pred_mask_dir (Path): Where predicted binary masks are written.
        results_dir (Path): Where figures and summaries are written.
        width, height (int): Target H×W for frames and masks.
        n_frames (int): Number of frames to process.
        ious, dices, norm_diffs, abs_diffs, errors (list[float]):
            Per-frame metrics (IoU, Dice, pixel-accuracy, pixel-error, alias of pixel-error).
        masks (list[np.ndarray]): Predicted boolean masks per frame.
        run_time (float | None): Total runtime in seconds after `run()`.
    """
    def __init__(
        self,
        syn_dir,
        msk_dir,
        pred_mask_dir,
        results_dir,
        width=256,
        height=256,
        n_frames=200,
        # MGAC evolution
        mgac_smooth=1,          
        mgac_balloon=0.20,
        mgac_thresh=0.35,
        mgac_iters=200,
        mgac_alpha=30.0,        # edge-stop strength for inverse_gaussian_gradient
        mgac_edge_sigma=1.0,
        # morphology / selection
        min_area=None,          # if None -> auto from first GT mask (~8%)
        top_k=1,
        closing_radius=1,       # gentler cleanup
        # init / preprocessing
        fg_bright=False,        # True if foreground is brighter than background
        init_offset=0.08,
        block_size=51,          # coerced to odd >=3
        # intensity fusion
        use_intensity=True,
        intensity_method="adaptive",   # "adaptive" or "global"
        intensity_combine="union",     # "union" or "intersect"
        # ROI evolution + leak guard
        evolve_roi_pad=32,      # padding (px) around seed for MGAC evolution
        area_leak_frac=0.35,    # if mask covers >35% of frame, treat as leak
        # visualization / mode
        debug_visuals=False,
        init_mode="track"
    ):
        """Initialize the MGAC tracker and load data.

        Args:
            syn_dir (str | Path): Frames directory.
            msk_dir (str | Path): Ground-truth masks directory.
            pred_mask_dir (str | Path): Output directory for predicted masks.
            results_dir (str | Path): Output directory for figures/plots.
            width (int): Resize width for processing/saving.
            height (int): Resize height for processing/saving.
            n_frames (int): Number of frames to process (from sorted filenames).
            mgac_smooth (int): Level-set smoothing iterations per step.
            mgac_balloon (float): Balloon force; >0 expands, <0 shrinks.
            mgac_thresh (float): Edge stopping threshold for MGAC.
            mgac_iters (int): Total MGAC iterations per frame.
            mgac_alpha (float): `inverse_gaussian_gradient` alpha (edge-stop).
            mgac_edge_sigma (float): Smoothing sigma for edge image.
            min_area (int | None): Minimum component area; if None, auto from GT.
            top_k (int): Keep top-K components after candidate ranking.
            closing_radius (int): Radius for `binary_closing` cleanup.
            fg_bright (bool): If True, foreground assumed brighter than background.
            init_offset (float): Offset for local adaptive thresholding.
            block_size (int): Window size for local thresholding (coerced to odd >= 3).
            use_intensity (bool): Fuse motion seed with intensity mask inside ROI.
            intensity_method (str): "adaptive" or "global" thresholding.
            intensity_combine (str): "union" or "intersect" with motion seed.
            evolve_roi_pad (int): Padding (px) around seed to define MGAC ROI.
            area_leak_frac (float): Leak guard; redo if mask coverage exceeds this fraction.
            debug_visuals (bool): If True, show debug plots per frame.
            init_mode (str): "track" (use prev mask) or "gt_each" (seed from GT per frame).

        Side Effects:
            - Creates `pred_mask_dir` and `results_dir` if they do not exist.
            - Loads frames and GT masks; sets `self.min_area` if not provided.
        """
        self.syn_dir = Path(syn_dir)
        self.msk_dir = Path(msk_dir)
        self.pred_mask_dir = Path(pred_mask_dir)
        self.results_dir = Path(results_dir)

        self.width = int(width)
        self.height = int(height)
        self.n_frames = int(n_frames)

        self.mgac_smooth = int(mgac_smooth)
        self.mgac_balloon = float(mgac_balloon)
        self.mgac_thresh  = float(mgac_thresh)
        self.mgac_iters   = int(mgac_iters)
        self.mgac_alpha   = float(mgac_alpha)
        self.mgac_edge_sigma = float(mgac_edge_sigma)

        self.min_area = None if min_area is None else int(min_area)
        self.top_k = int(top_k)
        self.closing_radius = int(closing_radius)

        self.fg_bright = bool(fg_bright)
        self.init_offset = float(init_offset)
        self.block_size = int(block_size)
        if self.block_size < 3: self.block_size = 3
        if self.block_size % 2 == 0: self.block_size += 1

        self.use_intensity = bool(use_intensity)
        self.intensity_method = str(intensity_method)
        self.intensity_combine = str(intensity_combine).lower()

        self.evolve_roi_pad = int(evolve_roi_pad)
        self.area_leak_frac = float(area_leak_frac)

        self.debug_visuals = bool(debug_visuals)
        self.init_mode = str(init_mode)

        self.pred_mask_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # load data
        self.gt_masks = self._load_all_gt_masks()
        self.frames = self._load_frames()

        if self.min_area is None:
            area0 = int(self.gt_masks[0].sum())
            self.min_area = max(50, int(0.08 * area0))

        # metrics
        self.ious = []
        self.dices = []
        self.norm_diffs = []
        self.errors = []
        self.masks = []
        self.abs_diffs = []

        self.run_time = None

        # tracking constraints
        self.track_roi_pad = 24
        self.max_jump_px   = 18
        self.area_ratio_lo = 0.4
        self.area_ratio_hi = 2.0
        self.prefer_darker = True

    # --------------------
    # Loading
    # --------------------
    def _load_all_gt_masks(self):
        """Load and binarize GT masks, resizing to (height, width).

        Returns:
            list[np.ndarray]: List of boolean masks (length >= 1). If fewer than
            `n_frames` masks exist, the last mask is replicated to reach `n_frames`.
        """
        exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
        gt_files = []
        for e in exts: gt_files.extend(self.msk_dir.glob(e))
        gt_files = sorted(gt_files, key=lambda f: f.name)
        assert len(gt_files) >= 1, f"No mask files found in {self.msk_dir}."

        H, W = int(self.height), int(self.width)

        def binarize(arr):
            if arr.ndim == 3: arr = arr[..., 0]
            if arr.dtype == np.uint8: m = arr >= 128
            else: m = arr > 0.5
            m = resize(m.astype(float), (H, W), order=0, preserve_range=True).astype(bool)
            return m

        masks = [binarize(imread(str(p))) for p in gt_files]

        
        if len(masks) < self.n_frames:
            masks.extend([masks[-1].copy() for _ in range(self.n_frames - len(masks))])
        else:
            masks = masks[:self.n_frames]
        assert all(m.shape == (H, W) for m in masks)
        return masks

    def _load_frames(self):

        """Load frames, resize to (height, width), and apply light preprocessing.

        Preprocessing:
            - Convert to grayscale if needed.
            - CLAHE contrast equalization (clip_limit=0.02).
            - Gaussian blur (sigma=1.0).
        """
        exts = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff')
        im_files = []
        for e in exts: im_files.extend(self.syn_dir.glob(e))
        im_files = sorted(im_files, key=lambda f: f.name)
        assert len(im_files) >= self.n_frames, f"Found only {len(im_files)} images, need {self.n_frames}"

        frames = []
        from skimage.exposure import equalize_adapthist
        for p in im_files[:self.n_frames]:
            im = img_as_float(imread(str(p)))
            if im.ndim == 3: im = rgb2gray(im[..., :3])
            im = resize(im, (self.height, self.width), order=1, preserve_range=True, anti_aliasing=True)
            im = equalize_adapthist(im, clip_limit=0.02)
            im = gaussian(im, sigma=1.0)
            frames.append(im)

        assert frames[0].shape == self.gt_masks[0].shape
        return frames

    # --------------------
    # Helpers
    # --------------------
    @staticmethod
    def _iou(a, b):

        """Intersection-over-Union (Jaccard) with empty-empty == 1.0."""
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        if union == 0:          
            return 1.0
        return inter / union

    @staticmethod
    def _dice(a, b):

        """Dice coefficient with empty-empty == 1.0."""
        inter = np.logical_and(a, b).sum()
        s = a.sum() + b.sum()
        if s == 0:              
            return 1.0
        return (2 * inter) / s


    def _signed_distance_from_mask(self, m):
        """Compute signed distance transform (inside positive, outside negative)."""
        return distance_transform_edt(m) - distance_transform_edt(~m)

    def _intensity_mask(self, frame):

        """Build an intensity-based foreground mask.

        Uses Otsu ("global") or local adaptive thresholding ("adaptive"), then
        removes small objects/holes relative to `min_area`.
        """
        if self.intensity_method == "global":
            t = threshold_otsu(frame)
            prelim = (frame > t) if self.fg_bright else (frame < t)
        else:
            adaptive = threshold_local(frame, block_size=self.block_size, offset=self.init_offset)
            prelim = (frame > adaptive) if self.fg_bright else (frame < adaptive)
        prelim = morphology.remove_small_objects(prelim, max(10, self.min_area // 4))
        prelim = morphology.remove_small_holes(prelim, max(10, self.min_area // 4))
        return prelim

    def _bbox_from_mask(self, m):
        """Compute tight bounding box around a mask (y0, x0, y1, x1)."""
        ys, xs = np.where(m)
        if ys.size == 0:
            return (0, 0, self.height, self.width)  
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        return (int(y0), int(x0), int(y1) + 1, int(x1) + 1)

    def _crop_with_pad(self, img, bbox, pad):
        """Crop image with padding while clamping to image bounds."""
        y0, x0, y1, x1 = bbox
        y0p = max(0, y0 - pad); x0p = max(0, x0 - pad)
        y1p = min(img.shape[0], y1 + pad); x1p = min(img.shape[1], x1 + pad)
        return img[y0p:y1p, x0p:x1p], (y0p, x0p)

    def _local_shift(self, frame, prev_frame, prev_mask):
        """Estimate local translation between consecutive frames in a ROI."""
        from skimage.registration import phase_cross_correlation
        bbox = self._bbox_from_mask(prev_mask)
        roi_prev, (oy, ox) = self._crop_with_pad(prev_frame, bbox, self.track_roi_pad)
        roi_curr, _         = self._crop_with_pad(frame,      bbox, self.track_roi_pad)
        if roi_prev.size == 0 or roi_curr.size == 0:
            return 0.0, 0.0
        shift, _, _ = phase_cross_correlation(roi_curr, roi_prev, upsample_factor=10)
        return float(shift[0]), float(shift[1])  

    def _edge_stop(self, frame):
        """Compute edge-stopping function for MGAC from a smoothed image."""
        grad_src = gaussian(frame, sigma=1.5)
        return inverse_gaussian_gradient(grad_src, alpha=self.mgac_alpha, sigma=self.mgac_edge_sigma)

    def _evolve_mgac_roi(self, gimage, phi, seed_mask, pad=None):
        """Run MGAC only inside a padded ROI around the seed; paste result back."""
        if pad is None: pad = self.evolve_roi_pad
        bbox = self._bbox_from_mask(seed_mask)
        g_roi, (oy, ox) = self._crop_with_pad(gimage, bbox, pad)
        phi_roi, _      = self._crop_with_pad(phi,     bbox, pad)

        ls_roi = morphological_geodesic_active_contour(
            g_roi, self.mgac_iters, init_level_set=phi_roi,
            smoothing=self.mgac_smooth, balloon=self.mgac_balloon, threshold=self.mgac_thresh
        )
        mask_full = np.zeros_like(seed_mask, bool)
        mask_full[oy:oy+ls_roi.shape[0], ox:ox+ls_roi.shape[1]] = (ls_roi > 0)
        return mask_full

    def _mean_intensity(self, img, comp_mask):
        """Mean intensity inside a component; lower means darker."""
        n = int(comp_mask.sum())
        return float(img[comp_mask].mean()) if n > 0 else np.inf

    def _seed_fallback(self, frame, prev_mask):
        """
        Fallback non-empty seed if ROI fusion yields empty.
        Prefers intensity mask; otherwise draws a small disk near the previous
        centroid or image center.
        """
        inten = self._intensity_mask(frame)
        if inten.any():
            return inten
        
        if prev_mask is not None and prev_mask.any():
            py, px = center_of_mass(prev_mask)
            py = int(py) if not np.isnan(py) else self.height // 2
            px = int(px) if not np.isnan(px) else self.width  // 2
        else:
            py, px = self.height // 2, self.width // 2
        seed = np.zeros((self.height, self.width), bool)
        yy, xx = np.ogrid[:self.height, :self.width]
        r = 6
        seed[(yy - py)**2 + (xx - px)**2 <= r*r] = True
        return seed

    # --------------------
    # Main
    # --------------------
    def run(self):
        """Run MGAC tracking over all frames. """
        logging.info("MGAC segmentation started.")
        t0 = time.time()
        self.ious, self.dices, self.errors, self.norm_diffs, self.masks = [], [], [], [], []
        self.abs_diffs = []

        mode = self.init_mode
        prev_mask, prev_frame = (None, None)
        if mode != "gt_each":
            prev_mask  = self.gt_masks[0].copy()
            prev_frame = self.frames[0].copy()

        # --- helper: intensity inside ROI around a seed ---
        def intensity_roi(img, seed_mask):
            bbox = self._bbox_from_mask(seed_mask)
            roi_img, (oy, ox) = self._crop_with_pad(img, bbox, self.track_roi_pad)
            if self.intensity_method == "global":
                t = threshold_otsu(roi_img)
                roi = (roi_img > t) if self.fg_bright else (roi_img < t)
            else:
                thr = threshold_local(roi_img, block_size=self.block_size, offset=self.init_offset)
                roi = (roi_img > thr) if self.fg_bright else (roi_img < thr)
            roi = morphology.remove_small_objects(roi, max(10, self.min_area // 4))
            roi = morphology.remove_small_holes(roi, max(10, self.min_area // 4))
            m = np.zeros_like(seed_mask, bool)
            m[oy:oy+roi.shape[0], ox:ox+roi.shape[1]] = roi
            return m

        for i, frame in enumerate(self.frames):
            gimage = self._edge_stop(frame)

            # --------------------------
            # Seed for this frame
            # --------------------------
            if mode == "gt_each":
                base_seed = self.gt_masks[i].copy()
            else:
                if i == 0:
                    base_seed = self.gt_masks[0].copy()
                else:
                    dy, dx = self._local_shift(frame, prev_frame, prev_mask)
                    r = float(np.hypot(dy, dx))
                    if r > self.max_jump_px and r > 0:
                        scale = self.max_jump_px / r
                        dy *= scale; dx *= scale
                    tform = AffineTransform(translation=(dx, dy))
                    base_seed = warp(prev_mask.astype(float), tform.inverse, order=0,
                                    preserve_range=True) > 0.5

            # fuse with intensity but **only in ROI**
            if self.use_intensity and base_seed.any():
                inten = intensity_roi(frame, base_seed)
                op = np.logical_and if self.intensity_combine == "intersect" else np.logical_or
                seed = op(base_seed, inten)
            else:
                seed = base_seed

            # never allow empty seed
            if not seed.any():
                seed = self._seed_fallback(frame, prev_mask)

            # --------------------------
            # MGAC (ROI-limited)
            # --------------------------
            phi = self._signed_distance_from_mask(seed)
            mask = self._evolve_mgac_roi(gimage, phi, seed, pad=self.evolve_roi_pad)

            # if empty, try global intensity once
            if not mask.any():
                inten_full = self._intensity_mask(frame)
                if inten_full.any():
                    phi_glob = self._signed_distance_from_mask(inten_full)
                    mask = self._evolve_mgac_roi(gimage, phi_glob, inten_full, pad=self.evolve_roi_pad)

            # --------------------------
            # Leak guard (redo from motion seed only)
            # --------------------------
            if mask.mean() > self.area_leak_frac and base_seed.any():
                tight_seed = base_seed  # ignore intensity for redo
                old_balloon = self.mgac_balloon
                self.mgac_balloon = min(0.12, old_balloon)
                phi_tight = self._signed_distance_from_mask(tight_seed)
                mask_tight = self._evolve_mgac_roi(gimage, phi_tight, tight_seed, pad=self.evolve_roi_pad)
                if mask_tight.any() and mask_tight.sum() < mask.sum():
                    mask = mask_tight
                self.mgac_balloon = old_balloon

            # --------------------------
            # Post-process & candidate selection
            # --------------------------
            lbl, n = ndi_label(mask, structure=np.ones((3, 3), int))
            min_area_track = max(50, self.min_area // 2)

            if n > 0:
                # Build list of previous component centroids+areas (for multi-target matching)
                prev_centroids_areas = []
                if prev_mask is not None and prev_mask.any():
                    lbl_prev, n_prev = ndi_label(prev_mask, structure=np.ones((3, 3), int))
                    for pid in range(1, n_prev + 1):
                        comp_prev = (lbl_prev == pid)
                        py, px = center_of_mass(comp_prev)
                        prev_centroids_areas.append((float(py), float(px), int(comp_prev.sum())))

                apply_strict = (mode != "gt_each" and i > 0)
                candidates = []
                for j in range(1, n + 1):
                    comp = (lbl == j)
                    area = int(comp.sum())
                    if area < min_area_track:
                        continue

                    cy, cx = center_of_mass(comp)

                    if prev_centroids_areas:
                        # distance to nearest previous component
                        dists = [float(np.hypot(cy - py, cx - px)) for (py, px, _) in prev_centroids_areas]
                        dist = min(dists)
                        kbest = int(np.argmin(dists))
                        prev_area_match = float(prev_centroids_areas[kbest][2])
                    else:
                        # first frame: permissive
                        dist = 0.0
                        prev_area_match = float(area)

                    if apply_strict and dist > self.max_jump_px:
                        continue
                    if apply_strict and prev_area_match > 0:
                        ratio = area / prev_area_match
                        if not (self.area_ratio_lo <= ratio <= self.area_ratio_hi):
                            continue

                    mean_int = self._mean_intensity(frame, comp) if self.prefer_darker else 0.0
                    # sort by: nearest to a previous object, larger area, darker (if prefer_darker)
                    candidates.append((j, dist, -area, mean_int))

                if candidates:
                    candidates.sort(key=lambda t: (t[1], t[2], t[3]))
                    keep_ids = [c[0] for c in candidates[:max(1, self.top_k)]]
                    mask = np.isin(lbl, keep_ids)
                    from skimage.morphology import binary_closing
                    if self.closing_radius > 0:
                        mask = binary_closing(mask, disk(self.closing_radius))
                    mask = morphology.remove_small_objects(mask, self.min_area)
                    mask = morphology.remove_small_holes(mask, area_threshold=4 * self.min_area)
                else:
                    # fallback instead of going empty
                    inten_full = self._intensity_mask(frame)
                    if inten_full.any():
                        phi_glob = self._signed_distance_from_mask(inten_full)
                        mask = self._evolve_mgac_roi(gimage, phi_glob, inten_full, pad=self.evolve_roi_pad)
                    else:
                        mask = np.zeros_like(mask, bool)


            # --------------------------
            # Local/global re-acquire if needed (track mode)
            # --------------------------
            if mode != "gt_each" and i > 0:
                overlap_prev = self._iou(mask, prev_mask)

                if overlap_prev < 0.20 and prev_mask is not None and prev_mask.any():
                    bbox = self._bbox_from_mask(prev_mask)
                    roi_img, (oy, ox) = self._crop_with_pad(frame, bbox, self.track_roi_pad)
                    if self.intensity_method == "global":
                        t = threshold_otsu(roi_img); prelim = (roi_img > t) if self.fg_bright else (roi_img < t)
                    else:
                        thr = threshold_local(roi_img, block_size=self.block_size, offset=self.init_offset)
                        prelim = (roi_img > thr) if self.fg_bright else (roi_img < thr)
                    prelim = morphology.remove_small_objects(prelim, max(10, min_area_track // 2))
                    prelim = morphology.remove_small_holes(prelim, max(10, min_area_track // 2))
                    tmp = np.zeros_like(mask, bool)
                    y0, x0 = oy, ox
                    tmp[y0:y0+prelim.shape[0], x0:x0+prelim.shape[1]] = prelim

                    phi_fb = self._signed_distance_from_mask(tmp)
                    mask2 = self._evolve_mgac_roi(gimage, phi_fb, tmp, pad=self.evolve_roi_pad)
                    if self._iou(mask2, prev_mask) > overlap_prev:
                        mask = mask2
                        overlap_prev = self._iou(mask, prev_mask)

                if (not mask.any()) or (overlap_prev < 0.20):
                    inten_full = self._intensity_mask(frame)
                    if inten_full.any():
                        lbl2, n2 = ndi_label(inten_full, structure=np.ones((3, 3), int))
                        if n2 > 1:
                            areas = [(k, (lbl2 == k).sum()) for k in range(1, n2 + 1)]
                            keep_id = max(areas, key=lambda t: t[1])[0]
                            inten_full = (lbl2 == keep_id)
                        phi_glob = self._signed_distance_from_mask(inten_full)
                        mask3 = self._evolve_mgac_roi(gimage, phi_glob, inten_full, pad=self.evolve_roi_pad)
                        if mask3.any():
                            mask = mask3

            # --------------------------
            # Save + metrics
            # --------------------------
            self.masks.append(mask)
            imageio.imwrite(self.pred_mask_dir / f"pred_mask_{i:03d}.png", (mask.astype(np.uint8) * 255))


            gt = self.gt_masks[i]
            self.ious.append(self._iou(mask, gt))
            self.dices.append(self._dice(mask, gt))
            diff = np.logical_xor(mask, gt)

            abs_diff  = diff.sum() / diff.size        # pixel error rate
            norm_diff = 1.0 - abs_diff                # pixel accuracy

            self.abs_diffs.append(abs_diff)
            self.norm_diffs.append(norm_diff)
            self.errors.append(abs_diff)


            prev_mask = mask.copy()
            prev_frame = frame

            if self.debug_visuals:
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1); plt.imshow(frame, cmap='gray'); plt.title(f"Frame {i}"); plt.axis('off')
                plt.subplot(1, 3, 2); plt.imshow(gimage, cmap='viridis'); plt.title("gimage"); plt.axis('off')
                plt.subplot(1, 3, 3); plt.imshow(frame, cmap='gray')
                plt.contour(mask.astype(float), levels=[0.5], colors=('r',), linewidths=2)
                plt.title("Prediction"); plt.axis('off')
                plt.tight_layout(); plt.show()

        #self.errors = [1.0 - i for i in self.ious]
        self.run_time = time.time() - t0
        logging.info("MGAC segmentation finished.")



    # --------------------
    # Viz & eval
    # --------------------

    def plot_performance(self, x_indices=None):

        """Plot Dice, IoU, and Pixel Accuracy\Error over time.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # choose x and label once (mirrors LSTM)
        if x_indices is None:
            x = np.arange(len(self.dices))
            xlabel = f"Prediction # (n={len(x)})"
        else:
            x = np.asarray(x_indices, dtype=int)
            xmin, xmax = int(x.min()), int(x.max())
            xlabel = f"Frame index ({xmin}–{xmax}, n={len(x)})"

        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

        # curves (same order/labels as LSTM)
        ax.plot(x, self.dices,      label="Dice",             linewidth=2)
        ax.plot(x, self.ious,       label="IoU",              linewidth=2)
        ax.plot(x, self.norm_diffs, label="Pixel Accuracy",  linewidth=2)

        # axis limits / ticks (same logic as LSTM)
        xmin, xmax = int(np.min(x)), int(np.max(x))
        if xmin == xmax:  # single-point edge case
            xmin -= 0.5; xmax += 0.5
        ax.set_xlim(xmin, xmax)

        nticks = min(6, max(2, len(np.unique(x))))
        ax.set_xticks(np.linspace(xmin, xmax, nticks, dtype=int))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Score")

        ax.set_ylim(0.3, 1.02)
        ax.set_yticks(np.arange(0.30, 1.03, 0.2))

        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=True, loc="lower left")

        # save with the same filenames as LSTM
        out = self.results_dir / "dice_iou_per_frame"
        plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def show_examples(self, indices=None):
        """Show qualitative examples with contours overlaid."""
        if indices is None:
            indices = [0, self.n_frames // 2, self.n_frames - 1]
        fig, axs = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4))
        if len(indices) == 1: axs = [axs]
        for ax, idx in zip(axs, indices):
            ax.imshow(self.frames[idx], cmap='gray')
            ax.contour(self.masks[idx].astype(float), levels=[0.5], colors=('r',), linewidths=2)
            ax.set_title(f"Frame {idx:03d}\nIoU={self.ious[idx]:.3f}, Dice={self.dices[idx]:.3f}")
            ax.axis('off')
        plt.tight_layout(); plt.show()

    def assert_results(self):
        """Sanity checks for lengths, ranges, and shapes."""
        assert len(self.ious) == self.n_frames
        assert len(self.errors) == self.n_frames
        assert len(self.dices) == self.n_frames
        assert all(0.0 <= iou <= 1.0 for iou in self.ious)
        assert all(0.0 <= err <= 1.0 for err in self.errors)
        assert all(0.0 <= d <= 1.0 for d in self.dices)
        assert all(mask.shape == (self.height, self.width) for mask in self.masks)
        print("All assertions passed.")

    def evaluate(self):
        """Print overall averages (optionally).

        Prints:
            - Average Pixel Accuracy and Pixel Error.
            - Average IoU and Dice.
            - Runtime summary if available.
        """
        if self.run_time is not None:
            mins = int(self.run_time // 60); secs = self.run_time % 60
            print(f"MGAC segmentation took {mins} min {secs:.1f} sec.")

        avg_accuracy = float(np.mean(self.norm_diffs)) if self.norm_diffs else 0.0
        avg_abs_diff = float(np.mean(self.abs_diffs))  if self.abs_diffs  else 0.0
        avg_iou = float(np.mean(self.ious)) if self.ious else 0.0
        avg_dice = float(np.mean(self.dices)) if self.dices else 0.0

        print("\n=== Overall Averages (vs Ground Truth) ===")
        print(f"Average Pixel Accuracy: {avg_accuracy:.4f}")
        print(f"Average Pixel Error:     {avg_abs_diff:.4f}")
        print(f"Average IoU (Jaccard):             {avg_iou:.4f}")
        print(f"Average Dice score:                {avg_dice:.4f}")


        single_mask_replicated = all(np.array_equal(self.gt_masks[0], m) for m in self.gt_masks)

        prev_overlaps = []; centroid_drifts = []; area_ratios = []
        if self.masks:
            prev = self.masks[0]
            py, px = center_of_mass(prev) if prev.any() else (0.0, 0.0)
            prev_area = max(1, int(prev.sum()))
            prev_overlaps.append(1.0); centroid_drifts.append(0.0); area_ratios.append(1.0)
            for k in range(1, len(self.masks)):
                curr = self.masks[k]
                overlap = self._iou(curr, prev); prev_overlaps.append(overlap)
                cy, cx = center_of_mass(curr) if curr.any() else (py, px)
                drift = float(np.hypot(cy - py, cx - px)); centroid_drifts.append(drift)
                curr_area = max(1, int(curr.sum())); area_ratios.append(curr_area / prev_area)
                prev, py, px, prev_area = curr, cy, cx, curr_area

        if single_mask_replicated and len(prev_overlaps) == len(self.masks):
            plt.figure(figsize=(8,3)); plt.plot(prev_overlaps, '-o', ms=3)
            plt.title("Tracking Health: IoU with Previous Frame"); plt.xlabel("Frame"); plt.ylabel("IoU (prev vs curr)")
            plt.grid(True); plt.tight_layout(); plt.savefig(self.results_dir / "prev_overlap_curve.png"); plt.show()

            plt.figure(figsize=(8,3)); plt.plot(centroid_drifts, '-o', ms=3)
            plt.title("Tracking Health: Centroid Drift (pixels)"); plt.xlabel("Frame"); plt.ylabel("Drift (px)")
            plt.grid(True); plt.tight_layout(); plt.savefig(self.results_dir / "centroid_drift_curve.png"); plt.show()

            plt.figure(figsize=(8,3)); plt.plot(area_ratios, '-o', ms=3)
            plt.title("Tracking Health: Area Ratio (curr / prev)"); plt.xlabel("Frame"); plt.ylabel("Area Ratio")
            plt.grid(True); plt.tight_layout(); plt.savefig(self.results_dir / "area_ratio_curve.png"); plt.show()

        highlight_indices = [0, self.n_frames // 2, self.n_frames - 1]
        highlight_indices = [i for i in highlight_indices if i < len(self.masks)]
        for idx in highlight_indices:
            frame = self.frames[idx]; pred = self.masks[idx]; gt = self.gt_masks[idx]
            overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
            both  = np.logical_and(gt, pred); extra = np.logical_and(pred == 1, gt == 0); miss  = np.logical_and(pred == 0, gt == 1)
            overlay[both]  = [255, 255, 255]; overlay[extra] = [0, 0, 255]; overlay[miss]  = [255, 0, 0]

            fig, axs = plt.subplots(1, 4, figsize=(16,4))
            axs[0].imshow(frame, cmap='gray'); axs[0].set_title("Original");     axs[0].axis('off')
            axs[1].imshow(gt, cmap='gray');    axs[1].set_title("Ground Truth");    axs[1].axis('off')
            axs[2].imshow(pred, cmap='gray');  axs[2].set_title("Prediction");   axs[2].axis('off')
            diff = np.logical_xor(pred, gt)
            abs_diff  = diff.sum() / diff.size
            norm_diff = 1.0 - abs_diff
            axs[3].imshow(overlay)
            axs[3].set_title(
                f"Overlay\nwhite=TP, blue=FP, red=FN\n"
                f"Accuracy={norm_diff:.3f} | Error={abs_diff:.3f}"
            )
            axs[3].axis('off')

            plt.tight_layout(); save_path = self.results_dir / f"compare_1x4_t{idx:03}.png"
            plt.savefig(save_path); print(f"Saved 1×4 comparison to: {save_path}"); plt.show()


