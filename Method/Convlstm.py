"""
ConvLSTM-based temporal segmentation with soft mask hints.

This module implements a two-stage model for video object segmentation:
(1) an encoder that fuses the grayscale image with a weak "hint" channel
    (seed mask or propagated mask), followed by a ConvLSTM that predicts a
    latent feature map for the next frame; and
(2) a UNet decoder that upsamples the latent map (concatenated with a
    downsampled grayscale target) into a per-pixel probability map.

Key features:
- Time-series cross-validation for training (`run`, `run_cross_validation`).
- Optional limited-hint simulation during training (first K frames per window).
- Test-time hint construction from the *first N ground-truth masks* with
  optical-flow propagation and exponential decay of hint strength.
- Simple postprocessing (Otsu + morphology) and framewise metrics.

Glossary:
hint
    A soft mask used as an auxiliary input channel. At test time it is
    built from the first N GT masks, optionally propagated and decayed.
decoded_edges
    Model outputs (H×W floats in [0,1]) prior to binarization.
cleaned mask
    A binarized + morphologically cleaned prediction.

"""

import os, re, time, random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import resize
from sklearn.metrics import jaccard_score
from sklearn.model_selection import TimeSeriesSplit


class LSTM_BASED:

    """
    ConvLSTM + UNet decoder for temporal binary segmentation with soft hints.

    Pipeline:
    1) Encoder: fuses [grayscale image, hint] → multi-scale latent (downsampled).
    2) ConvLSTM: autoregressively predicts next-frame latent features over time.
    3) Decoder (UNet): takes [predicted latent ⊕ downsampled target grayscale]
       and outputs a probability map for the target frame.
    4) Postprocessing: Otsu threshold + light morphology → final binary mask.

    Training uses time-series splits (no leakage) and can simulate limited
    annotations by exposing only the first K hints per sequence window.
    At test time, the first N GT masks can be used as seeds; they are
    optionally propagated with optical flow and exponentially decayed so the
    image channel remains influential.

    Attributes:
    syn_dir : pathlib.Path
        Directory with input frames for training.
    gt_mask_dir : pathlib.Path
        Directory with ground-truth masks aligned to frames.
    results_dir : pathlib.Path
        Output directory for predictions, figures, and CSVs.
    roi_x, roi_y, roi_w, roi_h : int
        ROI rectangle used to crop frames/masks for all processing.
    encoder, lstm_model, decoder : tf.keras.Model | None
        Trained sub-models (set after training or explicit loading).
    decoded_edges : np.ndarray | None
        Last set of decoder outputs (T, H, W) in [0,1].
    history : tf.keras.callbacks.History | None
        Training history for the ConvLSTM stage.
    """
    def __init__(
        self,
        syn_dir,
        gt_mask_dir,
        results_dir,
        roi_file=None,
        width=256,
        height=256,
        n_frames=200,
        window_size=5,
        lstm_units=128,
        batch_size=8,
        epochs=50,
        seed=42,
        force_full_frame=False,
        manual_roi=None,
        # training knobs (default: keep training strong)
        use_limited_hints=False,
        limited_hint_scale=0.6,
        limited_hint_N=None,
        # test-time knobs
        flow_warp=True,          # use optical flow to propagate first-N hints
        hint_min_scale=0.15,     # floor for decayed hints
        hint_decay_tau=50.0,      # decay length (frames)
    ):
        self.syn_dir = Path(syn_dir)
        self.gt_mask_dir = Path(gt_mask_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.roi_file = Path(roi_file) if roi_file else self.results_dir / "roi_coords.txt"

        self.width = width
        self.height = height
        self.n_frames = n_frames
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.force_full_frame = force_full_frame
        self.manual_roi = manual_roi
        self.training = True
        self.hint_scale_train = 0.35   # weaken hint during training
        self.hint_scale_test  = 0.10   # keep hint weak at test (set 0.0 for image-only)
        self.hint_dropout     = 0.10   # randomly drop hint entirely (training)
        self.hint_noise       = 0.02   # light Gaussian noise on hint (training)
        self.seq_overlap = 0.99  # sequence overlap ratio

        """
        Initialize the temporal segmentation model and basic I/O.

        Parameters:
       
        syn_dir : str | pathlib.Path
            Directory of training frames.
        gt_mask_dir : str | pathlib.Path
            Directory of ground-truth masks aligned with `syn_dir`.
        results_dir : str | pathlib.Path
            Output directory (created if missing).
        roi_file : str | pathlib.Path | None, default=None
            Text file with saved ROI coordinates (x, y, w, h). If not given,
            ROI is chosen via `select_roi()` (defaults to full frame).
        width, height : int, default=256
            Target frame size (used mainly for consistency; ROI can differ).
        n_frames : int, default=200
            Max number of frame–mask pairs to load for training.
        window_size : int, default=5
            Temporal window length for ConvLSTM sequences.
        lstm_units : int, default=128
            Hidden units for ConvLSTM (used inside `build_convlstm_model`).
        batch_size : int, default=8
            Batch size for training.
        epochs : int, default=50
            Max epochs for ConvLSTM training (early stopping enabled).
        seed : int, default=42
            Random seed for reproducibility.
        force_full_frame : bool, default=False
            If True, uses the full frame as ROI (ignores ROI file/manual ROI).
        manual_roi : tuple[int, int, int, int] | None, default=None
            Manually specified ROI `(x, y, w, h)`.
        use_limited_hints : bool, default=False
            If True, simulates limited supervision by keeping only the first
            K hints in each sliding window during training.
        limited_hint_scale : float, default=0.6
            Scaling applied to hints after the first K frames (when simulated).
        limited_hint_N : int | None, default=None
            K for limited hints; if None, uses `window_size`.
        flow_warp : bool, default=True
            Use optical flow to propagate hints forward at validation/test time.
        hint_min_scale : float, default=0.15
            Lower bound for decayed hint strength.
        hint_decay_tau : float, default=50.0
            Exponential decay constant (frames) for hint strength.
        """

        # Reproducibility
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Output dirs
        self.pred_masks_dir = self.results_dir / "predicted_masks"
        self.figures_dir = self.results_dir / "figures"
        self.pred_masks_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)

        # State
        self.all_imgs = None
        self.gt_paths = None
        self.roi_x = self.roi_y = self.roi_w = self.roi_h = None
        self.encoder = None
        self.lstm_model = None
        self.decoder = None
        self.history = None

        # LSTM/Decoder buffers
        self.h_lstm = self.w_lstm = self.c_lstm = None
        self.pred_encoded = None
        self.pred_encoded_plus_img = None
        self.decoder_targets = None
        self.decoded_edges = None
        self._last_indices_local = None  # local indices inside a val split
        self.split = 0  # for filenames (kept for compatibility)

        # knobs
        self.use_limited_hints = use_limited_hints
        self.limited_hint_scale = limited_hint_scale
        self.limited_hint_N = limited_hint_N

        self.flow_warp = flow_warp
        self.hint_min_scale = hint_min_scale
        self.hint_decay_tau = hint_decay_tau

    # ---------- Utilities ----------

    @staticmethod
    def apply_clahe(im):
        """Apply CLAHE to a normalized grayscale image [0,1]."""
        im_uint8 = (np.clip(im, 0, 1) * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(im_uint8)
        return eq.astype(np.float32) / 255.0

    def select_roi(self):
        if self.force_full_frame:
            self.roi_x, self.roi_y = 0, 0
            self.roi_h, self.roi_w = self.all_imgs.shape[1:3]
            print("Full-frame mode forced. ROI = image size.")
            return

        if self.manual_roi:
            self.roi_x, self.roi_y, self.roi_w, self.roi_h = map(int, self.manual_roi)
            print(f"Using manual ROI: {self.manual_roi}")
            return

        if self.roi_file.exists():
            self.roi_x, self.roi_y, self.roi_w, self.roi_h = np.loadtxt(self.roi_file, dtype=int)
            print(f"Loaded ROI from {self.roi_file}")
            return

        # Fallback to full frame if no GUI:
        self.roi_x, self.roi_y = 0, 0
        self.roi_h, self.roi_w = self.all_imgs.shape[1:3]
        print("No ROI file/GUI. Using full frame.")

    def crop_to_roi(self, image):
        return image[self.roi_y:self.roi_y + self.roi_h, self.roi_x:self.roi_x + self.roi_w]

    # ---------- Pairing / Loading (TRAIN) ----------

    @staticmethod
    def match_image_mask_files(img_dir, mask_dir, img_prefix="synth_frame_", mask_prefix="mask_"):
        """Pair image/mask paths by numeric suffix after given prefixes."""
        supported = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        imgs = [f for f in Path(img_dir).glob("*") if f.suffix.lower() in supported and f.name.startswith(img_prefix)]
        masks = [f for f in Path(mask_dir).glob("*") if f.suffix.lower() in supported and f.name.startswith(mask_prefix)]

        img_map = {f.stem.replace(img_prefix, ""): f for f in imgs if f.stem.replace(img_prefix, "").isdigit()}
        msk_map = {f.stem.replace(mask_prefix, ""): f for f in masks if f.stem.replace(mask_prefix, "").isdigit()}

        keys = sorted(set(img_map) & set(msk_map), key=lambda x: int(x))
        matched_imgs = [img_map[k] for k in keys]
        matched_masks = [msk_map[k] for k in keys]
        print(f"[TRAIN] Paired {len(keys)} image–mask files.")
        return matched_imgs, matched_masks

    # ---------- Mask processing ----------

    @staticmethod
    def get_particle_edges(image_crop, gt_crop):
        """
        Robust binarization of a GT mask and component filling. Returns float32 mask in {0,1}.
        """
        gt = gt_crop
        if gt.ndim == 3:
            gt = gt[..., 0]

        # Normalize to uint8
        if gt.dtype != np.uint8:
            scale = 255.0 if gt.max() <= 1.0 else 1.0
            gt8 = (gt.astype(np.float32) * scale).astype(np.uint8)
        else:
            gt8 = gt

        vals = np.unique(gt8)
        if vals.size <= 4:
            binary = (gt8 >= 128).astype(np.uint8)
        else:
            try:
                t = threshold_otsu(gt8)
            except ValueError:
                t = 1
            binary = (gt8 > t).astype(np.uint8)

        fg = binary.mean()
        if fg < 0.01 or fg > 0.99:
            binary = (1 - binary).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        num_labels, labels = cv2.connectedComponents(closed)
        filled = np.zeros_like(closed, dtype=np.uint8)
        for lbl in range(1, num_labels):
            filled[labels == lbl] = 1

        return filled.astype(np.float32)

    def build_filled_masks(self, frames, gt_paths):
        edges = []
        for i, frame in enumerate(frames):
            im = rgb2gray(frame) if frame.ndim == 3 else frame
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)
            crop = self.crop_to_roi(im)

            gt = imread(str(gt_paths[i]))
            if gt.ndim == 3:
                gt = gt[..., 0]
            gt_crop = self.crop_to_roi(gt)
            edges.append(self.get_particle_edges(crop, gt_crop))
        return np.stack(edges)

    def simulate_limited_hints(self, edges, N_hint=5, edge_scale=0.3):
        """
        Keep only the first N_hint frames' true hints per sliding window.
        """
        num_frames = len(edges)
        window = self.window_size
        step = max(1, int(round(window * (1 - 0.5))))  # default overlap=0.5

        edges_mod = edges.copy()
        for seq_start in range(0, num_frames - window, step):
            seq_end = seq_start + window
            last_hint = edges_mod[seq_start + N_hint - 1]
            for i in range(seq_start + N_hint, seq_end):
                edges_mod[i] = last_hint * edge_scale

        return edges_mod

    # ---------- Models ----------

    def build_encoder(self):
        """Two-channel encoder: [grayscale image, hint]."""
        inputs = layers.Input((self.roi_h, self.roi_w, 2))
        x1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
        x2 = layers.MaxPooling2D()(x1)
        x3 = layers.Conv2D(32, 3, activation='relu', padding='same')(x2)
        x4 = layers.MaxPooling2D()(x3)
        x5 = layers.Conv2D(64, 3, activation='relu', padding='same')(x4)

        target_h = max(1, self.roi_h // 4)
        target_w = max(1, self.roi_w // 4)
        x1_up = layers.Resizing(target_h, target_w)(x1)
        x3_up = layers.Resizing(target_h, target_w)(x3)
        x5_up = layers.Resizing(target_h, target_w)(x5)
        merged = layers.Concatenate()([x1_up, x3_up, x5_up])
        self.encoder = models.Model(inputs, merged)

    def encode_features(self, images, edges):
        """images/edges: (N,H,W) in [0,1]. Returns (N,h,w,c)."""
        assert images.shape == edges.shape, f"{images.shape} vs {edges.shape}"
        feats = []
        for i in range(len(images)):
            hint = np.clip(edges[i], 0.0, 1.0).astype(np.float32)

            # --- training-time corruption so model can't just copy hints ---
            if self.training and self.hint_dropout > 0 and np.random.rand() < self.hint_dropout:
                hint = np.zeros_like(hint, dtype=np.float32)
            if self.training and self.hint_noise > 0:
                hint = np.clip(hint + self.hint_noise*np.random.randn(*hint.shape).astype(np.float32), 0.0, 1.0)

            scale = self.hint_scale_train if self.training else self.hint_scale_test
            inp = np.stack([images[i], scale * hint], axis=-1)[None, ...]  # (1,H,W,2)
            feats.append(self.encoder.predict(inp, verbose=0)[0])
        return np.stack(feats, axis=0)

    @staticmethod
    def make_sequences(data, window, overlap=0.5, window_type=None, return_indices=False):
        X, y, y_idx = [], [], []
        step = max(1, int(round(window * (1 - overlap))))
        if window_type:
            weights = np.asarray(np.hanning(window))[:, None, None, None]
        else:
            weights = np.ones((window, 1, 1, 1))
        for i in range(0, len(data) - window, step):
            x_seq = data[i:i + window] * weights
            y_frame = data[i + window]
            X.append(x_seq); y.append(y_frame); y_idx.append(i + window)
        X = np.stack(X); y = np.stack(y); y_idx = np.array(y_idx, dtype=int)
        return (X, y, y_idx) if return_indices else (X, y)

    def build_convlstm_model(self, input_shape):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.ConvLSTM2D(64, (3,3), padding='same', return_sequences=True,
                              dropout=0.3, recurrent_dropout=0.3),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(64, (3,3), padding='same',
                              dropout=0.3, recurrent_dropout=0.3),
            layers.Dropout(0.3),
            layers.Conv2D(self.c_lstm, (1,1), activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def dice_loss(y_true, y_pred):
        smooth = 1e-6
        intersection = tf.reduce_sum(y_true * y_pred)
        return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

    @staticmethod
    def dice_bce_loss(y_true, y_pred):
        return tf.keras.losses.binary_crossentropy(y_true, y_pred) + LSTM_BASED.dice_loss(y_true, y_pred)

    def build_unet_decoder(self, input_shape):
        """
        UNet decoder. Input is **(latent ⊕ downsampled target grayscale)** with channels=input_shape[-1].
        """
        roi_h, roi_w = self.roi_h, self.roi_w

        def crop_to_match(skip, target):
            ch, cw = skip.shape[1] - target.shape[1], skip.shape[2] - target.shape[2]
            return layers.Cropping2D(((ch // 2, ch - ch // 2), (cw // 2, cw - cw // 2)))(skip)

        inputs = layers.Input(shape=input_shape)
        c1 = layers.Conv2D(64, 3, padding='same')(inputs); c1 = layers.BatchNormalization()(c1); c1 = layers.Activation('relu')(c1)
        c1 = layers.Conv2D(64, 3, padding='same')(c1);     c1 = layers.BatchNormalization()(c1); c1 = layers.Activation('relu')(c1)
        c1 = layers.Dropout(0.3)(c1); p1 = layers.MaxPooling2D()(c1)

        c2 = layers.Conv2D(128, 3, padding='same')(p1); c2 = layers.BatchNormalization()(c2); c2 = layers.Activation('relu')(c2)
        c2 = layers.Conv2D(128, 3, padding='same')(c2); c2 = layers.BatchNormalization()(c2); c2 = layers.Activation('relu')(c2)
        p2 = layers.MaxPooling2D()(c2)

        c3 = layers.Conv2D(256, 3, padding='same')(p2); c3 = layers.BatchNormalization()(c3); c3 = layers.Activation('relu')(c3)
        c3 = layers.Conv2D(256, 3, padding='same')(c3); c3 = layers.BatchNormalization()(c3); c3 = layers.Activation('relu')(c3)
        p3 = layers.MaxPooling2D()(c3)

        b = layers.Conv2D(512, 3, padding='same')(p3); b = layers.BatchNormalization()(b); b = layers.Dropout(0.3)(b); b = layers.Activation('relu')(b)

        u3 = layers.UpSampling2D()(b); u3 = layers.Concatenate()([u3, crop_to_match(c3, u3)])
        c6 = layers.Conv2D(256, 3, padding='same')(u3); c6 = layers.BatchNormalization()(c6); c6 = layers.Activation('relu')(c6)

        u2 = layers.UpSampling2D()(c6); u2 = layers.Concatenate()([u2, crop_to_match(c2, u2)])
        c7 = layers.Conv2D(128, 3, padding='same')(u2); c7 = layers.BatchNormalization()(c7); c7 = layers.Activation('relu')(c7)

        u1 = layers.UpSampling2D()(c7); u1 = layers.Concatenate()([u1, crop_to_match(c1, u1)])
        c8 = layers.Conv2D(64, 3, padding='same')(u1); c8 = layers.BatchNormalization()(c8); c8 = layers.Activation('relu')(c8)

        x = layers.Resizing(roi_h, roi_w)(c8)
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
        return models.Model(inputs, outputs)

    def train_lstm(self, use_validation=False):
        assert self.X_train_lstm.ndim == 5, f"LSTM input must be 5D, got {self.X_train_lstm.shape}"
        print("[Timing] LSTM training...")
        start = time.time()
        val_data = (self.X_val_lstm, self.y_val_lstm) if use_validation else None
        self.history = self.lstm_model.fit(
            self.X_train_lstm, self.y_train_lstm,
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=2,
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        m, s = divmod(time.time()-start, 60)
        print(f"[Timing] LSTM done in {int(m)}m {s:.1f}s.")

    def train_decoder(self):
        """
        Train decoder with concatenated latent+image input.
        Expects:
          - self.pred_encoded_plus_img: (N,h,w,c+1)
          - self.decoder_targets:       (N,H,W,1)
        """
        assert self.pred_encoded_plus_img is not None, "pred_encoded_plus_img not set."
        self.decoder = self.build_unet_decoder(self.pred_encoded_plus_img.shape[1:])
        self.decoder.compile(optimizer='adam', loss=self.dice_bce_loss)

        if self.decoder_targets.ndim == 3:
            self.decoder_targets = self.decoder_targets[..., np.newaxis]

        exp = (self.pred_encoded_plus_img.shape[0], self.roi_h, self.roi_w, 1)
        assert self.decoder_targets.shape == exp, f"Decoder target {self.decoder_targets.shape} != {exp}"

        print(f"[Decoder] fit X={self.pred_encoded_plus_img.shape}, y={self.decoder_targets.shape}")
        self.decoder.fit(
            self.pred_encoded_plus_img, self.decoder_targets,
            epochs=30, batch_size=8, verbose=2,
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        self.decoded_edges = self.decoder.predict(self.pred_encoded_plus_img, verbose=0).squeeze()

    # ---------- Postproc / Metrics ----------

    @staticmethod
    def clean_edges(pred_edge_map, thresh=0.30, min_area=400):
        norm = pred_edge_map.astype(np.float32)
        if norm.max() > 1.0 or norm.min() < 0.0:
            norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)

        # Otsu first
        norm8 = (norm * 255).astype(np.uint8)
        t, _ = cv2.threshold(norm8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fixed = (norm8 >= t).astype(np.uint8)
        if fixed.sum() == 0 and norm.max() > 0:
            fixed = (norm > thresh).astype(np.uint8)

        # light cleanup
        if fixed.sum() > 0:
            fixed = cv2.morphologyEx(fixed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        res = cv2.findContours(fixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]
        mask = np.zeros_like(fixed)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(mask, [cnt], -1, 1, -1)
        return mask

    @staticmethod
    def compute_metrics(gt_mask, pred_mask):
        gt = gt_mask.astype(np.uint8).flatten()
        pr = pred_mask.astype(np.uint8).flatten()
        inter = np.sum(gt * pr)
        s = np.sum(gt) + np.sum(pr)
        dice = 1.0 if s == 0 else (2.0 * inter) / s
        iou  = jaccard_score(gt, pr, average='binary', zero_division=1)
        return dice, iou

    

    def visualize_pixelwise_difference(self, idx, gt_is_clean_binary=True, original_name=None, save=True):
        e = self.decoded_edges[idx]
        pred_mask = self.clean_edges(e).astype(np.uint8)

        gt = imread(str(self.gt_paths[idx]))
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt_crop = self.crop_to_roi(gt)
        im_crop = self.crop_to_roi(self.all_imgs[idx])

        if gt_is_clean_binary:
            gt_bin = (gt_crop > 127).astype(np.uint8)
        else:
            gt_edge = self.get_particle_edges(im_crop, gt_crop)
            gt_bin = (gt_edge > 0).astype(np.uint8)

        if pred_mask.shape != gt_bin.shape:
            pred_mask = resize(pred_mask, gt_bin.shape, order=0, preserve_range=True).astype(np.uint8)

        both    = (gt_bin > 0) & (pred_mask > 0)
        extra   = (pred_mask > 0) & ~(gt_bin > 0)
        missing = ~(pred_mask > 0) & (gt_bin > 0)

        overlay = np.zeros((*gt_bin.shape, 3), dtype=np.uint8)
        overlay[both]    = [255, 255, 255]
        overlay[extra]   = [0, 0, 255]
        overlay[missing] = [255, 0, 0]

        #diff = np.logical_xor(gt_bin > 0, pred_mask > 0)
        #norm_diff = 1.0 - (diff.sum() / diff.size)
        
        diff = np.logical_xor(gt_bin > 0, pred_mask > 0)
        pixel_error    = diff.sum() / diff.size
        pixel_accuracy = 1.0 - pixel_error

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(im_crop, cmap='gray'); axs[0].set_title("Original"); axs[0].axis("off")
        axs[1].imshow(gt_bin, cmap='gray');  axs[1].set_title("Ground Truth"); axs[1].axis("off")
        axs[2].imshow(pred_mask, cmap='gray'); axs[2].set_title("Prediction"); axs[2].axis("off")
        axs[3].imshow(overlay); axs[3].set_title(f"Overlay\nwhite=TP, blue=FP, red=FN\n" f"Accuracy={pixel_accuracy:.3f} | Error={pixel_error:.3f}"); axs[3].axis("off")
        plt.tight_layout()

        if save:
            stem = Path(original_name).stem if original_name else f"frame_{idx:03d}"
            out = self.figures_dir / f"overlay_fp_fn_{stem}.png"
            plt.savefig(out, dpi=150)
            print(f"[Saved] {out}")
        plt.show()

    def evaluate(self, verbose_plots=True, original_filenames=None, gt_is_clean_binary=True, x_indices=None):

        dice_scores, iou_scores, pixel_accuracies, pixel_errors = [], [], [], []
        highlight = [0, len(self.decoded_edges)//2, len(self.decoded_edges)-1] if verbose_plots else []

        for i, (e, gt_path) in enumerate(zip(self.decoded_edges, self.gt_paths)):
            pred_mask = self.clean_edges(e).astype(np.uint8)

            gt = imread(str(gt_path))
            if gt.ndim == 3:
                gt = gt[..., 0]
            gt_crop = self.crop_to_roi(gt)
            im_crop = self.crop_to_roi(self.all_imgs[i])

            if gt_is_clean_binary:
                gt_binary = (gt_crop > 127).astype(np.uint8)
            else:
                gt_edge = self.get_particle_edges(im_crop, gt_crop)
                gt_binary = (gt_edge > 0).astype(np.uint8)

            if pred_mask.shape != gt_binary.shape:
                pred_mask = resize(pred_mask, gt_binary.shape, order=0, preserve_range=True).astype(np.uint8)

            dice, iou = self.compute_metrics(gt_binary, pred_mask)
            diff = np.logical_xor(gt_binary > 0, pred_mask > 0)
            pixel_error    = diff.sum() / diff.size
            pixel_accuracy = 1.0 - pixel_error

            if verbose_plots and i in highlight:
                label = (original_filenames[i] if original_filenames and i < len(original_filenames)
                        else f"Frame {i:03d}")
                print(f"{label}: Dice={dice:.4f}, IoU={iou:.4f}, PixelError={pixel_error:.4f}")
                self.visualize_pixelwise_difference(
                    idx=i,
                    gt_is_clean_binary=gt_is_clean_binary,
                    original_name=label,
                    save=True
                )

            dice_scores.append(dice)
            iou_scores.append(iou)
            pixel_accuracies.append(pixel_accuracy)
            pixel_errors.append(pixel_error)

        print(f"Average Dice: {np.mean(dice_scores):.4f} | IoU: {np.mean(iou_scores):.4f} "
            f"| Pixel Accuracy: {np.mean(pixel_accuracies):.4f} | Pixel Error: {np.mean(pixel_errors):.4f}")

        if verbose_plots:
            # choose x and label once
            if x_indices is None:
                x = np.arange(len(dice_scores))
                xlabel = f"Prediction # (n={len(x)})"
            else:
                x = np.asarray(x_indices, dtype=int)
                xmin, xmax = int(x.min()), int(x.max())
                xlabel = f"Frame index ({xmin}–{xmax}, n={len(x)})"

            fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
            ax.set_ylabel("Score")
            ax.set_xlabel(xlabel)

            ax.plot(x, dice_scores,        label="Dice",           linewidth=2)
            ax.plot(x, iou_scores,         label="IoU",            linewidth=2)
            ax.plot(x, pixel_accuracies,   label="Pixel Accuracy", linewidth=2)
            # If you prefer to show error instead (lower is better), plot it in a separate figure.

            xmin, xmax = int(np.min(x)), int(np.max(x))
            if xmin == xmax:
                xmin -= 0.5; xmax += 0.5
            ax.set_xlim(xmin, xmax)

            nticks = min(6, max(2, len(np.unique(x))))
            ax.set_xticks(np.linspace(xmin, xmax, nticks, dtype=int))
            ax.set_ylim(0.3, 1.02)
            ax.set_yticks(np.arange(0.3, 1.03, 0.20))
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(frameon=True, loc="lower left")

            out = self.figures_dir / "dice_iou_per_frame"
            plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
            plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

        return dice_scores, iou_scores, pixel_accuracies, pixel_errors



    # ---------- Training loop ----------

    def run(self, img_prefix="synth_frame_", mask_prefix="mask_", n_splits=5):
        """
        Train with time-series cross-validation and save artifacts.

        """
        img_paths, mask_paths = self.match_image_mask_files(
            self.syn_dir, self.gt_mask_dir, img_prefix=img_prefix, mask_prefix=mask_prefix
        )
        if len(img_paths) == 0:
            raise FileNotFoundError(f"No paired image/mask files in {self.syn_dir} / {self.gt_mask_dir}")
        if len(img_paths) < self.n_frames:
            print(f"[TRAIN] Only {len(img_paths)} pairs found. Adjusting n_frames.")
            self.n_frames = len(img_paths)

        imgs = []
        for fn in img_paths[:self.n_frames]:
            im = imread(str(fn))
            if im.ndim == 3:
                im = rgb2gray(im[..., :3])
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)
            im = self.apply_clahe(im)
            imgs.append(im)
        self.all_imgs = np.stack(imgs)
        self.gt_paths = mask_paths[:self.n_frames]

        self.select_roi()

        start_time = time.time()
        self.run_cross_validation(n_splits=n_splits)
        print(f"[Timing] Train+Val total: {time.time() - start_time:.2f} s")

    def run_cross_validation(self, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_imgs = np.array(self.all_imgs)
        all_masks = np.array(self.gt_paths)

        all_fold_metrics = []
        all_dice, all_iou, all_pixel_acc = [], [], []
        fold_num = 1

        for train_idx, val_idx in tscv.split(all_imgs):
            print(f"\n=== Fold {fold_num} ===")
            train_imgs, val_imgs = all_imgs[train_idx], all_imgs[val_idx]
            train_masks, val_masks = all_masks[train_idx], all_masks[val_idx]
            self.train_fold(train_imgs, train_masks, val_imgs, val_masks, fold_num, val_idx)

            d, j, n, a = self.evaluate_per_fold(val_imgs, val_masks, fold_num)
            all_dice.extend(d); all_iou.extend(j); all_pixel_acc.extend(n)

            all_fold_metrics.append({
                "Fold": fold_num,
                "Dice": np.mean(d),
                "IoU": np.mean(j),
                "PixelAccuracy": np.mean(n),
                "PixelError": np.mean(a)
            })

            fold_num += 1

        pd.DataFrame(all_fold_metrics).to_csv(self.results_dir / "fold_summary.csv", index=False)
        print("\n====== FINAL (All Folds) ======")
        print(f"Mean Dice: {np.mean(all_dice):.4f} | Mean IoU: {np.mean(all_iou):.4f} | "
            f"Mean PixelAccuracy: {np.mean(all_pixel_acc):.4f}")


    def train_fold(self, train_imgs, train_gts, val_imgs, val_gts, fold_num, val_idx):
        """
        One fold: build hints, fit ConvLSTM, train decoder on TRAIN only, infer VAL.

        Steps
        -----
        1) Build TRAIN hints from GT (optionally simulate limited hints).
        2) Build encoder and encode [image, hint] → latent sequences.
        3) Train ConvLSTM on TRAIN (validate on VAL features).
        4) Train decoder on TRAIN (targets = TRAIN hints).
        5) Infer VAL with decoder; store `decoded_edges` and save masks.
        6) Save models for this fold.
        """
        # 1) Build TRAIN hints from GT (ok to use full GT on TRAIN)
        train_edges = self.build_filled_masks(train_imgs, train_gts)

        if self.use_limited_hints:
            N_hint = self.limited_hint_N or self.window_size
            # safety clamp
            N_hint = max(1, min(N_hint, self.window_size))
            train_edges = self.simulate_limited_hints(train_edges, N_hint=N_hint, edge_scale=self.limited_hint_scale)

        # 2) Build encoder (expects ROI already chosen in run()->select_roi)
        self.build_encoder()

        # 3) Crop images to ROI
        train_imgs_roi = np.stack([self.crop_to_roi(im) for im in train_imgs])
        val_imgs_roi   = np.stack([self.crop_to_roi(im) for im in val_imgs])

        # ----------------- NEW: leak-free validation hints (first-K + flow + decay) -----------------
        # Choose K to mirror your test usage (you typically use first 5); fallback to window_size
        K = self.limited_hint_N or self.window_size
        K = max(1, min(K, len(val_imgs)))  # clamp

        # seed from GT for first K frames ONLY
        seed_edges = self.build_filled_masks(val_imgs[:K], val_gts[:K])
        val_edges = np.zeros((len(val_imgs), self.roi_h, self.roi_w), dtype=np.float32)
        val_edges[:K] = seed_edges

        # propagate from K to end via optical flow (or hold-last if flow_warp=False)
        if self.flow_warp:
            for t in range(K, len(val_imgs)):
                prev_img  = val_imgs_roi[t-1]
                curr_img  = val_imgs_roi[t]
                prev_hint = val_edges[t-1]
                val_edges[t] = self._warp_hint_by_flow(prev_img, curr_img, prev_hint)
        else:
            if len(val_imgs) > K:
                val_edges[K:] = val_edges[K-1]

        # decay hint strength over time (same idea as test)
        base = 0.3  # match your test edge_scale default; adjust if you use a different value
        tau  = float(self.hint_decay_tau)
        mn   = float(self.hint_min_scale)
        scales = np.empty(len(val_edges), dtype=np.float32)
        for t in range(len(val_edges)):
            if t < K:
                scales[t] = base
            else:
                scales[t] = max(mn, base * np.exp(-(t - (K - 1)) / tau))
        val_edges = (val_edges * scales[:, None, None]).astype(np.float32)
        # --------------------------------------------------------------------------------------------

        # 4) Encode [image, hint] with encoder
        self.training = True
        train_encoded = self.encode_features(train_imgs_roi, train_edges)

        # IMPORTANT: for VAL we already pre-scaled hints (above). Avoid double-weakening inside encoder.
        self.training = False
        _prev = self.hint_scale_test
        self.hint_scale_test = 1.0
        val_encoded = self.encode_features(val_imgs_roi, val_edges)
        self.hint_scale_test = _prev


        # restore for the rest of training
        self.training = True

        # 5) Make sequences (keep indices)
        train_X, train_y, train_y_idx = self.make_sequences(train_encoded, self.window_size, overlap=self.seq_overlap, return_indices=True)
        val_X,   val_y,   val_y_idx   = self.make_sequences(val_encoded,   self.window_size, overlap=self.seq_overlap, return_indices=True)

        # 6) Train ConvLSTM on TRAIN (validate with VAL)
        self.X_train_lstm, self.y_train_lstm = train_X, train_y
        self.X_val_lstm,   self.y_val_lstm   = val_X,   val_y
        h, w, c = train_y.shape[1:]
        self.h_lstm, self.w_lstm, self.c_lstm = h, w, c
        self.lstm_model = self.build_convlstm_model((self.window_size, h, w, c))
        self.train_lstm(use_validation=True)

        # 7) ---- Train decoder on TRAIN ONLY ----
        pred_encoded_train = self.lstm_model.predict(train_X, verbose=0)  # (N_train,h,w,c)
        train_target_imgs_roi = train_imgs_roi[train_y_idx]               # (N_train,H,W)
        img_ds_train = np.stack(
            [resize(fr, (h, w), order=1, preserve_range=True) for fr in train_target_imgs_roi],
            axis=0
        )[..., None].astype(np.float32)                                    # (N_train,h,w,1)

        self.pred_encoded_plus_img = np.concatenate([pred_encoded_train, img_ds_train], axis=-1)  # (N,h,w,c+1)
        self.decoder_targets = train_edges[train_y_idx][..., np.newaxis]                           # (N,H,W,1)
        self.train_decoder()  # fits decoder on TRAIN

        # 8) ---- Validation: infer ONLY ----
        pred_encoded_val = self.lstm_model.predict(val_X, verbose=0)      # (N_val,h,w,c)
        val_target_imgs_roi = val_imgs_roi[val_y_idx]
        img_ds_val = np.stack(
            [resize(fr, (h, w), order=1, preserve_range=True) for fr in val_target_imgs_roi],
            axis=0
        )[..., None].astype(np.float32)                                    # (N_val,h,w,1)

        pred_encoded_plus_img_val = np.concatenate([pred_encoded_val, img_ds_val], axis=-1)
        self.decoded_edges = self.decoder.predict(pred_encoded_plus_img_val, verbose=0).squeeze()

        # 9) Bookkeeping / save predictions for this fold
        self._last_indices_local = val_y_idx
        val_global_idx = val_idx[val_y_idx]
        self.save_predictions(indices=val_global_idx, subdir=f"predicted_masks_fold{fold_num}")

        # 10) Save models
        self.lstm_model.save(self.results_dir / f"lstm_model_fold{fold_num}.h5")
        self.decoder.save(self.results_dir / f"unet_decoder_fold{fold_num}.h5")
        self.encoder.save(self.results_dir / f"encoder_fold{fold_num}.h5")


    def save_predictions(self, indices=None, subdir="predicted_masks"):
        """
        Save raw decoder probabilities and the cleaned binary masks.
        If `indices` is provided, use those global frame indices for filenames.
        """
        if self.decoded_edges is None:
            print("[save_predictions] Nothing to save: decoded_edges is None.")
            return

        out_dir = self.results_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        decoded = np.asarray(self.decoded_edges)
        if decoded.ndim == 2:
            decoded = decoded[None, ...]
        assert decoded.ndim == 3, f"decoded_edges should be (N,H,W), got {decoded.shape}"

        for i, e in enumerate(decoded):
            idx = int(indices[i]) if indices is not None else (i + self.split + self.window_size)
            imsave(out_dir / f"pred_raw_{idx:03d}.png", (np.clip(e, 0, 1) * 255).astype(np.uint8))
            mask = self.clean_edges(e)
            imsave(out_dir / f"pred_mask_{idx:03d}.png", (mask * 255).astype(np.uint8))

    def evaluate_per_fold(self, val_imgs, val_gts, fold_num):
        idx = getattr(self, "_last_indices_local", None)
        if idx is None: idx = np.arange(len(val_imgs))
        self.all_imgs = val_imgs[idx]
        self.gt_paths = val_gts[idx]
        d, j, n, a = self.evaluate(verbose_plots=False)
        self._append_framewise_metrics(d, j, n, a, fold_num=fold_num, window_size=self.window_size)
        return d, j, n, a

    def _append_framewise_metrics(self, dice, iou, norm, absd, fold_num, window_size):
        df = pd.DataFrame({
            "FrameIndex": list(range(len(dice))),
            "Dice": dice, "IoU": iou,
            "PixelAccuracy": norm, "PixelError": absd,
            "Fold": fold_num, "WindowSize": window_size
        })

        out_csv = self.results_dir / "framewise_metrics.csv"
        if out_csv.exists():
            old = pd.read_csv(out_csv); df = pd.concat([old, df], ignore_index=True)
        df.to_csv(out_csv, index=False)

    # ---------- TTA + Testing ----------

    def tta_predict(self, encoded):
        test_X, _, y_idx = self.make_sequences(encoded, self.window_size, overlap=self.seq_overlap, return_indices=True)
        pred_orig = self.lstm_model.predict(test_X, verbose=0)
        test_X_flip = np.flip(test_X, axis=3)  # flip width
        pred_flip = self.lstm_model.predict(test_X_flip, verbose=0)
        pred_flip = np.flip(pred_flip, axis=2)
        pred_avg = (pred_orig + pred_flip) / 2.0
        return pred_avg, y_idx

    def _warp_hint_by_flow(self, prev_img, curr_img, prev_hint):
        """
        Warp prev_hint from t-1 to t using Farnebäck optical flow computed on images.
        Inputs are ROI-size (H,W) floats in [0,1]. Returns float32 (H,W) in [0,1].
        """
        # convert to uint8 for flow
        I0 = (np.clip(prev_img, 0, 1) * 255).astype(np.uint8)
        I1 = (np.clip(curr_img, 0, 1) * 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            I0, I1, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        h, w = I0.shape
        # flow gives (dx, dy) at each pixel mapping I0->I1
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        prev_hint_u8 = (np.clip(prev_hint, 0, 1) * 255).astype(np.uint8)
        warped = cv2.remap(prev_hint_u8, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return (warped.astype(np.float32) / 255.0)
    
    
    def predict_test_dataset(
        self,
        test_img_dir,
        test_gt_dir=None,
        evaluate_test=False,
        original_filenames=None,
        use_only_first_mask=False,
        use_first_n_masks=None,
        edge_scale=0.3,              # initial hint strength at test time
        gt_is_clean_binary=True,     # for evaluation
        return_arrays=False
    ):
        """
        Predict on a test dataset using the trained model.
        Uses optional optical-flow propagation for hints when only the first N masks are available,
        and decays hint strength over time so the image channel matters.
        """
        if self.encoder is None or self.lstm_model is None or self.decoder is None:
            raise RuntimeError("Load encoder/lstm/decoder .h5 before predicting.")

        SUP = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

        def _key(p: Path):
            m = re.findall(r'\d+', p.stem)
            return m[-1] if m else None

        self.training = False
        test_img_dir = Path(test_img_dir)
        test_gt_dir = Path(test_gt_dir) if test_gt_dir else None

        # Map files by numeric key
        img_map = {_key(f): f for f in test_img_dir.glob("*") if f.suffix.lower() in SUP and _key(f)}
        common_keys = sorted(img_map, key=lambda x: int(x))
        if not common_keys:
            raise FileNotFoundError(f"No test frames matched in {test_img_dir}")

        # for evaluation only; full GT map
        mask_map_all = {}
        if test_gt_dir:
            mask_map_all = {_key(f): f for f in test_gt_dir.glob("*")
                            if f.suffix.lower() in SUP and _key(f)}

        mask_map = {}
        if test_gt_dir:
            mask_map = {_key(f): f for f in test_gt_dir.glob("*") if f.suffix.lower() in SUP and _key(f)}

        # Load images in order
        test_imgs, ordered_names = [], []
        for k in common_keys:
            im = imread(str(img_map[k]))
            if im.ndim == 3:
                im = rgb2gray(im[..., :3])
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)
            im = self.apply_clahe(im)
            test_imgs.append(im)
            ordered_names.append(img_map[k].name)
        test_imgs = np.stack(test_imgs)
        if original_filenames is None:
            original_filenames = ordered_names

        # ROI (default to full frame if unset)
        if self.roi_x is None or self.roi_w is None:
            self.roi_x = self.roi_y = 0
            self.roi_h, self.roi_w = test_imgs.shape[1:3]

        # Build hint channel
        test_edges = np.zeros((len(test_imgs), self.roi_h, self.roi_w), dtype=np.float32)
        self.gt_paths = None

        # --------- HINT CONSTRUCTION (with optional flow + decay when only first N GT masks exist) ----------
        if use_first_n_masks is not None:
            if not test_gt_dir:
                raise ValueError("use_first_n_masks needs test_gt_dir.")
            N = int(use_first_n_masks)
            miss = [k for k in common_keys[:N] if k not in mask_map]
            if miss:
                raise FileNotFoundError(f"Missing masks for keys: {miss[:10]}")

            # build hints for first N frames
            firstN_imgs  = [test_imgs[i] for i in range(N)]
            firstN_paths = [mask_map[k] for k in common_keys[:N]]
            firstN_edges = self.build_filled_masks(firstN_imgs, firstN_paths)
            test_edges[:N] = firstN_edges

            # ROI images once (encoder sees ROI)
            test_imgs_roi = np.stack([self.crop_to_roi(im) for im in test_imgs])

            # propagate N-1 → end using optical flow
            if self.flow_warp:
                for t in range(N, len(test_imgs)):
                    prev_img  = test_imgs_roi[t-1]
                    curr_img  = test_imgs_roi[t]
                    prev_hint = test_edges[t-1]
                    test_edges[t] = self._warp_hint_by_flow(prev_img, curr_img, prev_hint)
            else:
                # no flow: just hold last hint
                if len(test_imgs) > N:
                    test_edges[N:] = firstN_edges[-1]

            # decay hint strength so the image matters more over time
            base = float(edge_scale)
            min_scale = float(self.hint_min_scale)
            tau = float(self.hint_decay_tau)
            scales = np.ones(len(test_edges), dtype=np.float32) * base
            for t in range(N, len(test_edges)):
                decay = base * np.exp(-(t - (N - 1)) / tau)
                scales[t] = max(min_scale, float(decay))
            for t in range(len(test_edges)):
                test_edges[t] = (test_edges[t].astype(np.float32) * scales[t]).astype(np.float32)

        elif test_gt_dir and use_only_first_mask:
            if not mask_map:
                raise FileNotFoundError(f"No masks in {test_gt_dir}")
            first_key = sorted(mask_map, key=lambda x: int(x))[0]
            edge0 = self.build_filled_masks([test_imgs[0]], [mask_map[first_key]])[0]
            if edge0.shape != (self.roi_h, self.roi_w):
                edge0 = resize(edge0, (self.roi_h, self.roi_w), order=1, preserve_range=True)
            test_edges[0] = edge0

            test_imgs_roi = np.stack([self.crop_to_roi(im) for im in test_imgs])

            if self.flow_warp:
                for t in range(1, len(test_imgs)):
                    prev_img  = test_imgs_roi[t-1]
                    curr_img  = test_imgs_roi[t]
                    prev_hint = test_edges[t-1]
                    test_edges[t] = self._warp_hint_by_flow(prev_img, curr_img, prev_hint)
            else:
                if len(test_imgs) > 1:
                    test_edges[1:] = edge0

            test_edges = (test_edges * float(edge_scale)).astype(np.float32)

            if evaluate_test:
                self.gt_paths = [mask_map[first_key]]
                if self.window_size >= 1:
                    print("[WARN] Only 1 GT mask; skipping evaluation.")
                    evaluate_test = False

        elif test_gt_dir:
            # Use whatever GT masks exist (no propagation)
            both = [k for k in common_keys if k in mask_map]
            if both:
                imgs_for_edges = [test_imgs[common_keys.index(k)] for k in both]
                gts_for_edges  = [mask_map[k] for k in both]
                built = self.build_filled_masks(imgs_for_edges, gts_for_edges)
                pos = {k: i for i, k in enumerate(both)}
                for i, k in enumerate(common_keys):
                    if k in pos:
                        test_edges[i] = built[pos[k]]
            if evaluate_test:
                self.gt_paths = [mask_map[k] for k in both]

        print("[DEBUG] test_edges stats per first frames:",
            [(float(test_edges[i].min()), float(test_edges[i].mean()), float(test_edges[i].max()))
            for i in range(min(5, len(test_edges)))])

        # --------- Encode → LSTM (TTA) → Decode ----------
        test_imgs_roi = np.stack([self.crop_to_roi(im) for im in test_imgs])

        # Avoid double-weakening of hints at TEST (mirror validation behavior)
        _prev = self.hint_scale_test
        self.hint_scale_test = 1.0
        try:
            test_encoded = self.encode_features(test_imgs_roi, test_edges)
        finally:
            self.hint_scale_test = _prev

        if len(test_encoded) <= self.window_size:
            raise ValueError(f"Need at least {self.window_size+1} frames; got {len(test_encoded)}.")
        pred_encoded, y_idx = self.tta_predict(test_encoded)  # (Npred,h,w,c)

        # concat downsampled target grayscale for those y_idx frames
        h2, w2 = pred_encoded.shape[1:3]
        target_imgs = test_imgs_roi[y_idx]  # (Npred, H, W)
        img_ds = np.stack([resize(fr, (h2, w2), order=1, preserve_range=True)
                        for fr in target_imgs], axis=0)[..., None].astype(np.float32)  # (Npred,h2,w2,1)
        pred_encoded_plus_img = np.concatenate([pred_encoded, img_ds], axis=-1)  # (Npred,h2,w2,c+1)

        decoded_masks = self.decoder.predict(pred_encoded_plus_img, verbose=0).squeeze()
        self.decoded_edges = decoded_masks
        print("[DEBUG] decoded min/mean/max:",
            float(self.decoded_edges.min()),
            float(self.decoded_edges.mean()),
            float(self.decoded_edges.max()))

        # --------- Save predictions (use true y indices) ----------
        out_dir = self.results_dir / "test_predictions"
        out_dir.mkdir(exist_ok=True)
        for i, e in enumerate(decoded_masks):
            idx = int(y_idx[i])
            imsave(out_dir / f"test_raw_{idx:03d}.png", (np.clip(e, 0, 1) * 255).astype(np.uint8))
            mask = self.clean_edges(e)
            imsave(out_dir / f"test_mask_{idx:03d}.png", (mask.astype(np.uint8) * 255))

        # --------- Optional evaluation on overlapping frames only ----------
 
        if evaluate_test and test_gt_dir:
            sel_pos, eval_gt, eval_imgs, eval_names, eval_idx = [], [], [], [], []
            for j, idx in enumerate(y_idx):              # j = position inside decoded_masks
                k = common_keys[int(idx)]                # filename key for frame idx
                if k in mask_map_all:                    # GT exists for this predicted frame
                    sel_pos.append(j)                    # keep this prediction position
                    eval_gt.append(mask_map_all[k])
                    eval_imgs.append(test_imgs[int(idx)])
                    eval_names.append(original_filenames[int(idx)])
                    eval_idx.append(int(idx))            # real frame index for x-axis

            if not sel_pos:
                print("[WARN] No overlapping GT for predicted frames; skipping evaluation.")
            else:
                # Temporarily align model state to just the evaluated frames
                _dec_bak, _gt_bak, _imgs_bak = self.decoded_edges, self.gt_paths, self.all_imgs
                self.decoded_edges = decoded_masks[sel_pos]             # align predictions
                self.gt_paths      = eval_gt
                self.all_imgs      = np.stack(eval_imgs)

                self.evaluate(original_filenames=eval_names,
                            gt_is_clean_binary=gt_is_clean_binary,
                            x_indices=eval_idx)

                # restore
                self.decoded_edges, self.gt_paths, self.all_imgs = _dec_bak, _gt_bak, _imgs_bak

        if return_arrays:
            return {
                "decoded": decoded_masks,         # (T_pred, H, W) raw probabilities
                "hints": test_edges,              # (T_all, H, W) hint frames actually used
                "y_idx": np.asarray(y_idx),       # indices of frames that were predicted
                "imgs": test_imgs_roi             # (T_all, H, W) ROI images
            }
