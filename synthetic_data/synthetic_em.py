#!/usr/bin/env python3
"""
Synthetic EM-like sequence generator: moving, deforming dark blobs (clusters)
on a reconstructed background. Saves frames as PNG and binary masks.

Usage:
  python synthetic_em.py \
    --img_dir data/Manualy_Labeled/images \
    --mask_dir data/Manualy_Labeled/masks_SK \
    --out_dir data/synthetic_results \
    --n_frames 200 --width 256 --height 256 --seed 42
"""
from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic EM-like sequences")
    p.add_argument("--img_dir", type=Path, required=True, help="Directory with real images")
    p.add_argument("--mask_dir", type=Path, required=True, help="Directory with binary masks aligned to images")
    p.add_argument("--out_dir",  type=Path, required=True, help="Output directory (frames + masks)")
    p.add_argument("--width",  type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--n_frames", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    # shape dynamics
    p.add_argument("--base_scale", type=float, default=1.25)
    p.add_argument("--grow_rate",  type=float, default=0.20)
    p.add_argument("--center_boost", type=float, default=1.3)
    p.add_argument("--l1_freq", type=int, default=2)
    p.add_argument("--l1_amp",  type=float, default=0.05)
    p.add_argument("--l2_freq", type=int, default=3)
    p.add_argument("--l2_amp",  type=float, default=0.03)

    # background parameters
    p.add_argument("--noise_alpha", type=float, default=0.01, help="Scaled Gaussian noise on inpainted BG")
    p.add_argument("--inpaint_radius", type=int, default=5)

    return p.parse_args()

def first_image_and_matching_mask(img_dir: Path, mask_dir: Path) -> tuple[Path, Path]:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if not imgs:
        raise FileNotFoundError(f"No images found in {img_dir}")
    img = imgs[0]
    # prefer exact stem match with common mask extensions
    mask_candidates = [mask_dir / f"{img.stem}.png", mask_dir / f"{img.stem}.tif", mask_dir / f"{img.stem}.jpg"]
    for m in mask_candidates:
        if m.exists():
            return img, m
    # fallback: first mask
    masks = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in (".png", ".tif", ".jpg", ".jpeg")])
    if not masks:
        raise FileNotFoundError(f"No masks found in {mask_dir}")
    return img, masks[0]

def main():
    a = parse_args()

    # Reproducibility
    np.random.seed(a.seed)
    random.seed(a.seed)

    WIDTH, HEIGHT, N_FRAMES = a.width, a.height, a.n_frames
    BASE_SCALE, GROW_RATE, CENTER_BOOST = a.base_scale, a.grow_rate, a.center_boost
    L1_FREQ, L1_AMP, L2_FREQ, L2_AMP = a.l1_freq, a.l1_amp, a.l2_freq, a.l2_amp
    NOISE_ALPHA, INPAINT_RADIUS = a.noise_alpha, a.inpaint_radius

    out_dir   = a.out_dir
    mask_out  = out_dir / "masks_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    # --- Load a background reference image + binary mask ---
    img0, mask0 = first_image_and_matching_mask(a.img_dir, a.mask_dir)
    orig = cv2.resize(cv2.imread(str(img0), cv2.IMREAD_GRAYSCALE), (WIDTH, HEIGHT))
    msk  = cv2.resize(cv2.imread(str(mask0), cv2.IMREAD_GRAYSCALE), (WIDTH, HEIGHT))
    msk  = (msk > 0).astype(np.uint8)

    # --- Background reconstruction (inpaint + mild noise) ---
    kernel = np.ones((3, 3), np.uint8)
    msk_bg = cv2.erode(msk, kernel, iterations=1)
    bg_only    = (orig * (1 - msk_bg)).astype(np.uint8)
    bg_inpaint = cv2.inpaint(bg_only, msk_bg, INPAINT_RADIUS, flags=cv2.INPAINT_TELEA).astype(np.float32)
    bg_inpaint += NOISE_ALPHA * bg_inpaint.std() * np.random.randn(*bg_inpaint.shape).astype(np.float32)
    bg = np.clip(bg_inpaint, 0, 255).astype(np.uint8)
    Image.fromarray(bg).save(out_dir / "background.png")

    # --- Build three clusters with different spreads & counts ---
    cluster_gap   = int(0.82 * HEIGHT)  # scaled to height
    center_top    = (HEIGHT // 2 - cluster_gap // 2, np.random.randint(60, WIDTH - 60))
    center_mid    = (HEIGHT // 2, WIDTH // 2)
    center_bottom = (HEIGHT // 2 + cluster_gap // 2, np.random.randint(60, WIDTH - 60))

    spread_top, spread_center, spread_bottom = (15, 30), (10, 10), (20, 40)
    n_top, n_center, n_bottom = np.random.randint(3, 8), np.random.randint(3, 6), np.random.randint(7, 10)

    def sample_cluster(center, spread, count):
        cy, cx = center; sy, sx = spread
        yy = np.clip(np.random.normal(cy, sy, size=(count,)), 0, HEIGHT - 1)
        xx = np.clip(np.random.normal(cx, sx, size=(count,)), 0, WIDTH - 1)
        return np.stack([yy, xx], axis=1)

    centers = np.concatenate([
        sample_cluster(center_top,    spread_top,    n_top),
        sample_cluster(center_mid,    spread_center, n_center),
        sample_cluster(center_bottom, spread_bottom, n_bottom),
    ], axis=0)
    n_bub = centers.shape[0]

    # motion: linear drift + sinusoidal jitter
    motion_velocity = np.random.uniform(-1.5, 1.5, size=(n_bub, 2))
    motion_jitter   = np.random.uniform(0.2, 0.7, size=(n_bub, 2))
    motion_phase    = np.random.rand(n_bub, 2) * 2 * np.pi

    def animate_centers(t: int) -> np.ndarray:
        drift = centers + t * motion_velocity
        jitter = np.sin(2 * np.pi * t / N_FRAMES + motion_phase) * motion_jitter * 10.0
        pos = drift + jitter
        pos[:, 0] = np.clip(pos[:, 0], 0, HEIGHT - 1)
        pos[:, 1] = np.clip(pos[:, 1], 0, WIDTH - 1)
        return pos

    # class radii
    radii = {'small': 17, 'medium': 28, 'large': 41}
    classes = np.random.choice(list(radii), size=n_bub)

    n_theta = 360
    thetas  = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi1    = np.random.rand(n_bub) * 2 * np.pi
    phi2    = np.random.rand(n_bub) * 2 * np.pi

    Y, X = np.ogrid[:HEIGHT, :WIDTH]

    for t in range(N_FRAMES):
        frame      = bg.astype(float)
        mask_frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

        # slow size pulsation
        theta_t = 2 * np.pi * t / N_FRAMES
        scale   = BASE_SCALE + GROW_RATE * np.sin(theta_t)

        current_centers = animate_centers(t)
        for i, cls in enumerate(classes):
            y0, x0 = current_centers[i]
            r0 = radii[cls] * scale

            # non-circular lobes
            lobes     = (np.sin(L1_FREQ * thetas + phi1[i]) * L1_AMP +
                         np.sin(L2_FREQ * thetas + phi2[i]) * L2_AMP)
            radii_arr = r0 * (1.0 + lobes)

            dx = X - x0
            dy = Y - y0
            ang = (np.arctan2(dy, dx) % (2 * np.pi))
            idx = (ang / (2 * np.pi) * n_theta).astype(int)
            R   = radii_arr[idx]

            disk = (dx * dx + dy * dy) <= (R * R)
            mask_frame[disk] = 255

            # soft dark blob with central boost
            bub = gaussian_filter(disk.astype(float), sigma=0.7)
            bub /= (bub.max() + 1e-6)
            sigma_c = r0 * 0.3
            spot    = np.exp(-(dx * dx + dy * dy) / (2 * sigma_c ** 2))
            bub    *= 1 + CENTER_BOOST * spot

            # intensity per class (small darker, large lighter)
            darkness = 90 * (1.2 if cls == "small" else 0.8 if cls == "large" else 1.0)
            frame   -= bub * darkness

        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # save frame + mask
        Image.fromarray(frame).save(out_dir / f"synth_frame_{t:03d}.png")
        Image.fromarray(mask_frame).save(mask_out / f"mask_{t:03d}.png")

    print(f"Saved {N_FRAMES} frames to {out_dir} (masks in {mask_out})")

if __name__ == "__main__":
    main()