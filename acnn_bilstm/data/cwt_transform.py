"""CWT transformation and multi-wavelength fusion.

Paper Section 2.2 & 3.2: Apply CWT with cgau1 wavelet, render as RGB via
jet colormap, fuse 4 wavelengths into 12-channel array.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count

import matplotlib.cm as cm
import numpy as np
import pywt
from PIL import Image

from acnn_bilstm.config import Config

logger = logging.getLogger(__name__)


def build_scales(cfg: Config) -> np.ndarray:
    """Build the CWT scale array.

    - ``"log"`` (default): logarithmically spaced scales covering the full
      0.5–8 Hz band.  scale = fc / (f * dt), so for 0.5 Hz → scale ≈ 400
      and 8 Hz → scale ≈ 25 (assuming fc ≈ 1.0 for cgau1).
    - ``"linear"``: np.arange(1, n_scales+1) — the notebook's original
      approach (only covers ≈ 1.56–200 Hz, misses the fundamental HR).
    """
    if cfg.cwt_scale_type == "log":
        # Cover the 0.5–8 Hz band fully
        # f = fc / (a * dt) → a = fc / (f * dt)
        # fc(cgau1) ≈ 1.0, dt = 1/fs
        a_min = cfg.fs / (cfg.highcut * 1.0)   # 200/8  = 25
        a_max = cfg.fs / (cfg.lowcut * 1.0)    # 200/0.5 = 400
        scales = np.logspace(
            np.log10(a_min), np.log10(a_max), cfg.n_scales
        )
    elif cfg.cwt_scale_type == "linear":
        scales = np.arange(1, cfg.n_scales + 1).astype(float)
    else:
        raise ValueError(f"Unknown cwt_scale_type: {cfg.cwt_scale_type!r}")
    return scales


def _normalise_01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def cwt_to_rgb(
    signal_1d: np.ndarray,
    scales: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Convert a 1-D PPG window → RGB CWT image (H, W, 3).

    Steps:
    1. CWT with cgau1 → complex coefficients (n_scales, 1000)
    2. Magnitude → normalise [0, 1]
    3. Jet colormap → RGB (n_scales, 1000, 3)
    4. Resize to (target_img_height, target_img_width)
    """
    coeffs, _ = pywt.cwt(signal_1d, scales, cfg.wavelet_name)
    magnitude = _normalise_01(np.abs(coeffs))

    cmap = cm.get_cmap(cfg.colormap)
    rgba = cmap(magnitude)          # (H_native, W_native, 4)
    rgb = rgba[..., :3]             # drop alpha

    rgb_uint8 = (rgb * 255).astype(np.uint8)
    pil_size = (cfg.target_img_width, cfg.target_img_height)  # PIL: (W, H)
    pil_img = Image.fromarray(rgb_uint8).resize(pil_size, Image.BILINEAR)
    return np.asarray(pil_img).astype(np.float32) / 255.0


def window_to_12ch(
    window_4ch: np.ndarray,
    scales: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Convert one 4-channel window (1000, 4) → fused (12, H, W)."""
    channels = []
    for ch in range(4):
        rgb = cwt_to_rgb(window_4ch[:, ch], scales, cfg)  # (H, W, 3)
        channels.append(rgb)
    # Stack along channel axis: 4 × (H, W, 3) → (H, W, 12)
    fused_hwc = np.concatenate(channels, axis=2)
    # Convert to (12, H, W) for PyTorch
    return np.transpose(fused_hwc, (2, 0, 1))


def convert_all_windows(
    windows: np.ndarray,
    cfg: Config,
    n_workers: int | None = None,
) -> np.ndarray:
    """Convert all windows (N, 1000, 4) → CWT images (N, 12, H, W).

    Uses ThreadPoolExecutor — NumPy/SciPy release the GIL so threads get
    real parallelism for the CWT computation.
    """
    scales = build_scales(cfg)
    if n_workers is None:
        n_workers = max(1, (cpu_count() or 2) - 1)

    N = windows.shape[0]
    logger.info(
        "CWT conversion: %d windows, %d threads, scale_type=%s",
        N, n_workers, cfg.cwt_scale_type,
    )

    def _convert(i: int) -> np.ndarray:
        return window_to_12ch(windows[i], scales, cfg)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_convert, range(N)))

    return np.stack(results, axis=0)  # (N, 12, H, W)
