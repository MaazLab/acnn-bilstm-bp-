"""Signal preprocessing — bandpass filtering, normalisation, windowing.

Paper Section 2.2: 0.5-8 Hz Butterworth 2nd-order bandpass, Z-score
normalisation per subject per channel, 5 s sliding window with 1 s stride,
outlier rejection.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from acnn_bilstm.config import Config

logger = logging.getLogger(__name__)


def bandpass_filter(
    x: np.ndarray, lowcut: float, highcut: float, fs: float, order: int
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    return filtfilt(b, a, x)


def filter_and_normalise(
    ppg_raw: pd.DataFrame, cfg: Config
) -> pd.DataFrame | None:
    """Apply bandpass filter and Z-score normalisation to 4-channel PPG.

    Returns None if any channel has zero variance after filtering.
    """
    ppg = pd.DataFrame(
        {
            ch: bandpass_filter(
                ppg_raw[ch].to_numpy(), cfg.lowcut, cfg.highcut, cfg.fs, cfg.filter_order
            )
            for ch in ppg_raw.columns
        }
    )
    for ch in ppg.columns:
        col = ppg[ch]
        std = col.std()
        if std < 1e-8:
            return None
        ppg[ch] = (col - col.mean()) / std
    return ppg


def _is_valid_window(
    window: np.ndarray, outlier_sigma: float
) -> bool:
    """Reject windows with NaN/Inf, flat channels, or extreme outliers."""
    if np.any(np.isnan(window)) or np.any(np.isinf(window)):
        return False
    for ch in range(window.shape[1]):
        col = window[:, ch]
        std = col.std()
        if std < 1e-6:
            return False
        if np.any(np.abs(col - col.mean()) > outlier_sigma * std):
            return False
    return True


def sliding_windows(
    ppg: pd.DataFrame, cfg: Config
) -> np.ndarray:
    """Segment filtered signal into overlapping windows.

    Returns array of shape ``(N, window_samples, 4)`` containing only
    valid (non-outlier) windows.
    """
    x = ppg.to_numpy(dtype=np.float32)
    n = x.shape[0]
    ws = cfg.window_samples
    ss = cfg.stride_samples

    if n < ws:
        return np.empty((0, ws, 4), dtype=np.float32)

    starts = np.arange(0, n - ws + 1, ss)
    valid = []
    for s in starts:
        w = x[s : s + ws]
        if _is_valid_window(w, cfg.outlier_sigma):
            valid.append(w)

    if not valid:
        return np.empty((0, ws, 4), dtype=np.float32)
    return np.stack(valid, axis=0)


def preprocess_dataset(
    dataset: List[Dict], cfg: Config
) -> Tuple[List[Dict], int]:
    """Filter, normalise and window every subject in *dataset*.

    Mutates each dict in-place, adding keys:
    - ``ppg_filtered``: filtered & normalised DataFrame
    - ``X``: windows array ``(N, 1000, 4)``
    - ``y``: labels array  ``(N, 2)``

    Returns the (mutated) dataset and total window count.
    """
    total_windows = 0
    kept: List[Dict] = []

    for item in dataset:
        ppg = filter_and_normalise(item["ppg_raw"], cfg)
        if ppg is None:
            logger.warning(
                "Subject %d: zero-variance channel after filtering – skipped",
                item["id"],
            )
            continue

        item["ppg_filtered"] = ppg
        windows = sliding_windows(ppg, cfg)
        if windows.shape[0] == 0:
            logger.warning("Subject %d: no valid windows – skipped", item["id"])
            continue

        labels = np.tile(
            np.array([item["sbp"], item["dbp"]], dtype=np.float32),
            (windows.shape[0], 1),
        )
        item["X"] = windows
        item["y"] = labels
        total_windows += windows.shape[0]
        kept.append(item)

    logger.info(
        "Preprocessing complete: %d subjects, %d total windows",
        len(kept),
        total_windows,
    )
    return kept, total_windows
