"""Plotting utilities — Bland-Altman plots, loss curves, CWT visualisation.

All plots are saved to the configured images directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def save_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Create and save a Bland-Altman plot (paper Figure 4)."""
    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_pred - y_true
    mean_diff = np.mean(diff_vals)
    sd_diff = np.std(diff_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean_vals, diff_vals, alpha=0.4, s=10, c="steelblue")
    ax.axhline(
        mean_diff, color="red", linestyle="--",
        label=f"Mean: {mean_diff:.3f}",
    )
    ax.axhline(
        mean_diff + 1.96 * sd_diff, color="gray", linestyle="--",
        label=f"+1.96 SD: {mean_diff + 1.96 * sd_diff:.3f}",
    )
    ax.axhline(
        mean_diff - 1.96 * sd_diff, color="gray", linestyle="--",
        label=f"-1.96 SD: {mean_diff - 1.96 * sd_diff:.3f}",
    )
    ax.set_xlabel("Mean (mmHg)")
    ax.set_ylabel("Difference (mmHg)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Bland-Altman plot: %s", save_path)


def save_training_curves(
    history: Dict[str, List[float]],
    save_path: Path,
    title: str = "Training Curve",
) -> None:
    """Save training & validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train MAE")
    ax.plot(history["val_loss"], label="Val MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves: %s", save_path)


def save_cwt_visualisation(
    cwt_sample: np.ndarray,
    subject_id: int,
    save_path: Path,
) -> None:
    """Visualise one 12-channel CWT sample as 4 wavelength RGB images.

    Parameters
    ----------
    cwt_sample : (12, H, W) array
    subject_id : int
    save_path : Path
    """
    # (12, H, W) → (H, W, 12)
    hwc = np.transpose(cwt_sample, (1, 2, 0))
    wavelengths = ["660 nm", "730 nm", "850 nm", "940 nm"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        rgb = hwc[:, :, i * 3 : (i + 1) * 3]
        axes[i].imshow(rgb)
        axes[i].set_title(f"Channel {i + 1} ({wavelengths[i]})")
        axes[i].axis("off")
    fig.suptitle(f"CWT Images - Subject {subject_id}, Window 0")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved CWT visualisation: %s", save_path)
