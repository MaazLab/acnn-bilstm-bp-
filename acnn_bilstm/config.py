"""Configuration dataclass for ACNN-BiLSTM pipeline.

Centralises every hyperparameter, path, and constant so that experiments are
fully reproducible and nothing is scattered across modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: Path = Path("Bioengineering_Paper_project_data/data")
    labels_xlsx: Path = Path(
        "Bioengineering_Paper_project_data/labels/Subjects Information.xlsx"
    )
    output_dir: Path = Path("outputs")
    images_dir: Path = Path("outputs/images")
    checkpoints_dir: Path = Path("outputs/checkpoints")

    # ── Signal acquisition (paper Section 2.1) ───────────────────────────
    fs: float = 200.0            # Sampling rate (Hz)
    lowcut: float = 0.5          # Bandpass low  (Hz)
    highcut: float = 8.0         # Bandpass high (Hz)
    filter_order: int = 2        # Butterworth order

    # ── Windowing (paper Section 2.2) ────────────────────────────────────
    window_sec: float = 5.0      # Window length in seconds
    stride_sec: float = 1.0      # Stride in seconds
    outlier_sigma: float = 5.0   # Reject windows with samples > N*sigma

    @property
    def window_samples(self) -> int:
        return int(self.window_sec * self.fs)  # 1000

    @property
    def stride_samples(self) -> int:
        return int(self.stride_sec * self.fs)  # 200

    # ── CWT (paper Section 2.2) ──────────────────────────────────────────
    wavelet_name: str = "cgau1"
    n_scales: int = 128
    cwt_scale_type: Literal["log", "linear"] = "log"
    # log  → np.logspace(log10(25), log10(400), 128)  covers 0.5-8 Hz fully
    # linear → np.arange(1, 129)  (notebook default, misses <1.56 Hz)

    target_img_height: int = 128  # CWT image height (= n_scales)
    target_img_width: int = 256   # CWT image width  (temporal axis, resampled)
    colormap: str = "jet"

    @property
    def target_img_size(self) -> tuple[int, int]:
        """(Height, Width) in PyTorch convention."""
        return (self.target_img_height, self.target_img_width)

    # ── Model architecture (paper Section 3.3, Figure 3) ────────────────
    in_channels: int = 12                   # 4 wavelengths × 3 RGB
    cnn_channels: List[int] = field(
        default_factory=lambda: [32, 32, 32, 64, 64, 128, 128, 256]
        # [init_conv] + [block1..block7]
    )
    cnn_strides: List[int] = field(
        default_factory=lambda: [2, 1, 2, 1, 2, 2, 2]
        # stride for each of the 7 ACNN blocks
    )
    se_reduction: int = 16                  # SE attention reduction ratio
    lstm_hidden: int = 256                  # BiLSTM hidden size
    lstm_layers: int = 2                    # Number of BiLSTM layers
    fc_hidden: int = 128                    # First FC layer output
    dropout: float = 0.2

    # ── Training (paper Section 3.3) ─────────────────────────────────────
    batch_size: int = 32
    num_epochs: int = 500
    patience: int = 150                     # Early stopping patience
    lr: float = 1e-3                        # Adam learning rate
    val_split: float = 0.1                  # fraction of train for early stopping
    test_size: float = 0.20                 # 80/20 subject-level split

    # ── Cross-validation ─────────────────────────────────────────────────
    n_folds: int = 10
    run_cv: bool = False

    # ── Misc ─────────────────────────────────────────────────────────────
    random_state: int = 42
    num_workers: int = 0                    # DataLoader workers (0 = main thread)
    cache_cwt: bool = True                  # Cache CWT images to disk

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for d in (self.output_dir, self.images_dir, self.checkpoints_dir):
            d.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        # Validate
        assert len(self.cnn_channels) == 8, (
            f"cnn_channels must have 8 entries (init + 7 blocks), got {len(self.cnn_channels)}"
        )
        assert len(self.cnn_strides) == 7, (
            f"cnn_strides must have 7 entries (one per ACNN block), got {len(self.cnn_strides)}"
        )
        assert self.n_scales == self.target_img_height, (
            "n_scales must equal target_img_height"
        )
