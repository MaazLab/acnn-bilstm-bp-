"""Z-score normaliser for blood pressure targets.

Fitted on training windows ONLY. Used to scale model targets before
training and to invert predictions back to mmHg for evaluation.

Why z-score and not min-max:
  - FC2 output is unconstrained (no sigmoid/tanh), so there is no risk of
    saturation from targets outside the training range.
  - Denormalisation is a pure linear transform: pred * std + mean.  No
    nonlinearity can compress the output range.
  - Test subjects with extreme BP (high z-score) are still handled correctly
    because the model just needs to output a larger float, not stay inside
    a fixed interval.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class BPScaler:
    """Per-column z-score scaler for BP label arrays of shape (N, 2).

    Column 0 = SBP, column 1 = DBP.  Statistics are computed and stored
    independently for each column so that differing BP ranges don't
    interfere with each other.

    Usage
    -----
    scaler = BPScaler().fit(y_train_windows)
    y_norm   = scaler.transform(y_raw)          # use for training
    y_mmhg   = scaler.inverse_transform(y_norm) # use for evaluation
    scaler.save(path / "bp_scaler.npz")         # persist for reproducibility
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self.mean_: np.ndarray = np.zeros(2)   # shape (2,) = [SBP_mean, DBP_mean]
        self.std_: np.ndarray = np.ones(2)     # shape (2,) = [SBP_std,  DBP_std ]

    # ── Fitting ──────────────────────────────────────────────────────────

    def fit(self, y: np.ndarray) -> "BPScaler":
        """Compute mean and std from training-set labels.

        Parameters
        ----------
        y : ndarray, shape (N, 2)
            ONLY pass training-window labels.  Never include validation or
            test data here to avoid data leakage.
        """
        self.mean_ = y.mean(axis=0)
        self.std_ = y.std(axis=0)
        # Guard against degenerate std (single unique value)
        self.std_ = np.where(self.std_ < 1e-6, 1.0, self.std_)
        self._fitted = True
        return self

    # ── Transform / inverse ──────────────────────────────────────────────

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Normalise raw BP labels to zero-mean, unit-variance.

        The model only needs to output values in roughly (-3, +3) rather
        than the raw mmHg range (≈ 80–200 SBP).
        """
        self._check_fitted()
        return (y - self.mean_) / self.std_

    def inverse_transform(self, y_norm: np.ndarray) -> np.ndarray:
        """Convert normalised predictions back to mmHg.

        This is a pure linear transform: y_mmhg = y_norm * std + mean.
        There is no nonlinearity, so extreme predictions are not clipped.
        """
        self._check_fitted()
        return y_norm * self.std_ + self.mean_

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save scaler statistics to a .npz file."""
        self._check_fitted()
        np.savez(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: Path) -> "BPScaler":
        """Load a previously saved scaler from a .npz file."""
        data = np.load(path)
        scaler = cls()
        scaler.mean_ = data["mean"]
        scaler.std_ = data["std"]
        return scaler

    # ── Internal ─────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "BPScaler has not been fitted yet. Call fit() first."
            )

    def __repr__(self) -> str:
        if not self._fitted:
            return "BPScaler(not fitted)"
        return (
            f"BPScaler("
            f"SBP mean={self.mean_[0]:.1f} std={self.std_[0]:.1f}, "
            f"DBP mean={self.mean_[1]:.1f} std={self.std_[1]:.1f})"
        )
