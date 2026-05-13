"""Diagnostic utilities to verify the training pipeline is not broken.

Run BEFORE the main training loop to answer two questions:

1. Mean-baseline diagnostic
   Is the model even beating a trivial predictor that always outputs the
   training-set mean?  If Train MAE >= baseline MAE the model has learnt
   nothing and is stuck predicting near the average BP.

2. 10-sample overfit diagnostic
   Can the model memorise a tiny dataset (1 window per subject, 10 subjects)?
   A 4.8 M parameter network should drive MAE close to 0 on 10 samples.
   Failure means there is a fundamental bug in the model, loss, labels, data
   shape, or preprocessing -- not just a hyper-parameter issue.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from acnn_bilstm.config import Config
from acnn_bilstm.data.dataset import BPDataset
from acnn_bilstm.model.acnn_bilstm import ACNNBiLSTM
from acnn_bilstm.training.trainer import get_device

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 1: Mean-baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_mean_baseline(
    y_train_windows: np.ndarray,
    y_val_windows: np.ndarray,
) -> Dict[str, float]:
    """Compute baseline MAE for a predictor that always returns the training mean.

    Parameters
    ----------
    y_train_windows : ndarray, shape (N_train, 2)
        BP labels [SBP, DBP] for every training window used in this split.
    y_val_windows : ndarray, shape (N_val, 2)
        BP labels [SBP, DBP] for the validation windows.

    Returns
    -------
    dict with keys: mae_sbp, mae_dbp, mae_overall, mean_sbp, mean_dbp
    """
    train_mean = y_train_windows.mean(axis=0)          # shape (2,) = [SBP, DBP]
    baseline = np.tile(train_mean, (len(y_val_windows), 1))

    mae_sbp     = float(np.abs(y_val_windows[:, 0] - baseline[:, 0]).mean())
    mae_dbp     = float(np.abs(y_val_windows[:, 1] - baseline[:, 1]).mean())
    mae_overall = float(np.abs(y_val_windows - baseline).mean())

    logger.info("-" * 66)
    logger.info("DIAGNOSTIC 1 | Mean-prediction baseline")
    logger.info("  Training-set mean   : SBP=%.1f mmHg, DBP=%.1f mmHg",
                train_mean[0], train_mean[1])
    logger.info("  Baseline MAE (val)  : SBP=%.2f  DBP=%.2f  Overall=%.2f mmHg",
                mae_sbp, mae_dbp, mae_overall)
    logger.info("  Interpretation")
    logger.info("    A model whose Train MAE >= %.2f mmHg is NOT learning any", mae_overall)
    logger.info("    BP-related features -- it is predicting near the global mean.")
    logger.info("    Current plateau ~24 mmHg vs baseline %.2f mmHg =>", mae_overall)
    if 24.0 >= mae_overall * 0.95:
        logger.info("    ** CONFIRMED: model is stuck at/above the mean-prediction floor **")
    else:
        logger.info("    Model is beating the mean baseline.")
    logger.info("-" * 66)

    return {
        "mae_sbp": mae_sbp,
        "mae_dbp": mae_dbp,
        "mae_overall": mae_overall,
        "mean_sbp": float(train_mean[0]),
        "mean_dbp": float(train_mean[1]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 2: 10-sample overfit
# ─────────────────────────────────────────────────────────────────────────────

def run_overfit_test(
    X_cwt: np.ndarray,
    y: np.ndarray,
    final_train_sids: np.ndarray,
    subject_window_map: dict,
    cfg: Config,
    n_samples: int = 10,
    max_epochs: int = 500,
) -> bool:
    """Train the model on 10 windows (one per subject) and check memorisation.

    The model has 4.8 M parameters.  On 10 samples it MUST be able to reduce
    MAE to near 0.  If it cannot, there is a fundamental bug in:
      - model output / final layer
      - loss function
      - label values
      - data tensor shape / dtype
      - optimizer
      - CWT / preprocessing

    One window per subject is chosen so that all 10 targets are different BP
    values, making random-chance agreement very unlikely.

    Parameters
    ----------
    X_cwt : ndarray, shape (N_total_train, 12, H, W)
    y     : ndarray, shape (N_total_train, 2)
    final_train_sids : subject IDs used in the final training split
    subject_window_map : maps subject ID -> array of window indices into X_cwt / y
    cfg   : Config
    n_samples  : number of samples to use (default 10)
    max_epochs : epochs to run (default 500)

    Returns
    -------
    passed : bool  (True if best MAE < 5 mmHg)
    """
    logger.info("-" * 66)
    logger.info("DIAGNOSTIC 2 | Overfit test: %d samples x %d epochs", n_samples, max_epochs)

    # Pick the first window of the first n_samples subjects so that all targets
    # are distinct (each subject has a unique SBP/DBP reading).
    chosen_sids = final_train_sids[:n_samples]
    overfit_idx = np.array([subject_window_map[sid][0] for sid in chosen_sids])

    X_over = X_cwt[overfit_idx]        # (n_samples, 12, H, W)
    y_over = y[overfit_idx]            # (n_samples, 2)

    logger.info("  Overfit sample BP targets:")
    for i in range(len(chosen_sids)):
        logger.info("    Subject %4d | SBP=%.1f | DBP=%.1f",
                    chosen_sids[i], y_over[i, 0], y_over[i, 1])
    target_range_sbp = y_over[:, 0].max() - y_over[:, 0].min()
    target_range_dbp = y_over[:, 1].max() - y_over[:, 1].min()
    logger.info("  Target range: SBP=%.1f mmHg, DBP=%.1f mmHg (higher = harder)",
                target_range_sbp, target_range_dbp)

    device = get_device()
    model = ACNNBiLSTM(cfg).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dataset = BPDataset(X_over, y_over)
    # Use full batch so every gradient step sees all 10 samples
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)

    log_at = {1, 10, 25, 50, 100, 200, 300, 400, 500}
    best_mae = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        mae = total_loss / len(dataset)
        if mae < best_mae:
            best_mae = mae
            best_state = copy.deepcopy(model.state_dict())

        if epoch in log_at:
            logger.info("  Overfit epoch %3d/%d | Train MAE: %.4f mmHg", epoch, max_epochs, mae)

        # Stop early if model has clearly memorised the data
        if mae < 1.0:
            logger.info("  Overfit epoch %3d/%d | Train MAE: %.4f mmHg  [converged early]",
                        epoch, max_epochs, mae)
            break

    # Final prediction check: show per-sample error with best weights
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_over, dtype=torch.float32).to(device)
        preds = model(X_t).cpu().numpy()

    logger.info("  Per-sample predictions (best weights, MAE=%.4f):", best_mae)
    for i in range(len(chosen_sids)):
        logger.info(
            "    Sample %2d | True SBP=%.1f DBP=%.1f | Pred SBP=%.1f DBP=%.1f | Err SBP=%.1f DBP=%.1f",
            i + 1,
            y_over[i, 0], y_over[i, 1],
            preds[i, 0], preds[i, 1],
            abs(preds[i, 0] - y_over[i, 0]),
            abs(preds[i, 1] - y_over[i, 1]),
        )

    passed = best_mae < 5.0
    logger.info("")
    if passed:
        logger.info("  RESULT: PASS  best MAE=%.4f < 5 mmHg", best_mae)
        logger.info("    Model CAN memorise small data => no fundamental code bug.")
        logger.info("    The plateau in full training is a learning / optimisation issue,")
        logger.info("    NOT a broken pipeline.  Most likely cause: missing target normalisation.")
    else:
        logger.info("  RESULT: FAIL  best MAE=%.4f >= 5 mmHg", best_mae)
        logger.info("    Model CANNOT memorise 10 samples with 4.8 M parameters!")
        logger.info("    Investigate: model output shape, loss function, label values,")
        logger.info("    data tensor dtype/range, optimizer, or CWT image generation.")
    logger.info("-" * 66)

    return passed
