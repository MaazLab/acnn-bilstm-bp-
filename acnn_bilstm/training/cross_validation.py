"""10-fold subject-level cross-validation.

Paper Section 3.3: "the data in the ratio of 0.8 was used for training and
10-fold cross validation".
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn

from acnn_bilstm.config import Config
from acnn_bilstm.data.dataset import BPDataset
from acnn_bilstm.model.acnn_bilstm import ACNNBiLSTM
from acnn_bilstm.training.trainer import evaluate, train_model

logger = logging.getLogger(__name__)


def run_cross_validation(
    X_cwt: np.ndarray,
    y_all: np.ndarray,
    subject_ids: np.ndarray,
    subject_window_map: Dict[int, np.ndarray],
    device: torch.device,
    cfg: Config,
) -> Dict[str, List[float]]:
    """Run 10-fold subject-level cross-validation.

    Parameters
    ----------
    X_cwt : (N, 12, H, W)
    y_all : (N, 2)
    subject_ids : (n_subjects,)
    subject_window_map : {sid: array of window indices}
    device : torch.device
    cfg : Config

    Returns
    -------
    cv_results : dict with lists of per-fold metrics
    """
    dataset = BPDataset(X_cwt, y_all)
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)

    cv_results: Dict[str, List[float]] = {
        "fold": [],
        "val_mae_sbp": [], "val_mae_dbp": [],
        "val_rmse_sbp": [], "val_rmse_dbp": [],
        "val_r2_sbp": [], "val_r2_dbp": [],
    }

    for fold_idx, (train_subj_idx, val_subj_idx) in enumerate(kf.split(subject_ids)):
        fold_train_sids = subject_ids[train_subj_idx]
        fold_val_sids = subject_ids[val_subj_idx]

        fold_train_win = np.concatenate(
            [subject_window_map[sid] for sid in fold_train_sids]
        )
        fold_val_win = np.concatenate(
            [subject_window_map[sid] for sid in fold_val_sids]
        )

        logger.info(
            "=" * 60 + "\nFold %d/%d | Train subjects: %d | Val subjects: %d | "
            "Train windows: %d | Val windows: %d\n" + "=" * 60,
            fold_idx + 1, cfg.n_folds,
            len(fold_train_sids), len(fold_val_sids),
            len(fold_train_win), len(fold_val_win),
        )

        train_loader = DataLoader(
            Subset(dataset, fold_train_win.tolist()),
            batch_size=cfg.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            Subset(dataset, fold_val_win.tolist()),
            batch_size=cfg.batch_size, shuffle=False,
        )

        fold_model = ACNNBiLSTM(cfg).to(device)
        fold_model, _, _ = train_model(
            fold_model, train_loader, val_loader, device, cfg
        )

        criterion = nn.L1Loss()
        _, val_preds, val_targets = evaluate(fold_model, val_loader, criterion, device)

        sbp_pred, dbp_pred = val_preds[:, 0], val_preds[:, 1]
        sbp_true, dbp_true = val_targets[:, 0], val_targets[:, 1]

        fold_metrics = {
            "val_mae_sbp": mean_absolute_error(sbp_true, sbp_pred),
            "val_mae_dbp": mean_absolute_error(dbp_true, dbp_pred),
            "val_rmse_sbp": float(np.sqrt(mean_squared_error(sbp_true, sbp_pred))),
            "val_rmse_dbp": float(np.sqrt(mean_squared_error(dbp_true, dbp_pred))),
            "val_r2_sbp": r2_score(sbp_true, sbp_pred),
            "val_r2_dbp": r2_score(dbp_true, dbp_pred),
        }

        cv_results["fold"].append(fold_idx + 1)
        for k, v in fold_metrics.items():
            cv_results[k].append(v)

        logger.info(
            "Fold %d Results:\n"
            "  SBP → MAE: %.2f, RMSE: %.2f, R²: %.4f\n"
            "  DBP → MAE: %.2f, RMSE: %.2f, R²: %.4f",
            fold_idx + 1,
            fold_metrics["val_mae_sbp"], fold_metrics["val_rmse_sbp"],
            fold_metrics["val_r2_sbp"],
            fold_metrics["val_mae_dbp"], fold_metrics["val_rmse_dbp"],
            fold_metrics["val_r2_dbp"],
        )

        del fold_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("10-Fold Subject-Level Cross-Validation Summary (mean ± std)")
    logger.info("=" * 60)
    for metric in ["val_mae", "val_rmse", "val_r2"]:
        for bp in ["sbp", "dbp"]:
            key = f"{metric}_{bp}"
            vals = np.array(cv_results[key])
            logger.info(
                "  %s %s: %.4f ± %.4f",
                bp.upper(), metric.split("_")[1].upper(),
                vals.mean(), vals.std(),
            )

    return cv_results
