"""Evaluate a saved ACNN-BiLSTM model on the test set.

Usage:
    python -m acnn_bilstm.evaluate --checkpoint outputs/checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from acnn_bilstm.config import Config
from acnn_bilstm.data.loader import load_all_subjects
from acnn_bilstm.data.preprocessing import preprocess_dataset
from acnn_bilstm.data.cwt_transform import convert_all_windows
from acnn_bilstm.data.dataset import (
    BPDataset,
    concat_windows,
    subject_level_split,
)
from acnn_bilstm.model.acnn_bilstm import ACNNBiLSTM
from acnn_bilstm.training.trainer import evaluate, get_device
from acnn_bilstm.evaluation.metrics import (
    compute_all_metrics,
    print_bhs_table,
    print_summary_table,
)
from acnn_bilstm.evaluation.plots import save_bland_altman

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Evaluate ACNN-BiLSTM model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--scale-type", choices=["log", "linear"], default="log",
        help="CWT scale type used during training",
    )
    args = parser.parse_args()

    cfg = Config()
    cfg.cwt_scale_type = args.scale_type
    cfg.ensure_dirs()

    device = get_device()
    logger.info("Device: %s", device)

    # ── Load & preprocess ────────────────────────────────────────────────
    dataset, _ = load_all_subjects(cfg)
    dataset, _ = preprocess_dataset(dataset, cfg)
    _, test_subjects = subject_level_split(dataset, cfg)
    X_test_raw, y_test = concat_windows(test_subjects)

    logger.info("Test subjects: %d, windows: %d", len(test_subjects), X_test_raw.shape[0])

    # ── CWT ──────────────────────────────────────────────────────────────
    X_test_cwt = convert_all_windows(X_test_raw, cfg)

    # ── Load model ───────────────────────────────────────────────────────
    model = ACNNBiLSTM(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    # ── Evaluate ─────────────────────────────────────────────────────────
    test_loader = DataLoader(
        BPDataset(X_test_cwt, y_test),
        batch_size=cfg.batch_size, shuffle=False,
    )
    criterion = torch.nn.L1Loss()
    test_loss, preds, targets = evaluate(model, test_loader, criterion, device)

    logger.info("Test MAE (overall): %.4f", test_loss)

    sbp_m = compute_all_metrics(targets[:, 0], preds[:, 0], "SBP (Systolic)")
    dbp_m = compute_all_metrics(targets[:, 1], preds[:, 1], "DBP (Diastolic)")
    print_summary_table(sbp_m, dbp_m)
    print_bhs_table(sbp_m, dbp_m)

    save_bland_altman(
        targets[:, 0], preds[:, 0],
        "Bland-Altman for SBP",
        cfg.images_dir / "eval_bland_altman_sbp.png",
    )
    save_bland_altman(
        targets[:, 1], preds[:, 1],
        "Bland-Altman for DBP",
        cfg.images_dir / "eval_bland_altman_dbp.png",
    )
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
