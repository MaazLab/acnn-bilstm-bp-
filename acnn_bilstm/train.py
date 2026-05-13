"""Train the ACNN-BiLSTM model.

Usage:
    python -m acnn_bilstm.train [--no-cv] [--scale-type log|linear]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from acnn_bilstm.config import Config
from acnn_bilstm.data.loader import load_all_subjects
from acnn_bilstm.data.preprocessing import preprocess_dataset
from acnn_bilstm.data.cwt_transform import convert_all_windows
from acnn_bilstm.data.dataset import (
    BPDataset,
    build_subject_window_map,
    concat_windows,
    subject_level_split,
)
from acnn_bilstm.data.scaler import BPScaler
from acnn_bilstm.model.acnn_bilstm import ACNNBiLSTM
from acnn_bilstm.training.trainer import evaluate, get_device, train_model
from acnn_bilstm.training.cross_validation import run_cross_validation
from acnn_bilstm.diagnostics import run_mean_baseline, run_overfit_test
from acnn_bilstm.evaluation.metrics import (
    compute_all_metrics,
    print_bhs_table,
    print_summary_table,
)
from acnn_bilstm.evaluation.plots import (
    save_bland_altman,
    save_cwt_visualisation,
    save_training_curves,
)

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure root logger with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ACNN-BiLSTM for BP prediction"
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Run 10-fold subject-level cross-validation before final training",
    )
    parser.add_argument(
        "--scale-type", choices=["log", "linear"], default="log",
        help="CWT scale spacing (default: log)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Run diagnostics: mean baseline + 10-sample overfit test before training",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    # ── Configuration ────────────────────────────────────────────────────
    cfg = Config()
    cfg.cwt_scale_type = args.scale_type
    if args.cv:
        cfg.run_cv = True
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr

    cfg.ensure_dirs()

    logger.info("=" * 70)
    logger.info("ACNN-BiLSTM Training Pipeline")
    logger.info("=" * 70)
    logger.info("CWT scale type: %s", cfg.cwt_scale_type)
    logger.info("Image size (HxW): %d x %d", cfg.target_img_height, cfg.target_img_width)
    logger.info("Epochs: %d, Batch: %d, LR: %s, Patience: %d",
                cfg.num_epochs, cfg.batch_size, cfg.lr, cfg.patience)
    logger.info("Run CV: %s, Folds: %d", cfg.run_cv, cfg.n_folds)

    device = get_device()
    logger.info("Device: %s", device)

    # ── Phase 1: Load data ───────────────────────────────────────────────
    t0 = time.time()
    logger.info("\n" + "=" * 70)
    logger.info("Phase 1: Loading data")
    logger.info("=" * 70)

    dataset, skipped = load_all_subjects(cfg)
    logger.info("Loaded %d subjects, skipped %d", len(dataset), len(skipped))

    # ── Phase 2: Preprocessing ───────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Phase 2: Preprocessing (filter, normalise, window)")
    logger.info("=" * 70)

    dataset, total_windows = preprocess_dataset(dataset, cfg)
    logger.info("Total valid windows: %d", total_windows)

    ex = dataset[0]
    logger.info(
        "Example: Subject %d | SBP=%.1f, DBP=%.1f | Windows: %s",
        ex["id"], ex["sbp"], ex["dbp"], ex["X"].shape,
    )

    # ── Phase 3: Subject-level split ─────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Phase 3: 80/20 subject-level split")
    logger.info("=" * 70)

    train_subjects, test_subjects = subject_level_split(dataset, cfg)
    X_train_raw, y_train = concat_windows(train_subjects)
    X_test_raw, y_test = concat_windows(test_subjects)

    logger.info("Train: %d subjects, %d windows", len(train_subjects), X_train_raw.shape[0])
    logger.info("Test:  %d subjects, %d windows", len(test_subjects), X_test_raw.shape[0])

    # ── Phase 4: CWT transformation ──────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Phase 4: CWT transformation & 12-channel fusion")
    logger.info("=" * 70)

    logger.info("Converting training windows...")
    X_train_cwt = convert_all_windows(X_train_raw, cfg)
    logger.info("X_train_cwt: %s", X_train_cwt.shape)

    logger.info("Converting test windows...")
    X_test_cwt = convert_all_windows(X_test_raw, cfg)
    logger.info("X_test_cwt: %s", X_test_cwt.shape)

    # Save CWT visualisation
    save_cwt_visualisation(
        X_train_cwt[0], train_subjects[0]["id"],
        cfg.images_dir / "cwt_sample.png",
    )

    # ── Phase 5: Model verification ──────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Phase 5: Model verification")
    logger.info("=" * 70)

    test_model = ACNNBiLSTM(cfg)
    dummy = torch.randn(2, cfg.in_channels, cfg.target_img_height, cfg.target_img_width)
    with torch.no_grad():
        out = test_model(dummy)
    assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"
    total_p, train_p = test_model.count_parameters()
    logger.info("Model output shape: %s [OK]", out.shape)
    logger.info("Total parameters: %s", f"{total_p:,}")
    logger.info("Trainable parameters: %s", f"{train_p:,}")
    del test_model

    # ── Phase 6: Cross-validation (optional) ─────────────────────────────
    if cfg.run_cv:
        logger.info("\n" + "=" * 70)
        logger.info("Phase 6: 10-Fold Subject-Level Cross-Validation")
        logger.info("=" * 70)

        train_subject_ids, subject_window_map = build_subject_window_map(train_subjects)
        cv_results = run_cross_validation(
            X_train_cwt, y_train, train_subject_ids,
            subject_window_map, device, cfg,
        )
    else:
        logger.info("\nPhase 6: Cross-validation SKIPPED (--no-cv)")

    # ── Phase 7: Final training ──────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Phase 7: Final model training")
    logger.info("=" * 70)

    # 90/10 split of training subjects for early stopping
    train_subject_ids, subject_window_map = build_subject_window_map(train_subjects)
    np.random.seed(cfg.random_state)
    n_subj = len(train_subject_ids)
    shuffled = np.random.permutation(n_subj)
    split_idx = int((1.0 - cfg.val_split) * n_subj)

    final_train_sids = train_subject_ids[shuffled[:split_idx]]
    final_val_sids = train_subject_ids[shuffled[split_idx:]]

    final_train_win = np.concatenate([subject_window_map[sid] for sid in final_train_sids])
    final_val_win = np.concatenate([subject_window_map[sid] for sid in final_val_sids])

    logger.info("Final train: %d subjects (%d windows)", len(final_train_sids), len(final_train_win))
    logger.info("Final val (early stop): %d subjects (%d windows)", len(final_val_sids), len(final_val_win))

    # ── Phase 7a: Diagnostics ────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Phase 7a: Diagnostics")
    logger.info("=" * 70)

    run_mean_baseline(
        y_train_windows=y_train[final_train_win],
        y_val_windows=y_train[final_val_win],
    )

    # ── Fit BP scaler on training windows only ───────────────────────────
    # Fit BEFORE the overfit test so the test uses the same normalised labels
    # that full training will use.
    scaler = BPScaler().fit(y_train[final_train_win])
    logger.info(
        "BP scaler | SBP mean=%.1f std=%.1f | DBP mean=%.1f std=%.1f",
        scaler.mean_[0], scaler.std_[0],
        scaler.mean_[1], scaler.std_[1],
    )
    logger.info(
        "Training MAE will be in normalised units. "
        "Approx: 1.0 norm ~ SBP %.1f mmHg | DBP %.1f mmHg",
        scaler.std_[0], scaler.std_[1],
    )
    scaler.save(cfg.checkpoints_dir / "bp_scaler.npz")

    y_train_norm = scaler.transform(y_train)
    y_test_norm  = scaler.transform(y_test)

    if args.diagnose:
        run_overfit_test(
            X_cwt=X_train_cwt,
            y=y_train_norm,
            final_train_sids=final_train_sids,
            subject_window_map=subject_window_map,
            cfg=cfg,
            scaler=scaler,
        )
    else:
        logger.info("Overfit test SKIPPED (pass --diagnose to enable)")

    train_dataset = BPDataset(X_train_cwt, y_train_norm)
    train_loader = DataLoader(
        Subset(train_dataset, final_train_win.tolist()),
        batch_size=cfg.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        Subset(train_dataset, final_val_win.tolist()),
        batch_size=cfg.batch_size, shuffle=False,
    )

    model = ACNNBiLSTM(cfg).to(device)
    checkpoint_path = cfg.checkpoints_dir / "best_model.pth"

    model, history, best_val = train_model(
        model, train_loader, val_loader, device, cfg,
        checkpoint_path=checkpoint_path,
    )
    logger.info("Best validation MAE: %.4f", best_val)

    # Save training curves
    save_training_curves(
        history, cfg.images_dir / "training_curves.png",
        title=f"Training Curve (LR={cfg.lr}, lstm_hidden={cfg.lstm_hidden})",
    )

    # ── Phase 8: Test evaluation ─────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Phase 8: Test set evaluation")
    logger.info("=" * 70)

    test_dataset = BPDataset(X_test_cwt, y_test_norm)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    criterion = torch.nn.L1Loss()
    test_loss_norm, test_preds_norm, _ = evaluate(model, test_loader, criterion, device)
    logger.info("Test MAE (normalised units): %.4f", test_loss_norm)

    # Denormalise predictions back to mmHg for all metrics and plots
    test_preds = scaler.inverse_transform(test_preds_norm)

    sbp_metrics = compute_all_metrics(y_test[:, 0], test_preds[:, 0], "SBP (Systolic)")
    dbp_metrics = compute_all_metrics(y_test[:, 1], test_preds[:, 1], "DBP (Diastolic)")

    print_summary_table(sbp_metrics, dbp_metrics)
    print_bhs_table(sbp_metrics, dbp_metrics)

    # ── Phase 9: Save plots ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Phase 9: Saving plots")
    logger.info("=" * 70)

    save_bland_altman(
        y_test[:, 0], test_preds[:, 0],
        "Bland-Altman for SBP prediction via Trial 3",
        cfg.images_dir / "bland_altman_sbp.png",
    )
    save_bland_altman(
        y_test[:, 1], test_preds[:, 1],
        "Bland-Altman for DBP prediction via Trial 3",
        cfg.images_dir / "bland_altman_dbp.png",
    )

    # ── Save final model ─────────────────────────────────────────────────
    final_path = cfg.checkpoints_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    logger.info("Final model saved to %s", final_path)

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
