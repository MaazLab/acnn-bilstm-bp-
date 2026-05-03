"""Evaluation metrics — R², ME ± SD, MAE, RMSE, AAMI, BHS grade.

Paper Section 3.4, Tables 1-4.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, name: str = ""
) -> Dict[str, float | bool | str]:
    """Compute full metric suite for one BP type (SBP or DBP).

    Returns dict with keys: r2, me, sd, mae, rmse, aami, bhs_grade,
    pct_5, pct_10, pct_15.
    """
    errors = y_pred - y_true

    r2 = r2_score(y_true, y_pred)
    me = float(np.mean(errors))
    sd = float(np.std(errors))
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # AAMI criteria: ME < ±5 mmHg, SD < 8 mmHg
    aami_pass = (abs(me) < 5) and (sd < 8)

    # BHS grade
    abs_errors = np.abs(errors)
    pct_5 = float(np.mean(abs_errors <= 5) * 100)
    pct_10 = float(np.mean(abs_errors <= 10) * 100)
    pct_15 = float(np.mean(abs_errors <= 15) * 100)

    if pct_5 >= 60 and pct_10 >= 85 and pct_15 >= 95:
        bhs_grade = "A"
    elif pct_5 >= 50 and pct_10 >= 75 and pct_15 >= 90:
        bhs_grade = "B"
    elif pct_5 >= 40 and pct_10 >= 65 and pct_15 >= 85:
        bhs_grade = "C"
    else:
        bhs_grade = "D"

    logger.info(
        "\n%s\n  %s Results\n%s\n"
        "  R2:        %.4f\n"
        "  ME +/- SD: %.2f +/- %.2f mmHg\n"
        "  MAE:       %.2f mmHg\n"
        "  RMSE:      %.2f mmHg\n"
        "  AAMI:      %s (ME<+/-5: %s, SD<8: %s)\n"
        "  BHS Grade: %s (<=5: %.1f%%, <=10: %.1f%%, <=15: %.1f%%)",
        "=" * 50, name, "=" * 50,
        r2, me, sd, mae, rmse,
        "Yes" if aami_pass else "No",
        abs(me) < 5, sd < 8,
        bhs_grade, pct_5, pct_10, pct_15,
    )

    return {
        "r2": r2, "me": me, "sd": sd, "mae": mae, "rmse": rmse,
        "aami": aami_pass, "bhs_grade": bhs_grade,
        "pct_5": pct_5, "pct_10": pct_10, "pct_15": pct_15,
    }


def print_summary_table(
    sbp_metrics: Dict, dbp_metrics: Dict
) -> None:
    """Print summary table matching paper Table 2 format."""
    header = (
        f"\n{'=' * 80}\n"
        f"{'Trial 3 (MWPPG Fusion)':^80}\n"
        f"{'=' * 80}\n"
        f"{'':15} {'R2':>6} {'ME +/- SD':>15} {'MAE':>8} {'RMSE':>8} {'AAMI':>6} {'BHS':>5}"
    )
    sbp_line = (
        f"{'SBP (mmHg)':15} {sbp_metrics['r2']:>6.2f} "
        f"{sbp_metrics['me']:>6.2f} +/- {sbp_metrics['sd']:<6.2f} "
        f"{sbp_metrics['mae']:>8.2f} {sbp_metrics['rmse']:>8.2f} "
        f"{'Yes' if sbp_metrics['aami'] else 'No':>6} "
        f"{sbp_metrics['bhs_grade']:>5}"
    )
    dbp_line = (
        f"{'DBP (mmHg)':15} {dbp_metrics['r2']:>6.2f} "
        f"{dbp_metrics['me']:>6.2f} +/- {dbp_metrics['sd']:<6.2f} "
        f"{dbp_metrics['mae']:>8.2f} {dbp_metrics['rmse']:>8.2f} "
        f"{'Yes' if dbp_metrics['aami'] else 'No':>6} "
        f"{dbp_metrics['bhs_grade']:>5}"
    )
    logger.info("%s\n%s\n%s", header, sbp_line, dbp_line)


def print_bhs_table(
    sbp_metrics: Dict, dbp_metrics: Dict
) -> None:
    """Print BHS grade tables matching paper Tables 3 & 4."""
    for bp, m in [("SBP", sbp_metrics), ("DBP", dbp_metrics)]:
        logger.info(
            "\nBHS Scheme - %s Prediction\n"
            "%15s %10s %10s %10s\n"
            "%15s %9.0f%% %9.0f%% %9.0f%%\n"
            "%15s %10s %10s %10s\n"
            "%15s %10s %10s %10s\n"
            "%15s %10s %10s %10s",
            bp,
            "", "<=5 mmHg", "<=10 mmHg", "<=15 mmHg",
            "Trial 3", m["pct_5"], m["pct_10"], m["pct_15"],
            "Grade A", "60%", "85%", "95%",
            "Grade B", "50%", "75%", "90%",
            "Grade C", "40%", "65%", "85%",
        )
