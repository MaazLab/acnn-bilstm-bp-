"""Raw data loading — CSV PPG files and Excel labels.

Handles varying column header formats across CSV files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from acnn_bilstm.config import Config

logger = logging.getLogger(__name__)

# Known header variants across CSV files
_COLUMN_CANDIDATES = [
    ["channel 1", "channel 2", "channel 3", "channel 4"],
    ["channel_1", "channel_2", "channel_3", "channel_4"],
    ["ch1", "ch2", "ch3", "ch4"],
]
_STANDARD_COLS = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]


def load_labels(labels_xlsx: Path) -> pd.DataFrame:
    """Load subject labels (ID, SBP, DBP) from Excel file.

    Returns DataFrame indexed by subject ID with columns SBP(mmHg), DBP(mmHg).
    """
    df = pd.read_excel(labels_xlsx)
    required = {"ID", "SBP(mmHg)", "DBP(mmHg)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in Excel: {missing}. Found: {list(df.columns)}"
        )
    df = df.copy()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df = df.dropna(subset=["ID"])
    df["ID"] = df["ID"].astype(int)
    df = df.set_index("ID")
    logger.info("Loaded labels for %d subjects", len(df))
    return df


def load_one_subject(csv_path: Path) -> pd.DataFrame:
    """Load a single subject's 4-channel PPG signal from CSV.

    Returns DataFrame with standardised column names:
    ``Channel 1`` … ``Channel 4``.
    """
    df = pd.read_csv(csv_path)
    norm = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=norm)

    chosen = None
    for cols in _COLUMN_CANDIDATES:
        if all(c in df.columns for c in cols):
            chosen = cols
            break
    if chosen is None:
        raise ValueError(
            f"{csv_path.name}: Could not match 4 channels. Found: {list(df.columns)}"
        )

    sig = df[chosen].astype(float).copy()
    sig.columns = _STANDARD_COLS
    return sig


def load_all_subjects(
    cfg: Config,
) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    """Load all subject CSVs and match with labels.

    Returns
    -------
    dataset : list of dict
        Each dict has keys: ``id``, ``sbp``, ``dbp``, ``ppg_raw`` (DataFrame).
    skipped : list of (filename, reason)
    """
    labels = load_labels(cfg.labels_xlsx)
    dataset: List[Dict] = []
    skipped: List[Tuple[str, str]] = []

    csv_files = sorted(cfg.data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cfg.data_dir}")

    discard_set = set(cfg.discard_ids) if cfg.discard_ids else set()

    for csv_path in csv_files:
        try:
            subject_id = int(csv_path.stem)
        except ValueError:
            skipped.append((csv_path.name, "Filename is not a number"))
            continue

        if subject_id in discard_set:
            skipped.append((csv_path.name, "Discarded: incomplete wavelength signal"))
            continue

        if subject_id not in labels.index:
            skipped.append((csv_path.name, f"ID={subject_id} not in labels"))
            continue

        try:
            ppg_raw = load_one_subject(csv_path)
        except ValueError as e:
            skipped.append((csv_path.name, str(e)))
            continue

        row = labels.loc[subject_id]
        dataset.append(
            {
                "id": subject_id,
                "sbp": float(row["SBP(mmHg)"]),
                "dbp": float(row["DBP(mmHg)"]),
                "ppg_raw": ppg_raw,
            }
        )

    logger.info(
        "Loaded %d subjects, skipped %d files", len(dataset), len(skipped)
    )
    if skipped:
        for name, reason in skipped[:10]:
            logger.warning("  Skipped %s: %s", name, reason)

    return dataset, skipped
