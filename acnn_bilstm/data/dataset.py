"""PyTorch Dataset and data-split utilities."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from acnn_bilstm.config import Config

logger = logging.getLogger(__name__)


class BPDataset(Dataset):
    """Simple in-memory dataset of CWT images and BP labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def subject_level_split(
    dataset: List[Dict], cfg: Config
) -> Tuple[List[Dict], List[Dict]]:
    """80/20 subject-level train/test split (paper Section 3.3).

    Returns (train_subjects, test_subjects).
    """
    ids = [item["id"] for item in dataset]
    train_ids, test_ids = train_test_split(
        ids, test_size=cfg.test_size, random_state=cfg.random_state, shuffle=True
    )
    train_set = set(train_ids)
    test_set = set(test_ids)

    train = [item for item in dataset if item["id"] in train_set]
    test = [item for item in dataset if item["id"] in test_set]

    logger.info(
        "Subject split: %d train, %d test", len(train), len(test)
    )
    return train, test


def concat_windows(subjects: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate windows and labels across subjects.

    Returns (X, y) with shapes (N, window_samples, 4) and (N, 2).
    """
    X = np.concatenate([s["X"] for s in subjects], axis=0)
    y = np.concatenate([s["y"] for s in subjects], axis=0)
    return X, y


def build_subject_window_map(
    subjects: List[Dict],
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Map subject IDs to window indices in the concatenated array.

    Returns (subject_ids_array, {sid: array_of_window_indices}).
    """
    ids = np.array([s["id"] for s in subjects])
    counts = np.array([s["X"].shape[0] for s in subjects])
    cum = np.cumsum(counts)
    starts = np.concatenate([[0], cum[:-1]])

    mapping = {}
    for i, sid in enumerate(ids):
        mapping[sid] = np.arange(starts[i], cum[i])

    return ids, mapping
