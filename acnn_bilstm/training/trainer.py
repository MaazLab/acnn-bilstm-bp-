"""Training loop with early stopping and checkpointing.

Paper Section 3.3: Adam optimizer, lr=0.001, MAE loss, early stopping.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from acnn_bilstm.config import Config

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Select the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    running = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        running += loss.item() * X_batch.size(0)
    return running / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model. Returns (avg_loss, predictions, targets)."""
    model.eval()
    running = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            running += criterion(outputs, y_batch).item() * X_batch.size(0)
            preds.append(outputs.cpu().numpy())
            targets.append(y_batch.cpu().numpy())
    avg_loss = running / len(loader.dataset)
    return avg_loss, np.concatenate(preds), np.concatenate(targets)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[nn.Module, Dict[str, List[float]], float]:
    """Full training loop with early stopping.

    Parameters
    ----------
    model : nn.Module
        Model already moved to *device*.
    train_loader, val_loader : DataLoader
        Training and validation data loaders.
    device : torch.device
    cfg : Config
    checkpoint_path : Path, optional
        If given, save best model state dict here.

    Returns
    -------
    model : nn.Module
        Model with best validation weights loaded.
    history : dict
        ``{"train_loss": [...], "val_loss": [...]}``.
    best_val_loss : float
    """
    criterion = nn.L1Loss()  # MAE — paper Section 3.3
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    t0 = time.time()
    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            patience_counter += 1

        # Log every 10 epochs or on improvement or at early stop
        if (epoch + 1) % 10 == 0 or improved or patience_counter >= cfg.patience:
            elapsed = time.time() - t0
            marker = " *" if improved else ""
            logger.info(
                "Epoch %3d/%d | Train MAE: %.4f | Val MAE: %.4f | "
                "Patience: %d/%d | %.0fs%s",
                epoch + 1, cfg.num_epochs, train_loss, val_loss,
                patience_counter, cfg.patience, elapsed, marker,
            )

        if patience_counter >= cfg.patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    model.load_state_dict(best_state)
    logger.info("Best validation MAE: %.4f", best_val_loss)
    return model, history, best_val_loss
