"""Squeeze-and-Excitation (SE) channel attention block.

Used between convolutional layers to learn inter-channel dependencies
(paper Section 3.1.1). Reduction ratio defaults to 16 following the
original SE-Net paper (Hu et al. 2018).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Parameters
    ----------
    channels : int
        Number of input (and output) channels.
    reduction : int
        Reduction ratio for the bottleneck FC layer.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze
        y = self.pool(x).view(b, c)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        # Scale
        return x * y
