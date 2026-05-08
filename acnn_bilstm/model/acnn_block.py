"""ACNN block: Conv2D → ReLU → BatchNorm → SE Attention.

One block as depicted in the paper's Figure 3 (inside the "×7" loop).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from acnn_bilstm.model.attention import SEBlock


class ACNNBlock(nn.Module):
    """Single Attention-based CNN block.

    Parameters
    ----------
    in_channels : int
        Number of input feature maps.
    out_channels : int
        Number of output feature maps.
    pool : bool
        If True, apply MaxPool2d(2, 2) after SE attention to halve spatial dims.
        MaxPool2d preserves the strongest activation in each 2x2 region
        (translational invariance), unlike strided conv which subsamples.
    se_reduction : int
        SE-block reduction ratio.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        # Conv2d always stride=1 — spatial reduction handled by MaxPool below
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.attn = SEBlock(out_channels, reduction=se_reduction)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.attn(x)
        if self.pool is not None:
            x = self.pool(x)
        return x
