"""ACNN block: Conv2D → BatchNorm → SELU → SE Attention.

Operation order follows the paper text (Section 3.3), NOT the figure label:
  - Paper text: "BN alleviates gradient dispersion... Then, the self-activation
    function is utilized" — BN comes first, SELU comes after.
  - Logical reason: BN normalizes raw conv output to zero-mean, unit-variance,
    which is the exact input condition required for SELU's self-normalization
    guarantee (Klambauer et al., NeurIPS 2017, arXiv:1706.03462) to hold.
    Placing BN after SELU (as Figure 3 label suggests) would undo SELU's
    normalization via BN's learned γ/β parameters, defeating its purpose.
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
        # SELU: inplace=True must NOT be used — it corrupts negative-region
        # gradients in autograd. SELU is self-normalizing (Klambauer 2017).
        self.act = nn.SELU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.attn = SEBlock(out_channels, reduction=se_reduction)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)    # BN first: normalises raw conv output to ~N(0,1)
        x = self.act(x)   # SELU second: self-normalisation guarantee requires
                          # near-standard-normal input (provided by BN above)
        x = self.attn(x)
        if self.pool is not None:
            x = self.pool(x)
        return x
