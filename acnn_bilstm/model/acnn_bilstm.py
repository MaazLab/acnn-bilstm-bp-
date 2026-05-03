"""Full ACNN-BiLSTM model (paper Section 3.3, Figure 3).

Architecture:
  Input (B, 12, 128, 256)
  → Initial Conv2D + ReLU + BN
  → 7 × ACNN Block (Conv2D + ReLU + BN + SE Attention)
  → Reshape to sequence (B, seq_len, feat_dim)
  → 2 × BiLSTM
  → FC → Dropout → FC → Dropout
  → Output (B, 2)  [SBP, DBP]
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from acnn_bilstm.config import Config
from acnn_bilstm.model.acnn_block import ACNNBlock

logger = logging.getLogger(__name__)


class ACNNBiLSTM(nn.Module):
    """Attention-based CNN–BiLSTM for blood pressure prediction.

    Builds from ``Config`` so the architecture is fully parameterised.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        ch = cfg.cnn_channels  # [32, 32, 32, 64, 64, 128, 128, 256]
        st = cfg.cnn_strides   # [2, 1, 2, 1, 2, 2, 2]

        # ── Initial Conv (before the ×7 ACNN loop) ──────────────────────
        self.init_conv = nn.Conv2d(
            cfg.in_channels, ch[0], kernel_size=3, padding=1
        )
        self.init_act = nn.ReLU(inplace=True)
        self.init_bn = nn.BatchNorm2d(ch[0])

        # ── 7 ACNN blocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList()
        for i in range(7):
            self.blocks.append(
                ACNNBlock(ch[i], ch[i + 1], stride=st[i], se_reduction=cfg.se_reduction)
            )

        # ── Compute BiLSTM input dimensions via a dummy forward pass ────
        with torch.no_grad():
            dummy = torch.zeros(
                1, cfg.in_channels, cfg.target_img_height, cfg.target_img_width
            )
            feat = self._forward_cnn(dummy)
            _, c, h, w = feat.shape
            self.seq_len = w       # width → sequence length for BiLSTM
            self.seq_feat = c * h  # channels × height → feature dimension

        logger.info(
            "CNN output: (%d, %d, %d) -> BiLSTM seq_len=%d, feat_dim=%d, "
            "compression ratio=%.1f:1",
            c, h, w, self.seq_len, self.seq_feat,
            self.seq_feat / cfg.lstm_hidden,
        )

        # ── 2-layer BiLSTM ──────────────────────────────────────────────
        self.bilstm1 = nn.LSTM(
            self.seq_feat, cfg.lstm_hidden, batch_first=True, bidirectional=True
        )
        self.bilstm2 = nn.LSTM(
            cfg.lstm_hidden * 2, cfg.lstm_hidden, batch_first=True, bidirectional=True
        )

        # ── FC head: FCN → Dropout → FCN → Dropout → BP ────────────────
        self.fc1 = nn.Linear(cfg.lstm_hidden * 2, cfg.fc_hidden)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(cfg.fc_hidden, 2)
        self.drop2 = nn.Dropout(cfg.dropout)

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_bn(self.init_act(self.init_conv(x)))
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_cnn(x)           # (B, C, H, W)

        b, c, h, w = x.shape
        # width → time steps, channels*height → features
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, W, C, H)
        x = x.view(b, w, c * h)                 # (B, seq_len, feat_dim)

        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)

        x = x[:, -1, :]                         # last timestep → (B, 2*hidden)

        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x

    def count_parameters(self) -> tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
