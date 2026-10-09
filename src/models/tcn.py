"""Temporal convolutional network with dilated causal convolutions.

Causality is structural rather than enforced by a mask: each convolution is
left-padded by ``(kernel_size - 1) * dilation`` and the right-hand overhang is
sliced off, so output position ``t`` is a function of inputs ``<= t`` only. There
is no way for a future transaction to reach the prediction even if the window
construction were wrong, which makes this the architecture where the temporal
guarantee is hardest to break by accident.

Dilations double per block, so ``n_layers`` blocks with kernel 3 cover
``1 + 2 * (2^n_layers - 1)`` positions. Four blocks reach 31 and five reach 63,
which brackets the 32-transaction window — the receptive field is checked against
the window length at construction and reported in :meth:`receptive_field`.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from src.models.base import ModelConfig, SequenceClassifier, init_weights

LOGGER = logging.getLogger(__name__)


class CausalConvBlock(nn.Module):
    """Two dilated causal convolutions with a residual connection."""

    def __init__(
        self, channels: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        """Build the block.

        Args:
            channels: Input and output channel count.
            kernel_size: Convolution width.
            dilation: Dilation factor.
            dropout: Dropout probability.
        """
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _causal(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        """Apply a convolution with left-only padding.

        Args:
            conv: The convolution to apply.
            x: Tensor ``(B, C, L)``.

        Returns:
            Tensor ``(B, C, L)`` where position t depends only on inputs <= t.
        """
        return conv(nn.functional.pad(x, (self.pad, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block.

        Args:
            x: Tensor ``(B, C, L)``.

        Returns:
            Tensor ``(B, C, L)``.
        """
        h = self.dropout(self.activation(self.norm1(self._causal(self.conv1, x))))
        h = self.dropout(self.activation(self.norm2(self._causal(self.conv2, h))))
        return x + h


class TCNClassifier(SequenceClassifier):
    """Stack of dilated causal convolution blocks, read at the final position."""

    family = "tcn"

    def __init__(self, config: ModelConfig) -> None:
        """Build the convolution stack.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.input_proj = (
            nn.Identity()
            if config.d_model == config.hidden
            else nn.Conv1d(config.d_model, config.hidden, 1)
        )
        self.blocks = nn.ModuleList(
            [
                CausalConvBlock(config.hidden, config.kernel_size, 2**i, config.dropout)
                for i in range(config.n_layers)
            ]
        )
        self.apply(init_weights)
        if self.receptive_field() < config.max_len:
            LOGGER.warning(
                "TCN receptive field is %d but the window is %d: the model cannot "
                "see the oldest transactions the other architectures get. Increase "
                "n_layers or kernel_size.",
                self.receptive_field(),
                config.max_len,
            )

    def receptive_field(self) -> int:
        """Number of past positions the final output can see."""
        k = self.config.kernel_size
        return 1 + sum(2 * (k - 1) * (2**i) for i in range(self.config.n_layers))

    def encode_sequence(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Convolve the window and read the final position.

        Args:
            x: Encoded positions ``(B, L, d_model)``.
            mask: Float mask ``(B, L)``.

        Returns:
            Tensor ``(B, hidden)``.
        """
        h = self.input_proj(x.transpose(1, 2))
        mask_c = mask.unsqueeze(1)
        for block in self.blocks:
            h = block(h) * mask_c
        return h[:, :, -1]
