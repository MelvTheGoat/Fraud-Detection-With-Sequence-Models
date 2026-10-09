"""Recurrent encoder (GRU or LSTM) over the transaction window.

The obvious sequence baseline, and the one most likely to be enough. The window
is rolled so real transactions start at index 0 and padding trails behind, the
recurrence runs left to right, and the state is read at ``length - 1``. Padding
is therefore never consumed before real data and never read after it — no packed
sequences needed, which keeps the model exportable to ONNX as a plain graph.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.base import ModelConfig, SequenceClassifier, gather_at, init_weights, roll_to_front


class RecurrentClassifier(SequenceClassifier):
    """GRU or LSTM over the encoded window.

    Attributes:
        family: ``"gru"`` or ``"lstm"`` depending on the configured cell.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the recurrent stack.

        Args:
            config: Model configuration; ``cell`` selects GRU or LSTM.

        Raises:
            ValueError: If ``config.cell`` is not ``"gru"`` or ``"lstm"``.
        """
        super().__init__(config)
        cell = config.cell.lower()
        if cell not in {"gru", "lstm"}:
            raise ValueError(f"cell must be 'gru' or 'lstm', got {config.cell!r}")
        self.family = cell
        rnn_class = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_class(
            input_size=config.d_model,
            hidden_size=config.hidden,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
        self.apply(init_weights)

    def encode_sequence(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the recurrence and read the state at the scored transaction.

        Args:
            x: Encoded positions ``(B, L, d_model)``.
            mask: Float mask ``(B, L)``.

        Returns:
            Tensor ``(B, hidden)``.
        """
        rolled, lengths = roll_to_front(x, mask)
        outputs, _ = self.rnn(rolled)
        return gather_at(outputs, lengths - 1)
