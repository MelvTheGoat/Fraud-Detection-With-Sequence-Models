"""Shared input encoder and the interface every sequence model implements.

All three architectures consume the same tensors and expose the same forward
signature, so the training loop, the ONNX exporter and the serving path are
written once. The only thing that differs between them is how the encoded
sequence is reduced to a single logit.

Embedding dimensions
--------------------
Sizes come from the standard heuristic ``1.6 * cardinality ** 0.56`` (the fastai
rule), rounded, floored at 2 and capped by ``emb_max_dim``. The cap is doing real
work here: the merchant vocabulary is ~2.9k, and the uncapped rule would ask for
139 dimensions — a 400k-parameter table fitted from roughly a thousand positive
training examples. Capping at 16 keeps the merchant table around 47k parameters,
which is still the bulk of the model and is reported as such. The rule and the
cap are hyperparameters like any other; the resulting sizes are printed by
:func:`describe_embeddings` and reproduced in the README.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import nn

from src.data.schema import PAD_INDEX, CategoricalSpec


def embedding_dim(cardinality: int, max_dim: int = 16) -> int:
    """Embedding width for a categorical of the given cardinality.

    Args:
        cardinality: Number of distinct codes including the reserved ones.
        max_dim: Upper bound on the returned width.

    Returns:
        Embedding width, at least 2.
    """
    rule = int(round(1.6 * cardinality**0.56))
    return int(max(2, min(max_dim, rule)))


@dataclass
class ModelConfig:
    """Configuration shared by every sequence model.

    Attributes:
        n_numeric: Number of numeric features per position.
        categorical: Specs of the embedded categorical columns.
        max_len: Sequence window length.
        d_model: Width of the shared representation.
        dropout: Dropout applied in the encoder and the head.
        emb_max_dim: Cap on any single embedding width.
        hidden: Architecture-specific hidden width (RNN state, TCN channels,
            Transformer feed-forward width).
        n_layers: Number of recurrent layers / TCN blocks / Transformer blocks.
        n_heads: Attention heads, Transformer only.
        kernel_size: Convolution kernel width, TCN only.
        cell: ``"gru"`` or ``"lstm"``, RNN only.
    """

    n_numeric: int
    categorical: list[CategoricalSpec]
    max_len: int
    d_model: int = 64
    dropout: float = 0.1
    emb_max_dim: int = 16
    hidden: int = 64
    n_layers: int = 2
    n_heads: int = 4
    kernel_size: int = 3
    cell: str = "gru"
    embedding_dims: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Fill in embedding widths if the caller did not supply them."""
        if not self.embedding_dims:
            self.embedding_dims = [
                embedding_dim(spec.cardinality, self.emb_max_dim) for spec in self.categorical
            ]


class InputEncoder(nn.Module):
    """Maps (numeric, categorical) positions to a ``d_model`` representation.

    Padded positions are zeroed on the way out so a downstream convolution or
    attention cannot pick up a bias term from a slot that does not exist.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialise embeddings and the projection.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(spec.cardinality, dim, padding_idx=PAD_INDEX)
                for spec, dim in zip(config.categorical, config.embedding_dims, strict=True)
            ]
        )
        concat_dim = config.n_numeric + sum(config.embedding_dims)
        self.norm = nn.LayerNorm(concat_dim)
        self.project = nn.Linear(concat_dim, config.d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    @property
    def output_dim(self) -> int:
        """Width of the encoder output."""
        return int(self.config.d_model)

    def forward(
        self, numeric: torch.Tensor, categorical: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode a batch of padded windows.

        Args:
            numeric: Float tensor ``(B, L, F)``.
            categorical: Long tensor ``(B, L, C)``.
            mask: Float tensor ``(B, L)``, 1 for real positions.

        Returns:
            Float tensor ``(B, L, d_model)``, zero at padded positions.
        """
        parts = [numeric]
        for i, embedding in enumerate(self.embeddings):
            parts.append(embedding(categorical[..., i]))
        x = torch.cat(parts, dim=-1)
        x = self.project(self.norm(x))
        x = self.dropout(self.activation(x))
        return x * mask.unsqueeze(-1)


class SequenceClassifier(nn.Module, ABC):
    """Base class for the three architectures.

    Every subclass reads its prediction off the final window position, which is
    always the transaction being scored (windows are left-padded). That keeps the
    causal contract identical across architectures.
    """

    family: str = "base"

    def __init__(self, config: ModelConfig) -> None:
        """Initialise the shared encoder and classification head.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.encoder = InputEncoder(config)
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, 1),
        )

    @abstractmethod
    def encode_sequence(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reduce an encoded window to one vector per row.

        Args:
            x: Encoded positions ``(B, L, d_model)``.
            mask: Float mask ``(B, L)``.

        Returns:
            Tensor ``(B, hidden)`` summarising each window at its final position.
        """

    def forward(
        self, numeric: torch.Tensor, categorical: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Score a batch of windows.

        Args:
            numeric: Float tensor ``(B, L, F)``.
            categorical: Long tensor ``(B, L, C)``.
            mask: Float tensor ``(B, L)``.

        Returns:
            Logits of shape ``(B,)``.
        """
        x = self.encoder(numeric, categorical, mask)
        pooled = self.encode_sequence(x, mask)
        return self.head(pooled).squeeze(-1)

    def attention_weights(
        self, numeric: torch.Tensor, categorical: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor | None:
        """Per-position attribution over the window, if the architecture has one.

        Returns:
            Tensor ``(B, L)`` of weights summing to 1 over unmasked positions, or
            ``None`` for architectures without an attention mechanism.
        """
        del numeric, categorical, mask
        return None


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Total number of parameters in a model.

    Args:
        model: Any module.
        trainable_only: Count only parameters that require gradients.

    Returns:
        Parameter count.
    """
    return int(
        sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
    )


def parameter_breakdown(model: SequenceClassifier) -> dict[str, int]:
    """Split the parameter count into embeddings and everything else.

    Useful because for these models the embedding tables dominate, and a reader
    comparing "160k parameters" against a GBM deserves to know that most of it is
    a lookup table rather than computation.

    Args:
        model: A sequence classifier.

    Returns:
        Dict with ``embeddings``, ``other`` and ``total``.
    """
    embedding_params = sum(p.numel() for p in model.encoder.embeddings.parameters())
    total = count_parameters(model)
    return {"embeddings": int(embedding_params), "other": int(total - embedding_params), "total": int(total)}


def describe_embeddings(config: ModelConfig) -> list[dict[str, int | str]]:
    """Per-column embedding sizes, for the write-up.

    Args:
        config: Model configuration.

    Returns:
        List of dicts with column, cardinality, dimension and parameter count.
    """
    return [
        {
            "column": spec.column,
            "cardinality": spec.cardinality,
            "dim": dim,
            "params": spec.cardinality * dim,
        }
        for spec, dim in zip(config.categorical, config.embedding_dims, strict=True)
    ]


def roll_to_front(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Move left-padding to the back so real positions start at index 0.

    Windows are left-padded because the scored transaction must sit at a fixed
    final position for the convolutional and attention models. A recurrent model
    wants the opposite: it should not consume a variable-length run of zeros
    before it sees any data. Rolling each row by its own padding length gives the
    RNN a clean right-padded sequence, and the true final state is then read at
    ``length - 1``.

    Args:
        x: Tensor ``(B, L, D)``.
        mask: Float mask ``(B, L)`` with padding at the front.

    Returns:
        Tuple of (rolled tensor, per-row lengths as long tensor).
    """
    b, length, _ = x.shape
    lengths = mask.sum(dim=1).long().clamp(min=1)
    pad = length - lengths
    index = (torch.arange(length, device=x.device).unsqueeze(0) + pad.unsqueeze(1)) % length
    rolled = torch.gather(x, 1, index.unsqueeze(-1).expand(b, length, x.size(-1)))
    return rolled, lengths


def gather_at(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Select one position per row from a ``(B, L, D)`` tensor.

    Args:
        x: Tensor ``(B, L, D)``.
        index: Long tensor ``(B,)`` of positions.

    Returns:
        Tensor ``(B, D)``.
    """
    b, _, d = x.shape
    return torch.gather(x, 1, index.view(b, 1, 1).expand(b, 1, d)).squeeze(1)


def init_weights(module: nn.Module) -> None:
    """Xavier-initialise linear layers and normal-initialise embeddings.

    Args:
        module: Module to initialise, intended for ``model.apply``.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.embedding_dim))
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0.0)
