"""Small Transformer encoder with causal masking and learned positional encoding.

Attention is implemented explicitly rather than through
``scaled_dot_product_attention`` because the weights themselves are part of the
deliverable: :mod:`src.explain` reads the final query position's attention over
the window to say which prior transactions the model looked at. At a 32-position
window the hand-written version costs nothing.

Positional encoding is learned and indexed by absolute slot, which — because
windows are left-padded — means "distance back from the transaction being
scored". That is the semantically useful frame for this problem: slot ``L-1`` is
always "now", slot ``L-2`` is always "the previous transaction", regardless of
how much history the customer has.

Masking has two parts. The causal mask stops a position attending to later ones
(belt and braces: the window already contains nothing later). The padding mask
stops any position attending to a padded slot. Their union can leave a padded
*query* with nothing to attend to at all, which produces NaN out of the softmax,
so the diagonal is always left open. Those rows are garbage by construction and
are never read, and because padded slots are masked as *keys* the garbage cannot
reach the scored position.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from src.models.base import ModelConfig, SequenceClassifier, init_weights


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention that returns its weights."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        """Build the projections.

        Args:
            d_model: Model width.
            n_heads: Number of heads; must divide ``d_model``.
            dropout: Dropout applied to the attention output.

        Raises:
            ValueError: If ``d_model`` is not divisible by ``n_heads``.
        """
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, blocked: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply attention.

        When the weights are not needed the fused kernel is used, which is worth
        roughly a factor of two in training wall-clock on CPU. The explicit path
        below it computes exactly the same function and is what
        :mod:`src.explain` calls.

        Args:
            x: Tensor ``(B, L, d_model)``.
            blocked: Bool tensor ``(B, L, L)``, True where a query may not attend
                to a key.
            need_weights: Return the attention matrix as well.

        Returns:
            Tuple of (output ``(B, L, d_model)``, weights ``(B, H, L, L)`` or
            ``None``).
        """
        b, length, d = x.shape
        qkv = self.qkv(x).view(b, length, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # Indexed rather than unpacked: unpacking a tensor along a dimension
        # makes the ONNX tracer warn about a fixed iteration count.
        q, k, v = qkv[0], qkv[1], qkv[2]

        if not need_weights:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=~blocked.unsqueeze(1)
            )
            return self.dropout(self.out(out.transpose(1, 2).reshape(b, length, d))), None

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(blocked.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(b, length, d)
        return self.dropout(self.out(out)), weights


class TransformerBlock(nn.Module):
    """Pre-norm attention block with a feed-forward sublayer."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, x: torch.Tensor, blocked: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run attention and feed-forward with residual connections.

        Args:
            x: Tensor ``(B, L, d_model)``.
            blocked: Bool attention mask ``(B, L, L)``.
            need_weights: Return the attention matrix as well.

        Returns:
            Tuple of (output, attention weights or ``None``).
        """
        attended, weights = self.attention(self.norm1(x), blocked, need_weights)
        x = x + attended
        x = x + self.feed_forward(self.norm2(x))
        return x, weights


class TransformerClassifier(SequenceClassifier):
    """Causal Transformer encoder read at the final window position."""

    family = "transformer"

    def __init__(self, config: ModelConfig) -> None:
        """Build the encoder stack.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.positional = nn.Embedding(config.max_len, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.to_hidden = (
            nn.Identity()
            if config.d_model == config.hidden
            else nn.Linear(config.d_model, config.hidden)
        )
        self.apply(init_weights)

    @staticmethod
    def _blocked_mask(mask: torch.Tensor) -> torch.Tensor:
        """Build the combined causal and padding mask.

        Args:
            mask: Float mask ``(B, L)``, 1 for real positions.

        Returns:
            Bool tensor ``(B, L, L)``, True where attention is forbidden. The
            diagonal is always permitted so no softmax row is fully masked.
        """
        length = mask.size(1)
        # Built from index comparisons rather than ``torch.triu`` / ``torch.eye``:
        # both export to ONNX operators that ONNX Runtime's CPU provider does not
        # implement, and this produces the identical mask from ops it does.
        positions = torch.arange(length, device=mask.device)
        query = positions.unsqueeze(1)
        key = positions.unsqueeze(0)
        causal = key > query
        diagonal = key == query

        pad_keys = mask.unsqueeze(1) < 0.5  # (B, 1, L)
        blocked = causal.unsqueeze(0) | pad_keys
        return blocked & ~diagonal.unsqueeze(0)

    def _run_blocks(
        self, x: torch.Tensor, mask: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply positional encoding and every block.

        Args:
            x: Encoded positions ``(B, L, d_model)``.
            mask: Float mask ``(B, L)``.
            need_weights: Collect per-layer attention matrices.

        Returns:
            Tuple of (final hidden states, per-layer attention weights; the list
            is empty when ``need_weights`` is False).
        """
        positions = torch.arange(x.size(1), device=x.device)
        h = x + self.positional(positions).unsqueeze(0)
        h = h * mask.unsqueeze(-1)
        blocked = self._blocked_mask(mask)
        weights: list[torch.Tensor] = []
        for block in self.blocks:
            h, w = block(h, blocked, need_weights)
            if w is not None:
                weights.append(w)
        return self.final_norm(h), weights

    def encode_sequence(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode the window and read the scored transaction's position.

        Args:
            x: Encoded positions ``(B, L, d_model)``.
            mask: Float mask ``(B, L)``.

        Returns:
            Tensor ``(B, hidden)``.
        """
        h, _ = self._run_blocks(x, mask)
        return self.to_hidden(h[:, -1, :])

    def attention_weights(
        self, numeric: torch.Tensor, categorical: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Attention paid by the scored transaction to each window position.

        Averaged over heads and layers. See :mod:`src.explain` for why this is a
        description of where the model looked and not a causal explanation of
        why it decided.

        Args:
            numeric: Float tensor ``(B, L, F)``.
            categorical: Long tensor ``(B, L, C)``.
            mask: Float tensor ``(B, L)``.

        Returns:
            Tensor ``(B, L)`` of weights, zero at padded positions.
        """
        x = self.encoder(numeric, categorical, mask)
        _, weights = self._run_blocks(x, mask, need_weights=True)
        # Each entry is (B, H, L, L); take the final query row and average.
        final_row = torch.stack([w[:, :, -1, :].mean(dim=1) for w in weights], dim=0)
        averaged = final_row.mean(dim=0)
        return averaged * mask
