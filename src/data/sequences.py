"""Windowed per-customer sequences with padding and masking.

For a transaction at position ``p`` in a customer's ordered history, the window is
``history[p - L + 1 : p + 1]`` — the last ``L`` transactions *including* the one
being scored, which sits at the final position. Windows are **left-padded**, so
position ``L - 1`` is always the transaction under decision. Every model in
:mod:`src.models` reads its prediction off that position, which makes the causal
contract identical across the GRU, the TCN and the Transformer.

The transaction being scored contributes only its authorisation-request fields
(amount, merchant, device, channel, ...). Its label is the target and the labels
of the prior transactions are never inputs: a chargeback lands weeks later, so a
production scorer does not know whether last Tuesday's purchase turned out to be
fraud.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.features import FeatureBuilder, FeatureSpec
from src.data.schema import PAD_INDEX, TransactionSchema

S = TransactionSchema

#: Value marking a padded slot in the window index array.
PAD_ROW = -1


def build_windows(
    customer_ids: np.ndarray, timestamps: np.ndarray, max_len: int
) -> np.ndarray:
    """Build left-padded index windows of the last ``max_len`` transactions.

    Args:
        customer_ids: Customer key per row, in the frame's row order.
        timestamps: Sortable timestamps per row (int64 nanoseconds or float
            seconds), in the frame's row order.
        max_len: Window length, including the scored transaction.

    Returns:
        Array of shape ``(n_rows, max_len)`` of row indices into the original
        frame, left-padded with :data:`PAD_ROW`. Row ``i``'s window ends with
        ``i`` itself and contains only earlier transactions of the same customer.

    Raises:
        ValueError: If ``max_len`` is not positive.
    """
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    n = len(customer_ids)
    windows = np.full((n, max_len), PAD_ROW, dtype=np.int64)
    if n == 0:
        return windows

    codes, _ = pd.factorize(customer_ids)
    # Stable sort by (customer, time) gives contiguous, chronologically ordered
    # blocks per customer while preserving the caller's tie-breaking.
    order = np.lexsort((timestamps, codes))
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1

    pad = np.full(max_len - 1, PAD_ROW, dtype=np.int64)
    for group in np.split(order, boundaries):
        padded = np.concatenate([pad, group])
        view = np.lib.stride_tricks.sliding_window_view(padded, max_len)
        windows[group] = view
    return windows


@dataclass
class SequenceData:
    """Dense arrays backing every sequence model.

    Attributes:
        numeric: Standardised numeric features, shape ``(n_rows, n_numeric)``.
        categorical: Integer codes, shape ``(n_rows, n_categorical)``.
        windows: Index windows from :func:`build_windows`.
        labels: Binary fraud labels per row.
        fraud_type: Mechanism label per row (``"none"`` for legitimate).
        timestamps: Row timestamps, for temporal analysis and monitoring.
        amounts: Raw transaction amounts, needed by the cost-sensitive policy.
        spec: Feature specification describing the column layout.
    """

    numeric: np.ndarray
    categorical: np.ndarray
    windows: np.ndarray
    labels: np.ndarray
    fraud_type: np.ndarray
    timestamps: np.ndarray
    amounts: np.ndarray
    spec: FeatureSpec

    def __len__(self) -> int:
        """Number of scorable rows."""
        return int(self.numeric.shape[0])

    @property
    def n_numeric(self) -> int:
        """Number of numeric features per sequence position."""
        return int(self.numeric.shape[1])

    @property
    def max_len(self) -> int:
        """Window length."""
        return int(self.windows.shape[1])


def build_sequence_data(
    frame: pd.DataFrame, builder: FeatureBuilder, max_len: int
) -> SequenceData:
    """Assemble :class:`SequenceData` from a built feature frame.

    Args:
        frame: Output of :meth:`FeatureBuilder.fit_transform` or ``transform``.
        builder: The fitted builder, used for the scaling statistics and spec.
        max_len: Window length.

    Returns:
        Populated :class:`SequenceData`.

    Raises:
        RuntimeError: If the builder has not been fitted.
    """
    if builder.spec is None:
        raise RuntimeError("FeatureBuilder must be fitted before building sequences")
    spec = builder.spec
    numeric = builder.scale(frame[spec.numeric].to_numpy(dtype=np.float64))
    categorical = frame[spec.categorical_columns].to_numpy(dtype=np.int64)
    windows = build_windows(
        frame[S.CUSTOMER_ID].to_numpy(),
        frame[S.TIMESTAMP].astype("int64").to_numpy(),
        max_len,
    )
    labels = (
        frame[S.IS_FRAUD].to_numpy(dtype=np.float32)
        if S.IS_FRAUD in frame.columns
        else np.zeros(len(frame), dtype=np.float32)
    )
    fraud_type = (
        frame[S.FRAUD_TYPE].to_numpy()
        if S.FRAUD_TYPE in frame.columns
        else np.full(len(frame), "none", dtype=object)
    )
    return SequenceData(
        numeric=numeric,
        categorical=categorical,
        windows=windows,
        labels=labels,
        fraud_type=fraud_type,
        timestamps=frame[S.TIMESTAMP].to_numpy(),
        amounts=frame[S.AMOUNT].to_numpy(dtype=np.float64),
        spec=spec,
    )


def gather(data: SequenceData, rows: np.ndarray) -> dict[str, torch.Tensor]:
    """Materialise a batch of padded sequences.

    Args:
        data: Backing arrays.
        rows: Row indices to score.

    Returns:
        Dict with ``numeric`` ``(B, L, F)``, ``categorical`` ``(B, L, C)``,
        ``mask`` ``(B, L)`` where 1 marks a real transaction, and ``label``
        ``(B,)``.
    """
    win = data.windows[rows]
    valid = win >= 0
    safe = np.where(valid, win, 0)

    numeric = data.numeric[safe]
    numeric[~valid] = 0.0
    categorical = data.categorical[safe]
    categorical[~valid] = PAD_INDEX

    return {
        "numeric": torch.from_numpy(numeric),
        "categorical": torch.from_numpy(categorical),
        "mask": torch.from_numpy(valid.astype(np.float32)),
        "label": torch.from_numpy(data.labels[rows]),
    }


class SequenceBatcher:
    """Minibatch iterator over rows of a :class:`SequenceData`.

    A hand-rolled iterator rather than ``DataLoader``: batches are pure numpy
    gathers, worker processes would only add overhead on CPU, and weighted
    sampling needs to be swappable per experiment arm.

    Attributes:
        drop_last: Whether to drop a final short batch (training only).
    """

    def __init__(
        self,
        data: SequenceData,
        rows: np.ndarray,
        batch_size: int,
        *,
        shuffle: bool = False,
        sample_weights: np.ndarray | None = None,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """Initialise the batcher.

        Args:
            data: Backing arrays.
            rows: Row indices this batcher may emit.
            batch_size: Rows per batch.
            shuffle: Shuffle row order each epoch.
            sample_weights: Per-row sampling weights aligned to ``rows``. When
                given, each epoch draws ``len(rows)`` rows *with replacement*
                proportional to the weights.
            seed: RNG seed.
            drop_last: Drop a trailing partial batch.
        """
        self.data = data
        self.rows = np.asarray(rows)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.rng = np.random.default_rng(seed)
        self.drop_last = drop_last

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self.rows)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Yield batches for one epoch."""
        if self.sample_weights is not None:
            p = self.sample_weights / self.sample_weights.sum()
            order = self.rng.choice(self.rows, size=len(self.rows), replace=True, p=p)
        elif self.shuffle:
            order = self.rng.permutation(self.rows)
        else:
            order = self.rows
        limit = len(order) - (len(order) % self.batch_size) if self.drop_last else len(order)
        for start in range(0, limit, self.batch_size):
            batch_rows = order[start : start + self.batch_size]
            yield gather(self.data, batch_rows), batch_rows


class SequenceDataset(Dataset):
    """``torch.utils.data.Dataset`` view over :class:`SequenceData`.

    Provided for interoperability and for tests that want the standard interface;
    training uses :class:`SequenceBatcher`.
    """

    def __init__(self, data: SequenceData, rows: np.ndarray) -> None:
        """Initialise the dataset.

        Args:
            data: Backing arrays.
            rows: Row indices exposed by this dataset.
        """
        self.data = data
        self.rows = np.asarray(rows)

    def __len__(self) -> int:
        """Number of rows."""
        return int(len(self.rows))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one padded sequence.

        Args:
            index: Position within ``rows``.

        Returns:
            Dict of tensors with the leading batch dimension removed.
        """
        batch = gather(self.data, self.rows[index : index + 1])
        return {k: v[0] for k, v in batch.items()}


def positive_sample_weights(
    labels: np.ndarray, rows: np.ndarray, positive_multiplier: float
) -> np.ndarray:
    """Sampling weights that oversample positives by a fixed multiplier.

    Args:
        labels: Binary labels for every row of the dataset.
        rows: Rows the sampler will draw from.
        positive_multiplier: Relative weight given to positives.

    Returns:
        Weight vector aligned to ``rows``.
    """
    y = labels[rows]
    return np.where(y > 0.5, positive_multiplier, 1.0).astype(np.float64)
