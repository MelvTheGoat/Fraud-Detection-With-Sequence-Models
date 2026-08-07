"""Temporal train/validation/test splitting.

The split is by wall-clock time and nothing else. A random split over transactions
lets a model train on a customer's Thursday and test on their Wednesday, which
inflates every metric and cannot be reproduced by a system that only ever sees the
past. An embargo gap is inserted between splits so that a transaction sitting just
after a boundary cannot build its rolling features out of transactions the model
was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import SplitConfig
from src.data.schema import TransactionSchema

S = TransactionSchema


@dataclass(frozen=True)
class TemporalSplit:
    """Boolean masks and boundaries for a temporal split.

    Attributes:
        train: Mask of training rows, aligned to the input frame's row order.
        val: Mask of validation rows.
        test: Mask of test rows.
        train_end: Last timestamp included in training.
        val_start: First timestamp eligible for validation.
        val_end: Last timestamp included in validation.
        test_start: First timestamp eligible for test.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a per-split row and fraud count table.

        Args:
            df: The frame the masks were built from.

        Returns:
            DataFrame with one row per split.
        """
        rows = []
        for name, mask in (("train", self.train), ("val", self.val), ("test", self.test)):
            sub = df.loc[mask]
            rows.append(
                {
                    "split": name,
                    "n": int(len(sub)),
                    "n_fraud": int(sub[S.IS_FRAUD].sum()),
                    "fraud_rate": float(sub[S.IS_FRAUD].mean()) if len(sub) else float("nan"),
                    "start": sub[S.TIMESTAMP].min() if len(sub) else pd.NaT,
                    "end": sub[S.TIMESTAMP].max() if len(sub) else pd.NaT,
                }
            )
        return pd.DataFrame(rows)


def make_temporal_split(df: pd.DataFrame, cfg: SplitConfig | None = None) -> TemporalSplit:
    """Split a transaction frame by time.

    Args:
        df: Transaction frame containing a ``timestamp`` column.
        cfg: Split configuration; defaults to 60/15/25 with a one-day embargo.

    Returns:
        A :class:`TemporalSplit` whose masks are aligned to ``df``'s row order.

    Raises:
        ValueError: If the fractions do not leave room for a test period.
    """
    cfg = cfg or SplitConfig()
    if cfg.test_frac <= 0:
        raise ValueError("train_frac + val_frac must be < 1 to leave a test period")

    ts = df[S.TIMESTAMP]
    t0, t1 = ts.min(), ts.max()
    span = t1 - t0
    embargo = pd.Timedelta(days=cfg.embargo_days)

    train_end = t0 + span * cfg.train_frac
    val_start = train_end + embargo
    val_end = t0 + span * (cfg.train_frac + cfg.val_frac)
    test_start = val_end + embargo

    return TemporalSplit(
        train=(ts <= train_end).to_numpy(),
        val=((ts >= val_start) & (ts <= val_end)).to_numpy(),
        test=(ts >= test_start).to_numpy(),
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
    )


def time_windows(
    ts: pd.Series, window_days: float, start: pd.Timestamp | None = None
) -> pd.Series:
    """Assign each timestamp to a fixed-width monitoring window.

    Args:
        ts: Timestamps.
        window_days: Window width in days.
        start: Window origin; defaults to the earliest timestamp.

    Returns:
        Integer window index per row.
    """
    origin = start if start is not None else ts.min()
    delta = (ts - origin).dt.total_seconds() / (window_days * 86400.0)
    return delta.astype(int).clip(lower=0)
