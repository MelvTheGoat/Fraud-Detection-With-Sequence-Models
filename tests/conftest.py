"""Shared fixtures. Everything here is synthetic; no test ever touches real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import SequenceConfig, SimulationConfig, SplitConfig
from src.data.features import FeatureBuilder
from src.data.sequences import SequenceData, build_sequence_data
from src.data.splits import TemporalSplit, make_temporal_split
from src.simulate import simulate

#: Small enough to keep the suite quick, large enough that every fraud mechanism
#: is present, splits are non-degenerate, and customers accumulate enough
#: post-training history for the serving-parity test to have full windows to
#: work with (it needs 63 consecutive transactions outside the training period).
TEST_SIM = SimulationConfig(
    n_customers=140,
    n_days=150,
    n_merchants=200,
    base_fraud_rate=0.02,
    seed=11,
)


@pytest.fixture(scope="session")
def transactions() -> pd.DataFrame:
    """A small synthetic transaction table."""
    df, _ = simulate(TEST_SIM)
    return df


@pytest.fixture(scope="session")
def metadata() -> dict:
    """Metadata for the small synthetic dataset."""
    _, meta = simulate(TEST_SIM)
    return meta


@pytest.fixture(scope="session")
def split(transactions: pd.DataFrame) -> TemporalSplit:
    """Temporal split over the small dataset."""
    return make_temporal_split(transactions, SplitConfig())


@pytest.fixture(scope="session")
def builder_and_frame(
    transactions: pd.DataFrame, split: TemporalSplit
) -> tuple[FeatureBuilder, pd.DataFrame]:
    """A fitted feature builder and the resulting feature frame."""
    builder = FeatureBuilder(min_merchant_count=2)
    frame = builder.fit_transform(transactions, split.train)
    return builder, frame


@pytest.fixture(scope="session")
def sequence_data(builder_and_frame: tuple[FeatureBuilder, pd.DataFrame]) -> SequenceData:
    """Sequence tensors for the small dataset."""
    builder, frame = builder_and_frame
    return build_sequence_data(frame, builder, SequenceConfig().max_len)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic generator for test-local randomness."""
    return np.random.default_rng(0)
