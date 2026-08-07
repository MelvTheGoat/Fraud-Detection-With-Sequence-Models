"""Feature and sequence-construction behaviour beyond the leakage guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    CAUSAL_FEATURES,
    NEVER_SECS,
    NUMERIC_FEATURES,
    FeatureBuilder,
    add_causal_features,
    customer_causal_features,
)
from src.data.schema import N_RESERVED_CODES, PAD_INDEX, UNK_INDEX, TransactionSchema
from src.data.sequences import SequenceBatcher, build_windows, gather, positive_sample_weights
from src.data.splits import make_temporal_split

S = TransactionSchema


def test_all_features_are_finite(builder_and_frame) -> None:
    _, frame = builder_and_frame
    values = frame[list(NUMERIC_FEATURES)].to_numpy()
    assert np.isfinite(values).all(), "non-finite feature values would poison training"


def test_first_transaction_has_neutral_history_features(transactions: pd.DataFrame) -> None:
    """A customer's first transaction should look unremarkable, not infinitely anomalous."""
    frame = add_causal_features(transactions)
    ordered = frame.sort_values([S.CUSTOMER_ID, S.TIMESTAMP])
    firsts = ordered.groupby(S.CUSTOMER_ID).head(1)
    assert (firsts["amount_z"] == 0).all()
    assert (firsts["log_hist_count"] == 0).all()
    assert (firsts["is_new_merchant"] == 1).all()
    assert (firsts["log_cnt_24h"] == 0).all()


def test_velocity_counts_are_correct() -> None:
    """Hand-checked window counts on a constructed sequence."""
    hour = 3600.0
    ts = np.array([0.0, 0.5 * hour, 0.9 * hour, 2.0 * hour, 30.0 * hour])
    amounts = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    keys = np.array(["a", "a", "b", "b", "c"], dtype=object)
    feats = customer_causal_features(
        ts, amounts, keys, keys, keys, keys, keys, np.zeros(5)
    )
    # Prior transactions in the half-open window [t - 1h, t):
    #   t=0.0h  -> nothing before it
    #   t=0.5h  -> the 0.0h transaction
    #   t=0.9h  -> the 0.0h and 0.5h transactions
    #   t=2.0h  -> window starts at 1.0h, and 0.9h falls just outside it
    #   t=30.0h -> nothing within the hour
    assert np.allclose(np.expm1(feats["log_cnt_1h"]), [0, 1, 2, 0, 0])
    # 24h window: everything accumulates until the 30h point, which drops it all.
    assert np.allclose(np.expm1(feats["log_cnt_24h"]), [0, 1, 2, 3, 0])
    assert np.allclose(np.expm1(feats["log_amt_1h"]), [0, 10, 30, 0, 0])
    assert np.allclose(np.expm1(feats["log_amt_24h"]), [0, 10, 30, 60, 0])


def test_entity_novelty_and_recency() -> None:
    ts = np.array([0.0, 100.0, 200.0, 300.0])
    merchants = np.array(["m1", "m2", "m1", "m3"], dtype=object)
    others = np.array(["x", "x", "x", "x"], dtype=object)
    feats = customer_causal_features(
        ts, np.ones(4), merchants, others, others, others, others, np.zeros(4)
    )
    assert list(feats["is_new_merchant"]) == [1.0, 1.0, 0.0, 1.0]
    # The third transaction returns to m1, 200 seconds after it was last seen.
    assert np.isclose(np.expm1(feats["log_secs_since_same_merchant"][2]), 200.0)
    # Never-seen entities get the sentinel rather than NaN.
    assert np.isclose(np.expm1(feats["log_secs_since_same_merchant"][0]), NEVER_SECS, rtol=1e-3)


def test_empty_input_returns_empty_features() -> None:
    empty = np.zeros(0)
    keys = np.array([], dtype=object)
    feats = customer_causal_features(empty, empty, keys, keys, keys, keys, keys, empty)
    assert set(feats) == set(CAUSAL_FEATURES)
    assert all(v.shape == (0,) for v in feats.values())


def test_unsorted_timestamps_are_rejected() -> None:
    ts = np.array([10.0, 5.0])
    keys = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="sorted"):
        customer_causal_features(ts, np.ones(2), keys, keys, keys, keys, keys, np.zeros(2))


def test_vocabulary_is_built_from_training_only(transactions: pd.DataFrame) -> None:
    """A merchant that appears only after the training window must map to UNK.

    Building the vocabulary on the whole dataset is a subtle leak: it tells the
    model which entities exist in the future.
    """
    split = make_temporal_split(transactions)
    builder = FeatureBuilder(min_merchant_count=2)
    frame = builder.fit_transform(transactions, split.train)

    train_merchants = set(transactions.loc[split.train, S.MERCHANT_ID])
    unseen = set(transactions.loc[split.test, S.MERCHANT_ID]) - train_merchants
    if not unseen:
        pytest.skip("every test merchant also appears in training")
    rows = frame[split.test & frame[S.MERCHANT_ID].isin(unseen).to_numpy()]
    assert (rows["merchant_code"] == UNK_INDEX).all()


def test_reserved_codes_are_never_assigned(builder_and_frame) -> None:
    builder, frame = builder_and_frame
    assert builder.spec is not None
    for column in builder.spec.categorical_columns:
        codes = frame[column].to_numpy()
        assert codes.min() >= UNK_INDEX, "PAD must stay reserved for padding"
        assert (codes != PAD_INDEX).all()
    for spec in builder.spec.categorical:
        assert spec.cardinality >= N_RESERVED_CODES


def test_scaling_uses_training_statistics_only(transactions: pd.DataFrame) -> None:
    """Standardisation is fitted on training rows and clipped at ten sigma."""
    split = make_temporal_split(transactions)
    builder = FeatureBuilder(min_merchant_count=2)
    frame = builder.fit_transform(transactions, split.train)
    train_values = frame.loc[split.train, list(NUMERIC_FEATURES)].to_numpy(dtype=float)

    assert builder.scaler_mean is not None and builder.scaler_std is not None
    np.testing.assert_allclose(builder.scaler_mean, train_values.mean(axis=0), rtol=1e-9)
    assert (builder.scaler_std > 0).all()

    scaled = builder.scale(train_values)
    assert np.all(np.abs(scaled) <= 10.0 + 1e-6), "clipping bound must hold"
    # Clipping shifts the mean of heavy-tailed features slightly off zero, which
    # is the intended trade: one pathological row must not dominate a minibatch.
    assert np.all(np.abs(scaled.mean(axis=0)) < 0.05)


def test_windows_are_deterministic(transactions: pd.DataFrame) -> None:
    args = (
        transactions[S.CUSTOMER_ID].to_numpy(),
        transactions[S.TIMESTAMP].astype("int64").to_numpy(),
        12,
    )
    np.testing.assert_array_equal(build_windows(*args), build_windows(*args))


def test_window_length_is_respected(transactions: pd.DataFrame) -> None:
    windows = build_windows(
        transactions[S.CUSTOMER_ID].to_numpy(),
        transactions[S.TIMESTAMP].astype("int64").to_numpy(),
        max_len=7,
    )
    assert windows.shape == (len(transactions), 7)
    assert (windows >= 0).sum(axis=1).max() <= 7


def test_invalid_window_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_windows(np.array(["a"]), np.array([1]), max_len=0)


def test_gather_masks_padding(sequence_data) -> None:
    batch = gather(sequence_data, np.arange(32))
    mask = batch["mask"].numpy()
    numeric = batch["numeric"].numpy()
    categorical = batch["categorical"].numpy()
    assert (numeric[mask == 0] == 0).all()
    assert (categorical[mask == 0] == PAD_INDEX).all()
    assert (mask[:, -1] == 1).all(), "the scored transaction is never padding"


def test_batcher_covers_every_row_once(sequence_data) -> None:
    rows = np.arange(500)
    batcher = SequenceBatcher(sequence_data, rows, batch_size=64, shuffle=True, seed=1)
    seen = np.concatenate([chunk for _, chunk in batcher])
    np.testing.assert_array_equal(np.sort(seen), rows)


def test_weighted_sampling_oversamples_positives(sequence_data) -> None:
    rows = np.arange(len(sequence_data))
    weights = positive_sample_weights(sequence_data.labels, rows, positive_multiplier=50.0)
    batcher = SequenceBatcher(
        sequence_data, rows, batch_size=256, sample_weights=weights, seed=0
    )
    drawn = np.concatenate([chunk for _, chunk in batcher])
    natural = float(sequence_data.labels[rows].mean())
    resampled = float(sequence_data.labels[drawn].mean())
    assert resampled > natural * 5, (
        f"weighted sampling barely changed the class ratio: {natural:.4f} -> {resampled:.4f}"
    )
