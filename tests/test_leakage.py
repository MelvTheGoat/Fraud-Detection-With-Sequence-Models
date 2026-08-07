"""Temporal leakage tests.

This is the file that matters most. A feature that quietly reads the future is
the bug that produces a 0.99 PR-AUC in a notebook and a worthless model in
production, and it never announces itself — the training loop converges happily,
the validation curve looks great, and only the live system disagrees.

The checks here are behavioural rather than structural: instead of inspecting the
feature code, they *perturb the future* and assert the past does not move. That
catches leakage regardless of how it was introduced.

``test_leakage_detector_has_teeth`` deliberately builds a leaky feature and
asserts the same check rejects it, so a future refactor cannot silently turn these
assertions into no-ops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    CAUSAL_FEATURES,
    FeatureBuilder,
    add_causal_features,
    customer_causal_features,
)
from src.data.schema import TransactionSchema
from src.data.sequences import PAD_ROW, build_windows
from src.data.splits import make_temporal_split

S = TransactionSchema


def _perturb_future(df: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """Return a copy of ``df`` with every transaction after ``cut`` scrambled.

    Amount, merchant, device, country and label are all changed. If any feature
    of an earlier transaction depends on any of these, the invariance assertion
    will fail.

    Args:
        df: Transaction frame.
        cut: Boundary timestamp.

    Returns:
        Perturbed copy with the same row order and txn_ids.
    """
    out = df.copy()
    future = (out[S.TIMESTAMP] > cut).to_numpy()
    out.loc[future, S.AMOUNT] = out.loc[future, S.AMOUNT] * 7.3 + 11.0
    out.loc[future, S.MERCHANT_ID] = "M_PERTURBED"
    out.loc[future, S.MCC] = "9999"
    out.loc[future, S.DEVICE_ID] = "D_PERTURBED"
    out.loc[future, S.DEVICE_TYPE] = "perturbed_device"
    out.loc[future, S.COUNTRY] = "ZZ"
    out.loc[future, S.CHANNEL] = "online"
    out.loc[future, S.IS_FRAUD] = 1 - out.loc[future, S.IS_FRAUD]
    return out


def _cut_at(df: pd.DataFrame, quantile: float) -> pd.Timestamp:
    """Timestamp at the given quantile of the timeline."""
    return df[S.TIMESTAMP].quantile(quantile)


# --------------------------------------------------------------------------------------
# 1. Causal features may not react to the future
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("quantile", [0.3, 0.6, 0.85])
def test_causal_features_ignore_future_transactions(
    transactions: pd.DataFrame, quantile: float
) -> None:
    """Scrambling every transaction after t must not move any feature before t."""
    cut = _cut_at(transactions, quantile)
    base = add_causal_features(transactions)
    perturbed = add_causal_features(_perturb_future(transactions, cut))

    past = (base[S.TIMESTAMP] <= cut).to_numpy()
    assert past.sum() > 100, "cut left too few rows to be a meaningful test"

    left = base.loc[past, list(CAUSAL_FEATURES)].to_numpy()
    right = perturbed.loc[past, list(CAUSAL_FEATURES)].to_numpy()
    bad = ~np.isclose(left, right, rtol=0, atol=0, equal_nan=True)
    if bad.any():
        offenders = np.array(CAUSAL_FEATURES)[bad.any(axis=0)]
        raise AssertionError(f"features leak future information: {offenders.tolist()}")


def test_causal_features_ignore_deleted_future_rows(transactions: pd.DataFrame) -> None:
    """Deleting the future entirely must not move any feature in the past.

    Perturbation catches value leakage; truncation additionally catches leakage
    through *counts* (a rolling window that accidentally looks forward changes
    size when later rows disappear).
    """
    cut = _cut_at(transactions, 0.5)
    past_mask = (transactions[S.TIMESTAMP] <= cut).to_numpy()

    full = add_causal_features(transactions).set_index(S.TXN_ID)
    truncated = add_causal_features(transactions.loc[past_mask].copy()).set_index(S.TXN_ID)

    common = truncated.index
    left = full.loc[common, list(CAUSAL_FEATURES)].to_numpy()
    right = truncated.loc[common, list(CAUSAL_FEATURES)].to_numpy()
    np.testing.assert_array_equal(left, right)


def test_labels_never_enter_causal_features(transactions: pd.DataFrame) -> None:
    """Flipping every label must leave every causal feature untouched.

    Prior fraud labels are not available at scoring time: a chargeback is
    confirmed weeks after the authorisation. Any feature that moves when labels
    move is reading information the live system will not have.
    """
    flipped = transactions.copy()
    flipped[S.IS_FRAUD] = 1 - flipped[S.IS_FRAUD]
    flipped[S.FRAUD_TYPE] = "none"

    base = add_causal_features(transactions)[list(CAUSAL_FEATURES)].to_numpy()
    other = add_causal_features(flipped)[list(CAUSAL_FEATURES)].to_numpy()
    np.testing.assert_array_equal(base, other)


def test_single_customer_features_are_prefix_stable(transactions: pd.DataFrame) -> None:
    """Features of a prefix must equal the first rows of the full sequence.

    This is the serving contract in miniature: at time t the online scorer has
    only the prefix, and it must produce exactly what the training pipeline
    produced for that same row.
    """
    cust = transactions[S.CUSTOMER_ID].value_counts().idxmax()
    sub = transactions[transactions[S.CUSTOMER_ID] == cust].sort_values(S.TIMESTAMP)
    args = (
        sub[S.TIMESTAMP].astype("int64").to_numpy() / 1e9,
        sub[S.AMOUNT].to_numpy(),
        sub[S.MERCHANT_ID].to_numpy(),
        sub[S.MCC].to_numpy(),
        sub[S.DEVICE_ID].to_numpy(),
        sub[S.DEVICE_TYPE].to_numpy(),
        sub[S.COUNTRY].to_numpy(),
        (sub[S.CHANNEL].to_numpy() == "online").astype(float),
    )
    full = customer_causal_features(*args)

    k = len(sub) // 2
    prefix = customer_causal_features(*(a[:k] for a in args))
    for name in CAUSAL_FEATURES:
        np.testing.assert_array_equal(
            full[name][:k], prefix[name], err_msg=f"{name} is not prefix-stable"
        )


# --------------------------------------------------------------------------------------
# 2. Sequence windows may only contain the customer's own past
# --------------------------------------------------------------------------------------


def test_windows_contain_only_own_customer_and_past(transactions: pd.DataFrame) -> None:
    """Every window slot is the same customer and no later than the scored row."""
    customer = transactions[S.CUSTOMER_ID].to_numpy()
    ts = transactions[S.TIMESTAMP].astype("int64").to_numpy()
    windows = build_windows(customer, ts, max_len=16)

    rows = np.arange(len(transactions))
    assert (windows[:, -1] == rows).all(), "the scored transaction must occupy the last slot"

    valid = windows >= 0
    safe = np.where(valid, windows, 0)

    same_customer = customer[safe] == customer[:, None]
    assert same_customer[valid].all(), "a window contained another customer's transaction"

    not_future = ts[safe] <= ts[:, None]
    assert not_future[valid].all(), "a window contained a later transaction"

    # Prior slots must be strictly earlier in the customer's ordering, i.e. lower
    # row index within the same customer, and monotonically increasing.
    prior = valid.copy()
    prior[:, -1] = False
    diffs = np.diff(np.where(valid, windows, np.iinfo(np.int64).max), axis=1)
    assert (diffs[valid[:, 1:] & valid[:, :-1]] > 0).all(), "windows are not ordered"


def test_padding_is_left_aligned_and_masked(sequence_data) -> None:
    """Padding sits at the front and the mask marks exactly the real rows."""
    windows = sequence_data.windows
    valid = windows >= 0
    # Once a slot is real, every later slot must be real too (left padding).
    first_real = valid.argmax(axis=1)
    for row in range(0, len(windows), max(len(windows) // 500, 1)):
        assert not valid[row, : first_real[row]].any()
        assert valid[row, first_real[row] :].all()
    assert (windows[valid] >= 0).all()
    assert (windows[~valid] == PAD_ROW).all()


def test_short_history_customers_are_still_scorable(transactions: pd.DataFrame) -> None:
    """A customer's very first transaction produces a fully padded window with one real slot."""
    ordered = transactions.sort_values([S.CUSTOMER_ID, S.TIMESTAMP])
    firsts = ordered.groupby(S.CUSTOMER_ID).head(1).index.to_numpy()
    windows = build_windows(
        transactions[S.CUSTOMER_ID].to_numpy(),
        transactions[S.TIMESTAMP].astype("int64").to_numpy(),
        max_len=8,
    )
    assert (windows[firsts, :-1] == PAD_ROW).all()
    assert (windows[firsts, -1] == firsts).all()


# --------------------------------------------------------------------------------------
# 3. Label-consuming encodings may only use the training period
# --------------------------------------------------------------------------------------


def test_risk_encodings_ignore_labels_outside_training(transactions: pd.DataFrame) -> None:
    """Flipping validation and test labels must not change any encoded value."""
    split = make_temporal_split(transactions)
    flipped = transactions.copy()
    outside = ~split.train
    flipped.loc[outside, S.IS_FRAUD] = 1 - flipped.loc[outside, S.IS_FRAUD]

    left = FeatureBuilder(min_merchant_count=2).fit_transform(transactions, split.train)
    right = FeatureBuilder(min_merchant_count=2).fit_transform(flipped, split.train)

    encoded = ["merchant_risk", "mcc_risk", "country_risk", "device_type_risk"]
    np.testing.assert_allclose(
        left[encoded].to_numpy(), right[encoded].to_numpy(), rtol=0, atol=0
    )


def test_training_risk_encoding_is_out_of_time_within_train(transactions: pd.DataFrame) -> None:
    """A training row's encoding must not depend on later training labels.

    Out-of-fold is not enough on its own: shuffled folds still let an early row
    borrow a late row's label. The encoder uses strictly earlier temporal blocks,
    so flipping the labels of the last training block must leave the first block's
    encodings alone.
    """
    split = make_temporal_split(transactions)
    train_ts = transactions.loc[split.train, S.TIMESTAMP]
    late_cut = train_ts.quantile(0.8)

    flipped = transactions.copy()
    late_train = split.train & (transactions[S.TIMESTAMP] > late_cut).to_numpy()
    flipped.loc[late_train, S.IS_FRAUD] = 1 - flipped.loc[late_train, S.IS_FRAUD]

    left = FeatureBuilder(min_merchant_count=2).fit_transform(transactions, split.train)
    right = FeatureBuilder(min_merchant_count=2).fit_transform(flipped, split.train)

    early_train = split.train & (transactions[S.TIMESTAMP] <= train_ts.quantile(0.2)).to_numpy()
    encoded = ["merchant_risk", "mcc_risk", "country_risk", "device_type_risk"]
    np.testing.assert_allclose(
        left.loc[early_train, encoded].to_numpy(),
        right.loc[early_train, encoded].to_numpy(),
        rtol=0,
        atol=0,
    )


# --------------------------------------------------------------------------------------
# 4. The detector itself must be able to fail
# --------------------------------------------------------------------------------------


def _leaky_feature(df: pd.DataFrame) -> pd.DataFrame:
    """A deliberately leaky feature: amount z-score against the *whole* history.

    This is the realistic version of the bug — someone computes a customer-level
    mean with ``groupby.transform('mean')`` instead of an expanding mean, and every
    row silently sees its own future.

    Args:
        df: Transaction frame.

    Returns:
        Frame with a single ``leaky_amount_z`` column added.
    """
    out = df.copy()
    grouped = out.groupby(S.CUSTOMER_ID)[S.AMOUNT]
    out["leaky_amount_z"] = (out[S.AMOUNT] - grouped.transform("mean")) / (
        grouped.transform("std").fillna(1.0) + 1e-6
    )
    return out


def test_leakage_detector_has_teeth(transactions: pd.DataFrame) -> None:
    """The future-perturbation check must reject a known-leaky feature.

    Without this, a refactor that made the checks vacuous would still pass.
    """
    cut = _cut_at(transactions, 0.5)
    base = _leaky_feature(transactions)
    perturbed = _leaky_feature(_perturb_future(transactions, cut))
    past = (base[S.TIMESTAMP] <= cut).to_numpy()

    differs = not np.allclose(
        base.loc[past, "leaky_amount_z"].to_numpy(),
        perturbed.loc[past, "leaky_amount_z"].to_numpy(),
        equal_nan=True,
    )
    assert differs, "the leakage check failed to notice a feature that reads the future"
