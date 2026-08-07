"""The simulator must produce a problem that is hard enough to be worth measuring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import SimulationConfig
from src.data.schema import FraudType, TransactionSchema
from src.simulate import simulate

S = TransactionSchema


def test_schema_and_ordering(transactions: pd.DataFrame) -> None:
    assert list(transactions.columns) == list(S.ALL)
    assert transactions[S.TIMESTAMP].is_monotonic_increasing
    assert transactions[S.TXN_ID].is_unique
    assert (transactions[S.AMOUNT] > 0).all()


def test_every_mechanism_is_present(transactions: pd.DataFrame) -> None:
    present = set(transactions[S.FRAUD_TYPE].unique())
    for mechanism in (
        FraudType.CARD_TESTING,
        FraudType.ACCOUNT_TAKEOVER,
        FraudType.MERCHANT_COMPROMISE,
    ):
        assert mechanism.value in present, f"{mechanism.value} was not generated"


def test_labels_and_mechanisms_agree(transactions: pd.DataFrame) -> None:
    """A transaction is fraudulent exactly when it carries a mechanism label."""
    labelled = transactions[S.FRAUD_TYPE] != FraudType.NONE.value
    assert (transactions[S.IS_FRAUD].astype(bool) == labelled).all()


def test_fraud_rate_is_realistic(transactions: pd.DataFrame) -> None:
    rate = transactions[S.IS_FRAUD].mean()
    assert 0.001 < rate < 0.06, f"fraud rate {rate:.4f} is not in a plausible range"


def test_merchant_cardinality_is_power_law(transactions: pd.DataFrame) -> None:
    """A few merchants carry most volume and a long tail is seen a handful of times.

    This is the regime that makes learned embeddings interesting; a uniform
    merchant distribution would make the whole premise moot.
    """
    counts = transactions[S.MERCHANT_ID].value_counts()
    head_share = counts.head(max(len(counts) // 20, 1)).sum() / counts.sum()
    assert head_share > 0.2, "the top 5% of merchants should carry a large share of volume"
    skew = counts.max() / max(counts.median(), 1)
    assert skew > 10, f"merchant volume is too uniform (max/median = {skew:.1f})"


def test_new_devices_are_common_for_legitimate_traffic(transactions: pd.DataFrame) -> None:
    """"Unseen device" must not be a giveaway.

    If legitimate customers never used an unfamiliar device, a one-line rule
    would solve the problem and the model comparison would measure nothing.
    """
    legit = transactions[transactions[S.IS_FRAUD] == 0]
    ordered = legit.sort_values([S.CUSTOMER_ID, S.TIMESTAMP])
    first_seen = ~ordered.duplicated(subset=[S.CUSTOMER_ID, S.DEVICE_ID])
    # Excludes each customer's genuine first transaction.
    not_first = ordered.duplicated(subset=[S.CUSTOMER_ID], keep="first")
    rate = float((first_seen & not_first).mean())
    assert rate > 0.01, f"only {rate:.4f} of legitimate transactions use a new device"


def test_device_type_is_informative_but_not_deterministic(transactions: pd.DataFrame) -> None:
    """Device fingerprint should carry signal without being a perfect separator."""
    table = pd.crosstab(transactions[S.DEVICE_TYPE], transactions[S.IS_FRAUD], normalize="columns")
    if 1 not in table.columns:
        pytest.skip("no fraud in this sample")
    fraud_share_in_risky = float(table.loc[table[1] > table[0], 1].sum())
    assert 0.1 < fraud_share_in_risky < 0.95, (
        "device type is either useless or a perfect tell; neither makes a useful benchmark"
    )


def test_amount_distributions_overlap(transactions: pd.DataFrame) -> None:
    """Fraud amounts must overlap legitimate ones, or amount alone would solve it."""
    legit = transactions.loc[transactions[S.IS_FRAUD] == 0, S.AMOUNT]
    fraud = transactions.loc[transactions[S.IS_FRAUD] == 1, S.AMOUNT]
    overlap = float(
        ((fraud >= legit.quantile(0.05)) & (fraud <= legit.quantile(0.95))).mean()
    )
    assert overlap > 0.4, f"only {overlap:.2f} of fraud amounts fall inside the legitimate range"


def test_drift_changes_the_fraud_distribution() -> None:
    """Post-drift fraud must actually differ from pre-drift fraud.

    The monitoring story depends on this; if the drift were cosmetic the monitor
    would have nothing to detect and the test of it would be vacuous.
    """
    cfg = SimulationConfig(n_customers=400, n_days=120, n_merchants=300, base_fraud_rate=0.02, seed=3)
    df, meta = simulate(cfg)
    drift_start = pd.Timestamp(meta["drift_start"])
    fraud = df[df[S.IS_FRAUD] == 1]
    before = fraud[fraud[S.TIMESTAMP] < drift_start]
    after = fraud[fraud[S.TIMESTAMP] >= drift_start]
    assert len(before) > 20 and len(after) > 20

    risky = {"headless_client", "emulator"}
    share_before = float(before[S.DEVICE_TYPE].isin(risky).mean())
    share_after = float(after[S.DEVICE_TYPE].isin(risky).mean())
    assert share_after < share_before, (
        "post-drift fraud should stop advertising itself through the device fingerprint"
    )

    assert after[S.AMOUNT].median() != before[S.AMOUNT].median()


def test_simulation_is_deterministic() -> None:
    cfg = SimulationConfig(n_customers=60, n_days=45, n_merchants=80, seed=99)
    first, _ = simulate(cfg)
    second, _ = simulate(cfg)
    pd.testing.assert_frame_equal(first, second)


def test_metadata_matches_the_frame(transactions: pd.DataFrame, metadata: dict) -> None:
    assert metadata["n_transactions"] == len(transactions)
    assert metadata["n_fraud"] == int(transactions[S.IS_FRAUD].sum())
    assert np.isclose(metadata["fraud_rate"], transactions[S.IS_FRAUD].mean())
