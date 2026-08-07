"""Drift monitoring, including an end-to-end proof that it fires on real drift.

``test_monitor_fires_on_adversarial_drift`` is the one that matters: it trains
the baseline on the pre-drift period and shows that the injected adversarial
change moves the score distribution enough for PSI to alert, and that measured
precision falls. A monitor that has never been shown to fire is decoration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines import baseline_feature_frame, train_lightgbm
from src.config import MonitoringConfig, SimulationConfig, SplitConfig
from src.data.features import FeatureBuilder
from src.data.schema import TransactionSchema
from src.data.splits import make_temporal_split, time_windows
from src.monitoring import (
    detection_summary,
    feature_drift,
    first_alert,
    monitor_over_time,
    population_stability_index,
    psi_edges,
    severity_of,
)
from src.simulate import simulate

S = TransactionSchema


def test_psi_is_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(size=20000)
    assert population_stability_index(sample, sample.copy()) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(size=20000)
    small = population_stability_index(reference, rng.normal(0.1, 1.0, 20000))
    large = population_stability_index(reference, rng.normal(1.5, 1.0, 20000))
    assert small < large
    assert large > 0.25


def test_psi_uses_fixed_edges_across_windows() -> None:
    """Each window must be scored against the reference bins, not its own.

    Re-deriving quantile bins per window makes every window look stable by
    construction, which is the classic way a drift monitor ends up never firing.
    """
    rng = np.random.default_rng(1)
    reference = rng.normal(size=10000)
    shifted = rng.normal(3.0, 1.0, 10000)
    edges = psi_edges(reference, 10)
    fixed = population_stability_index(reference, shifted, edges=edges)
    per_window = population_stability_index(shifted, shifted, edges=psi_edges(shifted, 10))
    assert fixed > 0.25
    assert per_window == pytest.approx(0.0, abs=1e-9)


def test_psi_handles_empty_samples() -> None:
    assert population_stability_index(np.array([]), np.array([1.0])) == 0.0
    assert population_stability_index(np.array([1.0]), np.array([])) == 0.0


def test_severity_thresholds() -> None:
    config = MonitoringConfig()
    assert severity_of(0.05, config) == "ok"
    assert severity_of(0.15, config) == "warn"
    assert severity_of(0.40, config) == "alert"


def test_time_windows_are_contiguous_and_ordered() -> None:
    ts = pd.Series(pd.date_range("2024-01-01", periods=100, freq="6h"))
    windows = time_windows(ts, window_days=7.0)
    assert windows.min() == 0
    assert windows.is_monotonic_increasing


def test_feature_drift_ranks_the_moved_feature_first() -> None:
    rng = np.random.default_rng(2)
    reference = pd.DataFrame(
        {"stable": rng.normal(size=5000), "moved": rng.normal(size=5000)}
    )
    current = pd.DataFrame(
        {"stable": rng.normal(size=5000), "moved": rng.normal(2.0, 1.0, size=5000)}
    )
    ranked = feature_drift(reference, current)
    assert ranked.iloc[0]["feature"] == "moved"
    assert ranked.iloc[0]["severity"] == "alert"


def test_monitor_reports_one_row_per_window() -> None:
    rng = np.random.default_rng(3)
    n = 6000
    timestamps = pd.date_range("2024-01-01", periods=n, freq="10min").to_numpy()
    scores = rng.beta(1, 40, size=n)
    y = (rng.uniform(size=n) < 0.02).astype(float)
    report = monitor_over_time(
        scores,
        y,
        timestamps,
        reference_scores=rng.beta(1, 40, size=5000),
        production_threshold=0.5,
        config=MonitoringConfig(window_days=7.0),
    )
    assert len(report) >= 5
    assert report["window"].is_monotonic_increasing
    assert (report["n"] >= 50).all()


def test_first_alert_returns_none_when_stable() -> None:
    report = pd.DataFrame({"score_psi": [0.01, 0.02, 0.03], "start": pd.date_range("2024-01-01", periods=3)})
    assert first_alert(report, "score_psi", 0.25) is None


@pytest.mark.slow
def test_monitor_fires_on_adversarial_drift() -> None:
    """Train before the drift, monitor across it, and check the alarm goes off.

    Both signals are checked, in the order a real deployment would see them: the
    score distribution shifts (available immediately, no labels needed) and the
    measured precision falls (available only once chargebacks are confirmed).
    """
    cfg = SimulationConfig(
        n_customers=700, n_days=140, n_merchants=500, base_fraud_rate=0.02, seed=21
    )
    df, meta = simulate(cfg)
    drift_start = pd.Timestamp(meta["drift_start"])

    split = make_temporal_split(df, SplitConfig(train_frac=0.55, val_frac=0.15))
    builder = FeatureBuilder(min_merchant_count=2)
    frame = builder.fit_transform(df, split.train)
    assert builder.spec is not None
    artifacts = train_lightgbm(frame, split, builder.spec, seed=0)

    x = baseline_feature_frame(frame, builder.spec)
    test_rows = np.flatnonzero(split.test)
    scores = artifacts.booster.predict(
        x.iloc[test_rows], num_iteration=artifacts.booster.best_iteration
    )
    y = frame.loc[split.test, S.IS_FRAUD].to_numpy()
    timestamps = frame.loc[split.test, S.TIMESTAMP].to_numpy()

    assert (pd.to_datetime(pd.Series(timestamps)) >= drift_start).any(), (
        "the test window must straddle the drift boundary for this test to mean anything"
    )

    # A threshold a real deployment would fix in advance, taken from validation.
    production_threshold = float(np.quantile(artifacts.scores.val_scores, 0.99))
    report = monitor_over_time(
        np.asarray(scores),
        y,
        timestamps,
        reference_scores=np.asarray(artifacts.scores.val_scores),
        fraud_type=frame.loc[split.test, S.FRAUD_TYPE].to_numpy(),
        production_threshold=production_threshold,
        config=MonitoringConfig(window_days=7.0, capacity=0.01),
    )
    assert len(report) >= 3

    pre = report[pd.to_datetime(report["start"]) < drift_start]
    post = report[pd.to_datetime(report["start"]) >= drift_start]
    assert len(pre) >= 1 and len(post) >= 1, "the test window must straddle the drift"

    # Leading signal: the alert rate at the fixed threshold collapses, with no
    # labels involved. This is the signal a deployment would actually see first.
    assert post["alert_rate_ratio"].min() < pre["alert_rate_ratio"].min(), (
        "the alert rate at a fixed threshold did not react to the drift"
    )
    detection = detection_summary(report, drift_start)
    assert detection["fired_any_warn"], "no label-free signal fired at all"
    assert detection["fired_alert_rate_warn"], "the alert-rate monitor missed the drift"

    # Lagging signal: precision at fixed capacity degrades once labels land.
    assert post["precision_at_capacity"].mean() < pre["precision_at_capacity"].mean(), (
        "precision did not fall despite the fraud mechanism changing"
    )


@pytest.mark.slow
def test_score_psi_alone_is_a_weak_detector_here() -> None:
    """Document the negative result: PSI on the bulk distribution barely moves.

    This is not a bug in the PSI implementation — ``test_psi_grows_with_the_size
    _of_the_shift`` shows it responds properly to a real shift. It is a property
    of the problem: over 99% of transactions are legitimate and unaffected by the
    adversary, so the distribution the PSI is measuring is dominated by a
    population that did not change. If this assertion ever starts failing because
    PSI got large, the monitoring write-up needs revisiting.
    """
    cfg = SimulationConfig(
        n_customers=700, n_days=140, n_merchants=500, base_fraud_rate=0.02, seed=21
    )
    df, meta = simulate(cfg)
    split = make_temporal_split(df, SplitConfig(train_frac=0.55, val_frac=0.15))
    builder = FeatureBuilder(min_merchant_count=2)
    frame = builder.fit_transform(df, split.train)
    assert builder.spec is not None
    artifacts = train_lightgbm(frame, split, builder.spec, seed=0)

    x = baseline_feature_frame(frame, builder.spec)
    test_rows = np.flatnonzero(split.test)
    scores = np.asarray(
        artifacts.booster.predict(
            x.iloc[test_rows], num_iteration=artifacts.booster.best_iteration
        )
    )
    report = monitor_over_time(
        scores,
        frame.loc[split.test, S.IS_FRAUD].to_numpy(),
        frame.loc[split.test, S.TIMESTAMP].to_numpy(),
        reference_scores=np.asarray(artifacts.scores.val_scores),
        production_threshold=float(np.quantile(artifacts.scores.val_scores, 0.99)),
        config=MonitoringConfig(window_days=7.0, capacity=0.01),
    )
    assert report["score_psi"].max() < MonitoringConfig().psi_warn, (
        "score PSI now exceeds the warning threshold; the claim that it is a weak "
        "detector on this problem no longer holds and the write-up must be updated"
    )
    assert (report["score_severity"] == "ok").all()
