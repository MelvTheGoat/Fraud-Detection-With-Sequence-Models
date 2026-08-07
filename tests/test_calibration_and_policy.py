"""Calibration measurement and cost-sensitive decisioning."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import (
    IsotonicCalibrator,
    TemperatureScaler,
    brier_score,
    calibrate_and_compare,
    expected_calibration_error,
    reliability_bins,
)
from src.config import CostConfig
from src.policy import (
    apply_policy,
    baseline_costs,
    choose_threshold,
    compare_policies,
    evaluate_threshold,
    sensitivity,
    sweep_thresholds,
)


def make_calibrated(n: int = 20000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities that are correct by construction."""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.6, 60.0, size=n)
    y = (rng.uniform(size=n) < p).astype(float)
    return y, p


def make_overconfident(n: int = 20000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Well-ranked but systematically overconfident probabilities.

    Built by sharpening the logits of a calibrated model, which is what a neural
    network trained to convergence on an imbalanced problem tends to do.
    """
    y, p = make_calibrated(n, seed)
    logit = np.log(p / (1 - p))
    return y, 1.0 / (1.0 + np.exp(-logit * 2.2))


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


def test_calibrated_probabilities_have_low_error() -> None:
    y, p = make_calibrated()
    assert expected_calibration_error(y, p) < 0.005
    assert brier_score(y, p) < 0.05


def test_overconfidence_is_detected() -> None:
    y, p = make_calibrated()
    _, sharp = make_overconfident()
    assert expected_calibration_error(y, sharp) > expected_calibration_error(y, p)


def test_quantile_bins_beat_uniform_bins_on_skewed_scores() -> None:
    """Equal-width bins hide miscalibration when scores mass near zero.

    Nearly every score is below 0.05, so a uniform binning puts them all in bin
    one and reports almost nothing regardless of how wrong the model is.
    """
    y, sharp = make_overconfident()
    quantile = expected_calibration_error(y, sharp, strategy="quantile")
    uniform = expected_calibration_error(y, sharp, strategy="uniform")
    assert quantile > uniform


def test_unknown_binning_strategy_is_rejected() -> None:
    y, p = make_calibrated(1000)
    with pytest.raises(ValueError, match="quantile"):
        expected_calibration_error(y, p, strategy="logarithmic")


def test_reliability_bins_are_well_formed() -> None:
    y, p = make_calibrated()
    bins = reliability_bins(y, p, n_bins=10)
    assert bins.count.sum() == len(y)
    assert (bins.mean_predicted >= 0).all() and (bins.mean_predicted <= 1).all()


def test_temperature_scaling_corrects_overconfidence() -> None:
    y, sharp = make_overconfident()
    scaler = TemperatureScaler().fit(y[:10000], sharp[:10000])
    fixed = scaler.transform(sharp[10000:])
    assert scaler.temperature > 1.2, "an overconfident model needs a temperature above 1"
    assert expected_calibration_error(y[10000:], fixed) < expected_calibration_error(
        y[10000:], sharp[10000:]
    )


def test_temperature_scaling_is_rank_preserving() -> None:
    """Monotone, so PR-AUC is unchanged by calibration.

    Checked as monotonicity rather than equal argsorts: probabilities clipped at
    the floor become exact ties, and the order within a tie is arbitrary.
    """
    y, p = make_overconfident(2000)
    scaler = TemperatureScaler().fit(y, p)
    transformed = scaler.transform(p)
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(transformed[order]) >= -1e-12)


def test_isotonic_is_rank_preserving_and_improves_calibration() -> None:
    y, sharp = make_overconfident()
    calibrator = IsotonicCalibrator().fit(y[:10000], sharp[:10000])
    fixed = calibrator.transform(sharp[10000:])
    assert expected_calibration_error(y[10000:], fixed) < expected_calibration_error(
        y[10000:], sharp[10000:]
    )
    # Monotone, so ordering (and therefore PR-AUC) is preserved.
    order = np.argsort(sharp[10000:])
    assert np.all(np.diff(fixed[order]) >= -1e-12)


def test_calibrate_and_compare_returns_all_three_methods() -> None:
    y, sharp = make_overconfident()
    reports, calibrated = calibrate_and_compare(
        y[:10000], sharp[:10000], y[10000:], sharp[10000:], model="test"
    )
    assert {r.method for r in reports} == {"raw", "temperature", "isotonic"}
    assert set(calibrated) == {"raw", "temperature", "isotonic"}
    assert all(len(v) == 10000 for v in calibrated.values())


# --------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------


@pytest.fixture
def costed() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels, scores and amounts where fraud is both rare and expensive."""
    rng = np.random.default_rng(3)
    n = 20000
    y = (rng.uniform(size=n) < 0.01).astype(float)
    scores = np.clip(rng.beta(1.0, 60.0, size=n) + 0.35 * y, 0, 1)
    amounts = np.where(y > 0.5, rng.lognormal(5.0, 0.8, n), rng.lognormal(3.0, 1.0, n))
    return y, scores, amounts


def test_threshold_outcome_arithmetic() -> None:
    y = np.array([1.0, 1.0, 0.0, 0.0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    amounts = np.array([100.0, 200.0, 50.0, 60.0])
    cost = CostConfig(
        false_positive_cost=10.0,
        false_negative_cost_fixed=5.0,
        false_negative_amount_multiplier=1.0,
        review_cost=1.0,
    )
    outcome = evaluate_threshold(y, scores, amounts, 0.5, cost)
    assert outcome.true_positives == 1 and outcome.false_positives == 1
    assert outcome.false_negatives == 1
    assert outcome.fraud_value_missed == pytest.approx(200.0)
    # 1 FP x 10 + 1 FN x 5 + 200 missed + 1 TP review x 1
    assert outcome.total_cost == pytest.approx(10 + 5 + 200 + 1)
    assert outcome.cost_per_transaction == pytest.approx(216 / 4)


def test_blocking_everything_and_nothing_bound_the_sweep(costed) -> None:
    y, scores, amounts = costed
    cost = CostConfig()
    sweep = sweep_thresholds(y, scores, amounts, cost)
    trivial = baseline_costs(y, amounts, cost)
    assert sweep["cost_per_transaction"].min() <= trivial["block_nothing"]
    assert sweep["alert_rate"].max() > 0.9
    assert sweep["alert_rate"].min() < 0.01


def test_chosen_threshold_beats_doing_nothing(costed) -> None:
    y, scores, amounts = costed
    cost = CostConfig()
    threshold, sweep = choose_threshold(y, scores, amounts, cost)
    outcome = evaluate_threshold(y, scores, amounts, threshold, cost)
    trivial = baseline_costs(y, amounts, cost)
    assert outcome.cost_per_transaction < trivial["block_nothing"]
    assert outcome.cost_per_transaction == pytest.approx(
        sweep["cost_per_transaction"].min(), rel=1e-9
    )


def test_capacity_cap_is_respected(costed) -> None:
    y, scores, amounts = costed
    threshold, _ = choose_threshold(y, scores, amounts, CostConfig(), max_alert_rate=0.005)
    outcome = evaluate_threshold(y, scores, amounts, threshold, CostConfig())
    assert outcome.alert_rate <= 0.005 + 1e-9


def test_threshold_is_selected_on_validation_not_test(costed) -> None:
    """The reported test cost must be at or above the oracle, never below.

    If a run ever reports a test cost lower than the best achievable test cost,
    the threshold was tuned on test.
    """
    y, scores, amounts = costed
    half = len(y) // 2
    result = apply_policy(
        "model",
        y[:half],
        scores[:half],
        amounts[:half],
        y[half:],
        scores[half:],
        amounts[half:],
        CostConfig(),
    )
    assert result.test.cost_per_transaction >= result.best_possible_test_cost - 1e-12
    assert result.selection_penalty >= 0
    assert result.savings_vs_block_nothing > 0


def test_higher_false_positive_cost_raises_the_threshold(costed) -> None:
    """The economics should move the operating point in the obvious direction."""
    y, scores, amounts = costed
    half = len(y) // 2
    table = sensitivity(
        y[:half],
        scores[:half],
        amounts[:half],
        y[half:],
        scores[half:],
        amounts[half:],
        model="m",
        fp_costs=(1.0, 50.0),
    )
    cheap, expensive = table.iloc[0], table.iloc[1]
    assert expensive["threshold"] >= cheap["threshold"]
    assert expensive["test_alert_rate"] <= cheap["test_alert_rate"]


def test_compare_policies_sorts_cheapest_first(costed) -> None:
    y, scores, amounts = costed
    half = len(y) // 2
    rng = np.random.default_rng(0)
    results = [
        apply_policy(
            name,
            y[:half],
            np.clip(scores[:half] + noise, 0, 1),
            amounts[:half],
            y[half:],
            np.clip(scores[half:] + noise_test, 0, 1),
            amounts[half:],
            CostConfig(),
        )
        for name, noise, noise_test in (
            ("good", np.zeros(half), np.zeros(len(y) - half)),
            (
                "noisy",
                rng.normal(0, 0.2, half),
                rng.normal(0, 0.2, len(y) - half),
            ),
        )
    ]
    table = compare_policies(results)
    assert list(table["model"]) == sorted(
        table["model"], key=lambda m: table.set_index("model").loc[m, "test_cost_per_txn"]
    )
    assert table["test_cost_per_txn"].is_monotonic_increasing
