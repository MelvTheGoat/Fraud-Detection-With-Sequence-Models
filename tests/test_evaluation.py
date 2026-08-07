"""Metric definitions, including the ones that exist to be dismissed."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import (
    aggregate_seeds,
    comparison_table,
    evaluate,
    format_mean_std,
    precision_recall_at_capacity,
    recall_by_fraud_type,
    top_k_threshold,
)


@pytest.fixture
def toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1000 rows, 10 frauds, a model that ranks 8 of them at the very top."""
    rng = np.random.default_rng(0)
    n = 1000
    y = np.zeros(n)
    y[:10] = 1.0
    scores = rng.uniform(0, 0.5, size=n)
    scores[:8] = np.linspace(0.99, 0.9, 8)
    scores[8:10] = 0.05
    fraud_type = np.where(y > 0.5, "card_testing", "none").astype(object)
    fraud_type[5:10] = "account_takeover"
    return y, scores, fraud_type


def test_top_k_threshold_respects_capacity() -> None:
    scores = np.linspace(0, 1, 1000)
    _, k = top_k_threshold(scores, 0.01)
    assert k == 10
    _, k = top_k_threshold(scores, 0.001)
    assert k == 1


def test_capacity_must_be_a_fraction() -> None:
    with pytest.raises(ValueError, match="capacity"):
        top_k_threshold(np.array([0.1, 0.2]), 1.5)


def test_precision_and_recall_at_capacity(toy) -> None:
    y, scores, _ = toy
    block = precision_recall_at_capacity(y, scores, 0.01)
    assert block["n_alerts"] == 10
    assert block["n_caught"] == 8
    assert block["precision"] == pytest.approx(0.8)
    assert block["recall"] == pytest.approx(0.8)
    assert block["lift"] == pytest.approx(0.8 / 0.01)


def test_alert_volume_is_exact_under_ties() -> None:
    """Tied scores must not blow the alert budget.

    Tree models produce many identical scores; a threshold-based implementation
    would flag every tied row and quietly hand the fraud team ten times the
    queue they asked for.
    """
    y = np.zeros(1000)
    y[:5] = 1.0
    scores = np.full(1000, 0.5)
    scores[:5] = 0.9
    block = precision_recall_at_capacity(y, scores, 0.05)
    assert block["n_alerts"] == 50


def test_per_mechanism_recall_splits_the_aggregate(toy) -> None:
    y, scores, fraud_type = toy
    by_type = recall_by_fraud_type(y, scores, fraud_type, 0.01)
    # The model ranks the first eight, which spans both mechanisms unevenly.
    assert by_type["card_testing"] == pytest.approx(1.0)
    assert by_type["account_takeover"] == pytest.approx(0.6)
    assert set(by_type) == {"card_testing", "account_takeover"}


def test_absent_mechanisms_are_omitted_not_zeroed() -> None:
    y = np.array([0.0, 1.0, 0.0, 0.0])
    scores = np.array([0.1, 0.9, 0.2, 0.3])
    fraud_type = np.array(["none", "card_testing", "none", "none"], dtype=object)
    by_type = recall_by_fraud_type(y, scores, fraud_type, 0.5)
    assert "account_takeover" not in by_type


def test_accuracy_is_reported_only_to_be_dismissed(toy) -> None:
    """The whole point: accuracy and always-legit accuracy are the same number."""
    y, scores, fraud_type = toy
    result = evaluate(y, scores, name="toy", fraud_type=fraud_type)
    assert result.accuracy_of_always_legit == pytest.approx(0.99)
    assert abs(result.accuracy - result.accuracy_of_always_legit) < 0.02


def test_evaluate_rejects_a_slice_with_no_positives() -> None:
    with pytest.raises(ValueError, match="no positives"):
        evaluate(np.zeros(10), np.linspace(0, 1, 10), name="empty")


def test_evaluate_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate(np.array([0.0, 1.0]), np.array([0.5]), name="bad")


def test_brier_is_nan_for_non_probabilities() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    logits = np.array([-2.0, 3.0, -1.0, 2.0])
    result = evaluate(y, logits, name="logits")
    assert np.isnan(result.brier)


def test_aggregate_seeds_reports_spread(toy) -> None:
    y, scores, fraud_type = toy
    rng = np.random.default_rng(1)
    results = [
        evaluate(
            y,
            np.clip(scores + rng.normal(0, 0.02, size=len(scores)), 0, 1),
            name="model",
            fraud_type=fraud_type,
            extra={"seed": seed},
        )
        for seed in range(3)
    ]
    agg = aggregate_seeds(results)
    assert len(agg) == 1
    assert agg["n_seeds"].iloc[0] == 3
    assert agg["pr_auc_std"].iloc[0] > 0, "three different runs must not report zero spread"

    table = comparison_table(results, capacity=0.005)
    assert "pr_auc_mean" in table.columns and "pr_auc_std" in table.columns


def test_format_mean_std() -> None:
    assert format_mean_std(0.5, 0.01) == "0.5000 +/- 0.0100"
    assert format_mean_std(0.5, float("nan")) == "0.5000"
