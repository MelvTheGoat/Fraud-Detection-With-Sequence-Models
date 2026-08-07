"""Metric regression gate.

CI runs this on a small synthetic dataset and fails if quality drops below a
floor. The floors are deliberately loose — roughly half the performance actually
observed — because the point is to catch a *break*, not to police noise. A gate
tuned tight enough to catch a 2% regression on a few hundred positives would fail
on seed variance instead, get muted within a week, and then catch nothing at all.

What each floor is protecting against, concretely:

* ``PR-AUC`` collapsing towards the base rate — a broken feature pipeline, a
  scrambled join, or labels shuffled relative to features.
* ``precision at capacity`` collapsing — the model still ranks but no longer
  concentrates fraud where a review queue would find it.
* The sequence model failing to beat the rules floor — the deep path is wired up
  wrongly, e.g. all-padding masks or a detached graph.

There is also an upper gate: PR-AUC of 0.999 on this data means something is
leaking, not that the model got good. Both directions are failures.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.baselines import rule_baseline, train_lightgbm
from src.config import TrainConfig
from src.data.schema import TransactionSchema
from src.evaluation import evaluate, precision_recall_at_capacity
from src.train import train_one

S = TransactionSchema

#: Floors, set at roughly half of observed performance on this fixture.
MIN_BASELINE_PR_AUC = 0.35
MIN_BASELINE_PRECISION_AT_1PCT = 0.30
MIN_SEQUENCE_PR_AUC = 0.30
MIN_RULES_PR_AUC = 0.05

#: A ceiling too. Near-perfect separation on this problem is evidence of leakage.
MAX_PLAUSIBLE_PR_AUC = 0.995


def test_rule_baseline_beats_random(builder_and_frame, split) -> None:
    _, frame = builder_and_frame
    scores = rule_baseline(frame, split)
    y_test = frame.loc[split.test, S.IS_FRAUD].to_numpy()
    pr_auc = average_precision_score(y_test, scores.test_scores)
    base_rate = float(y_test.mean())
    assert pr_auc > max(MIN_RULES_PR_AUC, base_rate * 2), (
        f"four obvious rules score PR-AUC {pr_auc:.4f} against a base rate of {base_rate:.4f}; "
        "the feature pipeline is probably broken"
    )


def test_lightgbm_baseline_meets_the_floor(builder_and_frame, split) -> None:
    builder, frame = builder_and_frame
    assert builder.spec is not None
    artifacts = train_lightgbm(frame, split, builder.spec, seed=0)
    y_test = frame.loc[split.test, S.IS_FRAUD].to_numpy()
    result = evaluate(
        y_test,
        artifacts.scores.test_scores,
        name="lightgbm",
        fraud_type=frame.loc[split.test, S.FRAUD_TYPE].to_numpy(),
    )
    assert result.pr_auc >= MIN_BASELINE_PR_AUC, (
        f"baseline PR-AUC {result.pr_auc:.4f} below the floor {MIN_BASELINE_PR_AUC}"
    )
    assert result.pr_auc <= MAX_PLAUSIBLE_PR_AUC, (
        f"baseline PR-AUC {result.pr_auc:.4f} is implausibly high; suspect leakage"
    )
    at_1pct = result.at_capacity["0.01"]["precision"]
    assert at_1pct >= MIN_BASELINE_PRECISION_AT_1PCT, (
        f"precision at 1% capacity {at_1pct:.4f} below the floor"
    )


def test_baseline_catches_every_fraud_mechanism(builder_and_frame, split) -> None:
    """No mechanism may be completely invisible to the model.

    Aggregate PR-AUC can look healthy while one attack type is never caught, and
    that is the failure an adversary finds first.
    """
    builder, frame = builder_and_frame
    assert builder.spec is not None
    artifacts = train_lightgbm(frame, split, builder.spec, seed=0)
    result = evaluate(
        frame.loc[split.test, S.IS_FRAUD].to_numpy(),
        artifacts.scores.test_scores,
        name="lightgbm",
        fraud_type=frame.loc[split.test, S.FRAUD_TYPE].to_numpy(),
    )
    by_type = result.by_fraud_type["0.01"]
    assert by_type, "the per-mechanism breakdown is empty"
    blind = [mechanism for mechanism, recall in by_type.items() if recall <= 0.0]
    assert not blind, f"the model catches nothing at all for: {blind}"


@pytest.mark.slow
def test_sequence_model_meets_the_floor(sequence_data, split) -> None:
    result = train_one(
        "gru",
        sequence_data,
        split,
        strategy="focal",
        seed=0,
        train_config=TrainConfig(max_epochs=3, batch_size=256),
    )
    rows_test = np.flatnonzero(split.test)
    y_test = sequence_data.labels[rows_test]
    pr_auc = average_precision_score(y_test, result.scores.test_scores)
    assert pr_auc >= MIN_SEQUENCE_PR_AUC, (
        f"sequence model PR-AUC {pr_auc:.4f} below the floor {MIN_SEQUENCE_PR_AUC}"
    )
    assert pr_auc <= MAX_PLAUSIBLE_PR_AUC, (
        f"sequence model PR-AUC {pr_auc:.4f} is implausibly high; suspect leakage"
    )
    at_1pct = precision_recall_at_capacity(y_test, result.scores.test_scores, 0.01)
    assert at_1pct["precision"] > 0.1


def test_parameter_budget_is_respected(sequence_data, split) -> None:
    """The models must stay small enough to train on a free CPU.

    This is a stated constraint of the project, so it gets a test rather than a
    promise.
    """
    from src.data.schema import CategoricalSpec
    from src.models import COMPARED_MODELS, build_model, count_parameters, make_config

    for architecture in COMPARED_MODELS:
        config = make_config(
            architecture,
            n_numeric=sequence_data.n_numeric,
            categorical=[
                CategoricalSpec(spec.column, spec.cardinality)
                for spec in sequence_data.spec.categorical
            ],
            max_len=sequence_data.max_len,
        )
        model = build_model(architecture, config)
        assert count_parameters(model) < 1_000_000, (
            f"{architecture} exceeds the one-million-parameter budget for a CPU-trainable model"
        )
