"""Training loop behaviour: loss, imbalance strategies, early stopping, seeds."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import TrainConfig
from src.train import STRATEGIES, focal_loss_with_logits, predict, set_seed, train_one


def test_focal_loss_reduces_to_weighted_cross_entropy_at_gamma_zero() -> None:
    logits = torch.tensor([2.0, -1.0, 0.5])
    targets = torch.tensor([1.0, 0.0, 1.0])
    focal = focal_loss_with_logits(logits, targets, alpha=0.5, gamma=0.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.allclose(focal, (0.5 * bce).mean())


def test_focal_loss_down_weights_easy_examples() -> None:
    """A confidently-correct example should contribute far less than a wrong one."""
    easy = focal_loss_with_logits(
        torch.tensor([8.0]), torch.tensor([1.0]), alpha=0.5, gamma=2.0
    )
    hard = focal_loss_with_logits(
        torch.tensor([-1.0]), torch.tensor([1.0]), alpha=0.5, gamma=2.0
    )
    assert easy < hard / 100


def test_focal_alpha_shifts_weight_towards_positives() -> None:
    logits = torch.tensor([0.0])
    positive = focal_loss_with_logits(logits, torch.tensor([1.0]), alpha=0.9, gamma=2.0)
    negative = focal_loss_with_logits(logits, torch.tensor([0.0]), alpha=0.9, gamma=2.0)
    assert positive > negative


def test_focal_loss_is_finite_at_extreme_logits() -> None:
    logits = torch.tensor([50.0, -50.0])
    targets = torch.tensor([0.0, 1.0])
    loss = focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0)
    assert torch.isfinite(loss)


def test_set_seed_makes_initialisation_reproducible() -> None:
    set_seed(5)
    first = torch.randn(4)
    set_seed(5)
    assert torch.equal(first, torch.randn(4))


@pytest.mark.slow
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_training_learns_something(sequence_data, split, strategy) -> None:
    """Every strategy must beat random ranking on held-out data.

    Deliberately a weak assertion: this is a two-epoch run on a few thousand
    transactions, so the point is that the plumbing works end to end, not that
    the model is good.
    """
    result = train_one(
        "gru",
        sequence_data,
        split,
        strategy=strategy,
        seed=0,
        train_config=TrainConfig(max_epochs=2, batch_size=256),
    )
    from sklearn.metrics import average_precision_score

    rows_val = np.flatnonzero(split.val)
    y_val = sequence_data.labels[rows_val]
    base_rate = float(y_val.mean())
    pr_auc = average_precision_score(y_val, result.scores.val_scores)
    assert pr_auc > base_rate * 2, f"{strategy}: PR-AUC {pr_auc:.4f} vs base rate {base_rate:.4f}"


@pytest.mark.slow
def test_weighted_sampling_inflates_predicted_probabilities(sequence_data, split) -> None:
    """Resampling breaks calibration in a predictable direction.

    The weighted arm is fitting P(fraud | x) in a resampled world where the base
    rate is far higher, so its probabilities should sit well above the true one.
    This is exactly why the calibration section exists.
    """
    config = TrainConfig(max_epochs=2, batch_size=256)
    weighted = train_one(
        "gru", sequence_data, split, strategy="weighted", seed=0, train_config=config
    )
    plain = train_one("gru", sequence_data, split, strategy="bce", seed=0, train_config=config)
    rows_val = np.flatnonzero(split.val)
    true_rate = float(sequence_data.labels[rows_val].mean())
    assert weighted.scores.val_scores.mean() > plain.scores.val_scores.mean()
    assert weighted.scores.val_scores.mean() > true_rate


@pytest.mark.slow
def test_scores_metadata_is_populated(sequence_data, split) -> None:
    result = train_one(
        "tcn",
        sequence_data,
        split,
        strategy="focal",
        seed=1,
        train_config=TrainConfig(max_epochs=2, batch_size=256),
    )
    scores = result.scores
    assert scores.name == "tcn_focal"
    assert scores.family == "tcn"
    assert scores.seed == 1
    assert scores.n_params > 0
    assert scores.train_seconds > 0
    assert scores.meta["imbalance_strategy"] == "focal"
    assert 1 <= scores.meta["best_epoch"] <= 2
    assert len(result.history) >= 1
    assert scores.val_scores.min() >= 0 and scores.val_scores.max() <= 1


@pytest.mark.slow
def test_different_seeds_give_different_models(sequence_data, split) -> None:
    """If seeds did not matter, reporting variance across them would be theatre."""
    config = TrainConfig(max_epochs=1, batch_size=256)
    first = train_one("gru", sequence_data, split, strategy="focal", seed=0, train_config=config)
    second = train_one("gru", sequence_data, split, strategy="focal", seed=1, train_config=config)
    assert not np.allclose(first.scores.val_scores, second.scores.val_scores)


@pytest.mark.slow
def test_same_seed_reproduces_the_run(sequence_data, split) -> None:
    config = TrainConfig(max_epochs=1, batch_size=256)
    first = train_one("gru", sequence_data, split, strategy="focal", seed=3, train_config=config)
    second = train_one("gru", sequence_data, split, strategy="focal", seed=3, train_config=config)
    np.testing.assert_allclose(first.scores.val_scores, second.scores.val_scores, rtol=0, atol=0)


def test_unknown_strategy_is_rejected(sequence_data, split) -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        train_one("gru", sequence_data, split, strategy="magic")  # type: ignore[arg-type]


@pytest.mark.slow
def test_predict_returns_probabilities(sequence_data, split) -> None:
    result = train_one(
        "gru",
        sequence_data,
        split,
        strategy="focal",
        seed=0,
        train_config=TrainConfig(max_epochs=1, batch_size=256),
    )
    rows = np.flatnonzero(split.test)[:500]
    scores = predict(result.model, sequence_data, rows, batch_size=128)
    assert scores.shape == rows.shape
    assert ((scores >= 0) & (scores <= 1)).all()
