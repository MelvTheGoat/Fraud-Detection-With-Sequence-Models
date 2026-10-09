"""Training loop: temporal splits, PR-AUC early stopping, and the imbalance study.

Three decisions here are load-bearing.

**Early stopping watches validation PR-AUC, not loss.** At a 0.65% positive rate
the loss is dominated by easy negatives and keeps drifting down long after
ranking has stopped improving; a loss-stopped model is reliably worse at the
operating point anyone cares about.

**Class imbalance is handled two ways and both are reported.** ``focal`` keeps the
natural class ratio and down-weights easy examples inside the loss. ``weighted``
oversamples positives so the model sees a roughly balanced stream. A third arm,
``bce``, does neither and exists as the calibration reference: resampling changes
the base rate the model is fitting, so a weighted model's outputs are not
probabilities of fraud, they are probabilities of fraud *in the resampled world*.
:mod:`src.calibration` quantifies how much damage each arm does.

**Every arm runs on at least three seeds.** On a problem with a few hundred test
positives, seed-to-seed spread is routinely comparable to the gap between two
architectures. A single run cannot support a claim about which model is better,
so the reporting path only ever exposes mean and standard deviation.
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from src.artifacts import ModelScores
from src.config import (
    ARTIFACT_DIR,
    RESULTS_DIR,
    SEEDS,
    SYNTHETIC_DIR,
    SequenceConfig,
    SplitConfig,
    TrainConfig,
)
from src.data.features import FeatureBuilder
from src.data.sequences import (
    SequenceBatcher,
    SequenceData,
    build_sequence_data,
    gather,
    positive_sample_weights,
)
from src.data.splits import TemporalSplit, make_temporal_split
from src.models import COMPARED_MODELS, ModelConfig, build_model, make_config, parameter_breakdown

LOGGER = logging.getLogger(__name__)

Strategy = Literal["bce", "focal", "weighted"]

#: Imbalance strategies compared in the write-up.
STRATEGIES: tuple[Strategy, ...] = ("focal", "weighted", "bce")


def focal_loss_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float
) -> torch.Tensor:
    """Binary focal loss.

    Down-weights examples the model already classifies confidently, so the
    gradient budget goes to the hard minority instead of the 99.35% of rows that
    are obviously legitimate.

    Args:
        logits: Raw model outputs ``(B,)``.
        targets: Binary targets ``(B,)``.
        alpha: Weight on the positive class in ``[0, 1]``.
        gamma: Focusing exponent; 0 recovers weighted cross-entropy.

    Returns:
        Scalar mean loss.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * bce).mean()


@dataclass
class TrainedModel:
    """A fitted sequence model and everything the rest of the pipeline needs.

    Attributes:
        model: The model, restored to its best-validation weights.
        config: The architecture configuration it was built with.
        scores: Validation and test scores plus timing and size.
        history: Per-epoch training loss and validation PR-AUC.
        best_epoch: Epoch whose weights were kept.
    """

    model: nn.Module
    config: ModelConfig
    scores: ModelScores
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0

    def save(self, directory: Path) -> Path:
        """Persist weights and configuration for serving and explanation.

        Args:
            directory: Destination directory.

        Returns:
            Path to the written checkpoint.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.scores.name}_seed{self.scores.seed}.pt"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_config": {
                    "n_numeric": self.config.n_numeric,
                    "categorical": [
                        {"column": c.column, "cardinality": c.cardinality}
                        for c in self.config.categorical
                    ],
                    "max_len": self.config.max_len,
                    "d_model": self.config.d_model,
                    "dropout": self.config.dropout,
                    "emb_max_dim": self.config.emb_max_dim,
                    "hidden": self.config.hidden,
                    "n_layers": self.config.n_layers,
                    "n_heads": self.config.n_heads,
                    "kernel_size": self.config.kernel_size,
                    "cell": self.config.cell,
                    "embedding_dims": self.config.embedding_dims,
                },
                "family": self.scores.family,
                "name": self.scores.name,
                "seed": self.scores.seed,
                "meta": self.scores.meta,
            },
            path,
        )
        return path


def set_seed(seed: int) -> None:
    """Seed every RNG that affects training.

    Args:
        seed: The seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(
    model: nn.Module, data: SequenceData, rows: np.ndarray, batch_size: int = 2048
) -> np.ndarray:
    """Score rows with a trained model.

    Args:
        model: A trained sequence classifier.
        data: Backing sequence arrays.
        rows: Row indices to score.
        batch_size: Inference batch size.

    Returns:
        Probabilities of shape ``(len(rows),)``.
    """
    model.eval()
    out = np.empty(len(rows), dtype=np.float64)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = gather(data, chunk)
        logits = model(batch["numeric"], batch["categorical"], batch["mask"])
        out[start : start + len(chunk)] = torch.sigmoid(logits).numpy()
    return out


def train_one(
    architecture: str,
    data: SequenceData,
    split: TemporalSplit,
    *,
    strategy: Strategy = "focal",
    seed: int = 0,
    train_config: TrainConfig | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> TrainedModel:
    """Train a single architecture / strategy / seed combination.

    Args:
        architecture: Key from :data:`~src.models.MODEL_REGISTRY`.
        data: Sequence tensors covering every row.
        split: Temporal split masks.
        strategy: Imbalance handling: ``"focal"``, ``"weighted"`` or ``"bce"``.
        seed: RNG seed.
        train_config: Optimisation settings.
        model_overrides: Extra architecture keyword arguments.

    Returns:
        The fitted model with best-validation weights restored.

    Raises:
        ValueError: If ``strategy`` is unknown.
    """
    if strategy not in {"bce", "focal", "weighted"}:
        raise ValueError(f"unknown strategy {strategy!r}")
    cfg = train_config or TrainConfig()
    set_seed(seed)

    config = make_config(
        architecture,
        n_numeric=data.n_numeric,
        categorical=list(data.spec.categorical),
        max_len=data.max_len,
        **(model_overrides or {}),
    )
    model = build_model(architecture, config)
    breakdown = parameter_breakdown(model)

    rows_train = np.flatnonzero(split.train)
    rows_val = np.flatnonzero(split.val)
    rows_test = np.flatnonzero(split.test)
    y_val = data.labels[rows_val]

    weights = (
        positive_sample_weights(data.labels, rows_train, cfg.pos_sample_weight)
        if strategy == "weighted"
        else None
    )
    batcher = SequenceBatcher(
        data,
        rows_train,
        cfg.batch_size,
        shuffle=True,
        sample_weights=weights,
        seed=seed,
        drop_last=True,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.max_epochs)

    name = f"{architecture}_{strategy}"
    best_pr = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch, _ in batcher:
            optimiser.zero_grad(set_to_none=True)
            logits = model(batch["numeric"], batch["categorical"], batch["mask"])
            target = batch["label"]
            if strategy == "focal":
                loss = focal_loss_with_logits(logits, target, cfg.focal_alpha, cfg.focal_gamma)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()
            total_loss += float(loss.detach())
            n_batches += 1
        scheduler.step()

        val_scores = predict(model, data, rows_val)
        pr_auc = float(average_precision_score(y_val, val_scores))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(n_batches, 1),
                "val_pr_auc": pr_auc,
            }
        )
        LOGGER.info(
            "%s seed=%d epoch %2d: loss %.5f  val PR-AUC %.4f%s",
            name,
            seed,
            epoch,
            history[-1]["train_loss"],
            pr_auc,
            "  *" if pr_auc > best_pr else "",
        )
        if pr_auc > best_pr:
            best_pr = pr_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        elif epoch - best_epoch >= cfg.patience:
            LOGGER.info("%s seed=%d: early stop at epoch %d", name, seed, epoch)
            break

    train_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)

    scores = ModelScores(
        name=name,
        family=architecture,
        seed=seed,
        val_scores=predict(model, data, rows_val),
        test_scores=predict(model, data, rows_test),
        train_seconds=train_seconds,
        n_params=breakdown["total"],
        meta={
            "imbalance_strategy": strategy,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "best_val_pr_auc": best_pr,
            "param_breakdown": breakdown,
            "embedding_dims": list(config.embedding_dims),
            "d_model": config.d_model,
            "hidden": config.hidden,
            "n_layers": config.n_layers,
            "max_len": config.max_len,
        },
    )
    LOGGER.info(
        "%s seed=%d: %d params, %.1fs, best val PR-AUC %.4f at epoch %d",
        name,
        seed,
        breakdown["total"],
        train_seconds,
        best_pr,
        best_epoch,
    )
    return TrainedModel(
        model=model, config=config, scores=scores, history=history, best_epoch=best_epoch
    )


def train_all(
    data: SequenceData,
    split: TemporalSplit,
    *,
    architectures: tuple[str, ...] = COMPARED_MODELS,
    strategies: tuple[Strategy, ...] = STRATEGIES,
    seeds: tuple[int, ...] = SEEDS,
    train_config: TrainConfig | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[ModelScores], dict[str, TrainedModel]]:
    """Train every architecture / strategy / seed combination.

    Args:
        data: Sequence tensors.
        split: Temporal split masks.
        architectures: Architectures to train.
        strategies: Imbalance strategies to compare.
        seeds: Seeds per arm.
        train_config: Optimisation settings.
        checkpoint_dir: If given, each model's weights are written here.

    Returns:
        Tuple of (score objects, trained models keyed by ``"<name>_seed<seed>"``).
    """
    all_scores: list[ModelScores] = []
    trained: dict[str, TrainedModel] = {}
    for architecture in architectures:
        for strategy in strategies:
            for seed in seeds:
                result = train_one(
                    architecture,
                    data,
                    split,
                    strategy=strategy,
                    seed=seed,
                    train_config=train_config,
                )
                key = f"{result.scores.name}_seed{seed}"
                trained[key] = result
                all_scores.append(result.scores)
                if checkpoint_dir is not None:
                    result.save(checkpoint_dir)
    return all_scores, trained


def prepare(
    dataset: str = "synthetic",
    data_dir: Path = SYNTHETIC_DIR,
    split_config: SplitConfig | None = None,
    sequence_config: SequenceConfig | None = None,
) -> tuple[Any, TemporalSplit, FeatureBuilder, SequenceData]:
    """Load a dataset and build everything the trainers need.

    Args:
        dataset: Dataset key.
        data_dir: Where the synthetic dataset lives.
        split_config: Split fractions.
        sequence_config: Window settings.

    Returns:
        Tuple of (feature frame, split, fitted builder, sequence data).
    """
    from src.data.loaders import load_dataset

    df, _ = load_dataset(dataset, data_dir=data_dir)
    split = make_temporal_split(df, split_config or SplitConfig())
    builder = FeatureBuilder()
    frame = builder.fit_transform(df, split.train)
    data = build_sequence_data(frame, builder, (sequence_config or SequenceConfig()).max_len)
    return frame, split, builder, data


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: train the sequence models and print test PR-AUC."""
    parser = argparse.ArgumentParser(description="Train the sequence models.")
    parser.add_argument("--data", type=Path, default=SYNTHETIC_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--checkpoints", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--models", nargs="+", default=list(COMPARED_MODELS))
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    frame, split, _, data = prepare(args.dataset, args.data)
    cfg = TrainConfig()
    if args.max_epochs is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "max_epochs": args.max_epochs})

    scores, _ = train_all(
        data,
        split,
        architectures=tuple(args.models),
        strategies=tuple(args.strategies),  # type: ignore[arg-type]
        seeds=tuple(args.seeds),
        train_config=cfg,
        checkpoint_dir=args.checkpoints,
    )

    from src.evaluation import evaluate, results_frame

    rows_test = np.flatnonzero(split.test)
    y_test = data.labels[rows_test]
    fraud_type = data.fraud_type[rows_test]
    results = [
        evaluate(
            y_test,
            s.test_scores,
            name=s.name,
            split="test",
            fraud_type=fraud_type,
            extra={"seed": s.seed, "train_seconds": s.train_seconds, "n_params": s.n_params},
        )
        for s in scores
    ]
    args.results.mkdir(parents=True, exist_ok=True)
    out = results_frame(results)
    out.to_csv(args.results / "sequence_results.csv", index=False)
    for s in scores:
        s.save(args.results / "scores")

    columns = ["model", "seed", "pr_auc", "precision@0.005", "recall@0.005", "train_seconds"]
    print(out[[c for c in columns if c in out.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
