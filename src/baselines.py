"""The honest baseline: LightGBM on hand-engineered aggregate features.

This is the number the sequence models have to beat, so it is built to win. A
crippled baseline proves nothing and is obvious to any reader: the usual tells
are default hyperparameters, no categorical handling, no velocity features, and
target encodings fitted on the whole dataset. None of those apply here.

What the baseline gets:

* Velocity counts and amounts over 1h / 24h / 7d windows.
* Amount z-scores against the customer's own expanding history, plus ratios to
  their running mean and maximum.
* Time since the previous transaction, since the same merchant, same MCC, same
  device.
* Entity novelty flags and distinct-entity counts.
* Merchant / MCC / country / device-type risk rates, target-encoded out-of-fold
  on strictly earlier temporal blocks.
* Raw high-cardinality categoricals passed to LightGBM as native categorical
  features, which is genuinely competitive with learned embeddings and is the
  fairest possible comparison.

In other words the baseline sees the same information the sequence models see —
just flattened into one row instead of a window. That is exactly the comparison
worth making: not "trees versus neural nets", but "is there signal in the
*ordering* of a customer's transactions that a well-built aggregate misses?"

Two arms are trained, mirroring the imbalance strategies used for the deep
models: a plain one that keeps its probabilities honest, and a
``scale_pos_weight`` one that trades calibration for ranking.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.artifacts import ModelScores
from src.config import RESULTS_DIR, SEEDS, SYNTHETIC_DIR, SplitConfig
from src.data.features import NUMERIC_FEATURES, FeatureBuilder, FeatureSpec
from src.data.schema import TransactionSchema
from src.data.splits import TemporalSplit, make_temporal_split
from src.evaluation import evaluate, log_accuracy_warning, results_frame

LOGGER = logging.getLogger(__name__)

S = TransactionSchema

#: Tuned by hand on the validation split, not by an automated search: the point
#: of the baseline is to be a fair strong reference, and an exhaustive sweep here
#: with none for the deep models would make the comparison dishonest in the other
#: direction. Both sides get a sensible, lightly hand-tuned configuration.
LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "min_data_per_group": 50,
    "verbosity": -1,
    "num_threads": 0,
}

NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 100


@dataclass
class BaselineArtifacts:
    """Everything a trained baseline produces.

    Attributes:
        booster: The fitted LightGBM model, needed by :mod:`src.explain`.
        scores: Validation and test scores plus timings.
        importance: Gain-based feature importance, descending.
        feature_names: Column order the booster was trained on.
        categorical_features: Which of those are categorical.
    """

    booster: lgb.Booster
    scores: ModelScores
    importance: pd.DataFrame
    feature_names: list[str]
    categorical_features: list[str]


def baseline_feature_frame(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Select and type the columns LightGBM trains on.

    Args:
        frame: Feature frame from :class:`~src.data.features.FeatureBuilder`.
        spec: Feature specification.

    Returns:
        DataFrame of numeric features plus categorical codes typed as ``category``
        so LightGBM uses its native categorical splits rather than treating the
        codes as ordinal integers.
    """
    columns = list(NUMERIC_FEATURES) + spec.categorical_columns
    out = frame[columns].copy()
    for column in spec.categorical_columns:
        out[column] = out[column].astype("category")
    return out


def train_lightgbm(
    frame: pd.DataFrame,
    split: TemporalSplit,
    spec: FeatureSpec,
    *,
    seed: int = 0,
    weighted: bool = False,
    name: str | None = None,
) -> BaselineArtifacts:
    """Train one LightGBM baseline arm.

    Early stopping is on validation average precision, never on log loss: the
    loss is dominated by the 99.35% of rows that are trivially negative, so it
    keeps improving long after ranking quality has stopped.

    Args:
        frame: Full feature frame in time order.
        split: Temporal split masks.
        spec: Feature specification.
        seed: RNG seed for bagging and feature sampling.
        weighted: Apply ``scale_pos_weight`` to rebalance the classes. This
            usually helps ranking a little and wrecks probability calibration,
            which :mod:`src.calibration` then measures.
        name: Override for the model name.

    Returns:
        Populated :class:`BaselineArtifacts`.
    """
    x = baseline_feature_frame(frame, spec)
    y = frame[S.IS_FRAUD].to_numpy()

    x_train, y_train = x.loc[split.train], y[split.train]
    x_val, y_val = x.loc[split.val], y[split.val]
    x_test = x.loc[split.test]

    params = dict(LGB_PARAMS)
    params["seed"] = seed
    params["bagging_seed"] = seed + 1
    params["feature_fraction_seed"] = seed + 2
    if weighted:
        n_pos = max(int(y_train.sum()), 1)
        params["scale_pos_weight"] = float((len(y_train) - n_pos) / n_pos)

    train_set = lgb.Dataset(x_train, label=y_train, categorical_feature=spec.categorical_columns)
    val_set = lgb.Dataset(
        x_val, label=y_val, reference=train_set, categorical_feature=spec.categorical_columns
    )

    started = time.perf_counter()
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    train_seconds = time.perf_counter() - started

    val_scores = booster.predict(x_val, num_iteration=booster.best_iteration)
    test_scores = booster.predict(x_test, num_iteration=booster.best_iteration)

    model_name = name or ("lightgbm_weighted" if weighted else "lightgbm")
    importance = (
        pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "gain": booster.feature_importance("gain"),
                "split": booster.feature_importance("split"),
            }
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    # "Parameters" for a GBM is a category error, but readers still want a size:
    # total leaf count is the honest analogue and is what we report.
    n_leaves = int(booster.num_trees() * params["num_leaves"])

    scores = ModelScores(
        name=model_name,
        family="lightgbm",
        seed=seed,
        val_scores=np.asarray(val_scores, dtype=np.float64),
        test_scores=np.asarray(test_scores, dtype=np.float64),
        train_seconds=train_seconds,
        n_params=n_leaves,
        meta={
            "best_iteration": int(booster.best_iteration),
            "n_trees": int(booster.num_trees()),
            "imbalance_strategy": "scale_pos_weight" if weighted else "none",
            "n_features": int(x.shape[1]),
            "params": {k: v for k, v in params.items() if k != "num_threads"},
        },
    )
    LOGGER.info(
        "%s seed=%d: %d trees in %.1fs (best iteration %d)",
        model_name,
        seed,
        booster.num_trees(),
        train_seconds,
        booster.best_iteration,
    )
    return BaselineArtifacts(
        booster=booster,
        scores=scores,
        importance=importance,
        feature_names=list(x.columns),
        categorical_features=list(spec.categorical_columns),
    )


def rule_baseline(frame: pd.DataFrame, split: TemporalSplit) -> ModelScores:
    """A hand-written rules score, standing in for a pre-ML fraud engine.

    Not a serious contender — the point is to establish the floor. A reader
    should be able to see how much of the achievable signal is picked up by four
    obvious rules, and therefore how much the learned models are actually adding.

    The rules: unusually large amount for this customer, a burst of transactions
    in the last hour, an unfamiliar device, and an unfamiliar merchant.

    Args:
        frame: Feature frame in time order.
        split: Temporal split masks.

    Returns:
        :class:`ModelScores` with the rule score on validation and test.
    """
    score = (
        1.5 * np.clip(frame["amount_z"].to_numpy(), 0, 8) / 8.0
        + 1.0 * np.clip(frame["log_cnt_1h"].to_numpy(), 0, 3) / 3.0
        + 0.8 * frame["is_new_device"].to_numpy()
        + 0.5 * frame["is_new_merchant"].to_numpy()
    )
    score = score / score.max()
    return ModelScores(
        name="rules",
        family="rules",
        seed=0,
        val_scores=score[split.val],
        test_scores=score[split.test],
        train_seconds=0.0,
        n_params=4,
        meta={"description": "four hand-written rules; establishes the floor"},
    )


def run_baselines(
    frame: pd.DataFrame,
    split: TemporalSplit,
    spec: FeatureSpec,
    *,
    seeds: tuple[int, ...] = SEEDS,
) -> tuple[list[ModelScores], dict[str, BaselineArtifacts]]:
    """Train every baseline arm across seeds.

    Args:
        frame: Feature frame in time order.
        split: Temporal split masks.
        spec: Feature specification.
        seeds: Seeds to run.

    Returns:
        Tuple of (all score objects, artefacts keyed by ``"<name>_seed<seed>"``
        for the models needed later by SHAP and serving).
    """
    all_scores: list[ModelScores] = [rule_baseline(frame, split)]
    artifacts: dict[str, BaselineArtifacts] = {}
    for weighted in (False, True):
        for seed in seeds:
            art = train_lightgbm(frame, split, spec, seed=seed, weighted=weighted)
            all_scores.append(art.scores)
            artifacts[f"{art.scores.name}_seed{seed}"] = art
    return all_scores, artifacts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: train the baselines and print their test metrics."""
    parser = argparse.ArgumentParser(description="Train the LightGBM baseline.")
    parser.add_argument("--data", type=Path, default=SYNTHETIC_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--dataset", default="synthetic", help="synthetic | ieee-cis | ulb")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from src.data.loaders import load_dataset

    df, _ = load_dataset(args.dataset, data_dir=args.data)
    split = make_temporal_split(df, SplitConfig())
    builder = FeatureBuilder()
    frame = builder.fit_transform(df, split.train)
    assert builder.spec is not None

    scores, _ = run_baselines(frame, split, builder.spec, seeds=tuple(args.seeds))

    y_test = frame.loc[split.test, S.IS_FRAUD].to_numpy()
    fraud_type = frame.loc[split.test, S.FRAUD_TYPE].to_numpy()
    results = [
        evaluate(
            y_test,
            s.test_scores,
            name=s.name,
            split="test",
            fraud_type=fraud_type,
            extra={"seed": s.seed, "train_seconds": s.train_seconds},
        )
        for s in scores
    ]
    log_accuracy_warning(results[0])

    args.results.mkdir(parents=True, exist_ok=True)
    frame_out = results_frame(results)
    frame_out.to_csv(args.results / "baseline_results.csv", index=False)
    for s in scores:
        s.save(args.results / "scores")

    columns = ["model", "seed", "pr_auc", "precision@0.005", "recall@0.005", "train_seconds"]
    print(frame_out[[c for c in columns if c in frame_out.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
