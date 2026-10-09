"""End-to-end experiment: train, evaluate, calibrate, cost, explain, monitor, deploy.

Running this produces every number in the README. It is deliberately one script
rather than a notebook, so the write-up cannot quietly drift away from the code
that produced it.

Order matters and is enforced here:

1. Split by time, then fit features on the training window only.
2. Train the baselines and the sequence models on identical inputs.
3. Evaluate on test, aggregating across seeds.
4. Fit calibrators on validation, evaluate on test.
5. Choose decision thresholds on validation, cost them on test.
6. Select a model to deploy on validation cost, never test.
7. Explain, monitor, export.

Step 6 is the one that is easy to get wrong: picking the deployed model by test
cost and then reporting that test cost is circular. The selection uses the
validation split, and the test number is reported afterwards as an out-of-sample
result.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts import ModelScores, load_all_scores, write_json
from src.baselines import BaselineArtifacts, baseline_feature_frame, run_baselines
from src.calibration import calibrate_and_compare, plot_reliability
from src.config import (
    ARTIFACT_DIR,
    RESULTS_DIR,
    SEEDS,
    SYNTHETIC_DIR,
    CostConfig,
    MonitoringConfig,
    SequenceConfig,
    ServingConfig,
    SplitConfig,
    TrainConfig,
    ensure_dirs,
)
from src.data.features import NUMERIC_FEATURES, FeatureBuilder
from src.data.loaders import load_dataset
from src.data.schema import TransactionSchema
from src.data.sequences import build_sequence_data
from src.data.splits import make_temporal_split
from src.evaluation import (
    aggregate_seeds,
    comparison_table,
    evaluate,
    log_accuracy_warning,
    results_frame,
)
from src.models import COMPARED_MODELS, describe_embeddings
from src.monitoring import detection_summary, feature_drift, monitor_over_time, plot_monitoring
from src.policy import (
    aggregate_policies,
    apply_policy,
    baseline_costs,
    compare_policies,
    plot_cost_frontier,
    sensitivity,
    sweep_thresholds,
)
from src.train import STRATEGIES, TrainedModel, train_all

LOGGER = logging.getLogger(__name__)

S = TransactionSchema


def _scores_by_name(scores: list[ModelScores]) -> dict[str, list[ModelScores]]:
    """Group score objects by model name."""
    grouped: dict[str, list[ModelScores]] = {}
    for entry in scores:
        grouped.setdefault(entry.name, []).append(entry)
    return grouped


def run(
    *,
    dataset: str = "synthetic",
    data_dir: Path = SYNTHETIC_DIR,
    results_dir: Path = RESULTS_DIR,
    artifact_dir: Path = ARTIFACT_DIR,
    seeds: tuple[int, ...] = SEEDS,
    architectures: tuple[str, ...] = COMPARED_MODELS,
    reuse_scores: bool = True,
    train_config: TrainConfig | None = None,
) -> dict[str, Any]:
    """Run the whole pipeline and write every artefact.

    Args:
        dataset: Dataset key.
        data_dir: Synthetic data directory.
        results_dir: Where tables and figures are written.
        artifact_dir: Where the deployment bundle is written.
        seeds: Seeds per arm.
        architectures: Sequence architectures to compare.
        reuse_scores: Reuse score files already in ``results/scores`` instead of
            retraining. Training the full grid takes hours on a CPU.
        train_config: Optimisation settings.

    Returns:
        A summary dict, also written to ``results/summary.json``.
    """
    ensure_dirs()
    results_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    summary: dict[str, Any] = {"dataset": dataset}

    # ---------------------------------------------------------------- data ----
    LOGGER.info("loading %s", dataset)
    df, data_meta = load_dataset(dataset, data_dir=data_dir)
    split = make_temporal_split(df, SplitConfig())
    LOGGER.info("\n%s", split.summary(df).to_string(index=False))
    summary["data"] = {
        "n_transactions": int(len(df)),
        "fraud_rate": float(df[S.IS_FRAUD].mean()),
        "splits": split.summary(df).to_dict(orient="records"),
        "metadata": {k: v for k, v in data_meta.items() if k != "compromised_merchants"},
    }

    builder = FeatureBuilder()
    frame = builder.fit_transform(df, split.train)
    assert builder.spec is not None
    sequence_config = SequenceConfig()
    data = build_sequence_data(frame, builder, sequence_config.max_len)

    rows_val = np.flatnonzero(split.val)
    rows_test = np.flatnonzero(split.test)
    y_val = data.labels[rows_val]
    y_test = data.labels[rows_test]
    amounts_val = data.amounts[rows_val]
    amounts_test = data.amounts[rows_test]
    fraud_type_test = data.fraud_type[rows_test]

    # ------------------------------------------------------------ training ----
    cached = _scores_by_name(load_all_scores(results_dir / "scores")) if reuse_scores else {}
    all_scores: list[ModelScores] = []
    baseline_artifacts: dict[str, BaselineArtifacts] = {}
    trained: dict[str, TrainedModel] = {}

    baseline_names = {"rules", "lightgbm", "lightgbm_weighted"}
    if reuse_scores and baseline_names <= set(cached):
        LOGGER.info("reusing cached baseline scores")
        for name in baseline_names:
            all_scores.extend(cached[name])
        # SHAP and the importance table still need a fitted booster.
        from src.baselines import train_lightgbm

        baseline_artifacts["lightgbm_seed0"] = train_lightgbm(
            frame, split, builder.spec, seed=0, weighted=False
        )
    else:
        LOGGER.info("training baselines")
        baseline_scores, baseline_artifacts = run_baselines(
            frame, split, builder.spec, seeds=seeds
        )
        all_scores.extend(baseline_scores)

    sequence_names = {f"{a}_{s}" for a in architectures for s in STRATEGIES}
    if reuse_scores and sequence_names <= set(cached):
        LOGGER.info("reusing cached sequence-model scores")
        for name in sequence_names:
            all_scores.extend(cached[name])
    else:
        LOGGER.info("training sequence models (this is the slow part)")
        sequence_scores, trained = train_all(
            data,
            split,
            architectures=architectures,
            seeds=seeds,
            train_config=train_config,
            checkpoint_dir=artifact_dir,
        )
        all_scores.extend(sequence_scores)

    for entry in all_scores:
        entry.save(results_dir / "scores")

    for entry in all_scores:
        if len(entry.test_scores) != len(rows_test):
            raise RuntimeError(
                f"cached scores for {entry.name} have {len(entry.test_scores)} test rows but the "
                f"current split has {len(rows_test)}. The data or split changed; delete "
                f"{results_dir / 'scores'} and rerun."
            )

    # ---------------------------------------------------------- evaluation ----
    LOGGER.info("evaluating on test")
    results = [
        evaluate(
            y_test,
            entry.test_scores,
            name=entry.name,
            split="test",
            fraud_type=fraud_type_test,
            extra={
                "seed": entry.seed,
                "family": entry.family,
                "train_seconds": entry.train_seconds,
                "n_params": entry.n_params,
                "imbalance_strategy": entry.meta.get("imbalance_strategy", "none"),
            },
        )
        for entry in all_scores
    ]
    log_accuracy_warning(results[0])

    per_seed = results_frame(results)
    per_seed.to_csv(results_dir / "results_per_seed.csv", index=False)
    aggregated = aggregate_seeds(results)
    aggregated.to_csv(results_dir / "results_aggregated.csv", index=False)
    headline = comparison_table(results, capacity=0.005)
    headline.to_csv(results_dir / "comparison.csv", index=False)
    LOGGER.info("\n%s", headline.to_string(index=False))

    # Drift-period breakdown: the test window straddles the adversarial change,
    # so a single test number averages two very different regimes.
    drift_start = pd.Timestamp(data_meta["drift_start"]) if data_meta.get("drift_start") else None
    if drift_start is not None:
        ts_test = pd.to_datetime(pd.Series(data.timestamps[rows_test]))
        pre = (ts_test < drift_start).to_numpy()
        regime_rows = []
        for entry in all_scores:
            for label, mask in (("pre_drift", pre), ("post_drift", ~pre)):
                if mask.sum() < 50 or y_test[mask].sum() == 0:
                    continue
                res = evaluate(
                    y_test[mask],
                    entry.test_scores[mask],
                    name=entry.name,
                    split=label,
                    fraud_type=fraud_type_test[mask],
                    extra={"seed": entry.seed},
                )
                regime_rows.append(res)
        regime = aggregate_seeds(regime_rows)
        regime.to_csv(results_dir / "results_by_drift_regime.csv", index=False)
        summary["drift_regime"] = regime.to_dict(orient="records")

    # --------------------------------------------------------- calibration ----
    LOGGER.info("assessing calibration")
    grouped = _scores_by_name(all_scores)
    calibration_rows: list[dict[str, Any]] = []
    calibrated_test: dict[str, dict[str, np.ndarray]] = {}
    calibrators: dict[str, Any] = {}
    # Calibrators are fitted per seed, not on a seed-averaged score. Averaging
    # first would silently turn every downstream number into an ensemble result
    # while the architecture table reports single models — and the artefact we
    # actually deploy is one checkpoint, not an ensemble. Per-seed keeps the two
    # tables comparable and keeps the headline honest about what ships.
    from src.calibration import IsotonicCalibrator, TemperatureScaler

    for name, entries in grouped.items():
        for entry in entries:
            reports, calibrated = calibrate_and_compare(
                y_val, entry.val_scores, y_test, entry.test_scores, model=name
            )
            for report in reports:
                row = report.to_row()
                row["seed"] = entry.seed
                calibration_rows.append(row)
            calibrated_test[f"{name}#{entry.seed}"] = calibrated
            calibrators[f"{name}#{entry.seed}"] = {
                "temperature": TemperatureScaler().fit(y_val, entry.val_scores),
                "isotonic": IsotonicCalibrator().fit(y_val, entry.val_scores),
            }
            if entry.seed == entries[0].seed and name in {
                "lightgbm",
                "gru_focal",
                "gru_weighted",
                "gru_bce",
            }:
                plot_reliability(
                    reports, figures / f"reliability_{name}.png", title=f"Reliability: {name}"
                )

    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(results_dir / "calibration_per_seed.csv", index=False)
    calibration_mean = (
        calibration.groupby(["model", "method"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["seed"], errors="ignore")
    )
    calibration_mean.to_csv(results_dir / "calibration.csv", index=False)
    LOGGER.info("\n%s", calibration_mean.round(5).to_string(index=False))

    # -------------------------------------------------------------- policy ----
    LOGGER.info("costing decisions")
    cost = CostConfig()
    policy_results = []
    sweeps: dict[str, pd.DataFrame] = {}
    for name, entries in grouped.items():
        for entry in entries:
            # The policy runs on the best-calibrated scores available, because
            # the threshold is an absolute probability and only means something
            # if the probabilities do.
            key = f"{name}#{entry.seed}"
            test_scores = calibrated_test[key]["isotonic"]
            val_scores = calibrators[key]["isotonic"].transform(entry.val_scores)
            result = apply_policy(
                name, y_val, val_scores, amounts_val, y_test, test_scores, amounts_test, cost
            )
            result.seed = entry.seed
            policy_results.append(result)
            if entry.seed == entries[0].seed:
                sweeps[name] = sweep_thresholds(y_test, test_scores, amounts_test, cost)

    policy_per_seed = compare_policies(policy_results)
    policy_per_seed.to_csv(results_dir / "policy_per_seed.csv", index=False)
    policy_table = aggregate_policies(policy_results)
    policy_table.to_csv(results_dir / "policy.csv", index=False)
    trivial = baseline_costs(y_test, amounts_test, cost)
    plot_cost_frontier(
        {k: v for k, v in sweeps.items() if k in headline["model"].head(6).tolist()},
        figures / "cost_frontier.png",
        block_nothing=trivial["block_nothing"],
    )
    LOGGER.info("\n%s", policy_table.round(5).to_string(index=False))
    summary["policy"] = {
        "cost_config": cost.__dict__,
        "trivial_policies": trivial,
        "table": policy_table.to_dict(orient="records"),
    }

    # Deployment is selected on *mean validation* cost across seeds, not on the
    # single luckiest run. Picking the best individual (model, seed) would be
    # selecting on noise: the seed spread is comparable to the gap between arms.
    mean_val_cost = (
        pd.DataFrame(
            [
                {"model": r.model, "val_cost": r.val.cost_per_transaction}
                for r in policy_results
            ]
        )
        .groupby("model")["val_cost"]
        .mean()
    )
    best_model = str(mean_val_cost.idxmin())
    # Among that arm's seeds, deploy the one whose validation cost is closest to
    # its arm's mean -- the representative run, not the flattering one.
    deploy_candidates = [r for r in policy_results if r.model == best_model]
    best_policy = min(
        deploy_candidates,
        key=lambda r: abs(r.val.cost_per_transaction - mean_val_cost[best_model]),
    )
    LOGGER.info(
        "selected %s seed %d for deployment (mean validation cost %.5f per transaction)",
        best_policy.model,
        best_policy.seed,
        float(mean_val_cost[best_model]),
    )
    deployed_key = f"{best_policy.model}#{best_policy.seed}"

    sensitivity_rows = []
    for name in ["lightgbm", best_policy.model]:
        if name not in grouped:
            continue
        entry = next(e for e in grouped[name] if e.seed == best_policy.seed)
        key = f"{name}#{entry.seed}"
        sensitivity_rows.append(
            sensitivity(
                y_val,
                calibrators[key]["isotonic"].transform(entry.val_scores),
                amounts_val,
                y_test,
                calibrated_test[key]["isotonic"],
                amounts_test,
                model=name,
            )
        )
    if sensitivity_rows:
        pd.concat(sensitivity_rows).to_csv(results_dir / "cost_sensitivity.csv", index=False)

    # ---------------------------------------------------------- monitoring ----
    LOGGER.info("running drift monitor")
    monitoring_config = MonitoringConfig()
    monitor_model = best_policy.model
    monitor_entry = next(e for e in grouped[monitor_model] if e.seed == best_policy.seed)
    monitor_scores = calibrated_test[deployed_key]["isotonic"]
    reference_scores = calibrators[deployed_key]["isotonic"].transform(
        monitor_entry.val_scores
    )
    watch_features = [
        "amount_z",
        "log_cnt_24h",
        "is_new_device",
        "log_secs_since_same_device",
        "device_type_risk",
        "merchant_risk",
        "is_new_merchant",
        "log_amount",
    ]
    monitor_report = monitor_over_time(
        monitor_scores,
        y_test,
        data.timestamps[rows_test],
        reference_scores=reference_scores,
        fraud_type=fraud_type_test,
        features=frame.loc[split.test, watch_features].reset_index(drop=True),
        reference_features=frame.loc[split.val, watch_features].reset_index(drop=True),
        production_threshold=best_policy.threshold,
        config=monitoring_config,
    )
    monitor_report.to_csv(results_dir / "monitoring.csv", index=False)
    if drift_start is not None:
        plot_monitoring(monitor_report, figures / "monitoring.png", drift_start=drift_start)
        detection = detection_summary(monitor_report, drift_start, monitoring_config)
        summary["monitoring"] = detection
        LOGGER.info("drift detection: %s", detection)

    drift_features = feature_drift(
        frame.loc[split.val, list(NUMERIC_FEATURES)],
        frame.loc[split.test, list(NUMERIC_FEATURES)],
        monitoring_config,
    )
    drift_features.to_csv(results_dir / "feature_drift.csv", index=False)

    # --------------------------------------------------------- explanation ----
    LOGGER.info("building the worked example")
    explanation_text = ""
    checkpoint_name = None
    for candidate in (best_policy.model, "transformer_focal", "gru_focal"):
        path = artifact_dir / f"{candidate}_seed0.pt"
        if path.exists():
            checkpoint_name = candidate
            break
    if checkpoint_name is not None:
        from src.api.service import load_checkpoint
        from src.explain import attention_alignment, plot_attention_alignment, worked_example

        model, model_config, _ = load_checkpoint(artifact_dir / f"{checkpoint_name}_seed0.pt")
        booster_key = "lightgbm_seed0"
        booster = baseline_artifacts.get(booster_key)
        explanation_text, _ = worked_example(
            model,
            data,
            frame,
            booster=booster.booster if booster else None,
            baseline_x=baseline_feature_frame(frame, builder.spec) if booster else None,
            rows=rows_test,
        )
        (results_dir / "worked_example.txt").write_text(explanation_text)
        LOGGER.info("\n%s", explanation_text)

        # The attention analysis is part of the write-up regardless of which arm
        # wins on cost, so it runs off the Transformer whenever one was trained
        # rather than only when the Transformer happens to be deployed.
        attention_checkpoint = artifact_dir / "transformer_focal_seed0.pt"
        if attention_checkpoint.exists():
            attention_model, attention_config, _ = load_checkpoint(attention_checkpoint)
            if attention_model.attention_weights(*_dummy_inputs(attention_config)) is not None:
                alignment = attention_alignment(attention_model, data, rows_test[:20000])
                alignment.to_csv(results_dir / "attention_alignment.csv", index=False)
                plot_attention_alignment(alignment, figures / "attention_alignment.png")
                if checkpoint_name != "transformer_focal":
                    attention_text, _ = worked_example(
                        attention_model, data, frame, rows=rows_test
                    )
                    (results_dir / "worked_example_transformer.txt").write_text(attention_text)

    # ---------------------------------------------------------- deployment ----
    LOGGER.info("building the deployment bundle")
    deployment = _deploy(
        best_policy.model,
        best_policy.seed,
        best_policy.threshold,
        grouped,
        calibrators,
        builder,
        artifact_dir,
        results_dir,
    )
    summary["deployment"] = deployment

    embedding_table = None
    if checkpoint_name is not None:
        embedding_table = describe_embeddings(model_config)
        pd.DataFrame(embedding_table).to_csv(results_dir / "embeddings.csv", index=False)

    summary["embeddings"] = embedding_table
    summary["models"] = [
        {
            "name": name,
            "family": entries[0].family,
            "n_params": entries[0].n_params,
            "mean_train_seconds": float(np.mean([e.train_seconds for e in entries])),
            "seeds": [e.seed for e in entries],
            "mean_epochs": float(
                np.mean([e.meta.get("epochs_run", 0) for e in entries])
            ),
        }
        for name, entries in grouped.items()
    ]
    summary["headline"] = headline.to_dict(orient="records")
    summary["calibration"] = calibration.to_dict(orient="records")
    summary["elapsed_seconds"] = time.perf_counter() - started
    write_json(summary, results_dir / "summary.json")
    LOGGER.info("done in %.1fs", summary["elapsed_seconds"])
    return summary


def _dummy_inputs(config: Any) -> tuple[Any, ...]:
    """Minimal inputs for probing whether a model exposes attention."""
    import torch

    return (
        torch.zeros(1, config.max_len, config.n_numeric),
        torch.zeros(1, config.max_len, len(config.categorical), dtype=torch.long),
        torch.ones(1, config.max_len),
    )


def _deploy(
    model_name: str,
    seed: int,
    threshold: float,
    grouped: dict[str, list[ModelScores]],
    calibrators: dict[str, Any],
    builder: FeatureBuilder,
    artifact_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """Write the serving bundle for the selected model and seed.

    The calibrator shipped here is the one fitted on *that seed's* validation
    scores. Shipping a calibrator fitted on a seed-average would map the
    deployed model's outputs through a curve fitted to a different model's
    score distribution -- the thresholds would be subtly wrong in production and
    nothing offline would notice.

    Args:
        model_name: Selected arm.
        seed: Seed of the checkpoint being deployed.
        threshold: Block threshold chosen on validation.
        grouped: Score objects by model name.
        calibrators: Fitted calibrators keyed ``"<name>#<seed>"``.
        builder: The fitted feature builder.
        artifact_dir: Bundle destination.
        results_dir: Results directory, for the latency report.

    Returns:
        Dict describing what was written.
    """
    from src.api.service import Deployment, export_onnx, load_checkpoint

    checkpoint = artifact_dir / f"{model_name}_seed{seed}.pt"
    if not checkpoint.exists():
        LOGGER.warning(
            "no checkpoint for %s; the deployment bundle needs a sequence model, "
            "skipping export",
            model_name,
        )
        return {"exported": False, "reason": f"no checkpoint for {model_name}"}

    model, config, payload = load_checkpoint(checkpoint)
    onnx_path = export_onnx(model, config, artifact_dir / "model.onnx")
    builder.save(artifact_dir / "feature_builder.pkl")

    with (artifact_dir / "calibrator.pkl").open("wb") as handle:
        pickle.dump(calibrators[f"{model_name}#{seed}"]["isotonic"], handle)

    entries = grouped[model_name]
    deployment = Deployment(
        model_name=model_name,
        model_family=payload["family"],
        model_version=ServingConfig().model_version,
        block_threshold=float(threshold),
        # Review tier at a fifth of the block threshold: alerts an analyst can
        # work, without declining the customer outright.
        review_threshold=float(threshold / 5.0),
        calibration="isotonic",
        n_parameters=int(entries[0].n_params),
        trained_on="synthetic (src/simulate.py)",
        selected_because=(
            f"lowest mean validation cost per transaction across seeds; "
            f"seed {seed} is the run closest to that arm's mean"
        ),
    )
    deployment.save(artifact_dir)

    report: dict[str, Any] = {
        "exported": True,
        "model_name": model_name,
        "onnx": str(onnx_path),
        "n_parameters": deployment.n_parameters,
    }
    try:
        from src.api.benchmark import benchmark

        latency = benchmark(artifact_dir, n=300)
        write_json(latency, results_dir / "latency.json")
        report["latency"] = {
            "p50_ms": next(
                r["p50_ms"] for r in latency["stats"] if r["path"].startswith("end_to_end")
            ),
            "p99_ms": latency["deployed_p99_ms"],
            "within_budget": latency["within_budget"],
            "onnx_speedup_p50": latency["onnx_speedup_p50"],
        }
    except Exception:  # pragma: no cover - benchmark is best-effort here
        LOGGER.exception("latency benchmark failed")
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the full experiment.")
    parser.add_argument("--data", type=Path, default=SYNTHETIC_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--models", nargs="+", default=list(COMPARED_MODELS))
    parser.add_argument(
        "--retrain", action="store_true", help="Ignore cached scores and retrain everything."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = run(
        dataset=args.dataset,
        data_dir=args.data,
        results_dir=args.results,
        artifact_dir=args.artifacts,
        seeds=tuple(args.seeds),
        architectures=tuple(args.models),
        reuse_scores=not args.retrain,
    )
    print(json.dumps({k: summary[k] for k in ("headline", "policy") if k in summary}, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
