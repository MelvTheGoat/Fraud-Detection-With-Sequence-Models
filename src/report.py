"""Render the results tables that go into the README.

The write-up quotes numbers; this module produces them from the artefacts on
disk so the two cannot drift. Run it after :mod:`src.experiment` and paste, or
diff it against the README to check nothing has gone stale.

Usage:
    python -m src.report --results results
    python -m src.report --results results --check README.md
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import RESULTS_DIR

LOGGER = logging.getLogger(__name__)


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a number for a markdown cell, tolerating missing values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pm(mean: Any, std: Any, digits: int = 4) -> str:
    """Format ``mean ± std`` for a markdown cell."""
    if mean is None or pd.isna(mean):
        return "-"
    if std is None or pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def architecture_table(results_dir: Path) -> str:
    """Architecture comparison: PR-AUC and precision/recall at 0.5% capacity."""
    path = results_dir / "results_aggregated.csv"
    if not path.exists():
        return "_(run `make experiment` to populate)_"
    agg = pd.read_csv(path)
    agg = agg[agg["split"] == "test"].sort_values("pr_auc_mean", ascending=False)

    summary_path = results_dir / "summary.json"
    sizes: dict[str, dict[str, Any]] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        sizes = {m["name"]: m for m in summary.get("models", [])}

    lines = [
        "| model | PR-AUC | precision @0.5% | recall @0.5% | params | train (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in agg.iterrows():
        info = sizes.get(row["model"], {})
        lines.append(
            "| `{model}` | {pr} | {prec} | {rec} | {params} | {secs} |".format(
                model=row["model"],
                pr=_pm(row.get("pr_auc_mean"), row.get("pr_auc_std")),
                prec=_pm(row.get("precision@0.005_mean"), row.get("precision@0.005_std")),
                rec=_pm(row.get("recall@0.005_mean"), row.get("recall@0.005_std")),
                params=_fmt(int(info["n_params"])) if info.get("n_params") else "-",
                secs=_fmt(info.get("mean_train_seconds"), 1) if info else "-",
            )
        )
    return "\n".join(lines)


def capacity_table(results_dir: Path) -> str:
    """Precision and recall at each review capacity."""
    path = results_dir / "results_aggregated.csv"
    if not path.exists():
        return "_(run `make experiment` to populate)_"
    agg = pd.read_csv(path)
    agg = agg[agg["split"] == "test"].sort_values("pr_auc_mean", ascending=False)
    caps = ["0.001", "0.005", "0.01"]
    header = "| model | " + " | ".join(
        f"P@{float(c) * 100:g}% / R@{float(c) * 100:g}%" for c in caps
    ) + " |"
    lines = [header, "| --- | " + " | ".join("---" for _ in caps) + " |"]
    for _, row in agg.iterrows():
        cells = []
        for cap in caps:
            precision = row.get(f"precision@{cap}_mean")
            recall = row.get(f"recall@{cap}_mean")
            cells.append(f"{_fmt(precision, 3)} / {_fmt(recall, 3)}")
        lines.append(f"| `{row['model']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mechanism_table(results_dir: Path, capacity: str = "0.005") -> str:
    """Per-fraud-mechanism recall at a fixed capacity."""
    path = results_dir / "results_aggregated.csv"
    if not path.exists():
        return "_(run `make experiment` to populate)_"
    agg = pd.read_csv(path)
    agg = agg[agg["split"] == "test"]
    mechanisms = [
        c.split(":")[-1]
        for c in agg.columns
        if c.startswith(f"recall@{capacity}:") and c.endswith("_mean")
    ]
    mechanisms = sorted({m.removesuffix("_mean") for m in mechanisms})
    if not mechanisms:
        return "_(per-mechanism labels unavailable for this dataset)_"
    agg = agg.sort_values("pr_auc_mean", ascending=False)
    lines = [
        "| model | " + " | ".join(m.replace("_", " ") for m in mechanisms) + " |",
        "| --- | " + " | ".join("---" for _ in mechanisms) + " |",
    ]
    for _, row in agg.iterrows():
        cells = [
            _pm(
                row.get(f"recall@{capacity}:{m}_mean"),
                row.get(f"recall@{capacity}:{m}_std"),
                3,
            )
            for m in mechanisms
        ]
        lines.append(f"| `{row['model']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def drift_table(results_dir: Path) -> str:
    """PR-AUC before and after the adversarial drift."""
    path = results_dir / "results_by_drift_regime.csv"
    if not path.exists():
        return "_(run `make experiment` to populate)_"
    agg = pd.read_csv(path)
    pivot = agg.pivot(index="model", columns="split", values="pr_auc_mean")
    if "pre_drift" not in pivot.columns or "post_drift" not in pivot.columns:
        return "_(the test window does not straddle the drift boundary)_"
    pivot["drop"] = pivot["pre_drift"] - pivot["post_drift"]
    pivot = pivot.sort_values("post_drift", ascending=False)
    lines = [
        "| model | PR-AUC pre-drift | PR-AUC post-drift | drop |",
        "| --- | --- | --- | --- |",
    ]
    for model, row in pivot.iterrows():
        lines.append(
            f"| `{model}` | {_fmt(row['pre_drift'])} | {_fmt(row['post_drift'])} "
            f"| {_fmt(row['drop'])} |"
        )
    return "\n".join(lines)


def cost_table(results_dir: Path) -> str:
    """The headline table: expected cost per transaction."""
    path = results_dir / "policy.csv"
    if not path.exists():
        return "_(run `make experiment` to populate)_"
    policy = pd.read_csv(path).sort_values("test_cost_per_txn_mean")
    summary_path = results_dir / "summary.json"
    trivial: dict[str, float] = {}
    if summary_path.exists():
        trivial = json.loads(summary_path.read_text()).get("policy", {}).get(
            "trivial_policies", {}
        )
    lines = [
        "| model | cost / txn | vs block-nothing | alert rate | precision | recall |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in policy.iterrows():
        saved = row["savings_vs_block_nothing_mean"]
        lines.append(
            "| `{m}` | {cost} | {saving} | {rate} | {prec} | {rec} |".format(
                m=row["model"],
                cost=_pm(row["test_cost_per_txn_mean"], row.get("test_cost_per_txn_std")),
                # Negative savings means the policy costs more than doing
                # nothing, which the rule baseline manages; show it as such.
                saving=("-" if saved > 0 else "+") + _fmt(abs(saved), 4),
                rate=_fmt(row["test_alert_rate_mean"], 4),
                prec=_fmt(row["test_precision_mean"], 3),
                rec=_fmt(row["test_recall_mean"], 3),
            )
        )
    if trivial:
        lines.append(
            f"| _block nothing_ | {_fmt(trivial.get('block_nothing'), 4)} | - | 0.0000 | - | 0.000 |"
        )
        lines.append(
            f"| _block everything_ | {_fmt(trivial.get('block_everything'), 4)} | - | 1.0000 | - | 1.000 |"
        )
    return "\n".join(lines)


def calibration_table(results_dir: Path) -> str:
    """Calibration before and after recalibration."""
    path = results_dir / "calibration.csv"
    if not path.exists():
        return "_(run `make experiment` to populate)_"
    calibration = pd.read_csv(path)
    pivot_ece = calibration.pivot(index="model", columns="method", values="ece")
    raw = calibration[calibration["method"] == "raw"].set_index("model")
    lines = [
        "| model | ECE raw | ECE temp | ECE isotonic | alert-region pred | alert-region actual | ratio |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in pivot_ece.index:
        ratio = raw["alert_confidence_ratio"].get(model, float("nan"))
        lines.append(
            "| `{m}` | {r} | {t} | {i} | {ap} | {ao} | {c} |".format(
                m=model,
                r=_fmt(pivot_ece.loc[model].get("raw"), 5),
                t=_fmt(pivot_ece.loc[model].get("temperature"), 5),
                i=_fmt(pivot_ece.loc[model].get("isotonic"), 5),
                ap=_fmt(raw["alert_mean_predicted"].get(model), 3),
                ao=_fmt(raw["alert_observed_rate"].get(model), 3),
                c=f"{ratio:.2f}x" if pd.notna(ratio) else "-",
            )
        )
    return "\n".join(lines)


def latency_table(results_dir: Path) -> str:
    """Serving latency percentiles."""
    path = results_dir / "latency.json"
    if not path.exists():
        return "_(run `make bench` to populate)_"
    report = json.loads(path.read_text())
    lines = ["| path | p50 (ms) | p95 (ms) | p99 (ms) |", "| --- | --- | --- | --- |"]
    for row in report["stats"]:
        lines.append(
            f"| `{row['path']}` | {row['p50_ms']:.3f} | {row['p95_ms']:.3f} "
            f"| {row['p99_ms']:.3f} |"
        )
    return "\n".join(lines)


def embedding_table(results_dir: Path) -> str:
    """Embedding widths and their parameter cost."""
    path = results_dir / "embeddings.csv"
    if not path.exists():
        return "_(run `make experiment` to populate)_"
    frame = pd.read_csv(path)
    lines = [
        "| column | cardinality | embedding dim | parameters |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| `{row['column']}` | {int(row['cardinality']):,} | {int(row['dim'])} "
            f"| {int(row['params']):,} |"
        )
    lines.append(f"| **total** | | | **{int(frame['params'].sum()):,}** |")
    return "\n".join(lines)


def monitoring_table(results_dir: Path) -> str:
    """Per-window monitoring signals."""
    path = results_dir / "monitoring.csv"
    if not path.exists():
        return "_(run `make experiment` to populate)_"
    report = pd.read_csv(path)
    columns = [
        c
        for c in (
            "start",
            "n",
            "score_psi",
            "alert_rate",
            "alert_rate_ratio",
            "precision_at_capacity",
            "recall_at_capacity",
            "severity",
        )
        if c in report.columns
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in report.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column == "start":
                cells.append(str(value)[:16])
            elif isinstance(value, float):
                cells.append(_fmt(value, 4))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


TABLES = {
    "architecture": architecture_table,
    "capacity": capacity_table,
    "mechanism": mechanism_table,
    "drift": drift_table,
    "cost": cost_table,
    "calibration": calibration_table,
    "latency": latency_table,
    "embeddings": embedding_table,
    "monitoring": monitoring_table,
}


def render_all(results_dir: Path = RESULTS_DIR) -> str:
    """Render every table with a heading, as one markdown document."""
    blocks = []
    for name, fn in TABLES.items():
        blocks.append(f"### {name}\n\n{fn(results_dir)}\n")
    return "\n".join(blocks)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Render results tables for the README.")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--table", choices=sorted(TABLES), default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.table:
        print(TABLES[args.table](args.results))
    else:
        print(render_all(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
