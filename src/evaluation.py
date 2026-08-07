"""Metrics for a problem where accuracy is meaningless.

**Accuracy is meaningless here, and this module says so in numbers.** At a 0.65%
fraud rate a model that declines nothing scores 99.35% accurate, which is also
roughly what a good model scores, so the metric cannot distinguish them. Every
report produced here includes ``accuracy`` alongside ``accuracy_of_always_legit``
precisely so the reader can see the two are the same and stop looking at either.

What is measured instead:

* **PR-AUC** — ranking quality where the positive class is rare. ROC-AUC is
  reported too, but it is optimistic under extreme imbalance because the false
  positive rate barely moves when negatives dominate.
* **Precision and recall at a fixed review capacity** — the operating point a
  fraud team actually has. An analyst queue absorbs a fixed number of alerts per
  day; "top 0.1% / 0.5% / 1% of scored transactions" is that budget. A model that
  wins on PR-AUC but loses at 0.1% capacity is the worse model to deploy.
* **Per-mechanism recall** — on synthetic data the attack behind each label is
  known, so we can see *which* mechanisms a model catches. Aggregate metrics
  hide a model that is completely blind to one attack type while excelling at
  another, and that blindness is what an adversary finds and exploits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import REVIEW_CAPACITIES
from src.data.schema import FRAUD_TYPES, FraudType

LOGGER = logging.getLogger(__name__)


def top_k_threshold(scores: np.ndarray, capacity: float) -> tuple[float, int]:
    """Score threshold that flags the top ``capacity`` fraction of transactions.

    Args:
        scores: Model scores, higher is more suspicious.
        capacity: Fraction of the population that can be reviewed, in ``(0, 1]``.

    Returns:
        Tuple of (threshold, number of alerts). The threshold is inclusive: a
        transaction is flagged when ``score >= threshold``.

    Raises:
        ValueError: If ``capacity`` is outside ``(0, 1]``.
    """
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")
    n = len(scores)
    k = max(int(np.ceil(capacity * n)), 1)
    order = np.argsort(-scores, kind="stable")
    threshold = float(scores[order[k - 1]])
    return threshold, k


def precision_recall_at_capacity(
    y_true: np.ndarray, scores: np.ndarray, capacity: float
) -> dict[str, float]:
    """Precision, recall and lift when only the top ``capacity`` fraction is reviewed.

    Ties are broken by taking exactly ``k`` transactions in score order, so the
    alert volume matches the stated budget even when many scores collide (which
    they do for tree models).

    Args:
        y_true: Binary labels.
        scores: Model scores.
        capacity: Review capacity as a fraction of the population.

    Returns:
        Dict with ``precision``, ``recall``, ``n_alerts``, ``n_caught``,
        ``threshold`` and ``lift`` (precision divided by the base rate).
    """
    n = len(y_true)
    _, k = top_k_threshold(scores, capacity)
    order = np.argsort(-scores, kind="stable")[:k]
    caught = float(y_true[order].sum())
    total_positive = float(y_true.sum())
    base_rate = total_positive / n if n else 0.0
    precision = caught / k if k else 0.0
    return {
        "precision": precision,
        "recall": caught / total_positive if total_positive else float("nan"),
        "n_alerts": float(k),
        "n_caught": caught,
        "threshold": float(scores[order[-1]]) if k else float("nan"),
        "lift": precision / base_rate if base_rate > 0 else float("nan"),
    }


def recall_by_fraud_type(
    y_true: np.ndarray,
    scores: np.ndarray,
    fraud_type: np.ndarray,
    capacity: float,
) -> dict[str, float]:
    """Per-mechanism recall at a fixed review capacity.

    Args:
        y_true: Binary labels.
        scores: Model scores.
        fraud_type: Mechanism label per row; ``"none"`` for legitimate rows.
        capacity: Review capacity as a fraction of the population.

    Returns:
        Dict mapping mechanism name to recall. Mechanisms absent from the data
        are omitted rather than reported as zero.
    """
    _, k = top_k_threshold(scores, capacity)
    flagged = np.zeros(len(y_true), dtype=bool)
    flagged[np.argsort(-scores, kind="stable")[:k]] = True

    out: dict[str, float] = {}
    for mechanism in FRAUD_TYPES:
        mask = fraud_type == mechanism
        total = int(mask.sum())
        if total == 0:
            continue
        out[mechanism] = float(flagged[mask].sum()) / total
    return out


def fraud_type_counts(fraud_type: np.ndarray) -> dict[str, int]:
    """Count positives of each mechanism present in the data."""
    return {
        mechanism: int((fraud_type == mechanism).sum())
        for mechanism in FRAUD_TYPES
        if (fraud_type == mechanism).sum() > 0
    }


@dataclass
class EvaluationResult:
    """Metrics for one model on one dataset slice.

    Attributes:
        name: Model identifier.
        split: Which slice this covers (``train``/``val``/``test``/...).
        pr_auc: Average precision.
        roc_auc: Area under the ROC curve.
        brier: Brier score of the raw scores, when they are probabilities.
        n: Number of rows scored.
        n_positive: Number of frauds present.
        base_rate: Fraud rate in the slice.
        accuracy: Accuracy at a 0.5 cutoff — reported only to be dismissed.
        accuracy_of_always_legit: Accuracy of predicting "never fraud".
        at_capacity: Per-capacity precision/recall/lift blocks.
        by_fraud_type: Per-mechanism recall, keyed by capacity.
        extra: Anything a caller wants to carry alongside (seed, timings, ...).
    """

    name: str
    split: str
    pr_auc: float
    roc_auc: float
    brier: float
    n: int
    n_positive: int
    base_rate: float
    accuracy: float
    accuracy_of_always_legit: float
    at_capacity: dict[str, dict[str, float]] = field(default_factory=dict)
    by_fraud_type: dict[str, dict[str, float]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Flatten to a single dict suitable for a DataFrame row."""
        row: dict[str, Any] = {
            "model": self.name,
            "split": self.split,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "brier": self.brier,
            "n": self.n,
            "n_positive": self.n_positive,
            "base_rate": self.base_rate,
        }
        for capacity, block in self.at_capacity.items():
            row[f"precision@{capacity}"] = block["precision"]
            row[f"recall@{capacity}"] = block["recall"]
            row[f"lift@{capacity}"] = block["lift"]
        for capacity, block in self.by_fraud_type.items():
            for mechanism, value in block.items():
                row[f"recall@{capacity}:{mechanism}"] = value
        row.update(self.extra)
        return row


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    name: str,
    split: str = "test",
    fraud_type: np.ndarray | None = None,
    capacities: tuple[float, ...] = REVIEW_CAPACITIES,
    extra: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Compute the full metric set for one model on one slice.

    Args:
        y_true: Binary labels.
        scores: Model scores; probabilities where available.
        name: Model identifier.
        split: Slice name.
        fraud_type: Optional per-row mechanism labels for the breakdown.
        capacities: Review capacities to report.
        extra: Extra fields to attach to the result.

    Returns:
        Populated :class:`EvaluationResult`.

    Raises:
        ValueError: If inputs have mismatched lengths or no positives.
    """
    y_true = np.asarray(y_true).astype(np.float64)
    scores = np.asarray(scores).astype(np.float64)
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if y_true.sum() == 0:
        raise ValueError("slice contains no positives; metrics would be undefined")

    at_capacity = {
        f"{c:g}": precision_recall_at_capacity(y_true, scores, c) for c in capacities
    }
    by_type: dict[str, dict[str, float]] = {}
    if fraud_type is not None:
        known = set(np.unique(fraud_type)) - {FraudType.NONE.value}
        if known and known != {FraudType.UNKNOWN.value}:
            by_type = {
                f"{c:g}": recall_by_fraud_type(y_true, scores, fraud_type, c)
                for c in capacities
            }

    # Only meaningful when ``scores`` are probabilities; harmless otherwise, and
    # the calibration module is where this is taken seriously.
    in_unit_interval = bool(np.all((scores >= 0) & (scores <= 1)))
    brier = float(np.mean((scores - y_true) ** 2)) if in_unit_interval else float("nan")
    predicted = (scores >= 0.5).astype(np.float64) if in_unit_interval else np.zeros_like(y_true)

    return EvaluationResult(
        name=name,
        split=split,
        pr_auc=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        brier=brier,
        n=int(len(y_true)),
        n_positive=int(y_true.sum()),
        base_rate=float(y_true.mean()),
        accuracy=float((predicted == y_true).mean()),
        accuracy_of_always_legit=float((y_true == 0).mean()),
        at_capacity=at_capacity,
        by_fraud_type=by_type,
        extra=extra or {},
    )


def results_frame(results: list[EvaluationResult]) -> pd.DataFrame:
    """Flatten a list of results into a DataFrame."""
    return pd.DataFrame([r.to_row() for r in results])


def aggregate_seeds(
    results: list[EvaluationResult], group_keys: tuple[str, ...] = ("model", "split")
) -> pd.DataFrame:
    """Mean and standard deviation of every metric across seeds.

    A single training run is not a result. Seed-to-seed spread on an imbalanced
    problem is routinely larger than the gap between two architectures, so a
    comparison without it cannot support any claim about which model is better.

    Args:
        results: Results from repeated runs, each carrying a ``seed`` in ``extra``.
        group_keys: Columns identifying one experimental arm.

    Returns:
        DataFrame with ``<metric>_mean``, ``<metric>_std`` and ``n_seeds``
        columns, one row per arm.
    """
    frame = results_frame(results)
    metric_columns = [
        c
        for c in frame.columns
        if c not in set(group_keys) | {"seed", "n", "n_positive"}
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    grouped = frame.groupby(list(group_keys), dropna=False)
    agg = grouped[metric_columns].agg(["mean", "std"])
    # ``agg`` returns a (metric, statistic) MultiIndex; flatten it to
    # ``pr_auc_mean`` / ``pr_auc_std`` style names.
    flattened: list[str] = []
    for column in agg.columns:
        parts = column if isinstance(column, tuple) else (column,)
        flattened.append("_".join(str(part) for part in parts))
    agg.columns = flattened
    agg["n_seeds"] = grouped.size()
    return agg.reset_index()


def comparison_table(
    results: list[EvaluationResult], capacity: float = 0.005
) -> pd.DataFrame:
    """Headline architecture comparison at one review capacity.

    Args:
        results: Evaluation results across models and seeds.
        capacity: Capacity to report precision and recall at.

    Returns:
        One row per model with mean +/- std across seeds.
    """
    key = f"{capacity:g}"
    agg = aggregate_seeds(results)
    columns = [
        "model",
        "split",
        "pr_auc_mean",
        "pr_auc_std",
        f"precision@{key}_mean",
        f"precision@{key}_std",
        f"recall@{key}_mean",
        f"recall@{key}_std",
        "n_seeds",
    ]
    present = [c for c in columns if c in agg.columns]
    return agg[present].sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    """Format a mean and standard deviation as ``0.1234 +/- 0.0056``."""
    if np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


ACCURACY_WARNING = (
    "Accuracy is not a usable metric on this problem: predicting 'never fraud' "
    "scores {always:.4%}, and any model worth deploying scores about the same. "
    "Read PR-AUC, precision at review capacity, and expected cost instead."
)


def log_accuracy_warning(result: EvaluationResult) -> None:
    """Log the accuracy comparison so it appears in every run's output."""
    LOGGER.info(ACCURACY_WARNING.format(always=result.accuracy_of_always_legit))
