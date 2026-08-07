"""Cost-sensitive decisioning: the comparison that actually decides anything.

PR-AUC ranks models. It does not tell you whether to deploy one, because it
assumes a false positive and a false negative are equally interesting and they
are not. Declining a legitimate £40 grocery payment costs a support call and some
goodwill. Letting a £900 fraud through costs £900 plus a chargeback fee. A model
that wins on PR-AUC can lose on expected cost if its errors sit on the expensive
side of that asymmetry.

So the headline number in this project is **expected cost per transaction**, with
both cost parameters exposed so a reader can substitute their own economics and
re-derive the ranking.

Two rules keep this honest:

* **The threshold is chosen on validation and applied unchanged to test.** A
  threshold tuned on the test set is a second fitting step and reports its own
  training error. The gap between the validation-chosen threshold's test cost and
  the best achievable test cost is reported so the size of that effect is visible.
* **Costs use the actual transaction amount.** A flat per-fraud cost would hide
  the thing that matters most, which is that the expensive frauds are large and a
  model can be excellent at catching cheap ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CostConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class DecisionOutcome:
    """Aggregate outcome of applying one threshold.

    Attributes:
        threshold: Score at or above which a transaction is blocked.
        n: Number of transactions.
        n_blocked: How many were blocked.
        alert_rate: Fraction blocked — the operational load.
        true_positives: Frauds blocked.
        false_positives: Legitimate transactions blocked.
        false_negatives: Frauds allowed.
        precision: Share of blocks that were fraud.
        recall: Share of fraud that was blocked.
        fraud_value_blocked: Currency value of blocked fraud.
        fraud_value_missed: Currency value of fraud allowed through.
        total_cost: Expected cost under the cost model.
        cost_per_transaction: ``total_cost / n``.
    """

    threshold: float
    n: int
    n_blocked: int
    alert_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    fraud_value_blocked: float
    fraud_value_missed: float
    total_cost: float
    cost_per_transaction: float

    def to_row(self) -> dict[str, float]:
        """Flatten to a DataFrame row."""
        return {
            "threshold": self.threshold,
            "alert_rate": self.alert_rate,
            "n_blocked": float(self.n_blocked),
            "true_positives": float(self.true_positives),
            "false_positives": float(self.false_positives),
            "false_negatives": float(self.false_negatives),
            "precision": self.precision,
            "recall": self.recall,
            "fraud_value_blocked": self.fraud_value_blocked,
            "fraud_value_missed": self.fraud_value_missed,
            "total_cost": self.total_cost,
            "cost_per_transaction": self.cost_per_transaction,
        }


def evaluate_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    cost: CostConfig,
) -> DecisionOutcome:
    """Apply one threshold and cost the resulting decisions.

    Args:
        y_true: Binary labels.
        scores: Model scores.
        amounts: Transaction amounts, used to value missed fraud.
        threshold: Block when ``score >= threshold``.
        cost: Cost parameters.

    Returns:
        Populated :class:`DecisionOutcome`.
    """
    blocked = scores >= threshold
    fraud = y_true > 0.5

    tp = blocked & fraud
    fp = blocked & ~fraud
    fn = ~blocked & fraud

    missed_value = float(amounts[fn].sum())
    blocked_value = float(amounts[tp].sum())

    total = (
        cost.false_positive_cost * float(fp.sum())
        + cost.false_negative_cost_fixed * float(fn.sum())
        + cost.false_negative_amount_multiplier * missed_value
        + cost.review_cost * float(tp.sum())
        # Blocking fraud recovers the amount, but not always all of it: a partly
        # settled or already-shipped order still loses something. At the default
        # recovery of 1.0 this term is exactly zero, so it changes no reported
        # number — it exists so the documented knob actually does something when
        # a reader turns it down.
        + (1.0 - cost.true_positive_recovery) * blocked_value
    )
    n = int(len(y_true))
    n_blocked = int(blocked.sum())
    return DecisionOutcome(
        threshold=float(threshold),
        n=n,
        n_blocked=n_blocked,
        alert_rate=n_blocked / n if n else 0.0,
        true_positives=int(tp.sum()),
        false_positives=int(fp.sum()),
        false_negatives=int(fn.sum()),
        precision=float(tp.sum()) / n_blocked if n_blocked else 0.0,
        recall=float(tp.sum()) / float(fraud.sum()) if fraud.any() else float("nan"),
        fraud_value_blocked=blocked_value,
        fraud_value_missed=missed_value,
        total_cost=total,
        cost_per_transaction=total / n if n else 0.0,
    )


def sweep_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    cost: CostConfig,
    n_points: int = 400,
) -> pd.DataFrame:
    """Cost the full range of thresholds.

    Candidate thresholds are score quantiles rather than a uniform grid, because
    fraud scores pile up near zero and a uniform grid spends every point in a
    region where no decision changes.

    Args:
        y_true: Binary labels.
        scores: Model scores.
        amounts: Transaction amounts.
        cost: Cost parameters.
        n_points: Number of thresholds to evaluate.

    Returns:
        DataFrame of :class:`DecisionOutcome` rows, ascending in threshold.
    """
    quantiles = np.unique(
        np.quantile(scores, np.linspace(0.0, 1.0, n_points), method="linear")
    )
    candidates = np.unique(np.concatenate([quantiles, [scores.max() + 1e-9]]))
    rows = [
        evaluate_threshold(y_true, scores, amounts, float(t), cost).to_row() for t in candidates
    ]
    return pd.DataFrame(rows)


def choose_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    cost: CostConfig,
    *,
    max_alert_rate: float | None = None,
    n_points: int = 400,
) -> tuple[float, pd.DataFrame]:
    """Pick the cost-minimising threshold, optionally under a capacity cap.

    Args:
        y_true: Labels of the selection set — validation, never test.
        scores: Scores on that set.
        amounts: Amounts on that set.
        cost: Cost parameters.
        max_alert_rate: If set, only thresholds whose alert rate is at or below
            this are eligible. Use it when the review queue is a hard constraint
            rather than a cost.
        n_points: Threshold grid resolution.

    Returns:
        Tuple of (chosen threshold, the full sweep).
    """
    sweep = sweep_thresholds(y_true, scores, amounts, cost, n_points=n_points)
    eligible = sweep
    if max_alert_rate is not None:
        eligible = sweep[sweep["alert_rate"] <= max_alert_rate]
        if eligible.empty:
            eligible = sweep.nsmallest(1, "alert_rate")
    best = eligible.loc[eligible["total_cost"].idxmin()]
    return float(np.asarray(best["threshold"]).item()), sweep


def baseline_costs(
    y_true: np.ndarray, amounts: np.ndarray, cost: CostConfig
) -> dict[str, float]:
    """Cost of the two trivial policies, as reference points.

    Every model has to beat "block nothing" to be worth running at all, and the
    gap to it is the value the model creates. "Block everything" is included
    because it bounds the other end and is occasionally, embarrassingly, the
    cheaper of the two under aggressive cost assumptions.

    Args:
        y_true: Binary labels.
        amounts: Transaction amounts.
        cost: Cost parameters.

    Returns:
        Dict of cost per transaction for each trivial policy.
    """
    fraud = y_true > 0.5
    n = len(y_true)
    block_nothing = (
        cost.false_negative_cost_fixed * float(fraud.sum())
        + cost.false_negative_amount_multiplier * float(amounts[fraud].sum())
    ) / n
    block_everything = (
        cost.false_positive_cost * float((~fraud).sum()) + cost.review_cost * float(fraud.sum())
    ) / n
    return {
        "block_nothing": block_nothing,
        "block_everything": block_everything,
    }


@dataclass
class PolicyResult:
    """Cost outcome for one model, threshold chosen on validation.

    Attributes:
        model: Model name.
        threshold: Threshold chosen on the validation split.
        val: Outcome on validation at that threshold.
        test: Outcome on test at that threshold.
        best_possible_test_cost: Lowest test cost achievable by any threshold.
        selection_penalty: How much choosing on validation costs versus the
            unattainable oracle threshold, per transaction.
        savings_vs_block_nothing: Cost per transaction saved against doing nothing.
        seed: Seed of the run this outcome came from, so costs can be aggregated
            across seeds the same way every other metric is.
    """

    model: str
    threshold: float
    val: DecisionOutcome
    test: DecisionOutcome
    best_possible_test_cost: float
    selection_penalty: float
    savings_vs_block_nothing: float
    seed: int = 0

    def to_row(self) -> dict[str, float | str]:
        """Flatten to a DataFrame row."""
        return {
            "model": self.model,
            "seed": self.seed,
            "threshold": self.threshold,
            "test_cost_per_txn": self.test.cost_per_transaction,
            "test_alert_rate": self.test.alert_rate,
            "test_precision": self.test.precision,
            "test_recall": self.test.recall,
            "test_fraud_value_missed": self.test.fraud_value_missed,
            "oracle_cost_per_txn": self.best_possible_test_cost,
            "selection_penalty": self.selection_penalty,
            "savings_vs_block_nothing": self.savings_vs_block_nothing,
        }


def apply_policy(
    model: str,
    y_val: np.ndarray,
    val_scores: np.ndarray,
    val_amounts: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    test_amounts: np.ndarray,
    cost: CostConfig,
    *,
    max_alert_rate: float | None = None,
) -> PolicyResult:
    """Choose a threshold on validation and report its cost on test.

    Args:
        model: Model name.
        y_val: Validation labels.
        val_scores: Validation scores.
        val_amounts: Validation amounts.
        y_test: Test labels.
        test_scores: Test scores.
        test_amounts: Test amounts.
        cost: Cost parameters.
        max_alert_rate: Optional hard capacity cap.

    Returns:
        Populated :class:`PolicyResult`.
    """
    threshold, _ = choose_threshold(
        y_val, val_scores, val_amounts, cost, max_alert_rate=max_alert_rate
    )
    val_outcome = evaluate_threshold(y_val, val_scores, val_amounts, threshold, cost)
    test_outcome = evaluate_threshold(y_test, test_scores, test_amounts, threshold, cost)

    test_sweep = sweep_thresholds(y_test, test_scores, test_amounts, cost)
    oracle = float(test_sweep["cost_per_transaction"].min())
    reference = baseline_costs(y_test, test_amounts, cost)["block_nothing"]

    return PolicyResult(
        model=model,
        threshold=threshold,
        val=val_outcome,
        test=test_outcome,
        best_possible_test_cost=oracle,
        selection_penalty=test_outcome.cost_per_transaction - oracle,
        savings_vs_block_nothing=reference - test_outcome.cost_per_transaction,
    )


def aggregate_policies(results: list[PolicyResult]) -> pd.DataFrame:
    """Mean and standard deviation of every cost metric across seeds.

    The cost table is the headline of the whole project, so it has to be
    aggregated the same way the accuracy tables are. Costing a seed-*averaged*
    score instead would quietly report an ensemble result — better than any
    single model, and not what the deployed artefact does.

    Args:
        results: One result per (model, seed).

    Returns:
        One row per model, cheapest first, with ``_mean``/``_std`` columns.
    """
    frame = pd.DataFrame([r.to_row() for r in results])
    metrics = [
        c for c in frame.columns if c not in {"model", "seed"} and pd.api.types.is_numeric_dtype(frame[c])
    ]
    grouped = frame.groupby("model")
    agg = grouped[metrics].agg(["mean", "std"])
    flattened: list[str] = []
    for column in agg.columns:
        parts = column if isinstance(column, tuple) else (column,)
        flattened.append("_".join(str(part) for part in parts))
    agg.columns = flattened
    agg["n_seeds"] = grouped.size()
    return agg.reset_index().sort_values("test_cost_per_txn_mean").reset_index(drop=True)


def compare_policies(results: list[PolicyResult]) -> pd.DataFrame:
    """Rank models by test cost per transaction.

    Args:
        results: One result per model.

    Returns:
        DataFrame sorted by cost, cheapest first.
    """
    return (
        pd.DataFrame([r.to_row() for r in results])
        .sort_values("test_cost_per_txn")
        .reset_index(drop=True)
    )


def plot_cost_frontier(
    sweeps: dict[str, pd.DataFrame], path: Path, block_nothing: float | None = None
) -> Path:
    """Plot expected cost against alert rate for each model.

    Args:
        sweeps: Threshold sweeps keyed by model name.
        path: Output image path.
        block_nothing: Cost per transaction of blocking nothing, drawn as a
            reference line.

    Returns:
        The written path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for name, sweep in sweeps.items():
        ordered = sweep.sort_values("alert_rate")
        ax.plot(ordered["alert_rate"], ordered["cost_per_transaction"], linewidth=1.4, label=name)
    if block_nothing is not None:
        ax.axhline(block_nothing, color="k", linestyle="--", linewidth=1, label="block nothing")
    ax.set_xscale("log")
    ax.set_xlabel("alert rate (fraction of transactions blocked)")
    ax.set_ylabel("expected cost per transaction")
    ax.set_title("Cost frontier")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def sensitivity(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    val_amounts: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    test_amounts: np.ndarray,
    *,
    model: str,
    fp_costs: tuple[float, ...] = (2.0, 5.0, 8.0, 15.0, 30.0),
) -> pd.DataFrame:
    """How the chosen operating point moves as the false-positive cost changes.

    The cost of declining a good customer is the parameter nobody can measure
    precisely, so the right response is to show how much the answer depends on
    it rather than to pick a number and hope.

    Args:
        y_val: Validation labels.
        val_scores: Validation scores.
        val_amounts: Validation amounts.
        y_test: Test labels.
        test_scores: Test scores.
        test_amounts: Test amounts.
        model: Model name.
        fp_costs: False-positive costs to sweep.

    Returns:
        One row per assumed false-positive cost.
    """
    rows = []
    for fp_cost in fp_costs:
        cost = CostConfig(false_positive_cost=fp_cost)
        result = apply_policy(
            model,
            y_val,
            val_scores,
            val_amounts,
            y_test,
            test_scores,
            test_amounts,
            cost,
        )
        row = result.to_row()
        row["false_positive_cost"] = fp_cost
        rows.append(row)
    return pd.DataFrame(rows)
