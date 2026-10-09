"""Central configuration: paths, split boundaries, cost model and training defaults.

Every knob that materially changes a reported number lives here so the write-up
and the code cannot drift apart. Values chosen for CPU-trainability are marked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"
AUDIT_DIR = REPO_ROOT / "audit"

#: Seeds used for every multi-seed report. Three is the stated minimum; a single
#: run is not a result.
SEEDS: tuple[int, ...] = (0, 1, 2)

#: Review capacities a real fraud team can actually work: fraction of the
#: scored population an analyst queue can absorb per day.
REVIEW_CAPACITIES: tuple[float, ...] = (0.001, 0.005, 0.01)


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters of the synthetic transaction generator.

    Attributes:
        n_customers: Number of distinct cardholders.
        n_days: Length of the simulated period in days.
        start_date: ISO date of the first simulated day.
        n_merchants: Size of the merchant universe (power-law popularity).
        base_fraud_rate: Approximate share of transactions that are fraudulent.
        card_testing_share: Share of fraud coming from card-testing bursts.
        ato_share: Share of fraud coming from account takeover.
        merchant_compromise_share: Share of fraud from compromised merchants.
        drift_start_frac: Fraction of the timeline after which adversarial
            drift begins. Fraud mechanics change from here on.
        unreported_fraud_rate: Share of fraudulent transactions the customer
            never disputes, so they carry a legitimate label. Every real fraud
            label set has these; they put a ceiling on achievable recall and
            stop the synthetic problem from being cleanly separable.
        seed: RNG seed.
    """

    n_customers: int = 2500
    n_days: int = 180
    start_date: str = "2023-01-01"
    n_merchants: int = 3000
    base_fraud_rate: float = 0.0065
    card_testing_share: float = 0.40
    ato_share: float = 0.35
    merchant_compromise_share: float = 0.25
    drift_start_frac: float = 0.85
    unreported_fraud_rate: float = 0.04
    seed: int = 7


@dataclass(frozen=True)
class SplitConfig:
    """Temporal split boundaries, expressed as fractions of the timeline.

    The split is always by time, never random: a random split lets the model see
    a customer's future and is the single most common way fraud papers report
    numbers they cannot reproduce in production.

    Attributes:
        train_frac: Fraction of the timeline used for training.
        val_frac: Fraction used for validation (early stopping, calibration fit,
            threshold selection).
        embargo_days: Gap inserted between splits so that a transaction near a
            boundary cannot use history that straddles it.
    """

    train_frac: float = 0.60
    val_frac: float = 0.15
    embargo_days: float = 1.0

    @property
    def test_frac(self) -> float:
        """Fraction of the timeline reserved for test."""
        return 1.0 - self.train_frac - self.val_frac


@dataclass(frozen=True)
class SequenceConfig:
    """Windowed sequence construction.

    Attributes:
        max_len: Number of transactions per window, including the transaction
            being scored (which occupies the final position). Rows with less
            history than this are still scored -- they are padded -- because a
            real system has to score a customer's second transaction too.
    """

    max_len: int = 32


@dataclass(frozen=True)
class TrainConfig:
    """Training defaults. Sized so every model trains on a laptop CPU.

    Attributes:
        batch_size: Minibatch size.
        max_epochs: Upper bound on epochs; early stopping usually fires first.
        patience: Early-stopping patience in epochs, on validation PR-AUC.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        grad_clip: Global gradient-norm clip.
        focal_gamma: Focusing parameter for focal loss.
        focal_alpha: Positive-class weight inside focal loss.
        pos_sample_weight: Oversampling multiplier applied to positives under
            the ``weighted`` strategy.
    """

    batch_size: int = 512
    max_epochs: int = 12
    patience: int = 3
    lr: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    pos_sample_weight: float = 40.0


@dataclass(frozen=True)
class CostConfig:
    """Cost model for the decisioning policy.

    A blocked legitimate transaction and a missed fraud are not symmetric, and
    neither is a constant: the cost of a false positive is dominated by customer
    friction/churn, while a false negative costs the transaction value plus
    handling. Both are parameterised so a reader can substitute their own.

    Attributes:
        false_positive_cost: Cost of declining a legitimate transaction
            (friction, call-centre handling, churn hazard), in currency units.
        false_negative_cost_fixed: Fixed handling cost of a missed fraud
            (chargeback fee, investigation) on top of the amount lost.
        false_negative_amount_multiplier: Share of the transaction amount lost
            when fraud is not blocked. 1.0 means the full amount is written off.
        review_cost: Marginal cost of a manual review, if a review tier is used.
        true_positive_recovery: Share of the amount recovered by blocking.
    """

    false_positive_cost: float = 8.0
    false_negative_cost_fixed: float = 15.0
    false_negative_amount_multiplier: float = 1.0
    review_cost: float = 2.0
    true_positive_recovery: float = 1.0


@dataclass(frozen=True)
class ServingConfig:
    """Serving budget and runtime options.

    Attributes:
        latency_budget_p99_ms: Hard budget for end-to-end scoring at p99. Card
            authorisation gives the issuer a low-hundreds-of-ms window in total;
            the model is one hop inside it, so we claim 50 ms.
        model_version: Version string reported by ``/model-info`` and written to
            every audit record.
        onnx_intra_op_threads: ONNX Runtime intra-op thread count. 1 is right
            for per-request scoring: batch size 1 gets no benefit from threads
            and thread hand-off adds tail latency.
    """

    latency_budget_p99_ms: float = 50.0
    model_version: str = "0.1.0"
    onnx_intra_op_threads: int = 1


@dataclass(frozen=True)
class MonitoringConfig:
    """Drift monitoring thresholds.

    Attributes:
        psi_bins: Number of quantile bins for the population stability index.
        psi_warn: PSI above this is a warning (conventional 0.1).
        psi_alert: PSI above this is an alert (conventional 0.25).
        alert_rate_ratio_warn: Warn when the alert rate at the fixed production
            threshold falls below this multiple of its reference value (or rises
            above its reciprocal).
        alert_rate_ratio_alert: Alert at this multiple.
        window_days: Width of each monitoring window in days.
        capacity: Review capacity at which rolling precision is tracked.
    """

    psi_bins: int = 10
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    alert_rate_ratio_warn: float = 0.70
    alert_rate_ratio_alert: float = 0.50
    window_days: float = 7.0
    capacity: float = 0.005


def ensure_dirs() -> None:
    """Create the directories the pipeline writes into."""
    for path in (RAW_DIR, SYNTHETIC_DIR, ARTIFACT_DIR, RESULTS_DIR, AUDIT_DIR):
        path.mkdir(parents=True, exist_ok=True)
