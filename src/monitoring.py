"""Drift monitoring, and why fraud drift is not ordinary drift.

Ordinary covariate drift is incidental. Customers gradually change what they buy,
a new merchant grows, a country is onboarded; the data moves and the model decays
because the world moved on without reference to the model.

Fraud drift is adversarial and *targeted*. The attacker is optimising against
the deployed model. When card testing stops looking like card testing, it is
because the velocity rule worked and the attacker changed to defeat it. Three
consequences follow, and they drive every design choice in this module:

1. **Degradation concentrates in the segment you were best at.** Aggregate PR-AUC
   can look almost unchanged while recall on one mechanism collapses, because the
   adversary attacks the specific pattern the model relies on. So this module
   tracks per-window recall by mechanism where the labels allow it, not just an
   overall number.

2. **Degradation is abrupt, not gradual.** A new technique appears when it
   appears. Smoothing over a long window to reduce false alarms delays detection
   of exactly the event worth detecting.

3. **The labels arrive too late to be the alarm.** A chargeback is confirmed
   weeks after the authorisation, so precision measured on confirmed labels is a
   *lagging* indicator — it tells you about a loss you have already taken. The
   score distribution and the input features are available immediately, so the
   leading signal has to be built from those.

A finding worth stating plainly, because it contradicts the standard playbook:
**PSI over the full score distribution is close to useless on this problem.**
More than 99% of transactions are legitimate and the adversary does not change
their behaviour at all, so the bulk of the distribution sits still while the
fraud tail is rewritten underneath it. In the runs here the injected adversarial
drift moves score PSI by about 0.01 — a tenth of the conventional "warning"
threshold — while precision at fixed capacity falls by more than half. A monitor
built on score PSI alone would have missed it entirely.

What does work, and is equally label-free, is watching the region the attacker is
actually operating in: the **alert rate at the fixed production threshold**. When
fraud learns to score low, the alert rate at an unchanged threshold falls, and it
falls immediately. Both signals are computed and reported; the alert-rate ratio
is the one that fires.

The synthetic data has adversarial drift injected in its final period, so the
monitor can be tested rather than merely asserted: see
:func:`monitor_over_time` and ``tests/test_monitoring.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.config import MonitoringConfig
from src.data.schema import FRAUD_TYPES

LOGGER = logging.getLogger(__name__)

Severity = Literal["ok", "warn", "alert"]

_EPS = 1e-6


def psi_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile bin edges derived from a reference sample.

    Quantile rather than equal-width bins, because fraud scores are massed near
    zero and equal-width bins would put everything in one bucket and report a PSI
    of approximately nothing no matter what happened.

    Args:
        reference: Reference sample.
        n_bins: Requested number of bins.

    Returns:
        Monotone edge array with open ends.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        edges = np.array([reference.min() - _EPS, reference.max() + _EPS])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10, edges: np.ndarray | None = None
) -> float:
    """Population stability index between a reference and a current sample.

    ``PSI = sum((c_i - r_i) * ln(c_i / r_i))`` over bins. The conventional
    reading is < 0.1 stable, 0.1-0.25 warrants a look, > 0.25 significant shift.
    Those cut-offs are industry convention rather than anything principled, which
    is why they live in :class:`~src.config.MonitoringConfig` where they can be
    changed.

    Args:
        reference: Reference sample.
        current: Current sample.
        n_bins: Bin count when ``edges`` is not supplied.
        edges: Precomputed bin edges; pass these to keep binning fixed across
            windows, otherwise each window is scored against its own bins and the
            comparison is meaningless.

    Returns:
        The PSI. Zero when either sample is empty.
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    if edges is None:
        edges = psi_edges(reference, n_bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_frac = np.maximum(ref_counts / max(ref_counts.sum(), 1), _EPS)
    cur_frac = np.maximum(cur_counts / max(cur_counts.sum(), 1), _EPS)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def severity_of(psi: float, config: MonitoringConfig) -> Severity:
    """Classify a PSI value against the configured thresholds.

    Args:
        psi: The PSI value.
        config: Monitoring thresholds.

    Returns:
        ``"ok"``, ``"warn"`` or ``"alert"``.
    """
    if psi >= config.psi_alert:
        return "alert"
    if psi >= config.psi_warn:
        return "warn"
    return "ok"


def severity_of_alert_rate(ratio: float, config: MonitoringConfig) -> Severity:
    """Classify an alert-rate ratio against the configured thresholds.

    A ratio below 1 means the model is flagging less than it used to at the same
    threshold — the signature of an adversary that has learned to score low. A
    ratio above 1 means it is flagging more, which is an operational problem in
    its own right because the review queue has a fixed size. Both directions are
    treated symmetrically.

    Args:
        ratio: Window alert rate divided by the reference alert rate.
        config: Monitoring thresholds.

    Returns:
        ``"ok"``, ``"warn"`` or ``"alert"``.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        return "alert"
    deviation = min(ratio, 1.0 / ratio)
    if deviation <= config.alert_rate_ratio_alert:
        return "alert"
    if deviation <= config.alert_rate_ratio_warn:
        return "warn"
    return "ok"


def worst_severity(*severities: Severity) -> Severity:
    """Return the most serious of several severities."""
    order = {"ok": 0, "warn": 1, "alert": 2}
    return max(severities, key=lambda s: order[s])


@dataclass
class WindowReport:
    """Monitoring signals for a single time window.

    Attributes:
        window: Window index.
        start: First timestamp in the window.
        end: Last timestamp in the window.
        n: Transactions scored.
        n_fraud: Confirmed frauds (lagging; unavailable in real time).
        score_psi: PSI of the score distribution against the reference.
        score_severity: Severity classification of ``score_psi``.
        mean_score: Mean score in the window.
        mean_score_ratio: Mean score relative to the reference period.
        alert_rate_ratio: Alert rate relative to the reference period. This is
            the leading signal that actually works on this problem — see the
            module docstring.
        alert_rate_severity: Severity classification of ``alert_rate_ratio``.
        severity: Worst of the two label-free severities.
        alert_rate: Fraction flagged at the fixed production threshold.
        precision_at_capacity: Precision within the window's own top-capacity
            slice.
        recall_at_capacity: Recall within that slice.
        precision_at_threshold: Precision at the fixed production threshold.
        recall_by_type: Per-mechanism recall at capacity, where known.
        max_feature_psi: Largest PSI across monitored features.
        worst_feature: Which feature that was.
    """

    window: int
    start: pd.Timestamp
    end: pd.Timestamp
    n: int
    n_fraud: int
    score_psi: float
    score_severity: Severity
    mean_score: float
    mean_score_ratio: float
    alert_rate_ratio: float
    alert_rate_severity: Severity
    severity: Severity
    alert_rate: float
    precision_at_capacity: float
    recall_at_capacity: float
    precision_at_threshold: float
    recall_by_type: dict[str, float]
    max_feature_psi: float = 0.0
    worst_feature: str = ""

    def to_row(self) -> dict[str, object]:
        """Flatten to a DataFrame row."""
        row: dict[str, object] = {
            "window": self.window,
            "start": self.start,
            "end": self.end,
            "n": self.n,
            "n_fraud": self.n_fraud,
            "score_psi": self.score_psi,
            "score_severity": self.score_severity,
            "mean_score": self.mean_score,
            "mean_score_ratio": self.mean_score_ratio,
            "alert_rate_ratio": self.alert_rate_ratio,
            "alert_rate_severity": self.alert_rate_severity,
            "severity": self.severity,
            "alert_rate": self.alert_rate,
            "precision_at_capacity": self.precision_at_capacity,
            "recall_at_capacity": self.recall_at_capacity,
            "precision_at_threshold": self.precision_at_threshold,
            "max_feature_psi": self.max_feature_psi,
            "worst_feature": self.worst_feature,
        }
        for mechanism, value in self.recall_by_type.items():
            row[f"recall:{mechanism}"] = value
        return row


def _precision_recall_at_capacity(
    y: np.ndarray, scores: np.ndarray, capacity: float
) -> tuple[float, float, np.ndarray]:
    """Precision, recall and the flag mask for a window's own top slice."""
    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), np.zeros(0, dtype=bool)
    k = max(int(np.ceil(capacity * n)), 1)
    order = np.argsort(-scores, kind="stable")[:k]
    flagged = np.zeros(n, dtype=bool)
    flagged[order] = True
    caught = float(y[flagged].sum())
    total = float(y.sum())
    return caught / k, (caught / total if total else float("nan")), flagged


def monitor_over_time(
    scores: np.ndarray,
    y_true: np.ndarray,
    timestamps: np.ndarray,
    *,
    reference_scores: np.ndarray,
    fraud_type: np.ndarray | None = None,
    features: pd.DataFrame | None = None,
    reference_features: pd.DataFrame | None = None,
    production_threshold: float | None = None,
    config: MonitoringConfig | None = None,
) -> pd.DataFrame:
    """Run the monitor over consecutive time windows.

    Args:
        scores: Model scores for the monitored period.
        y_true: Labels for that period. Real-time monitoring does not have these;
            they are used here for the lagging precision/recall signals.
        timestamps: Timestamps for that period.
        reference_scores: Scores from the reference period (validation), which
            fix the PSI bins.
        fraud_type: Optional mechanism labels for the per-mechanism breakdown.
        features: Optional feature frame for the monitored period.
        reference_features: Optional feature frame for the reference period.
        production_threshold: Fixed threshold the deployed policy uses. Alert
            rate and precision at this threshold are what an operations team
            actually watches.
        config: Monitoring thresholds and window width.

    Returns:
        One row per window, in time order.
    """
    cfg = config or MonitoringConfig()
    ts = pd.to_datetime(pd.Series(timestamps))
    origin = ts.min()
    window_index = ((ts - origin).dt.total_seconds() / (cfg.window_days * 86400.0)).astype(int)

    edges = psi_edges(reference_scores, cfg.psi_bins)
    reference_mean = float(np.mean(reference_scores)) if len(reference_scores) else float("nan")
    reference_alert_rate = (
        float(np.mean(reference_scores >= production_threshold))
        if production_threshold is not None and len(reference_scores)
        else float("nan")
    )
    feature_edges: dict[str, np.ndarray] = {}
    if features is not None and reference_features is not None:
        for column in features.columns:
            feature_edges[column] = psi_edges(
                reference_features[column].to_numpy(), cfg.psi_bins
            )

    reports: list[WindowReport] = []
    for w in sorted(window_index.unique()):
        mask = (window_index == w).to_numpy()
        if mask.sum() < 50:
            continue
        w_scores, w_y = scores[mask], y_true[mask]
        precision, recall, flagged = _precision_recall_at_capacity(w_y, w_scores, cfg.capacity)

        by_type: dict[str, float] = {}
        if fraud_type is not None:
            w_types = fraud_type[mask]
            for mechanism in FRAUD_TYPES:
                type_mask = w_types == mechanism
                if type_mask.sum() == 0:
                    continue
                by_type[mechanism] = float(flagged[type_mask].sum()) / int(type_mask.sum())

        if production_threshold is not None:
            at_threshold = w_scores >= production_threshold
            alert_rate = float(at_threshold.mean())
            precision_at_threshold = (
                float(w_y[at_threshold].sum()) / int(at_threshold.sum())
                if at_threshold.any()
                else float("nan")
            )
        else:
            alert_rate = float("nan")
            precision_at_threshold = float("nan")

        max_feature_psi, worst_feature = 0.0, ""
        if features is not None and reference_features is not None:
            for column, column_edges in feature_edges.items():
                value = population_stability_index(
                    reference_features[column].to_numpy(),
                    features.loc[mask, column].to_numpy(),
                    edges=column_edges,
                )
                if value > max_feature_psi:
                    max_feature_psi, worst_feature = value, column

        score_psi = population_stability_index(reference_scores, w_scores, edges=edges)
        mean_score = float(w_scores.mean())
        mean_ratio = mean_score / reference_mean if reference_mean else float("nan")
        rate_ratio = (
            alert_rate / reference_alert_rate
            if np.isfinite(reference_alert_rate) and reference_alert_rate > 0
            else float("nan")
        )
        psi_severity = severity_of(score_psi, cfg)
        rate_severity = (
            severity_of_alert_rate(rate_ratio, cfg) if np.isfinite(rate_ratio) else "ok"
        )
        reports.append(
            WindowReport(
                window=int(w),
                start=ts[mask].min(),
                end=ts[mask].max(),
                n=int(mask.sum()),
                n_fraud=int(w_y.sum()),
                score_psi=score_psi,
                score_severity=psi_severity,
                mean_score=mean_score,
                mean_score_ratio=mean_ratio,
                alert_rate_ratio=rate_ratio,
                alert_rate_severity=rate_severity,
                severity=worst_severity(psi_severity, rate_severity),
                alert_rate=alert_rate,
                precision_at_capacity=precision,
                recall_at_capacity=recall,
                precision_at_threshold=precision_at_threshold,
                recall_by_type=by_type,
                max_feature_psi=max_feature_psi,
                worst_feature=worst_feature,
            )
        )
    return pd.DataFrame([r.to_row() for r in reports])


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, config: MonitoringConfig | None = None
) -> pd.DataFrame:
    """PSI for every column, most-drifted first.

    Args:
        reference: Reference-period features.
        current: Current-period features.
        config: Monitoring thresholds.

    Returns:
        DataFrame of feature, psi and severity.
    """
    cfg = config or MonitoringConfig()
    rows = []
    for column in reference.columns:
        if column not in current.columns:
            continue
        value = population_stability_index(
            reference[column].to_numpy(), current[column].to_numpy(), cfg.psi_bins
        )
        rows.append({"feature": column, "psi": value, "severity": severity_of(value, cfg)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def first_alert(report: pd.DataFrame, column: str = "score_psi", threshold: float = 0.25) -> dict[str, object] | None:
    """First window whose value crosses a threshold.

    Args:
        report: Output of :func:`monitor_over_time`.
        column: Column to test.
        threshold: Crossing value.

    Returns:
        The first breaching row as a dict, or ``None`` if the monitor never fired.
    """
    breaches = report[report[column] >= threshold]
    if breaches.empty:
        return None
    return dict(breaches.iloc[0])


def _first_where(report: pd.DataFrame, column: str, values: set[str]) -> dict[str, object] | None:
    """First row whose ``column`` is one of ``values``."""
    if column not in report.columns:
        return None
    hits = report[report[column].isin(values)]
    if hits.empty:
        return None
    return dict(hits.iloc[0])


def detection_summary(
    report: pd.DataFrame, drift_start: pd.Timestamp, config: MonitoringConfig | None = None
) -> dict[str, object]:
    """Did the monitor fire, on which signal, and how long after the drift began?

    Both label-free signals are reported separately, because on this problem they
    do not behave the same way and the difference is the point. PSI over the full
    score distribution is nearly blind here: 99%+ of transactions are legitimate
    and unaffected by the adversary, so the bulk of the distribution barely moves
    even as the fraud tail changes completely. The alert rate at the fixed
    production threshold looks only at the region the attacker is actually
    operating in, and moves sharply.

    Args:
        report: Output of :func:`monitor_over_time`.
        drift_start: When the adversarial change actually began.
        config: Monitoring thresholds.

    Returns:
        Dict describing the first firing of each signal and the detection delay.
    """
    cfg = config or MonitoringConfig()
    signals: dict[str, dict[str, object] | None] = {
        "psi_warn": first_alert(report, "score_psi", cfg.psi_warn),
        "psi_alert": first_alert(report, "score_psi", cfg.psi_alert),
        "alert_rate_warn": _first_where(report, "alert_rate_severity", {"warn", "alert"}),
        "alert_rate_alert": _first_where(report, "alert_rate_severity", {"alert"}),
        "any_warn": _first_where(report, "severity", {"warn", "alert"}),
        "any_alert": _first_where(report, "severity", {"alert"}),
    }
    out: dict[str, object] = {"drift_start": drift_start}
    for label, row in signals.items():
        out[f"fired_{label}"] = row is not None
        if row is None:
            continue
        detected_at = pd.Timestamp(str(row["start"]))
        out[f"{label}_window_start"] = detected_at
        out[f"{label}_delay_days"] = (detected_at - drift_start).total_seconds() / 86400.0
    if signals["any_warn"] is not None:
        row = signals["any_warn"]
        out["first_warning_psi"] = float(np.asarray(row["score_psi"], dtype=np.float64).item())
        if "alert_rate_ratio" in row:
            out["first_warning_alert_rate_ratio"] = float(
                np.asarray(row["alert_rate_ratio"], dtype=np.float64).item()
            )
    return out


def plot_monitoring(report: pd.DataFrame, path: Path, drift_start: pd.Timestamp | None = None) -> Path:
    """Plot PSI and rolling precision over time.

    Args:
        report: Output of :func:`monitor_over_time`.
        path: Output image path.
        drift_start: Optional vertical marker for when drift actually began.

    Returns:
        The written path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True)
    x = pd.to_datetime(report["start"])

    ax1.plot(x, report["score_psi"], marker="o", markersize=4, label="score PSI")
    if "max_feature_psi" in report.columns and report["max_feature_psi"].any():
        ax1.plot(
            x, report["max_feature_psi"], marker="s", markersize=3, alpha=0.7,
            label="worst feature PSI",
        )
    ax1.axhline(0.10, color="orange", linestyle="--", linewidth=1, label="warn (0.10)")
    ax1.axhline(0.25, color="red", linestyle="--", linewidth=1, label="alert (0.25)")
    ax1.set_ylabel("PSI")
    ax1.set_title("Leading signal: distribution drift (no labels required)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(x, report["precision_at_capacity"], marker="o", markersize=4, label="precision @ capacity")
    if "recall_at_capacity" in report.columns:
        ax2.plot(x, report["recall_at_capacity"], marker="s", markersize=3, alpha=0.7, label="recall @ capacity")
    ax2.set_ylabel("rate")
    ax2.set_xlabel("window start")
    ax2.set_title("Lagging signal: needs confirmed labels, arrives weeks later")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    if drift_start is not None:
        for ax in (ax1, ax2):
            ax.axvline(drift_start, color="purple", linestyle=":", linewidth=1.5)
        ax1.text(drift_start, ax1.get_ylim()[1] * 0.95, " adversarial drift begins", fontsize=8, color="purple")

    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
