"""Calibration: are the scores probabilities, or just rankings?

Ranking is not enough here. The decision policy in :mod:`src.policy` compares an
expected loss against an expected cost of friction, and that arithmetic is only
valid if ``p = 0.02`` genuinely means "two in a hundred of these are fraud". A
model can rank perfectly and still be systematically overconfident, at which
point every threshold derived from its output is wrong.

Two mechanisms in this project are expected to damage calibration, and both are
measured rather than assumed:

* **Neural networks are typically overconfident.** Typically — not always, and
  not by a fixed amount. The direction and size are reported per model.
* **Resampling changes the base rate the model is fitting.** A model trained on a
  stream where positives were oversampled 40x is estimating
  ``P(fraud | x, resampled world)``, which is much larger than
  ``P(fraud | x)``. Its ranking may be fine and its probabilities will be
  badly wrong. This is why :mod:`src.train` also trains a plain cross-entropy
  arm: it is the reference for what an undistorted probability looks like.

Three metrics, because they disagree in informative ways:

* **Brier score** — mean squared error of the probabilities. Proper, but at a
  0.6% base rate it is dominated by how close the negatives are to zero, so a
  model that predicts ~0 everywhere scores well.
* **Expected calibration error** — mean absolute gap between confidence and
  accuracy across bins. Reported with *quantile* bins, because equal-width bins
  put ~99% of this data in the first bin and report near-zero error for
  everything.
* **Reliability diagrams** — the picture, because the direction of the error
  matters and a single number hides it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

LOGGER = logging.getLogger(__name__)

_EPS = 1e-12


def _clip(p: np.ndarray) -> np.ndarray:
    """Clip probabilities away from 0 and 1 so logits stay finite."""
    return np.clip(p, 1e-7, 1 - 1e-7)


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error between probabilities and outcomes.

    Args:
        y_true: Binary labels.
        probabilities: Predicted probabilities.

    Returns:
        The Brier score; lower is better.
    """
    return float(np.mean((probabilities - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
    strategy: str = "quantile",
) -> float:
    """Weighted mean gap between predicted probability and observed frequency.

    Args:
        y_true: Binary labels.
        probabilities: Predicted probabilities.
        n_bins: Number of bins.
        strategy: ``"quantile"`` for equal-count bins (the right choice on
            skewed score distributions) or ``"uniform"`` for equal-width bins.

    Returns:
        The ECE; lower is better.

    Raises:
        ValueError: If ``strategy`` is not recognised.
    """
    bins = reliability_bins(y_true, probabilities, n_bins=n_bins, strategy=strategy)
    if not bins.count.size:
        return float("nan")
    weights = bins.count / bins.count.sum()
    return float(np.sum(weights * np.abs(bins.mean_predicted - bins.observed_rate)))


def alert_region_calibration(
    y_true: np.ndarray, probabilities: np.ndarray, capacity: float = 0.005
) -> tuple[float, float]:
    """Mean predicted probability and observed rate inside the alert region.

    Aggregate calibration on this problem is dominated by the 99%+ of
    transactions scored near zero, all of which are indeed legitimate. A model
    can therefore look beautifully calibrated overall while being badly wrong in
    the only band where a decision is ever taken. This restricts the comparison
    to the top ``capacity`` fraction — the alerts a fraud team would actually
    work — which is where a threshold on an absolute probability has to be right.

    Args:
        y_true: Binary labels.
        probabilities: Predicted probabilities.
        capacity: Fraction of the population treated as the alert region.

    Returns:
        Tuple of (mean predicted probability, observed fraud rate) in that region.
    """
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")
    k = max(int(np.ceil(capacity * n)), 1)
    top = np.argsort(-np.asarray(probabilities), kind="stable")[:k]
    return float(np.mean(probabilities[top])), float(np.mean(np.asarray(y_true)[top]))


def maximum_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15
) -> float:
    """Largest absolute calibration gap over quantile bins.

    Args:
        y_true: Binary labels.
        probabilities: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        The maximum calibration error.
    """
    bins = reliability_bins(y_true, probabilities, n_bins=n_bins)
    if not bins.count.size:
        return float("nan")
    return float(np.max(np.abs(bins.mean_predicted - bins.observed_rate)))


@dataclass
class ReliabilityBins:
    """Binned reliability data underlying a diagram.

    Attributes:
        mean_predicted: Mean predicted probability per bin.
        observed_rate: Observed positive rate per bin.
        count: Number of rows per bin.
        edges: Bin edges used.
    """

    mean_predicted: np.ndarray
    observed_rate: np.ndarray
    count: np.ndarray
    edges: np.ndarray


def reliability_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
    strategy: str = "quantile",
) -> ReliabilityBins:
    """Bin predictions and compare predicted probability with observed frequency.

    Args:
        y_true: Binary labels.
        probabilities: Predicted probabilities.
        n_bins: Number of bins.
        strategy: ``"quantile"`` or ``"uniform"``.

    Returns:
        Populated :class:`ReliabilityBins`, with empty bins dropped.

    Raises:
        ValueError: If ``strategy`` is not recognised.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(probabilities, quantiles))
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    if edges.size < 2:
        edges = np.array([probabilities.min(), probabilities.max() + _EPS])

    index = np.clip(np.searchsorted(edges, probabilities, side="right") - 1, 0, len(edges) - 2)
    mean_pred, observed, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = index == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred.append(float(probabilities[mask].mean()))
        observed.append(float(y_true[mask].mean()))
        counts.append(n)
    return ReliabilityBins(
        mean_predicted=np.asarray(mean_pred),
        observed_rate=np.asarray(observed),
        count=np.asarray(counts, dtype=np.float64),
        edges=edges,
    )


# --------------------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------------------


class TemperatureScaler:
    """Single-parameter logit rescaling.

    Divides the logit by a learned temperature. ``T > 1`` softens an
    overconfident model. One parameter cannot fix a mis-shaped curve, but it
    cannot overfit either, which makes it the right first thing to try when the
    calibration set is small — and a fraud calibration set is always small,
    because it is bounded by the number of confirmed frauds in the validation
    window.
    """

    def __init__(self) -> None:
        """Initialise with an identity temperature."""
        self.temperature: float = 1.0

    def fit(self, y_true: np.ndarray, probabilities: np.ndarray) -> TemperatureScaler:
        """Fit the temperature by minimising negative log likelihood.

        A golden-section search over ``log T`` is used rather than gradient
        descent: the objective is one-dimensional and unimodal, and this avoids
        pulling an optimiser into the dependency graph for one scalar.

        Args:
            y_true: Binary labels from a held-out set.
            probabilities: Predicted probabilities on that set.

        Returns:
            ``self``, fitted.
        """
        logits = np.log(_clip(probabilities) / (1 - _clip(probabilities)))
        y_true = np.asarray(y_true, dtype=np.float64)

        def nll(log_t: float) -> float:
            scaled = logits / np.exp(log_t)
            # log(1 + exp(x)) computed stably.
            log_sig = -np.logaddexp(0.0, -scaled)
            log_one_minus = -np.logaddexp(0.0, scaled)
            return float(-np.mean(y_true * log_sig + (1 - y_true) * log_one_minus))

        lo, hi = -3.0, 3.0
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
        fa, fb = nll(a), nll(b)
        for _ in range(60):
            if fa < fb:
                hi, b, fb = b, a, fa
                a = hi - phi * (hi - lo)
                fa = nll(a)
            else:
                lo, a, fa = a, b, fb
                b = lo + phi * (hi - lo)
                fb = nll(b)
        self.temperature = float(np.exp((lo + hi) / 2.0))
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature.

        Args:
            probabilities: Raw predicted probabilities.

        Returns:
            Rescaled probabilities.
        """
        p = _clip(np.asarray(probabilities, dtype=np.float64))
        logits = np.log(p / (1 - p)) / self.temperature
        return 1.0 / (1.0 + np.exp(-logits))


class IsotonicCalibrator:
    """Monotone piecewise-constant recalibration.

    Strictly more flexible than temperature scaling and rank-preserving, so
    PR-AUC is unchanged by construction. The cost is variance: with a few hundred
    positives it will happily fit steps that are noise, and it cannot extrapolate
    beyond the score range it was fitted on.
    """

    def __init__(self) -> None:
        """Initialise the underlying regressor."""
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, y_true: np.ndarray, probabilities: np.ndarray) -> IsotonicCalibrator:
        """Fit on a held-out set.

        Args:
            y_true: Binary labels.
            probabilities: Predicted probabilities.

        Returns:
            ``self``, fitted.
        """
        self.model.fit(np.asarray(probabilities, dtype=np.float64), np.asarray(y_true))
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply the fitted mapping.

        Args:
            probabilities: Raw predicted probabilities.

        Returns:
            Recalibrated probabilities.
        """
        return np.asarray(
            self.model.predict(np.asarray(probabilities, dtype=np.float64)), dtype=np.float64
        )


@dataclass
class CalibrationReport:
    """Calibration metrics for one model under each calibration method.

    Attributes:
        model: Model name.
        method: ``"raw"``, ``"temperature"`` or ``"isotonic"``.
        brier: Brier score on the evaluation set.
        ece: Expected calibration error (quantile bins).
        mce: Maximum calibration error.
        mean_predicted: Mean predicted probability over the whole slice.
        observed_rate: Observed positive rate over the whole slice.
        alert_mean_predicted: Mean predicted probability inside the alert region.
        alert_observed_rate: Observed fraud rate inside the alert region.
        temperature: Fitted temperature, where applicable.
        bins: Reliability bins for plotting.
    """

    model: str
    method: str
    brier: float
    ece: float
    mce: float
    mean_predicted: float
    observed_rate: float
    alert_mean_predicted: float = float("nan")
    alert_observed_rate: float = float("nan")
    temperature: float | None = None
    bins: ReliabilityBins | None = field(default=None, repr=False)

    def to_row(self) -> dict[str, float | str | None]:
        """Flatten to a DataFrame row."""
        return {
            "model": self.model,
            "method": self.method,
            "brier": self.brier,
            "ece": self.ece,
            "mce": self.mce,
            "mean_predicted": self.mean_predicted,
            "observed_rate": self.observed_rate,
            "confidence_ratio": (
                self.mean_predicted / self.observed_rate if self.observed_rate else float("nan")
            ),
            "alert_mean_predicted": self.alert_mean_predicted,
            "alert_observed_rate": self.alert_observed_rate,
            "alert_confidence_ratio": (
                self.alert_mean_predicted / self.alert_observed_rate
                if self.alert_observed_rate
                else float("nan")
            ),
            "temperature": self.temperature,
        }


def assess(
    y_true: np.ndarray, probabilities: np.ndarray, *, model: str, method: str, temperature: float | None = None
) -> CalibrationReport:
    """Compute every calibration metric for one set of probabilities.

    Args:
        y_true: Binary labels.
        probabilities: Predicted probabilities.
        model: Model name for the report.
        method: Calibration method name.
        temperature: Fitted temperature, if any.

    Returns:
        Populated :class:`CalibrationReport`.
    """
    alert_predicted, alert_observed = alert_region_calibration(y_true, probabilities)
    return CalibrationReport(
        model=model,
        method=method,
        brier=brier_score(y_true, probabilities),
        ece=expected_calibration_error(y_true, probabilities),
        mce=maximum_calibration_error(y_true, probabilities),
        mean_predicted=float(np.mean(probabilities)),
        observed_rate=float(np.mean(y_true)),
        alert_mean_predicted=alert_predicted,
        alert_observed_rate=alert_observed,
        temperature=temperature,
        bins=reliability_bins(y_true, probabilities),
    )


def calibrate_and_compare(
    y_val: np.ndarray,
    val_probabilities: np.ndarray,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
    *,
    model: str,
) -> tuple[list[CalibrationReport], dict[str, np.ndarray]]:
    """Fit calibrators on validation and evaluate all three on test.

    The calibrator is fitted on the validation split and never on test, for the
    same reason the model is: a calibration map fitted on the evaluation data
    reports its own training error.

    Args:
        y_val: Validation labels.
        val_probabilities: Validation probabilities.
        y_test: Test labels.
        test_probabilities: Test probabilities.
        model: Model name.

    Returns:
        Tuple of (reports for raw / temperature / isotonic, calibrated test
        probabilities keyed by method).
    """
    temperature = TemperatureScaler().fit(y_val, val_probabilities)
    isotonic = IsotonicCalibrator().fit(y_val, val_probabilities)

    calibrated = {
        "raw": np.asarray(test_probabilities, dtype=np.float64),
        "temperature": temperature.transform(test_probabilities),
        "isotonic": isotonic.transform(test_probabilities),
    }
    reports = [
        assess(
            y_test,
            values,
            model=model,
            method=method,
            temperature=temperature.temperature if method == "temperature" else None,
        )
        for method, values in calibrated.items()
    ]
    LOGGER.info(
        "%s: raw ECE %.5f -> temperature %.5f (T=%.3f) -> isotonic %.5f",
        model,
        reports[0].ece,
        reports[1].ece,
        temperature.temperature,
        reports[2].ece,
    )
    return reports, calibrated


def plot_reliability(
    reports: list[CalibrationReport], path: Path, title: str = "Reliability"
) -> Path:
    """Write a reliability diagram comparing calibration methods.

    Axes are log-scaled: at a 0.6% base rate every interesting probability lives
    below 0.05, and a linear diagram is a dot in the corner.

    Args:
        reports: Reports to plot; each contributes one curve.
        path: Output image path.
        title: Figure title.

    Returns:
        The written path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    lo = 1e-5
    ax.plot([lo, 1], [lo, 1], "k--", linewidth=1, label="perfect calibration")
    for report in reports:
        if report.bins is None or not report.bins.count.size:
            continue
        ax.plot(
            np.maximum(report.bins.mean_predicted, lo),
            np.maximum(report.bins.observed_rate, lo),
            marker="o",
            markersize=4,
            linewidth=1.2,
            label=f"{report.method} (ECE {report.ece:.4f})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo, 1)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraud rate")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
