"""Interpretability: attention over prior transactions, SHAP on the baseline.

An analyst working an alert needs to know what to look at, and a model-risk
reviewer needs to know what the model keyed on. Those are different questions and
neither is answered by "the score was 0.87".

**On attention as an explanation — the honest position.** Attention weights show
where the model *read from*, not why it decided. They are not a causal account:
a low-weight position can still be decisive (its information may already be
carried by the residual stream), and a high-weight position can be irrelevant
(attention is one factor in a product with values that may be near zero). Multiple
attention distributions can produce the same output, so no single one is *the*
explanation. Reviewers know this, and presenting attention as causal attribution
is the fastest way to lose their confidence.

What attention is genuinely good for here is narrower and still useful: it points
at which prior transactions the model conditioned on, which is exactly the
"show me the related activity" affordance a fraud analyst wants. This module
therefore reports attention as *evidence pointers* and pairs every attention
explanation with a gradient-based attribution, which does answer a
counterfactual-flavoured question — "how much does the score move if this input
moves?" — and can disagree with attention. Where they disagree, that is worth
knowing.

**SHAP on the LightGBM baseline** is a different kind of object: exact for tree
ensembles (via TreeSHAP), additive, and in log-odds units. Comparing the two
sides is instructive because they attribute to different things — SHAP to
engineered aggregate features, attention to positions in a sequence — and a
reader should not be led to believe those are the same currency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.features import NUMERIC_FEATURES
from src.data.schema import TransactionSchema
from src.data.sequences import SequenceData, gather
from src.models import SequenceClassifier

LOGGER = logging.getLogger(__name__)

S = TransactionSchema

ATTENTION_CAVEAT = (
    "Attention weights show which prior transactions the model read from, not "
    "why it decided. They are not a causal attribution: a low-weight position "
    "can still be decisive and a high-weight one can be inert. Treat them as "
    "pointers to related activity for an analyst to review, and read the "
    "gradient attributions alongside them."
)


@dataclass
class SequenceExplanation:
    """Explanation of one scored transaction from a sequence model.

    Attributes:
        row: Row index in the feature frame.
        score: Predicted probability.
        label: True label, when known.
        attention: Per-position attention from the scored transaction, or
            ``None`` for architectures without attention.
        position_influence: Gradient-times-input attribution per position — how
            much the logit moves with the inputs at that position.
        feature_influence: Gradient-times-input attribution per numeric feature
            at the scored position.
        history: The window's transactions as a readable table.
    """

    row: int
    score: float
    label: float
    attention: np.ndarray | None
    position_influence: np.ndarray
    feature_influence: dict[str, float]
    history: pd.DataFrame = field(repr=False)

    def top_positions(self, k: int = 5) -> pd.DataFrame:
        """Prior transactions ranked by attention, most attended first.

        Args:
            k: How many to return.

        Returns:
            Slice of ``history`` with attention and influence columns attached.
        """
        table = self.history.copy()
        if self.attention is not None:
            table["attention"] = self.attention[-len(table) :]
        table["influence"] = self.position_influence[-len(table) :]
        sort_key = "attention" if self.attention is not None else "influence"
        return table.sort_values(sort_key, ascending=False).head(k)

    def top_features(self, k: int = 8) -> pd.DataFrame:
        """Numeric features ranked by absolute influence on the logit.

        Args:
            k: How many to return.

        Returns:
            DataFrame of feature and signed influence.
        """
        frame = pd.DataFrame(
            {"feature": list(self.feature_influence), "influence": list(self.feature_influence.values())}
        )
        return frame.reindex(frame["influence"].abs().sort_values(ascending=False).index).head(k)


def explain_sequence_model(
    model: SequenceClassifier,
    data: SequenceData,
    frame: pd.DataFrame,
    row: int,
) -> SequenceExplanation:
    """Explain one transaction scored by a sequence model.

    Args:
        model: A trained sequence classifier.
        data: Sequence tensors.
        frame: Feature frame, for rendering the readable history.
        row: Row index to explain.

    Returns:
        Populated :class:`SequenceExplanation`.
    """
    model.eval()
    batch = gather(data, np.asarray([row]))
    numeric = batch["numeric"].clone().requires_grad_(True)
    categorical = batch["categorical"]
    mask = batch["mask"]

    logit = model(numeric, categorical, mask)
    score = float(torch.sigmoid(logit).item())
    model.zero_grad(set_to_none=True)
    logit.backward()

    # Gradient x input: a first-order estimate of how much the logit moves with
    # each input. Cheap, signed, and it answers a different question from
    # attention, which is the point of showing both.
    gradient = numeric.grad
    if gradient is None:  # pragma: no cover - backward() always populates it
        raise RuntimeError("no gradient was produced for the numeric inputs")
    contribution = (gradient * numeric).detach()[0].numpy()
    position_influence = contribution.sum(axis=1)

    attention_tensor = model.attention_weights(numeric.detach(), categorical, mask)
    attention = attention_tensor[0].detach().numpy() if attention_tensor is not None else None

    window = data.windows[row]
    valid = window[window >= 0]
    history = frame.iloc[valid][
        [
            S.TIMESTAMP,
            S.AMOUNT,
            S.MERCHANT_ID,
            S.MCC,
            S.DEVICE_ID,
            S.DEVICE_TYPE,
            S.CHANNEL,
            S.COUNTRY,
        ]
    ].copy()
    history.insert(0, "position", np.arange(-len(valid) + 1, 1))

    final_features = dict(
        zip(NUMERIC_FEATURES, contribution[-1].tolist(), strict=True)
    )
    return SequenceExplanation(
        row=int(row),
        score=score,
        label=float(data.labels[row]),
        attention=attention,
        position_influence=position_influence,
        feature_influence=final_features,
        history=history,
    )


def attention_alignment(
    model: SequenceClassifier,
    data: SequenceData,
    rows: np.ndarray,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Average attention by position offset, split by outcome.

    Aggregating over many transactions says something an individual explanation
    cannot: whether the model looks further back for fraud than for legitimate
    activity. If attention on fraud is concentrated on the immediately preceding
    transactions, the model is essentially doing velocity detection and the
    sequence machinery is not earning its keep.

    Args:
        model: A trained model with attention.
        data: Sequence tensors.
        rows: Rows to aggregate over.
        batch_size: Batch size for the sweep.

    Returns:
        DataFrame of offset (0 = the scored transaction, -1 = previous) with mean
        attention for fraudulent and legitimate rows.

    Raises:
        ValueError: If the model has no attention mechanism.
    """
    model.eval()
    totals = {0.0: np.zeros(data.max_len), 1.0: np.zeros(data.max_len)}
    counts = {0.0: 0, 1.0: 0}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = gather(data, chunk)
            weights = model.attention_weights(
                batch["numeric"], batch["categorical"], batch["mask"]
            )
            if weights is None:
                raise ValueError("model has no attention mechanism to aggregate")
            values = weights.numpy()
            labels = data.labels[chunk]
            for key in (0.0, 1.0):
                sel = labels == key
                if sel.any():
                    totals[key] += values[sel].sum(axis=0)
                    counts[key] += int(sel.sum())
    return pd.DataFrame(
        {
            "offset": np.arange(-data.max_len + 1, 1),
            "mean_attention_legit": totals[0.0] / max(counts[0.0], 1),
            "mean_attention_fraud": totals[1.0] / max(counts[1.0], 1),
        }
    )


# --------------------------------------------------------------------------------------
# SHAP on the LightGBM baseline
# --------------------------------------------------------------------------------------


@dataclass
class ShapExplanation:
    """TreeSHAP output for one baseline prediction.

    Attributes:
        row: Row index explained.
        score: Predicted probability.
        base_value: Model expected value in log-odds.
        contributions: Per-feature SHAP values in log-odds.
    """

    row: int
    score: float
    base_value: float
    contributions: pd.Series

    def top(self, k: int = 10) -> pd.DataFrame:
        """Features ranked by absolute SHAP value.

        Args:
            k: How many to return.

        Returns:
            DataFrame of feature, value and contribution.
        """
        ordered = self.contributions.reindex(
            self.contributions.abs().sort_values(ascending=False).index
        ).head(k)
        return pd.DataFrame(
            {"feature": ordered.index.astype(str), "shap_log_odds": ordered.to_numpy()}
        )


def explain_baseline(
    booster: Any, x: pd.DataFrame, row: int
) -> ShapExplanation:
    """TreeSHAP attribution for one LightGBM prediction.

    Args:
        booster: A fitted LightGBM booster.
        x: The feature frame the booster was trained on.
        row: Positional index of the row to explain.

    Returns:
        Populated :class:`ShapExplanation`.
    """
    import shap

    explainer = shap.TreeExplainer(booster)
    sample = x.iloc[[row]]
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[-1]
    contributions = pd.Series(np.asarray(values).reshape(-1), index=x.columns)
    base = explainer.expected_value
    base_value = float(base[-1] if isinstance(base, (list, np.ndarray)) else base)
    score = float(booster.predict(sample)[0])
    return ShapExplanation(
        row=int(row), score=score, base_value=base_value, contributions=contributions
    )


def baseline_global_importance(booster: Any, x: pd.DataFrame, sample_size: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Mean absolute SHAP value per feature over a sample.

    Args:
        booster: A fitted LightGBM booster.
        x: Feature frame.
        sample_size: Rows to sample for the estimate.
        seed: Sampling seed.

    Returns:
        DataFrame of feature and mean absolute SHAP, descending.
    """
    import shap

    rng = np.random.default_rng(seed)
    take = min(sample_size, len(x))
    rows = rng.choice(len(x), size=take, replace=False)
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(x.iloc[rows])
    if isinstance(values, list):
        values = values[-1]
    mean_abs = np.abs(np.asarray(values)).mean(axis=0)
    return (
        pd.DataFrame({"feature": x.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------------------
# Worked example
# --------------------------------------------------------------------------------------


def worked_example(
    model: SequenceClassifier,
    data: SequenceData,
    frame: pd.DataFrame,
    booster: Any | None = None,
    baseline_x: pd.DataFrame | None = None,
    *,
    rows: np.ndarray | None = None,
    prefer_fraud_type: str | None = None,
) -> tuple[str, SequenceExplanation]:
    """Explain one flagged transaction end to end, as readable text.

    Picks a high-scoring true positive, renders the customer's recent history,
    the attention the model paid to each prior transaction, the gradient
    attribution over features, and — when a booster is supplied — the SHAP
    decomposition of the baseline's opinion of the same transaction.

    Args:
        model: A trained sequence model.
        data: Sequence tensors.
        frame: Feature frame.
        booster: Optional fitted LightGBM booster for the side-by-side.
        baseline_x: Feature frame the booster was trained on.
        rows: Candidate rows; defaults to every row.
        prefer_fraud_type: Prefer a case of this mechanism if one is available.

    Returns:
        Tuple of (rendered text, the underlying explanation).

    Raises:
        ValueError: If no fraudulent candidate row exists.
    """
    from src.train import predict

    candidates = np.arange(len(data)) if rows is None else np.asarray(rows)
    labels = data.labels[candidates]
    positives = candidates[labels > 0.5]
    if prefer_fraud_type is not None:
        typed = positives[data.fraud_type[positives] == prefer_fraud_type]
        if typed.size:
            positives = typed
    if not positives.size:
        raise ValueError("no fraudulent rows available to explain")

    scores = predict(model, data, positives)
    chosen = int(positives[int(np.argmax(scores))])
    explanation = explain_sequence_model(model, data, frame, chosen)

    lines: list[str] = []
    row = frame.iloc[chosen]
    lines.append("=" * 78)
    lines.append("WORKED EXAMPLE: one flagged transaction, explained end to end")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Transaction   {row[S.TXN_ID]}")
    lines.append(f"Customer      {row[S.CUSTOMER_ID]}")
    lines.append(f"Timestamp     {row[S.TIMESTAMP]}")
    lines.append(f"Amount        {row[S.AMOUNT]:.2f}")
    lines.append(f"Merchant      {row[S.MERCHANT_ID]}  (MCC {row[S.MCC]})")
    lines.append(f"Device        {row[S.DEVICE_ID]}  ({row[S.DEVICE_TYPE]})")
    lines.append(f"Channel       {row[S.CHANNEL]} / {row[S.ENTRY_MODE]} / {row[S.COUNTRY]}")
    lines.append("")
    lines.append(f"Model score   {explanation.score:.4f}")
    lines.append(f"True label    {'FRAUD' if explanation.label > 0.5 else 'legitimate'}"
                 f"  (mechanism: {row[S.FRAUD_TYPE]})")
    lines.append("")

    has_attention = explanation.attention is not None
    lines.append("-" * 78)
    lines.append(
        "PRIOR TRANSACTIONS THE MODEL READ FROM"
        if has_attention
        else "PRIOR TRANSACTIONS RANKED BY INFLUENCE ON THE SCORE"
    )
    lines.append("-" * 78)
    top = explanation.top_positions(6)
    display = top.copy()
    display[S.TIMESTAMP] = display[S.TIMESTAMP].dt.strftime("%Y-%m-%d %H:%M")
    display[S.AMOUNT] = display[S.AMOUNT].map(lambda v: f"{v:,.2f}")
    for column in ("attention", "influence"):
        if column in display.columns:
            display[column] = display[column].map(lambda v: f"{v:+.4f}")
    lines.append(display.to_string(index=False))
    lines.append("")
    if has_attention:
        lines.append("Caveat: " + ATTENTION_CAVEAT)
    else:
        lines.append(
            "This architecture has no attention mechanism, so positions are ranked "
            "by gradient x input: how much the logit moves with the inputs at that "
            "position. First-order and signed, and like any local attribution it "
            "describes sensitivity at this point, not a causal account."
        )
    lines.append("")

    lines.append("-" * 78)
    lines.append("FEATURE ATTRIBUTION AT THE SCORED TRANSACTION (gradient x input, logit units)")
    lines.append("-" * 78)
    features = explanation.top_features(10)
    features["influence"] = features["influence"].map(lambda v: f"{v:+.4f}")
    lines.append(features.to_string(index=False))
    lines.append("")

    if booster is not None and baseline_x is not None:
        lines.append("-" * 78)
        lines.append("THE SAME TRANSACTION, AS THE LIGHTGBM BASELINE SEES IT (TreeSHAP, log-odds)")
        lines.append("-" * 78)
        shap_explanation = explain_baseline(booster, baseline_x, chosen)
        lines.append(f"baseline score {shap_explanation.score:.4f}, "
                     f"expected value {shap_explanation.base_value:+.4f} log-odds")
        table = shap_explanation.top(10)
        table["shap_log_odds"] = table["shap_log_odds"].map(lambda v: f"{v:+.4f}")
        lines.append(table.to_string(index=False))
        lines.append("")
        lines.append(
            "The two attributions are not in the same currency: SHAP decomposes a "
            "log-odds prediction over engineered aggregate features and is exact "
            "for tree ensembles, while the sequence attributions are first-order "
            "and spread over positions in a window. Agreement between them is "
            "reassuring; disagreement usually means the sequence model is using "
            "ordering the aggregates flattened away."
        )

    return "\n".join(lines), explanation


def plot_attention_alignment(alignment: pd.DataFrame, path: Path) -> Path:
    """Plot mean attention by position offset for fraud and legitimate rows.

    Args:
        alignment: Output of :func:`attention_alignment`.
        path: Output image path.

    Returns:
        The written path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(alignment["offset"], alignment["mean_attention_legit"], marker="o", markersize=3, label="legitimate")
    ax.plot(alignment["offset"], alignment["mean_attention_fraud"], marker="s", markersize=3, label="fraud")
    ax.set_xlabel("position offset (0 = transaction being scored)")
    ax.set_ylabel("mean attention weight")
    ax.set_title("Where the Transformer looks")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
