"""Causal feature engineering with a hard temporal-integrity guarantee.

Two rules govern everything in this module.

1. **A feature for a transaction at time t may read only transactions strictly
   before t, for that customer.** Every history-derived quantity here is computed
   from position ``[0, i)`` or from a time window that excludes ``t`` itself.
   :mod:`tests.test_leakage` fails the build if this is ever violated.

2. **Labels are not features, not even prior labels.** A chargeback is confirmed
   weeks after the authorisation, so at scoring time the fraud status of the
   customer's previous transactions is genuinely unknown. Anything that consumes
   labels — the risk encodings — is fitted on the training period only, and
   within training it is fitted out-of-fold on strictly earlier blocks.

The per-customer computation is written once, as a function over ordered numpy
arrays, and is used by *both* the batch pipeline and the serving path. That is
deliberate: the classic production bug is a fast re-implementation of the feature
code for online scoring that quietly disagrees with the training code.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import (
    N_RESERVED_CODES,
    UNK_INDEX,
    CategoricalSpec,
    TransactionSchema,
)

LOGGER = logging.getLogger(__name__)

S = TransactionSchema

#: Numeric features produced by :func:`customer_causal_features`, in order.
CAUSAL_FEATURES: tuple[str, ...] = (
    # Request-time facts (no history needed).
    "log_amount",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_night",
    "is_weekend",
    "is_online",
    # History depth / recency.
    "log_hist_count",
    "log_secs_since_prev",
    "log_window_span_secs",
    # Amount anomaly relative to this customer.
    "amount_z",
    "amount_vs_mean",
    "amount_vs_max",
    # Velocity.
    "log_cnt_1h",
    "log_cnt_24h",
    "log_cnt_7d",
    "log_amt_1h",
    "log_amt_24h",
    "log_amt_7d",
    "amt_share_24h",
    "log_distinct_merchants_24h",
    "log_distinct_devices_7d",
    # Entity novelty and recency.
    "is_new_merchant",
    "is_new_mcc",
    "is_new_device",
    "is_new_device_type",
    "is_new_country",
    "log_secs_since_same_merchant",
    "log_secs_since_same_mcc",
    "log_secs_since_same_device",
    "log_merchant_seen_count",
    "log_device_seen_count",
    "log_distinct_merchants",
    "log_distinct_devices",
    # Behavioural mix.
    "prior_online_rate",
    "hour_deviation",
)

#: Population-level features fitted on the training period.
POPULATION_FEATURES: tuple[str, ...] = (
    "merchant_risk",
    "mcc_risk",
    "country_risk",
    "device_type_risk",
    "log_merchant_freq",
)

NUMERIC_FEATURES: tuple[str, ...] = CAUSAL_FEATURES + POPULATION_FEATURES

#: Categorical columns that receive a learned embedding, and the raw column they
#: are coded from.
EMBEDDED_CATEGORICALS: tuple[tuple[str, str], ...] = (
    ("merchant_code", S.MERCHANT_ID),
    ("mcc_code", S.MCC),
    ("device_type_code", S.DEVICE_TYPE),
    ("country_code", S.COUNTRY),
    ("channel_code", S.CHANNEL),
    ("entry_mode_code", S.ENTRY_MODE),
)

#: Sentinel for "this entity has never been seen": ~10 years in seconds. Using a
#: large finite number rather than NaN keeps the tensors dense and lets a tree
#: split on it cleanly.
NEVER_SECS = 3.2e8

_EPS = 1e-6

#: Timestamps are normalised to microseconds before any arithmetic. The offline
#: path reads pandas timestamps (nanosecond resolution) while the online path
#: receives JSON datetimes (microsecond resolution at best), and without a common
#: quantisation the two would disagree in the last decimal place of every time
#: delta. The difference is numerically trivial and operationally fatal: it is
#: exactly the sort of silent online/offline skew that degrades a deployed model
#: while every offline check still passes. ``tests/test_serving.py`` asserts the
#: two paths agree, and this is what makes that possible.
_NS_PER_US = 1000

#: How many prior transactions any feature may look at. This is a *bounded*
#: history on purpose, and it is the single decision that makes the serving path
#: honest.
#:
#: The natural implementation computes expanding statistics over a customer's
#: entire history: lifetime transaction count, tenure, mean spend since signup,
#: every merchant ever used. Offline that is easy. Online it is impossible — the
#: authorisation request cannot carry an unbounded history, so a service either
#: keeps a stateful per-customer store (a second system that will drift from the
#: training code) or silently computes something different from what the model
#: was trained on. The second failure is the common one and it is invisible:
#: every offline check passes while the deployed model degrades.
#:
#: Bounding every feature at ``DEFAULT_LOOKBACK`` prior transactions gives up
#: long-horizon signal (true tenure, lifetime distinct merchants) and buys exact
#: online/offline parity from a stateless service. The trade is recorded in
#: DECISIONS.md, and ``tests/test_serving.py`` holds it to account.
DEFAULT_LOOKBACK = 31


def to_epoch_seconds(nanoseconds: np.ndarray) -> np.ndarray:
    """Convert integer nanosecond timestamps to seconds at microsecond resolution.

    Args:
        nanoseconds: Int64 nanoseconds since the epoch.

    Returns:
        Float seconds, truncated to whole microseconds.
    """
    return (np.asarray(nanoseconds, dtype=np.int64) // _NS_PER_US).astype(np.float64) / 1e6


# --------------------------------------------------------------------------------------
# Per-customer causal features (single implementation, used by batch and serving)
# --------------------------------------------------------------------------------------


def _window_view(values: np.ndarray, lookback: int, pad_value: Any) -> np.ndarray:
    """Matrix whose row ``i`` holds the ``lookback`` entries before position ``i``.

    Args:
        values: Per-transaction values in time order.
        lookback: Window width.
        pad_value: Value used where a row has less history than ``lookback``.

    Returns:
        Array of shape ``(len(values), lookback)``. Row ``i`` is
        ``values[i - lookback : i]``, left-padded with ``pad_value``.
    """
    padded = np.concatenate([np.full(lookback, pad_value, dtype=values.dtype), values])
    view = np.lib.stride_tricks.sliding_window_view(padded, lookback)
    return view[: len(values)]


def _distinct_per_row(codes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Count distinct codes per row, ignoring invalid entries.

    Args:
        codes: Integer codes, shape ``(n, k)``.
        valid: Boolean mask of the same shape.

    Returns:
        Per-row distinct count.
    """
    masked = np.where(valid, codes, -1)
    ordered = np.sort(masked, axis=1)
    is_first = np.empty(ordered.shape, dtype=bool)
    is_first[:, :1] = ordered[:, :1] != -1
    is_first[:, 1:] = (ordered[:, 1:] != ordered[:, :-1]) & (ordered[:, 1:] != -1)
    return is_first.sum(axis=1).astype(np.float64)


def customer_causal_features(
    ts_sec: np.ndarray,
    amount: np.ndarray,
    merchant_key: np.ndarray,
    mcc_key: np.ndarray,
    device_key: np.ndarray,
    device_type_key: np.ndarray,
    country_key: np.ndarray,
    is_online: np.ndarray,
    lookback: int = DEFAULT_LOOKBACK,
) -> dict[str, np.ndarray]:
    """Compute causal features for one customer's ordered transactions.

    Every returned value at index ``i`` is a function of the ``lookback``
    transactions strictly before ``i`` plus the request-time fields of ``i``
    itself. Two properties are enforced elementwise rather than assumed:

    * **Strictly past.** A window entry counts only when its timestamp is
      strictly less than ``t_i``, so simultaneous transactions never inform each
      other.
    * **Bounded.** Nothing reads further back than ``lookback`` transactions.
      This is what makes the online path in
      :meth:`FeatureBuilder.transform_window` able to reproduce these values
      exactly from a request payload — see the module note on why unbounded
      history would make the deployed features quietly different from the
      trained ones.

    Args:
        ts_sec: Unix timestamps in seconds, ascending.
        amount: Transaction amounts.
        merchant_key: Keys identifying the merchant.
        mcc_key: Keys identifying the MCC.
        device_key: Keys identifying the physical device/token.
        device_type_key: Keys identifying the device type.
        country_key: Keys identifying the country.
        is_online: 1.0 where the transaction is card-not-present.
        lookback: Number of prior transactions any feature may consider.

    Returns:
        Dict mapping each name in :data:`CAUSAL_FEATURES` to a float32 array.

    Raises:
        ValueError: If ``ts_sec`` is not sorted ascending or ``lookback`` < 1.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    n = int(ts_sec.shape[0])
    if n == 0:
        return {name: np.zeros(0, dtype=np.float32) for name in CAUSAL_FEATURES}
    if np.any(np.diff(ts_sec) < 0):
        raise ValueError("ts_sec must be sorted ascending")

    ts_sec = ts_sec.astype(np.float64)
    amount = amount.astype(np.float64)
    log_amount = np.log1p(np.maximum(amount, 0.0))
    online = is_online.astype(np.float64)
    k = lookback

    # --- the window, and which of its slots are real and strictly earlier -------
    win_t = _window_view(ts_sec, k, -np.inf)
    valid = np.isfinite(win_t) & (win_t < ts_sec[:, None])
    n_prior = valid.sum(axis=1).astype(np.float64)
    has_hist = n_prior > 0
    denom = np.maximum(n_prior, 1.0)

    win_amount = _window_view(amount, k, 0.0)
    win_log_amount = _window_view(log_amount, k, 0.0)
    win_online = _window_view(online, k, 0.0)

    def masked_sum(values: np.ndarray) -> np.ndarray:
        return np.where(valid, values, 0.0).sum(axis=1)

    # --- amount anomaly relative to this customer's recent behaviour ------------
    sum_log = masked_sum(win_log_amount)
    sum_log_sq = masked_sum(win_log_amount**2)
    prior_mean = sum_log / denom
    prior_var = np.maximum(sum_log_sq / denom - prior_mean**2, 0.0)
    prior_std = np.sqrt(prior_var)
    # With no history the anomaly is zero, not infinite: a customer's first
    # transaction is not evidence of anything.
    prior_mean = np.where(has_hist, prior_mean, log_amount)
    # A floor on the denominator stops a customer with two near-identical
    # transactions from producing z-scores in the hundreds.
    amount_z = (log_amount - prior_mean) / (prior_std + 0.35)
    amount_vs_mean = log_amount - prior_mean

    prior_max = np.where(valid, win_amount, -np.inf).max(axis=1)
    prior_max = np.where(has_hist, prior_max, amount)
    amount_vs_max = log_amount - np.log1p(np.maximum(prior_max, 0.0))

    # --- recency ----------------------------------------------------------------
    last_t = np.where(valid, win_t, -np.inf).max(axis=1)
    secs_since_prev = np.where(has_hist, ts_sec - last_t, NEVER_SECS)
    oldest_t = np.where(valid, win_t, np.inf).min(axis=1)
    window_span = np.where(has_hist, ts_sec - oldest_t, 0.0)

    # --- time-bounded velocity, additionally capped by the lookback -------------
    counts: dict[str, np.ndarray] = {}
    sums: dict[str, np.ndarray] = {}
    # Keep the per-window masks: the distinct-entity features below reuse them,
    # and binding them inside the loop under an ``if name == ...`` leaves them
    # merely *probably* defined — one edit to the window list away from a
    # NameError that only fires on some inputs.
    within: dict[str, np.ndarray] = {}
    for name, width in (("1h", 3600.0), ("24h", 86400.0), ("7d", 7 * 86400.0)):
        in_window = valid & (win_t >= (ts_sec - width)[:, None])
        within[name] = in_window
        counts[name] = in_window.sum(axis=1).astype(np.float64)
        sums[name] = np.where(in_window, win_amount, 0.0).sum(axis=1)

    # --- entity novelty, recency and diversity ----------------------------------
    def entity(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (is_new, seconds since last seen, times seen, window codes)."""
        codes = pd.factorize(keys)[0].astype(np.int64)
        win_codes = _window_view(codes, k, -1)
        match = (win_codes == codes[:, None]) & valid
        seen = match.any(axis=1)
        last_seen = np.where(match, win_t, -np.inf).max(axis=1)
        since = np.where(seen, ts_sec - last_seen, NEVER_SECS)
        return (~seen).astype(np.float64), since, match.sum(axis=1).astype(np.float64), win_codes

    is_new_merchant, since_merchant, merchant_seen, win_merchant = entity(merchant_key)
    is_new_mcc, since_mcc, _, _ = entity(mcc_key)
    is_new_device, since_device, device_seen, win_device = entity(device_key)
    is_new_device_type, _, _, _ = entity(device_type_key)
    is_new_country, _, _, _ = entity(country_key)

    distinct_merchants = _distinct_per_row(win_merchant, valid)
    distinct_devices = _distinct_per_row(win_device, valid)
    distinct_merchants_24h = _distinct_per_row(win_merchant, within["24h"])
    distinct_devices_7d = _distinct_per_row(win_device, within["7d"])

    # --- behavioural mix ---------------------------------------------------------
    prior_online_rate = np.where(has_hist, masked_sum(win_online) / denom, 0.0)

    hours = (ts_sec / 3600.0) % 24.0
    angle = 2.0 * np.pi * hours / 24.0
    hour_sin, hour_cos = np.sin(angle), np.cos(angle)
    mean_sin = masked_sum(_window_view(hour_sin, k, 0.0)) / denom
    mean_cos = masked_sum(_window_view(hour_cos, k, 0.0)) / denom
    resultant = np.sqrt(mean_sin**2 + mean_cos**2)
    # 0 = right on the customer's usual hour, up to 2 = diametrically opposite.
    # Scaled by the resultant length, so a customer with no clear rhythm
    # contributes about nothing.
    hour_deviation = np.where(
        has_hist, resultant - (hour_sin * mean_sin + hour_cos * mean_cos), 0.0
    )

    dow = ((ts_sec // 86400.0) + 3) % 7  # 1970-01-01 was a Thursday
    dow_angle = 2.0 * np.pi * dow / 7.0

    out: dict[str, np.ndarray] = {
        "log_amount": log_amount,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": np.sin(dow_angle),
        "dow_cos": np.cos(dow_angle),
        "is_night": ((hours < 6.0) | (hours >= 23.0)).astype(np.float64),
        "is_weekend": (dow >= 5).astype(np.float64),
        "is_online": online,
        "log_hist_count": np.log1p(n_prior),
        "log_secs_since_prev": np.log1p(np.maximum(secs_since_prev, 0.0)),
        "log_window_span_secs": np.log1p(np.maximum(window_span, 0.0)),
        "amount_z": amount_z,
        "amount_vs_mean": amount_vs_mean,
        "amount_vs_max": amount_vs_max,
        "log_cnt_1h": np.log1p(counts["1h"]),
        "log_cnt_24h": np.log1p(counts["24h"]),
        "log_cnt_7d": np.log1p(counts["7d"]),
        "log_amt_1h": np.log1p(sums["1h"]),
        "log_amt_24h": np.log1p(sums["24h"]),
        "log_amt_7d": np.log1p(sums["7d"]),
        "amt_share_24h": amount / (amount + sums["24h"] + _EPS),
        "log_distinct_merchants_24h": np.log1p(distinct_merchants_24h),
        "log_distinct_devices_7d": np.log1p(distinct_devices_7d),
        "is_new_merchant": is_new_merchant,
        "is_new_mcc": is_new_mcc,
        "is_new_device": is_new_device,
        "is_new_device_type": is_new_device_type,
        "is_new_country": is_new_country,
        "log_secs_since_same_merchant": np.log1p(np.maximum(since_merchant, 0.0)),
        "log_secs_since_same_mcc": np.log1p(np.maximum(since_mcc, 0.0)),
        "log_secs_since_same_device": np.log1p(np.maximum(since_device, 0.0)),
        "log_merchant_seen_count": np.log1p(merchant_seen),
        "log_device_seen_count": np.log1p(device_seen),
        "log_distinct_merchants": np.log1p(distinct_merchants),
        "log_distinct_devices": np.log1p(distinct_devices),
        "prior_online_rate": prior_online_rate,
        "hour_deviation": hour_deviation,
    }
    return {key: np.asarray(value, dtype=np.float32) for key, value in out.items()}


def add_causal_features(
    df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK
) -> pd.DataFrame:
    """Add causal features to a transaction frame, per customer.

    The frame is processed group by group; each group is sorted by timestamp and
    handed to :func:`customer_causal_features`. Results are written back on the
    original row order.

    Args:
        df: Transaction frame with at least the request fields.
        lookback: Prior transactions any feature may consider.

    Returns:
        A copy of ``df`` with the :data:`CAUSAL_FEATURES` columns appended.
    """
    work = df.copy()
    work["_pos"] = np.arange(len(work))
    work = work.sort_values([S.CUSTOMER_ID, S.TIMESTAMP], kind="mergesort")

    ts_all = to_epoch_seconds(work[S.TIMESTAMP].astype("int64").to_numpy())
    amount_all = work[S.AMOUNT].to_numpy()
    online_all = (work[S.CHANNEL].to_numpy() == "online").astype(np.float64)
    keys = {
        "merchant": work[S.MERCHANT_ID].to_numpy(),
        "mcc": work[S.MCC].to_numpy(),
        "device": work[S.DEVICE_ID].to_numpy(),
        "device_type": work[S.DEVICE_TYPE].to_numpy(),
        "country": work[S.COUNTRY].to_numpy(),
    }

    out = {name: np.zeros(len(work), dtype=np.float32) for name in CAUSAL_FEATURES}
    # ``indices`` gives positional slices per customer; the frame is already sorted
    # by customer then time, so each group is contiguous and ascending.
    codes, _ = pd.factorize(work[S.CUSTOMER_ID].to_numpy())
    order = np.argsort(codes, kind="stable")
    boundaries = np.flatnonzero(np.diff(codes[order])) + 1
    for group in np.split(order, boundaries):
        feats = customer_causal_features(
            ts_all[group],
            amount_all[group],
            keys["merchant"][group],
            keys["mcc"][group],
            keys["device"][group],
            keys["device_type"][group],
            keys["country"][group],
            online_all[group],
            lookback=lookback,
        )
        for name, values in feats.items():
            out[name][group] = values

    for name in CAUSAL_FEATURES:
        work[name] = out[name]
    work = work.sort_values("_pos", kind="mergesort").drop(columns=["_pos"])
    return work.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Population-level encodings, fitted on the training period only
# --------------------------------------------------------------------------------------


def _temporal_oof_target_encode(
    keys: np.ndarray,
    y: np.ndarray,
    block: np.ndarray,
    smoothing: float,
    initial_prior: float,
) -> np.ndarray:
    """Out-of-fold target encoding using strictly earlier temporal blocks.

    Standard shuffled K-fold target encoding is out-of-fold but not out-of-time:
    a row in block 2 would be encoded partly with labels from block 5. Here block
    ``k`` is encoded using only blocks ``< k``, which is both out-of-fold and
    causally honest.

    The smoothing prior is expanding for the same reason. A prior computed over
    the whole training window is a weak dependency — one scalar summarising every
    label — but it is still a dependency on the future, and
    ``test_training_risk_encoding_is_out_of_time_within_train`` rejects it. Block
    ``k`` therefore smooths towards the observed rate over blocks ``< k``, and
    block 0 smooths towards ``initial_prior``, an ex-ante estimate fixed before
    looking at the data.

    Args:
        keys: Entity keys, one per training row.
        y: Binary labels.
        block: Temporal block index per row, ascending in time.
        smoothing: Pseudo-count strength.
        initial_prior: Positive rate assumed before any block has been observed.

    Returns:
        Encoded values aligned to ``keys``.
    """
    out = np.full(len(keys), initial_prior, dtype=np.float64)
    sum_y: dict[Any, float] = {}
    cnt: dict[Any, float] = {}
    seen_pos = 0.0
    seen_n = 0.0
    for b in np.unique(block):
        mask = block == b
        prior_b = seen_pos / seen_n if seen_n > 0 else initial_prior
        k_arr = keys[mask]
        s = np.array([sum_y.get(k, 0.0) for k in k_arr], dtype=np.float64)
        c = np.array([cnt.get(k, 0.0) for k in k_arr], dtype=np.float64)
        out[mask] = (s + smoothing * prior_b) / (c + smoothing)
        for k, yi in zip(k_arr, y[mask], strict=True):
            sum_y[k] = sum_y.get(k, 0.0) + float(yi)
            cnt[k] = cnt.get(k, 0.0) + 1.0
        seen_pos += float(y[mask].sum())
        seen_n += float(mask.sum())
    return out


def _full_target_map(
    keys: np.ndarray, y: np.ndarray, prior: float, smoothing: float
) -> dict[Any, float]:
    """Smoothed target-encoding map over all supplied (training) rows."""
    frame = pd.DataFrame({"k": keys, "y": y})
    agg = frame.groupby("k")["y"].agg(["sum", "count"])
    values = (agg["sum"] + smoothing * prior) / (agg["count"] + smoothing)
    return values.to_dict()


@dataclass
class FeatureSpec:
    """Description of the built feature matrix.

    Attributes:
        numeric: Ordered numeric feature column names.
        categorical: Ordered embedded categorical specs.
    """

    numeric: list[str]
    categorical: list[CategoricalSpec]

    @property
    def categorical_columns(self) -> list[str]:
        """Names of the integer-coded categorical columns."""
        return [c.column for c in self.categorical]


@dataclass
class FeatureBuilder:
    """Fits vocabularies, risk encodings and scaling on the training period.

    Attributes:
        lookback: Prior transactions any feature may consider. Must match what
            the serving path can supply; see :data:`DEFAULT_LOOKBACK`.
        min_merchant_count: Merchants seen fewer times than this in training are
            folded into the shared "rare" code. Their identity carries no
            reliable signal and a per-merchant embedding row would just memorise
            noise; the rare bucket is itself informative.
        smoothing: Target-encoding pseudo-count.
        n_blocks: Number of temporal blocks used for out-of-fold encoding.
        initial_prior: Fraud rate assumed for the first training block, fixed
            before looking at the data so that no training row's encoding depends
            on a label that arrives after it.
        max_merchant_vocab: Hard cap on the merchant vocabulary.
    """

    lookback: int = DEFAULT_LOOKBACK
    min_merchant_count: int = 3
    smoothing: float = 50.0
    n_blocks: int = 5
    initial_prior: float = 0.005
    max_merchant_vocab: int = 20000

    vocab: dict[str, dict[Any, int]] = field(default_factory=dict)
    risk_maps: dict[str, dict[Any, float]] = field(default_factory=dict)
    freq_maps: dict[str, dict[Any, float]] = field(default_factory=dict)
    prior: float = 0.0
    scaler_mean: np.ndarray | None = None
    scaler_std: np.ndarray | None = None
    spec: FeatureSpec | None = None

    # -- vocabularies ---------------------------------------------------------------

    def _fit_vocab(self, train: pd.DataFrame) -> None:
        """Build integer vocabularies from the training rows only."""
        for coded, raw in EMBEDDED_CATEGORICALS:
            counts = train[raw].value_counts()
            if raw == S.MERCHANT_ID:
                counts = counts[counts >= self.min_merchant_count]
                counts = counts.iloc[: self.max_merchant_vocab]
            self.vocab[coded] = {
                key: i + N_RESERVED_CODES for i, key in enumerate(counts.index.tolist())
            }

    def _apply_vocab(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw categorical columns to integer codes, unseen values to UNK."""
        for coded, raw in EMBEDDED_CATEGORICALS:
            mapping = self.vocab[coded]
            df[coded] = df[raw].map(mapping).fillna(UNK_INDEX).astype(np.int64)
        return df

    # -- risk encodings -------------------------------------------------------------

    def _fit_risk(self, df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
        """Attach smoothed risk encodings, out-of-fold within the training period."""
        train = df.loc[train_mask]
        y = train[S.IS_FRAUD].to_numpy()
        self.prior = float(y.mean()) if len(y) else 0.0

        # Temporal blocks inside the training window.
        ranks = train[S.TIMESTAMP].rank(method="first").to_numpy()
        block = np.floor((ranks - 1) / max(len(train) / self.n_blocks, 1)).astype(int)
        block = np.clip(block, 0, self.n_blocks - 1)

        columns = {
            "merchant_risk": S.MERCHANT_ID,
            "mcc_risk": S.MCC,
            "country_risk": S.COUNTRY,
            "device_type_risk": S.DEVICE_TYPE,
        }
        for feature, raw in columns.items():
            keys = train[raw].to_numpy()
            self.risk_maps[feature] = _full_target_map(keys, y, self.prior, self.smoothing)
            values = np.full(len(df), self.prior, dtype=np.float64)
            values[train_mask] = _temporal_oof_target_encode(
                keys, y, block, self.smoothing, self.initial_prior
            )
            other = ~train_mask
            if other.any():
                mapped = df.loc[other, raw].map(self.risk_maps[feature])
                values[other] = mapped.fillna(self.prior).to_numpy()
            df[feature] = values.astype(np.float32)

        # Frequency encoding needs no labels, so the full training counts are fine
        # everywhere; it captures "how established is this merchant".
        freq = train[S.MERCHANT_ID].value_counts().to_dict()
        self.freq_maps["log_merchant_freq"] = {k: float(v) for k, v in freq.items()}
        df["log_merchant_freq"] = (
            np.log1p(df[S.MERCHANT_ID].map(self.freq_maps["log_merchant_freq"]).fillna(0.0))
        ).astype(np.float32)
        return df

    def _apply_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach risk encodings using the fitted full-training maps (serving path)."""
        columns = {
            "merchant_risk": S.MERCHANT_ID,
            "mcc_risk": S.MCC,
            "country_risk": S.COUNTRY,
            "device_type_risk": S.DEVICE_TYPE,
        }
        for feature, raw in columns.items():
            df[feature] = (
                df[raw].map(self.risk_maps[feature]).fillna(self.prior).astype(np.float32)
            )
        df["log_merchant_freq"] = (
            np.log1p(df[S.MERCHANT_ID].map(self.freq_maps["log_merchant_freq"]).fillna(0.0))
        ).astype(np.float32)
        return df

    # -- public API -----------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
        """Build the full feature frame and fit all training-period state.

        Args:
            df: Transaction frame in time order.
            train_mask: Boolean mask of training rows, aligned to ``df``.

        Returns:
            Feature frame: the original columns plus numeric features and codes.
        """
        out = add_causal_features(df, lookback=self.lookback)
        train = out.loc[train_mask]
        self._fit_vocab(train)
        out = self._apply_vocab(out)
        out = self._fit_risk(out, train_mask)

        self.spec = FeatureSpec(
            numeric=list(NUMERIC_FEATURES),
            categorical=[
                CategoricalSpec(coded, len(self.vocab[coded]) + N_RESERVED_CODES)
                for coded, _ in EMBEDDED_CATEGORICALS
            ],
        )
        values = out.loc[train_mask, list(NUMERIC_FEATURES)].to_numpy(dtype=np.float64)
        self.scaler_mean = values.mean(axis=0)
        std = values.std(axis=0)
        self.scaler_std = np.where(std < 1e-8, 1.0, std)
        LOGGER.info(
            "features: %d numeric, %d categorical (merchant vocab %d)",
            len(NUMERIC_FEATURES),
            len(self.spec.categorical),
            self.spec.categorical[0].cardinality,
        )
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformation to new data (serving / drift replay).

        Args:
            df: Transaction frame in time order. Label columns are not required.

        Returns:
            Feature frame with the same columns ``fit_transform`` produces.

        Raises:
            RuntimeError: If called before :meth:`fit_transform`.
        """
        if not self.vocab:
            raise RuntimeError("FeatureBuilder.transform called before fit_transform")
        out = add_causal_features(df, lookback=self.lookback)
        out = self._apply_vocab(out)
        out = self._apply_risk(out)
        return out

    def transform_window(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """Feature matrices for one customer's ordered transactions, without pandas.

        This is the serving path. It calls exactly the same
        :func:`customer_causal_features` the batch pipeline calls and then applies
        the same fitted vocabularies, risk maps and scaling — the difference is
        only that it skips the DataFrame machinery, which dominates latency on a
        window of a few dozen rows. ``tests/test_serving.py`` asserts the two
        paths agree to floating-point equality.

        Args:
            records: Transactions in ascending time order, oldest first, with
                keys matching :data:`~src.data.schema.TransactionSchema`
                request fields. The last record is the one being scored. Pass the
                caller's full available history, not just the model window: rows
                near the start of the window need their own ``lookback`` of
                history behind them to reproduce the training-time features.

        Returns:
            Tuple of (scaled numeric ``(n, n_numeric)``, categorical codes
            ``(n, n_categorical)``).

        Raises:
            RuntimeError: If the builder has not been fitted.
            ValueError: If ``records`` is empty.
        """
        if self.spec is None:
            raise RuntimeError("FeatureBuilder.transform_window called before fit_transform")
        if not records:
            raise ValueError("records must not be empty")

        n = len(records)
        ts = to_epoch_seconds(
            np.array([pd.Timestamp(r[S.TIMESTAMP]).value for r in records], dtype=np.int64)
        )
        amount = np.array([float(r[S.AMOUNT]) for r in records], dtype=np.float64)
        merchant = np.array([r[S.MERCHANT_ID] for r in records], dtype=object)
        mcc = np.array([r[S.MCC] for r in records], dtype=object)
        device = np.array([r[S.DEVICE_ID] for r in records], dtype=object)
        device_type = np.array([r[S.DEVICE_TYPE] for r in records], dtype=object)
        country = np.array([r[S.COUNTRY] for r in records], dtype=object)
        online = np.array([1.0 if r[S.CHANNEL] == "online" else 0.0 for r in records])

        causal = customer_causal_features(
            ts, amount, merchant, mcc, device, device_type, country, online,
            lookback=self.lookback,
        )

        raw_lookup = {
            "merchant_risk": merchant,
            "mcc_risk": mcc,
            "country_risk": country,
            "device_type_risk": device_type,
        }
        population = {
            name: np.array(
                [float(self.risk_maps[name].get(key, self.prior)) for key in keys]
            )
            for name, keys in raw_lookup.items()
        }
        freq = self.freq_maps["log_merchant_freq"]
        population["log_merchant_freq"] = np.log1p(
            np.array([float(freq.get(key, 0.0)) for key in merchant])
        )

        numeric = np.empty((n, len(NUMERIC_FEATURES)), dtype=np.float64)
        for i, name in enumerate(NUMERIC_FEATURES):
            numeric[:, i] = causal[name] if name in causal else population[name]

        raw_columns = {
            "merchant_code": merchant,
            "mcc_code": mcc,
            "device_type_code": device_type,
            "country_code": country,
            "channel_code": np.array(
                [r[S.CHANNEL] for r in records], dtype=object
            ),
            "entry_mode_code": np.array([r[S.ENTRY_MODE] for r in records], dtype=object),
        }
        categorical = np.empty((n, len(self.spec.categorical)), dtype=np.int64)
        for j, coded in enumerate(self.spec.categorical_columns):
            mapping = self.vocab[coded]
            keys = raw_columns[coded]
            categorical[:, j] = [mapping.get(key, UNK_INDEX) for key in keys]

        return self.scale(numeric), categorical

    def save(self, path: Path) -> Path:
        """Persist the fitted state so serving uses the training-time mappings.

        Uses :mod:`pickle`, so only load files this pipeline produced.

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load(path: Path) -> FeatureBuilder:
        """Load a builder written by :meth:`save`.

        Args:
            path: File to read.

        Returns:
            The restored builder.
        """
        with path.open("rb") as handle:
            builder: FeatureBuilder = pickle.load(handle)
        return builder

    def scale(self, values: np.ndarray) -> np.ndarray:
        """Standardise a numeric feature matrix with the fitted training statistics.

        Args:
            values: Array of shape ``(..., n_numeric)``.

        Returns:
            Standardised array of the same shape, clipped to +/- 10 sigma so a
            single pathological row cannot dominate a minibatch.

        Raises:
            RuntimeError: If called before :meth:`fit_transform`.
        """
        if self.scaler_mean is None or self.scaler_std is None:
            raise RuntimeError("FeatureBuilder.scale called before fit_transform")
        scaled = (values - self.scaler_mean) / self.scaler_std
        return np.clip(scaled, -10.0, 10.0).astype(np.float32)
