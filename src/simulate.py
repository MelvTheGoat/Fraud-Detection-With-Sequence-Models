"""Synthetic transaction generator with known fraud mechanisms and adversarial drift.

Why synthetic first: the fraud mechanism is the thing we actually want to measure
against, and no public dataset tells you *which* attack produced a given label.
Here we know, so :mod:`src.evaluation` can report which mechanisms each model
catches and which it is blind to — the breakdown that aggregate PR-AUC hides.

Three mechanisms are injected:

``card_testing``
    A stolen card number is validated with a burst of tiny authorisations in
    quick succession, usually at online digital-goods merchants, from a device
    the customer has never used. Signal: velocity + amount anomaly.

``account_takeover``
    The customer's account is taken over; behaviour breaks sharply from their own
    history — unfamiliar device, out-of-pattern hours, amounts far above their
    personal distribution, merchants they have never used. Signal: deviation from
    *this customer's* baseline, not from the population's.

``merchant_compromise``
    A merchant's card data is breached. Cards that transacted there are cloned
    and used *elsewhere* days later. Signal is cross-customer and lives in the
    prior sequence: "this card recently shopped at the compromised merchant".

Realistic confounders are injected too, otherwise the problem collapses: phone
upgrades create legitimate new devices, travel creates legitimate foreign
countries and unfamiliar merchants, and a heavy amount tail creates legitimate
large purchases. Without these, "new device" alone would score near-perfectly and
the whole comparison would be meaningless.

Adversarial drift
-----------------
In the final ``drift_start_frac`` of the timeline every mechanism adapts against
the obvious defences: card testing raises amounts into the normal range and slows
its cadence; takeover mimics the victim's own amount distribution and reuses a
known device; merchant compromise fans out across many small merchants with a
longer delay. This is what makes :mod:`src.monitoring` testable — the degradation
is targeted, not random.

Usage:
    python -m src.simulate --out data/synthetic --customers 3000 --days 180
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import SYNTHETIC_DIR, SimulationConfig
from src.data.schema import FraudType, TransactionSchema

LOGGER = logging.getLogger(__name__)

S = TransactionSchema

# --------------------------------------------------------------------------------------
# Static universe definitions
# --------------------------------------------------------------------------------------

#: (mcc, label, amount multiplier, online propensity, "high risk" flag).
#: High-risk MCCs are the ones fraud disproportionately targets: resellable goods,
#: stored value and money movement.
MCC_TABLE: tuple[tuple[str, str, float, float, bool], ...] = (
    ("5411", "grocery", 1.00, 0.05, False),
    ("5812", "restaurant", 0.65, 0.10, False),
    ("5814", "fast_food", 0.25, 0.15, False),
    ("5541", "fuel", 0.90, 0.02, False),
    ("5912", "pharmacy", 0.45, 0.08, False),
    ("5311", "department_store", 1.30, 0.35, False),
    ("5691", "clothing", 1.10, 0.45, False),
    ("4121", "rideshare", 0.35, 0.95, False),
    ("5732", "electronics", 3.20, 0.55, True),
    ("5999", "misc_retail", 0.80, 0.40, False),
    ("4899", "streaming", 0.20, 1.00, False),
    ("7995", "gambling", 1.60, 0.90, True),
    ("5967", "direct_marketing", 0.90, 1.00, True),
    ("6051", "quasi_cash", 4.00, 0.75, True),
    ("5816", "digital_goods", 0.30, 1.00, True),
    ("4111", "transit", 0.30, 0.30, False),
    ("7011", "hotel", 2.60, 0.70, False),
    ("4511", "airline", 3.80, 0.85, False),
    ("5399", "warehouse_club", 1.70, 0.20, False),
    ("8062", "medical", 1.90, 0.15, False),
)

COUNTRIES: tuple[str, ...] = ("GB", "US", "FR", "DE", "ES", "NL", "IE", "PL")
#: Home-country weights; most cardholders in the portfolio are domestic.
COUNTRY_WEIGHTS: tuple[float, ...] = (0.55, 0.14, 0.07, 0.07, 0.06, 0.04, 0.04, 0.03)

CHANNELS: tuple[str, ...] = ("in_store", "online")
ENTRY_MODES: tuple[str, ...] = ("chip", "contactless", "ecom", "magstripe", "manual")

#: Device *types* are shared across customers, unlike device IDs which are
#: effectively customer-unique. The embedding lives on this column; the raw
#: device ID is used only relationally (novelty, recency).
DEVICE_TYPES: tuple[str, ...] = (
    "android_chrome",
    "ios_safari",
    "windows_chrome",
    "ios_app",
    "android_app",
    "macos_safari",
    "windows_edge",
    "windows_firefox",
    "android_samsung",
    "linux_firefox",
    "other_mobile",
    "other_desktop",
    "smart_tv",
    "console",
    "headless_client",
    "emulator",
)
#: Consumer popularity of each device type (power law, head-heavy).
DEVICE_TYPE_WEIGHTS: tuple[float, ...] = (
    0.22, 0.19, 0.15, 0.09, 0.08, 0.07, 0.05, 0.04,
    0.035, 0.02, 0.015, 0.012, 0.008, 0.005, 0.003, 0.002,
)
#: Device types an automated fraud stack over-uses before it learns to blend in.
FRAUD_DEVICE_TYPES: tuple[str, ...] = (
    "headless_client",
    "emulator",
    "other_desktop",
    "linux_firefox",
    "windows_chrome",
)
FRAUD_DEVICE_WEIGHTS: tuple[float, ...] = (0.30, 0.24, 0.20, 0.16, 0.10)

POS_DEVICE_TYPE = "pos_terminal"


def _normalised(weights: tuple[float, ...]) -> np.ndarray:
    """Return the weights as a probability vector summing to exactly 1."""
    arr = np.asarray(weights, dtype=np.float64)
    return arr / arr.sum()


_DEVICE_TYPE_P = _normalised(DEVICE_TYPE_WEIGHTS)
_FRAUD_DEVICE_P = _normalised(FRAUD_DEVICE_WEIGHTS)

#: MCCs a card-testing burst prefers: cheap, instant, digital.
CARD_TESTING_MCCS: tuple[str, ...] = ("5816", "4899", "5967")
#: MCCs a takeover prefers: high value, resellable, or cash-like.
ATO_MCCS: tuple[str, ...] = ("5732", "6051", "7995", "5311", "4511")


# --------------------------------------------------------------------------------------
# Universe construction
# --------------------------------------------------------------------------------------


def _power_law_weights(n: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    """Return normalised Zipf-like popularity weights with the ordering shuffled.

    Args:
        n: Number of entities.
        exponent: Zipf exponent; higher concentrates mass on the head.
        rng: Random generator.

    Returns:
        Array of shape ``(n,)`` summing to 1.
    """
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = ranks ** (-exponent)
    rng.shuffle(weights)
    return weights / weights.sum()


def _build_merchants(cfg: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Build the merchant universe with power-law popularity.

    Merchant cardinality is the point of the exercise: a handful of merchants
    carry most volume and a long tail is seen a handful of times each. That is
    exactly the regime where a learned embedding has something to offer over
    one-hot (too wide) or target encoding (too noisy in the tail).

    Args:
        cfg: Simulation configuration.
        rng: Random generator.

    Returns:
        DataFrame indexed 0..n_merchants-1 with mcc, popularity and price level.
    """
    n = cfg.n_merchants
    mcc_codes = np.array([row[0] for row in MCC_TABLE])
    mcc_mult = np.array([row[2] for row in MCC_TABLE])
    mcc_online = np.array([row[3] for row in MCC_TABLE])

    # MCC popularity is itself skewed: far more restaurants than airlines.
    mcc_pop = _power_law_weights(len(MCC_TABLE), 0.6, rng)
    mcc_idx = rng.choice(len(MCC_TABLE), size=n, p=mcc_pop)

    return pd.DataFrame(
        {
            "merchant_id": [f"M{i:05d}" for i in range(n)],
            "mcc": mcc_codes[mcc_idx],
            "amount_mult": mcc_mult[mcc_idx] * rng.lognormal(0.0, 0.35, size=n),
            "online_p": np.clip(mcc_online[mcc_idx] + rng.normal(0, 0.05, size=n), 0.0, 1.0),
            "popularity": _power_law_weights(n, 0.95, rng),
        }
    )


def _build_customers(cfg: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Build the customer universe with per-customer spending personalities.

    Args:
        cfg: Simulation configuration.
        rng: Random generator.

    Returns:
        DataFrame of customer attributes.
    """
    n = cfg.n_customers
    # Log-normal amount personality: median spend spans roughly 8x across the book.
    log_mu = rng.normal(2.9, 0.55, size=n)
    log_sigma = rng.uniform(0.45, 1.05, size=n)
    # Transaction rate per day, gamma so a minority are very active.
    rate = rng.gamma(shape=2.2, scale=0.55, size=n) + 0.20
    home = rng.choice(len(COUNTRIES), size=n, p=_normalised(COUNTRY_WEIGHTS))

    # Real people use several devices — a phone, a laptop, a tablet, a work
    # machine — with lopsided frequencies, and occasionally something entirely
    # new. Giving each customer a single device would make "unseen device" an
    # almost perfect fraud rule and the whole comparison would be about that one
    # feature.
    device_ids: list[list[str]] = []
    device_types: list[list[str]] = []
    device_weights: list[np.ndarray] = []
    for i in range(n):
        k = int(rng.integers(2, 5))
        device_ids.append([f"D{i:06d}_{j}" for j in range(k)])
        device_types.append(
            list(rng.choice(np.array(DEVICE_TYPES), size=k, p=_DEVICE_TYPE_P))
        )
        w = rng.dirichlet(np.full(k, 0.7))
        device_weights.append(w)

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "log_mu": log_mu,
            "log_sigma": log_sigma,
            "rate": rate,
            "home_country": np.array(COUNTRIES)[home],
            "peak_hour": rng.uniform(10.0, 20.0, size=n),
            "night_owl": rng.random(size=n) < 0.12,
            # Card token used for card-present transactions.
            "device_card": [f"D{i:06d}c" for i in range(n)],
            "device_ids": device_ids,
            "device_types": device_types,
            "device_weights": device_weights,
            # A phone upgrade mid-period retires the primary device and
            # introduces a legitimate new one.
            "upgrade_day": np.where(
                rng.random(size=n) < 0.20, rng.uniform(0.1, 0.95, size=n) * cfg.n_days, np.inf
            ),
            "upgrade_device": [f"D{i:06d}_up" for i in range(n)],
            "upgrade_device_type": rng.choice(np.array(DEVICE_TYPES), size=n, p=_DEVICE_TYPE_P),
            # Some customers shop in short bursts of small purchases — in-app
            # top-ups, a transit run, a round of coffees. This is the legitimate
            # behaviour card testing hides behind.
            "bursty": rng.random(size=n) < 0.10,
        }
    )


def _hour_intensity(hours: np.ndarray, peak: float, night_owl: bool) -> np.ndarray:
    """Circadian acceptance probability for candidate transaction times.

    Args:
        hours: Hour-of-day values in ``[0, 24)``.
        peak: Customer's peak spending hour.
        night_owl: Whether the customer also transacts late.

    Returns:
        Acceptance probabilities in ``(0, 1]``.
    """
    delta = np.minimum(np.abs(hours - peak), 24.0 - np.abs(hours - peak))
    base = np.exp(-0.5 * (delta / 4.0) ** 2)
    floor = 0.18 if night_owl else 0.04
    return np.clip(base, floor, 1.0)


# --------------------------------------------------------------------------------------
# Legitimate traffic
# --------------------------------------------------------------------------------------


def _simulate_customer(
    cust: pd.Series,
    merchants: pd.DataFrame,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate one customer's legitimate transactions.

    Times come from a thinned Poisson process: candidates are drawn uniformly then
    accepted with a circadian/weekly intensity, which yields realistic clustering
    without an expensive point-process simulation.

    Args:
        cust: Row of the customer table.
        merchants: Merchant universe.
        cfg: Simulation configuration.
        rng: Random generator.

    Returns:
        Dict of column arrays for this customer.
    """
    n_days = cfg.n_days
    n_candidates = rng.poisson(cust["rate"] * n_days * 2.0)
    if n_candidates < 3:
        n_candidates = 3
    t_days = np.sort(rng.uniform(0.0, n_days, size=n_candidates))

    hours = (t_days % 1.0) * 24.0
    dow = (np.floor(t_days).astype(int)) % 7
    weekend_boost = np.where(dow >= 5, 1.25, 1.0)
    accept = _hour_intensity(hours, float(cust["peak_hour"]), bool(cust["night_owl"]))
    accept = np.clip(accept * weekend_boost * 0.5, 0.0, 1.0)
    keep = rng.random(size=n_candidates) < accept
    t_days = t_days[keep]
    if t_days.size < 2:
        t_days = np.sort(rng.uniform(0.0, n_days, size=2))

    # Legitimate bursts: a run of small purchases inside an hour. Card testing
    # looks exactly like this, which is the point — without it, "several small
    # transactions in a few minutes" would be a perfect fraud rule.
    burst_flags = np.zeros(t_days.size, dtype=bool)
    if bool(cust["bursty"]) and t_days.size > 5:
        for _ in range(int(rng.integers(1, 6))):
            anchor = float(rng.choice(t_days))
            k = int(rng.integers(3, 9))
            extra = anchor + np.sort(rng.uniform(0.0, rng.uniform(0.2, 1.2) / 24.0, size=k))
            t_days = np.concatenate([t_days, extra])
            burst_flags = np.concatenate([burst_flags, np.ones(k, dtype=bool)])
        order = np.argsort(t_days)
        t_days, burst_flags = t_days[order], burst_flags[order]
        t_days = np.clip(t_days, 0.0, n_days - 1e-6)
    n = t_days.size

    # Merchant affinity: a personal repertoire plus occasional exploration.
    n_fav = int(rng.integers(4, 14))
    fav = rng.choice(len(merchants), size=n_fav, p=merchants["popularity"].to_numpy())
    fav_w = _power_law_weights(n_fav, 0.7, rng)
    from_fav = rng.random(size=n) < 0.62
    m_idx = np.where(
        from_fav,
        rng.choice(fav, size=n, p=fav_w),
        rng.choice(len(merchants), size=n, p=merchants["popularity"].to_numpy()),
    )

    amt_mult = merchants["amount_mult"].to_numpy()[m_idx]
    base = np.exp(cust["log_mu"] + cust["log_sigma"] * rng.normal(size=n))
    amount = base * amt_mult
    # Legitimate heavy tail: ~2% of purchases are large one-offs. Without this,
    # "large amount" would be an almost perfect fraud rule.
    big = rng.random(size=n) < 0.02
    amount = np.where(big, amount * rng.uniform(6.0, 28.0, size=n), amount)
    # Burst purchases are small by nature.
    amount = np.where(burst_flags, rng.uniform(0.8, 9.0, size=n), amount)
    amount = np.clip(np.round(amount, 2), 0.5, 20000.0)

    online = rng.random(size=n) < merchants["online_p"].to_numpy()[m_idx]
    channel = np.where(online, "online", "in_store")
    entry = np.where(
        online,
        "ecom",
        rng.choice(["chip", "contactless", "magstripe"], size=n, p=[0.45, 0.5, 0.05]),
    )

    # Device: card token in store; one of the customer's own devices online. A
    # phone upgrade retires the primary device, and ~5% of online transactions
    # come from something never seen before (a new browser, a borrowed laptop,
    # a work machine). Both keep "unseen device" from being a giveaway.
    pool_ids = np.array(cust["device_ids"], dtype=object)
    pool_types = np.array(cust["device_types"], dtype=object)
    pool_w = np.asarray(cust["device_weights"], dtype=np.float64)
    picks = rng.choice(len(pool_ids), size=n, p=pool_w / pool_w.sum())

    upgraded = t_days >= cust["upgrade_day"]
    online_id = np.where(
        upgraded & (picks == 0), cust["upgrade_device"], pool_ids[picks]
    )
    online_type = np.where(
        upgraded & (picks == 0), cust["upgrade_device_type"], pool_types[picks]
    )
    one_off = rng.random(size=n) < 0.05
    if one_off.any():
        k = int(one_off.sum())
        online_id[one_off] = [f"D{rng.integers(0, 10**9):09d}o" for _ in range(k)]
        online_type[one_off] = rng.choice(np.array(DEVICE_TYPES), size=k, p=_DEVICE_TYPE_P)

    device = np.where(online, online_id, cust["device_card"])
    device_type = np.where(online, online_type, POS_DEVICE_TYPE)

    country = np.full(n, cust["home_country"], dtype=object)
    # Legitimate travel: 25% of customers take 1-2 trips with foreign spend and
    # unfamiliar merchants. This is the main confounder for geo/merchant novelty.
    if rng.random() < 0.25:
        for _ in range(int(rng.integers(1, 3))):
            start = rng.uniform(0, max(n_days - 10, 1))
            span = rng.uniform(3.0, 10.0)
            on_trip = (t_days >= start) & (t_days < start + span)
            if not on_trip.any():
                continue
            dest = rng.choice([c for c in COUNTRIES if c != cust["home_country"]])
            country[on_trip] = dest
            k = int(on_trip.sum())
            m_idx[on_trip] = rng.choice(
                len(merchants), size=k, p=merchants["popularity"].to_numpy()
            )
            amount[on_trip] = np.clip(np.round(amount[on_trip] * rng.uniform(1.1, 2.2), 2), 0.5, 20000.0)

    return {
        S.CUSTOMER_ID: np.full(n, cust["customer_id"], dtype=object),
        "t_days": t_days,
        S.AMOUNT: amount,
        S.MERCHANT_ID: merchants["merchant_id"].to_numpy()[m_idx],
        S.MCC: merchants["mcc"].to_numpy()[m_idx],
        S.DEVICE_ID: device.astype(object),
        S.DEVICE_TYPE: device_type.astype(object),
        S.COUNTRY: country,
        S.CHANNEL: channel.astype(object),
        S.ENTRY_MODE: entry.astype(object),
        S.IS_FRAUD: np.zeros(n, dtype=np.int8),
        S.FRAUD_TYPE: np.full(n, FraudType.NONE.value, dtype=object),
    }


# --------------------------------------------------------------------------------------
# Fraud mechanisms
# --------------------------------------------------------------------------------------


def _fresh_device(rng: np.random.Generator) -> str:
    """Return a device token that no legitimate customer owns."""
    return f"DF{rng.integers(0, 10**9):09d}"


def _fraud_device_type(drifted: bool, rng: np.random.Generator) -> str:
    """Pick a device type for a fraudulent session.

    Before drift the attacker's automation leaves a fingerprint: headless clients
    and emulators are over-represented, but only in about a third of sessions —
    the rest already look like ordinary consumer devices. Making the fingerprint
    deterministic would hand every model a single perfect feature and there would
    be nothing left to compare. After drift the stack spoofs consumer
    fingerprints throughout and device type stops carrying signal at all.

    Args:
        drifted: Whether the adversarial variant is used.
        rng: Random generator.

    Returns:
        A device-type string.
    """
    if drifted or rng.random() > 0.35:
        return str(rng.choice(np.array(DEVICE_TYPES), p=_DEVICE_TYPE_P))
    return str(rng.choice(np.array(FRAUD_DEVICE_TYPES), p=_FRAUD_DEVICE_P))


def _attacker_device(cust: pd.Series, rng: np.random.Generator, reuse_p: float) -> str:
    """Device token for a fraudulent session.

    With probability ``reuse_p`` the attacker is operating from a device the
    customer has genuinely used (malware on the victim's own machine, a hijacked
    session), otherwise from a device nobody has seen.

    Args:
        cust: Victim customer row.
        rng: Random generator.
        reuse_p: Probability of reusing one of the victim's devices.

    Returns:
        A device token.
    """
    if rng.random() < reuse_p:
        return str(rng.choice(np.array(cust["device_ids"], dtype=object)))
    return _fresh_device(rng)


def _mcc_merchant_pool(merchants: pd.DataFrame, mccs: tuple[str, ...]) -> np.ndarray:
    """Indices of merchants whose MCC is in ``mccs``."""
    mask = merchants["mcc"].isin(mccs).to_numpy()
    idx = np.flatnonzero(mask)
    return idx if idx.size else np.arange(len(merchants))


def _card_testing_burst(
    cust: pd.Series,
    merchants: pd.DataFrame,
    start_day: float,
    drifted: bool,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate one card-testing burst.

    Pre-drift the burst is the textbook pattern: many sub-£5 authorisations inside
    a few minutes. Post-drift the adversary has read the same rulebook — amounts
    move into the customer's normal range and the burst is stretched over hours,
    which defeats a fixed velocity-plus-small-amount rule.

    Args:
        cust: Victim customer row.
        merchants: Merchant universe.
        start_day: Burst start time in days since the simulation epoch.
        drifted: Whether the adversarial variant is used.
        rng: Random generator.

    Returns:
        Dict of column arrays.
    """
    pool = _mcc_merchant_pool(merchants, CARD_TESTING_MCCS)
    if drifted:
        n = int(rng.integers(4, 9))
        span_days = rng.uniform(3.0, 8.0) / 24.0
        amount = np.round(
            np.exp(cust["log_mu"] + cust["log_sigma"] * rng.normal(size=n)) * rng.uniform(0.6, 1.4),
            2,
        )
        # Blends in with ordinary retail rather than sitting in digital goods.
        pool = np.arange(len(merchants))
    else:
        n = int(rng.integers(5, 15))
        span_days = rng.uniform(4.0, 60.0) / (24.0 * 60.0)
        amount = np.round(rng.uniform(0.5, 9.0, size=n), 2)

    offsets = np.sort(rng.uniform(0.0, span_days, size=n))
    m_idx = rng.choice(pool, size=n, p=_power_law_weights(pool.size, 0.5, rng))
    device = _attacker_device(cust, rng, 0.15)
    country = rng.choice(COUNTRIES) if rng.random() < 0.6 else cust["home_country"]

    return {
        S.CUSTOMER_ID: np.full(n, cust["customer_id"], dtype=object),
        "t_days": start_day + offsets,
        S.AMOUNT: np.clip(amount, 0.5, 20000.0),
        S.MERCHANT_ID: merchants["merchant_id"].to_numpy()[m_idx],
        S.MCC: merchants["mcc"].to_numpy()[m_idx],
        S.DEVICE_ID: np.full(n, device, dtype=object),
        S.DEVICE_TYPE: np.full(n, _fraud_device_type(drifted, rng), dtype=object),
        S.COUNTRY: np.full(n, country, dtype=object),
        S.CHANNEL: np.full(n, "online", dtype=object),
        S.ENTRY_MODE: np.full(n, "ecom", dtype=object),
        S.IS_FRAUD: np.ones(n, dtype=np.int8),
        S.FRAUD_TYPE: np.full(n, FraudType.CARD_TESTING.value, dtype=object),
    }


def _account_takeover(
    cust: pd.Series,
    merchants: pd.DataFrame,
    start_day: float,
    drifted: bool,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate an account-takeover episode.

    Pre-drift this is a hard behavioural break: unknown device, night hours,
    amounts 5-20x the victim's median, high-risk MCCs. Post-drift the attacker
    mimics — amounts are drawn from the victim's own distribution and the session
    reuses a device the victim has actually used, so the break is visible only in
    merchant/MCC novelty and timing.

    Args:
        cust: Victim customer row.
        merchants: Merchant universe.
        start_day: Takeover start time in days.
        drifted: Whether the adversarial variant is used.
        rng: Random generator.

    Returns:
        Dict of column arrays.
    """
    n = int(rng.integers(3, 13))
    span_days = rng.uniform(2.0, 20.0) / 24.0
    offsets = np.sort(rng.uniform(0.0, span_days, size=n))
    pool = _mcc_merchant_pool(merchants, ATO_MCCS)

    median_amt = float(np.exp(cust["log_mu"]))
    if drifted:
        amount = np.round(
            np.exp(cust["log_mu"] + cust["log_sigma"] * rng.normal(size=n)) * rng.uniform(0.9, 2.0),
            2,
        )
        # Session hijack: a device the victim has genuinely used.
        idx = int(rng.integers(0, len(cust["device_ids"])))
        device = str(cust["device_ids"][idx])
        device_type = str(cust["device_types"][idx])
        pool = np.arange(len(merchants))
    else:
        amount = np.round(median_amt * rng.uniform(3.0, 10.0, size=n), 2)
        device = _attacker_device(cust, rng, 0.25)
        device_type = _fraud_device_type(drifted, rng)

    m_idx = rng.choice(pool, size=n, p=_power_law_weights(pool.size, 0.5, rng))
    # Out-of-pattern hours: shift the episode towards the small hours.
    hour_shift = (start_day % 1.0)
    t = np.floor(start_day) + ((hour_shift + rng.uniform(0.05, 0.20)) % 1.0) + offsets

    return {
        S.CUSTOMER_ID: np.full(n, cust["customer_id"], dtype=object),
        "t_days": t,
        S.AMOUNT: np.clip(amount, 0.5, 20000.0),
        S.MERCHANT_ID: merchants["merchant_id"].to_numpy()[m_idx],
        S.MCC: merchants["mcc"].to_numpy()[m_idx],
        S.DEVICE_ID: np.full(n, device, dtype=object),
        S.DEVICE_TYPE: np.full(n, device_type, dtype=object),
        S.COUNTRY: np.full(n, cust["home_country"], dtype=object),
        S.CHANNEL: np.full(n, "online", dtype=object),
        S.ENTRY_MODE: np.full(n, "ecom", dtype=object),
        S.IS_FRAUD: np.ones(n, dtype=np.int8),
        S.FRAUD_TYPE: np.full(n, FraudType.ACCOUNT_TAKEOVER.value, dtype=object),
    }


def _merchant_compromise(
    victims: pd.DataFrame,
    merchants: pd.DataFrame,
    exposure_days: np.ndarray,
    drifted: bool,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate cloned-card fraud following a merchant data breach.

    The compromised merchant is *not* where the fraud happens — the breach happens
    there and the cloned cards are used elsewhere days later. The only trace in
    the victim's own history is that they shopped at the breached merchant
    recently, which is precisely the kind of cross-transaction signal a sequence
    model can pick up and a per-transaction model cannot.

    Post-drift the delay lengthens and the fraud fans out across long-tail
    merchants, which erodes any merchant-risk-rate feature.

    Args:
        victims: Customers exposed at the breached merchant.
        merchants: Merchant universe.
        exposure_days: Time (in days) at which each victim was exposed.
        drifted: Whether the adversarial variant is used.
        rng: Random generator.

    Returns:
        Dict of column arrays.
    """
    rows: list[dict[str, np.ndarray]] = []
    if drifted:
        pool = np.argsort(merchants["popularity"].to_numpy())[: max(len(merchants) // 2, 1)]
        delay_range = (10.0, 30.0)
    else:
        pool = _mcc_merchant_pool(merchants, ("5732", "5311", "5999", "5691"))
        delay_range = (1.0, 14.0)

    for (_, cust), exposed_at in zip(victims.iterrows(), exposure_days, strict=True):
        n = int(rng.integers(1, 5))
        delay = rng.uniform(*delay_range)
        offsets = np.sort(rng.uniform(0.0, 0.5, size=n))
        m_idx = rng.choice(pool, size=n)
        amount = np.round(
            np.exp(cust["log_mu"]) * rng.uniform(1.0, 4.5, size=n),
            2,
        )
        rows.append(
            {
                S.CUSTOMER_ID: np.full(n, cust["customer_id"], dtype=object),
                "t_days": exposed_at + delay + offsets,
                S.AMOUNT: np.clip(amount, 0.5, 20000.0),
                S.MERCHANT_ID: merchants["merchant_id"].to_numpy()[m_idx],
                S.MCC: merchants["mcc"].to_numpy()[m_idx],
                S.DEVICE_ID: np.full(n, _attacker_device(cust, rng, 0.20), dtype=object),
                S.DEVICE_TYPE: np.full(n, _fraud_device_type(drifted, rng), dtype=object),
                S.COUNTRY: np.full(n, cust["home_country"], dtype=object),
                S.CHANNEL: np.full(n, "online", dtype=object),
                S.ENTRY_MODE: np.full(n, "ecom", dtype=object),
                S.IS_FRAUD: np.ones(n, dtype=np.int8),
                S.FRAUD_TYPE: np.full(n, FraudType.MERCHANT_COMPROMISE.value, dtype=object),
            }
        )
    if not rows:
        return {}
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def _concat(chunks: list[dict[str, np.ndarray]]) -> pd.DataFrame:
    """Concatenate per-episode column dicts into a DataFrame."""
    chunks = [c for c in chunks if c]
    if not chunks:
        return pd.DataFrame()
    keys = list(chunks[0].keys())
    return pd.DataFrame({k: np.concatenate([c[k] for c in chunks]) for k in keys})


def simulate(cfg: SimulationConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate the full synthetic dataset.

    Args:
        cfg: Simulation configuration; defaults are CPU-sized (~300k transactions).

    Returns:
        Tuple of (transactions sorted by timestamp, metadata dict). Metadata
        records the drift boundary and per-mechanism counts so downstream code
        never has to hard-code them.
    """
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)

    merchants = _build_merchants(cfg, rng)
    customers = _build_customers(cfg, rng)
    LOGGER.info("universe: %d customers, %d merchants", len(customers), len(merchants))

    legit_chunks = [
        _simulate_customer(customers.iloc[i], merchants, cfg, rng) for i in range(len(customers))
    ]
    legit = _concat(legit_chunks)
    n_legit = len(legit)
    LOGGER.info("legitimate transactions: %d", n_legit)

    drift_start_day = cfg.drift_start_frac * cfg.n_days
    target_fraud = int(round(cfg.base_fraud_rate * n_legit / (1 - cfg.base_fraud_rate)))

    fraud_chunks: list[dict[str, np.ndarray]] = []

    # --- card testing -------------------------------------------------------------
    target_ct = int(target_fraud * cfg.card_testing_share)
    produced = 0
    while produced < target_ct:
        cust = customers.iloc[int(rng.integers(0, len(customers)))]
        start_day = float(rng.uniform(2.0, cfg.n_days - 1.0))
        drifted = start_day >= drift_start_day
        chunk = _card_testing_burst(cust, merchants, start_day, drifted, rng)
        fraud_chunks.append(chunk)
        produced += len(chunk["t_days"])

    # --- account takeover ---------------------------------------------------------
    target_ato = int(target_fraud * cfg.ato_share)
    produced = 0
    while produced < target_ato:
        cust = customers.iloc[int(rng.integers(0, len(customers)))]
        start_day = float(rng.uniform(5.0, cfg.n_days - 1.0))
        drifted = start_day >= drift_start_day
        chunk = _account_takeover(cust, merchants, start_day, drifted, rng)
        fraud_chunks.append(chunk)
        produced += len(chunk["t_days"])

    # --- merchant compromise ------------------------------------------------------
    target_mc = int(target_fraud * cfg.merchant_compromise_share)
    produced = 0
    compromised_merchants: list[dict[str, Any]] = []
    # Mid-popularity merchants make the most interesting breaches: enough exposed
    # cards to matter, not so many that the pattern is trivial.
    pop_rank = np.argsort(-merchants["popularity"].to_numpy())
    candidate_merchants = pop_rank[int(0.02 * len(merchants)) : int(0.25 * len(merchants))]
    legit_by_merchant = legit.groupby(S.MERCHANT_ID).indices
    cust_index = customers.set_index("customer_id")

    guard = 0
    while produced < target_mc and guard < 500:
        guard += 1
        m_i = int(rng.choice(candidate_merchants))
        m_id = merchants["merchant_id"].iloc[m_i]
        idx = legit_by_merchant.get(m_id)
        if idx is None or len(idx) < 12:
            continue
        window_start = float(rng.uniform(1.0, max(cfg.n_days - 25.0, 2.0)))
        window_end = window_start + float(rng.uniform(3.0, 12.0))
        sub = legit.iloc[idx]
        in_window = sub[(sub["t_days"] >= window_start) & (sub["t_days"] < window_end)]
        if in_window.empty:
            continue
        exposed = in_window.drop_duplicates(subset=[S.CUSTOMER_ID])
        take = min(len(exposed), int(rng.integers(4, 25)))
        exposed = exposed.sample(n=take, random_state=int(rng.integers(0, 2**31)))
        victims = cust_index.loc[exposed[S.CUSTOMER_ID].to_numpy()].reset_index()
        drifted = window_end >= drift_start_day
        chunk = _merchant_compromise(
            victims, merchants, exposed["t_days"].to_numpy(), drifted, rng
        )
        if not chunk:
            continue
        fraud_chunks.append(chunk)
        produced += len(chunk["t_days"])
        compromised_merchants.append(
            {
                "merchant_id": str(m_id),
                "window_start_day": window_start,
                "window_end_day": window_end,
                "n_exposed_customers": int(len(victims)),
                "drifted": bool(drifted),
            }
        )

    fraud = _concat(fraud_chunks)
    LOGGER.info("fraudulent transactions: %d", len(fraud))

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df[(df["t_days"] >= 0.0) & (df["t_days"] < cfg.n_days)].copy()

    # Unreported fraud. Some victims never notice or never dispute, so the label
    # says "legitimate" while the behaviour was fraudulent. This is not noise
    # added for its own sake: it is the reason real fraud models plateau well
    # short of perfect separation, and without it the synthetic problem is
    # solvable to PR-AUC 1.0 and the architecture comparison measures nothing.
    df = df.reset_index(drop=True)
    fraud_rows = np.flatnonzero(df[S.IS_FRAUD].to_numpy() == 1)
    n_unreported = int(round(cfg.unreported_fraud_rate * fraud_rows.size))
    if n_unreported:
        hidden = np.zeros(len(df), dtype=bool)
        hidden[rng.choice(fraud_rows, size=n_unreported, replace=False)] = True
        df.loc[hidden, S.IS_FRAUD] = np.int8(0)
        df.loc[hidden, S.FRAUD_TYPE] = FraudType.NONE.value

    start = pd.Timestamp(cfg.start_date)
    df[S.TIMESTAMP] = start + pd.to_timedelta(df["t_days"] * 86400.0, unit="s")
    df = df.sort_values([S.TIMESTAMP, S.CUSTOMER_ID], kind="mergesort").reset_index(drop=True)
    df[S.TXN_ID] = [f"T{i:08d}" for i in range(len(df))]
    df = df.drop(columns=["t_days"])
    df = df[list(S.ALL)]

    drift_start_ts = start + pd.Timedelta(seconds=drift_start_day * 86400.0)
    metadata: dict[str, Any] = {
        "source": "synthetic",
        "config": asdict(cfg),
        "n_transactions": int(len(df)),
        "n_fraud": int(df[S.IS_FRAUD].sum()),
        "fraud_rate": float(df[S.IS_FRAUD].mean()),
        "start": str(df[S.TIMESTAMP].min()),
        "end": str(df[S.TIMESTAMP].max()),
        "drift_start": str(drift_start_ts),
        "fraud_type_counts": df[S.FRAUD_TYPE].value_counts().to_dict(),
        "n_customers": int(df[S.CUSTOMER_ID].nunique()),
        "n_merchants": int(df[S.MERCHANT_ID].nunique()),
        "n_devices": int(df[S.DEVICE_ID].nunique()),
        "compromised_merchants": compromised_merchants,
    }
    return df, metadata


def write_dataset(df: pd.DataFrame, metadata: dict[str, Any], out_dir: Path) -> Path:
    """Write transactions and metadata to ``out_dir``.

    Args:
        df: Transaction table.
        metadata: Metadata dict from :func:`simulate`.
        out_dir: Destination directory.

    Returns:
        Path to the written parquet file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "transactions.parquet"
    df.to_parquet(path, index=False)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic fraud data.")
    parser.add_argument("--out", type=Path, default=SYNTHETIC_DIR, help="Output directory.")
    parser.add_argument("--customers", type=int, default=None, help="Number of customers.")
    parser.add_argument("--days", type=int, default=None, help="Days to simulate.")
    parser.add_argument("--merchants", type=int, default=None, help="Merchant universe size.")
    parser.add_argument("--fraud-rate", type=float, default=None, help="Target fraud rate.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    base = SimulationConfig()
    cfg = SimulationConfig(
        n_customers=args.customers if args.customers is not None else base.n_customers,
        n_days=args.days if args.days is not None else base.n_days,
        n_merchants=args.merchants if args.merchants is not None else base.n_merchants,
        base_fraud_rate=args.fraud_rate if args.fraud_rate is not None else base.base_fraud_rate,
        seed=args.seed if args.seed is not None else base.seed,
    )
    df, metadata = simulate(cfg)
    path = write_dataset(df, metadata, args.out)
    LOGGER.info(
        "wrote %d transactions (%.3f%% fraud) to %s",
        metadata["n_transactions"],
        100 * metadata["fraud_rate"],
        path,
    )
    for k, v in metadata["fraud_type_counts"].items():
        LOGGER.info("  %-22s %d", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
