"""Synthetic fraud episodes for the demo, mirroring the simulator's mechanisms.

The first version of the demo let a visitor edit the single transaction being
authorised, and the smoke test caught the problem: editing one transaction
barely moves the score. That is not a bug, it is the thesis. ``_card_testing_burst``
and ``_account_takeover`` in :mod:`src.simulate` both emit *episodes* of three to
fifteen transactions, and the features the model reads — velocity in the last
hour, merchant novelty, device novelty, amount z-score against the customer's own
history — are properties of a sequence. A model that fired on one unusual
transaction would be a model that declines every holiday.

So the presets inject an episode: several synthetic transactions appended to the
customer's real history, with the last one as the candidate. The amounts, spans,
merchant pools, device behaviour and channel all follow the pre-drift variants in
:mod:`src.simulate`, so what the visitor sees is the pattern the model was
actually trained against rather than something invented to make a demo look good.

The injected rows are clearly marked in the response and labelled as synthetic in
the page. They are fiction, and a demo that blurred that line would be worse than
no demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

#: MCCs each mechanism prefers, copied from src.simulate rather than imported:
#: the demo image ships src/, but pinning the values here means a change to the
#: simulator shows up as a demo that no longer matches, instead of silently
#: changing what the page claims the model was trained against.
CARD_TESTING_MCCS = ("5816", "4899", "5967")
ATO_MCCS = ("5732", "6051", "7995", "5311", "4511")

#: Device types over-represented in automated fraud sessions.
FRAUD_DEVICE_TYPES = ("headless_chrome", "emulator_android", "linux_curl")


@dataclass(frozen=True)
class Episode:
    """A synthetic burst appended to a customer's real history.

    Attributes:
        mechanism: ``"card_testing"`` or ``"account_takeover"``.
        injected: The synthetic prior transactions, oldest first.
        candidate: The transaction to score, the last of the burst.
        explanation: What the burst is and what the model is expected to read.
    """

    mechanism: str
    injected: list[dict[str, Any]]
    candidate: dict[str, Any]
    explanation: str

    def to_payload(self) -> dict[str, Any]:
        """JSON shape returned by ``/demo/episode``."""
        return {
            "mechanism": self.mechanism,
            "injected": self.injected,
            "candidate": self.candidate,
            "explanation": self.explanation,
        }


def _iso(moment: datetime) -> str:
    """Format a timestamp at millisecond precision.

    The dataset carries nanoseconds, which Pydantic parses happily but Safari's
    date parser rejects. Milliseconds are enough for a burst measured in minutes
    and survive the round trip through the browser.
    """
    return moment.isoformat(timespec="milliseconds")


def _parse(stamp: str) -> datetime:
    """Parse a dataset timestamp, tolerating nanosecond precision."""
    head, _, frac = stamp.partition(".")
    micros = int((frac or "0")[:6].ljust(6, "0"))
    return datetime.fromisoformat(head).replace(microsecond=micros)


def _unfamiliar_merchants(
    pool: list[dict[str, str]], history: list[dict[str, Any]], n: int, rng: np.random.Generator
) -> list[dict[str, str]]:
    """Pick ``n`` distinct merchants the customer has not used.

    Novelty is the point: ``is_new_merchant`` and ``distinct_merchants_24h`` are
    among the features the model reads, and a burst across merchants the customer
    already shops at would be testing a different hypothesis.

    Args:
        pool: Candidate merchants as ``{"merchant_id", "mcc"}`` dicts.
        history: The customer's real transactions.
        n: How many to pick.
        rng: Random generator.

    Returns:
        Up to ``n`` merchants, falling back to repeats only if the pool is small.
    """
    seen = {t["merchant_id"] for t in history}
    fresh = [m for m in pool if m["merchant_id"] not in seen]
    if len(fresh) < n:  # pragma: no cover - the pool is sized to make this unreachable
        fresh = fresh or pool
        return [fresh[int(i)] for i in rng.integers(0, len(fresh), size=n)]
    picked = rng.choice(len(fresh), size=n, replace=False)
    return [fresh[int(i)] for i in picked]


def card_testing(
    case: dict[str, Any], pool: list[dict[str, str]], seed: int = 0
) -> Episode:
    """A card-testing burst ending in the transaction to be scored.

    Follows the pre-drift variant in :mod:`src.simulate`: six authorisations
    inside about twenty minutes, amounts uniform on £0.50-£9, each at a different
    merchant the customer has never used, all from one device nobody has seen,
    online and card-not-present.

    Six rather than the simulator's five-to-fifteen because six is enough for the
    1h velocity and distinct-merchant features to move and still leaves most of
    the 32-position window showing the customer's genuine behaviour. A burst that
    filled the window would be a different and much easier problem.

    Args:
        case: A demo case.
        pool: Merchants whose MCC is in :data:`CARD_TESTING_MCCS`.
        seed: RNG seed, so a given case and preset always give the same burst.

    Returns:
        The episode.
    """
    rng = np.random.default_rng(seed)
    n = 6
    end = _parse(case["candidate"]["timestamp"])
    # Ascending gaps of a few minutes, ending at the candidate's own timestamp.
    offsets = np.sort(rng.uniform(0.5, 20.0, size=n - 1))[::-1]
    device = f"D{rng.integers(100000, 999999)}_x"
    device_type = str(rng.choice(FRAUD_DEVICE_TYPES))
    country = case["candidate"]["country"]
    merchants = _unfamiliar_merchants(pool, case["history"], n, rng)
    amounts = np.round(rng.uniform(0.5, 9.0, size=n), 2)

    rows: list[dict[str, Any]] = []
    for i in range(n):
        at = end - timedelta(minutes=float(offsets[i])) if i < n - 1 else end
        rows.append(
            {
                "txn_id": f"SYN-CT-{i + 1:02d}",
                "timestamp": _iso(at),
                "amount": float(amounts[i]),
                "merchant_id": merchants[i]["merchant_id"],
                "mcc": merchants[i]["mcc"],
                "device_id": device,
                "device_type": device_type,
                "country": country,
                "channel": "online",
                "entry_mode": "ecom",
            }
        )

    return Episode(
        mechanism="card_testing",
        injected=rows[:-1],
        candidate=rows[-1],
        explanation=(
            f"Six authorisations in about {int(offsets[0])} minutes, £"
            f"{amounts.min():.2f}-£{amounts.max():.2f}, at six merchants this "
            "customer has never used, all from one unseen device. The model is "
            "not reading any single one of these — it is reading the rate, the "
            "merchant novelty and the amounts against this customer's own median "
            f"of £{case['customer_median_amount']:.2f}."
        ),
    )


def account_takeover(
    case: dict[str, Any], pool: list[dict[str, str]], seed: int = 0
) -> Episode:
    """An account-takeover episode ending in the transaction to be scored.

    Follows the pre-drift variant: five transactions over a few hours in the small
    hours, amounts three to ten times the customer's median, high-value and
    cash-like MCCs, from a device nobody has seen, online.

    Args:
        case: A demo case.
        pool: Merchants whose MCC is in :data:`ATO_MCCS`.
        seed: RNG seed.

    Returns:
        The episode.
    """
    rng = np.random.default_rng(seed + 7)
    n = 5
    end = _parse(case["candidate"]["timestamp"])
    offsets = np.sort(rng.uniform(0.3, 6.0, size=n - 1))[::-1]
    device = f"D{rng.integers(100000, 999999)}_x"
    device_type = str(rng.choice(FRAUD_DEVICE_TYPES))
    median = float(case["customer_median_amount"])
    merchants = _unfamiliar_merchants(pool, case["history"], n, rng)
    amounts = np.round(median * rng.uniform(3.0, 10.0, size=n), 2)

    rows: list[dict[str, Any]] = []
    for i in range(n):
        at = end - timedelta(hours=float(offsets[i])) if i < n - 1 else end
        rows.append(
            {
                "txn_id": f"SYN-ATO-{i + 1:02d}",
                "timestamp": _iso(at),
                "amount": float(amounts[i]),
                "merchant_id": merchants[i]["merchant_id"],
                "mcc": merchants[i]["mcc"],
                "device_id": device,
                "device_type": device_type,
                "country": case["candidate"]["country"],
                "channel": "online",
                "entry_mode": "ecom",
            }
        )

    return Episode(
        mechanism="account_takeover",
        injected=rows[:-1],
        candidate=rows[-1],
        explanation=(
            f"Five transactions over about {offsets[0]:.1f} hours, £"
            f"{amounts.min():.2f}-£{amounts.max():.2f} against a median of £"
            f"{median:.2f}, at merchants this customer has never used, from a "
            "device nobody has seen. Note what is *not* changed: the country is "
            "the customer's own. Takeover does not need to look foreign, and a "
            "model that leaned on country would miss this."
        ),
    )


BUILDERS = {
    "card_testing": card_testing,
    "account_takeover": account_takeover,
}


def build(
    mechanism: str, case: dict[str, Any], pools: dict[str, list[dict[str, str]]], seed: int = 0
) -> Episode:
    """Build an episode by mechanism name.

    Args:
        mechanism: Key in :data:`BUILDERS`.
        case: The demo case to attach it to.
        pools: Merchant pools keyed by mechanism.
        seed: RNG seed.

    Returns:
        The episode.

    Raises:
        KeyError: If the mechanism is unknown.
    """
    return BUILDERS[mechanism](case, pools[mechanism], seed)
