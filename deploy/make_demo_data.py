"""Build the demo case file served by the public demo.

The scoring API needs ``max_len - 1 + lookback`` prior transactions to reproduce
its training-time inputs exactly, which is 62 for the default configuration. That
is more than anyone will type into a form, so the demo ships a pool of real
customer histories taken from the *test* period of the simulated dataset — the
part the model never trained on — and lets a visitor edit only the transaction
being authorised.

Ground-truth labels are carried in the case file so the UI can reveal them after
scoring, but they are never part of a scoring request: the demo posts to the same
``/score`` endpoint a production caller would use, and that schema has no
``is_fraud`` field on the candidate or on any prior transaction.

Usage::

    python -m deploy.make_demo_data --data data/synthetic --out deploy/demo_data.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import SYNTHETIC_DIR, SequenceConfig, SplitConfig
from src.data.features import DEFAULT_LOOKBACK
from src.data.schema import FraudType, TransactionSchema
from src.data.splits import make_temporal_split

LOGGER = logging.getLogger(__name__)
S = TransactionSchema

#: How many prior transactions each case carries. One short of the full parity
#: depth is pointless, and more than the depth is ignored by the service, so this
#: is exactly ``max_len - 1 + lookback``.
HISTORY_DEPTH = SequenceConfig().max_len - 1 + DEFAULT_LOOKBACK

#: Fraud cases per mechanism, and legitimate cases overall. Fraud is 0.65% of the
#: data, so an unweighted sample would be ~20 legitimate cases and no fraud at
#: all; the demo oversamples deliberately and says so in the UI.
PER_MECHANISM = 3
N_LEGITIMATE = 6

MECHANISM_NOTES = {
    FraudType.CARD_TESTING.value: (
        "Card testing. A compromised card number is probed with small amounts at "
        "unfamiliar online merchants in quick succession. The giveaway is the "
        "rate and the novelty of the merchants, not any single amount."
    ),
    FraudType.ACCOUNT_TAKEOVER.value: (
        "Account takeover. Someone else is holding the credentials: a new device, "
        "often a new country, and amounts well above this customer's own history."
    ),
    FraudType.MERCHANT_COMPROMISE.value: (
        "Merchant compromise. The card details leaked from a merchant this "
        "customer legitimately used, and are now being spent elsewhere. The "
        "hardest of the three, because the customer's own behaviour looks normal "
        "right up to the fraudulent transaction."
    ),
}

LEGITIMATE_NOTE = (
    "Ordinary spending. Nothing here should be flagged. If you push the amount "
    "far enough above this customer's own history, or switch the device, you "
    "should be able to make the score climb — that is the model working, not a "
    "false positive."
)


def _rows_to_requests(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert transaction rows to the API's ``Transaction`` shape.

    Only ``REQUEST_FIELDS`` are emitted, minus ``customer_id`` which lives on the
    request rather than on each transaction. Dropping the label columns here, at
    the point of serialisation, is what guarantees the demo cannot accidentally
    post a label.

    Args:
        frame: Rows from the simulated dataset.

    Returns:
        One dict per row, JSON-serialisable.
    """
    fields = [f for f in S.REQUEST_FIELDS if f != S.CUSTOMER_ID]
    out: list[dict[str, Any]] = []
    for row in frame[fields].to_dict("records"):
        record: dict[str, Any] = {str(k): v for k, v in row.items()}
        record[S.TIMESTAMP] = pd.Timestamp(record[S.TIMESTAMP]).isoformat()
        record[S.AMOUNT] = round(float(record[S.AMOUNT]), 2)
        out.append(record)
    return out


def _build_case(
    df: pd.DataFrame, index: int, case_id: str, note: str
) -> dict[str, Any] | None:
    """Assemble one demo case around the transaction at ``index``.

    Args:
        df: The full dataset, sorted by timestamp.
        index: Positional index of the candidate transaction.
        case_id: Identifier used in URLs.
        note: Explanatory text shown in the UI.

    Returns:
        The case dict, or ``None`` when the customer has too little history for
        the window to be meaningful.
    """
    candidate = df.iloc[index]
    customer = candidate[S.CUSTOMER_ID]
    # Prior transactions only, and strictly before the candidate. Taking them by
    # position rather than by timestamp comparison would include same-second ties
    # that the online path would not have.
    prior = df.iloc[:index]
    prior = prior[prior[S.CUSTOMER_ID] == customer].tail(HISTORY_DEPTH)
    if len(prior) < 12:
        return None

    return {
        "id": case_id,
        "customer_id": str(customer),
        "history": _rows_to_requests(prior),
        "candidate": _rows_to_requests(df.iloc[[index]])[0],
        "truth": {
            "is_fraud": bool(candidate[S.IS_FRAUD]),
            "fraud_type": str(candidate[S.FRAUD_TYPE]),
        },
        "note": note,
        "history_depth": int(len(prior)),
        "customer_median_amount": round(float(prior[S.AMOUNT].median()), 2),
        "customer_devices": sorted({str(d) for d in prior[S.DEVICE_ID]}),
        "customer_countries": sorted({str(c) for c in prior[S.COUNTRY]}),
    }


def build(data_dir: Path, seed: int = 11) -> dict[str, Any]:
    """Sample demo cases from the test period of the simulated dataset.

    Cases are drawn from test rows only. A case taken from the training period
    would be one the model has memorised to some degree, and a demo that quietly
    shows you training-set performance is a demo that lies.

    Args:
        data_dir: Directory containing ``transactions.parquet``.
        seed: RNG seed for case selection.

    Returns:
        The case file payload.

    Raises:
        FileNotFoundError: If the dataset has not been simulated yet.
    """
    path = data_dir / "transactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make simulate` first")

    df = pd.read_parquet(path).sort_values(S.TIMESTAMP, kind="mergesort").reset_index(drop=True)
    split = make_temporal_split(df, SplitConfig())
    test_positions = np.flatnonzero(split.test)
    rng = np.random.default_rng(seed)

    cases: list[dict[str, Any]] = []

    for mechanism, note in MECHANISM_NOTES.items():
        eligible = test_positions[
            (df.loc[test_positions, S.FRAUD_TYPE].to_numpy() == mechanism)
        ]
        rng.shuffle(eligible)
        taken = 0
        for position in eligible:
            case = _build_case(
                df, int(position), f"{mechanism}-{taken + 1}", note
            )
            if case is not None:
                cases.append(case)
                taken += 1
            if taken >= PER_MECHANISM:
                break
        LOGGER.info("%s: %d cases from %d eligible", mechanism, taken, len(eligible))

    legit = test_positions[~df.loc[test_positions, S.IS_FRAUD].to_numpy().astype(bool)]
    rng.shuffle(legit)
    taken = 0
    for position in legit:
        case = _build_case(df, int(position), f"legitimate-{taken + 1}", LEGITIMATE_NOTE)
        if case is not None:
            cases.append(case)
            taken += 1
        if taken >= N_LEGITIMATE:
            break

    # The vocabulary drives the UI's dropdowns. It is taken from the whole dataset
    # rather than from the sampled cases so a visitor can select a value the model
    # has genuinely never seen for this customer, which is the interesting case.
    vocab = {
        column: sorted({str(v) for v in df[column].unique()})[:400]
        for column in (S.MCC, S.DEVICE_TYPE, S.COUNTRY, S.CHANNEL, S.ENTRY_MODE)
    }
    vocab["merchant_id"] = sorted({str(v) for v in df[S.MERCHANT_ID].unique()})[:200]

    return {
        "dataset": {
            "n_transactions": int(len(df)),
            "fraud_rate": round(float(df[S.IS_FRAUD].mean()), 6),
            "test_start": str(split.test_start),
            "n_test": int(split.test.sum()),
        },
        "history_depth": HISTORY_DEPTH,
        "cases": cases,
        "vocab": vocab,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the demo case file.")
    parser.add_argument("--data", type=Path, default=SYNTHETIC_DIR)
    parser.add_argument("--out", type=Path, default=Path("deploy/demo_data.json"))
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    payload = build(args.data, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    LOGGER.info(
        "wrote %s: %d cases, %d prior transactions each, %.1f KB",
        args.out,
        len(payload["cases"]),
        payload["history_depth"],
        args.out.stat().st_size / 1024,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
