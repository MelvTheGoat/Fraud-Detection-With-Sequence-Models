"""Canonical transaction schema shared by the simulator, the loaders and the models.

One schema keeps synthetic and real datasets interchangeable: a real dataset is
adapted *into* these columns by its loader, so nothing downstream needs to know
which source it is looking at.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FraudType(StrEnum):
    """Known fraud mechanisms in the synthetic data.

    Real data has no such column; loaders set every fraudulent row to
    ``UNKNOWN`` so the per-mechanism breakdown degrades gracefully rather than
    silently reporting nonsense.
    """

    NONE = "none"
    CARD_TESTING = "card_testing"
    ACCOUNT_TAKEOVER = "account_takeover"
    MERCHANT_COMPROMISE = "merchant_compromise"
    UNKNOWN = "unknown"


#: The mechanisms that are actually fraud, in report order.
FRAUD_TYPES: tuple[str, ...] = (
    FraudType.CARD_TESTING.value,
    FraudType.ACCOUNT_TAKEOVER.value,
    FraudType.MERCHANT_COMPROMISE.value,
    FraudType.UNKNOWN.value,
)


@dataclass(frozen=True)
class CategoricalSpec:
    """A categorical column that gets a learned embedding.

    Attributes:
        column: Name of the integer-coded column in the feature frame.
        cardinality: Number of distinct codes, including the two reserved
            indices: 0 for padding and 1 for unknown/rare.
    """

    column: str
    cardinality: int


class TransactionSchema:
    """Column names for the canonical transaction table.

    A row is one authorisation request. Everything in ``REQUEST_FIELDS`` is known
    at scoring time; everything in ``LABEL_FIELDS`` arrives later (a chargeback
    is typically confirmed weeks after the authorisation) and must never be used
    as a feature, including for prior transactions in a sequence.
    """

    TXN_ID = "txn_id"
    CUSTOMER_ID = "customer_id"
    TIMESTAMP = "timestamp"
    AMOUNT = "amount"
    MERCHANT_ID = "merchant_id"
    MCC = "mcc"
    DEVICE_ID = "device_id"
    DEVICE_TYPE = "device_type"
    COUNTRY = "country"
    CHANNEL = "channel"
    ENTRY_MODE = "entry_mode"

    IS_FRAUD = "is_fraud"
    FRAUD_TYPE = "fraud_type"

    #: Fields present in the authorisation request itself.
    REQUEST_FIELDS: tuple[str, ...] = (
        TXN_ID,
        CUSTOMER_ID,
        TIMESTAMP,
        AMOUNT,
        MERCHANT_ID,
        MCC,
        DEVICE_ID,
        DEVICE_TYPE,
        COUNTRY,
        CHANNEL,
        ENTRY_MODE,
    )

    #: Fields that only exist after the fact.
    LABEL_FIELDS: tuple[str, ...] = (IS_FRAUD, FRAUD_TYPE)

    ALL: tuple[str, ...] = REQUEST_FIELDS + LABEL_FIELDS


#: Index 0 is padding, index 1 is unknown/rare. Both are learned like any other
#: embedding row, which is what we want: "a merchant I have never seen" is a
#: genuine and informative state, not a missing value.
PAD_INDEX = 0
UNK_INDEX = 1
N_RESERVED_CODES = 2
