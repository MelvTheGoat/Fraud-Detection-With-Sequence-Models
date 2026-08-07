"""Dataset loading: synthetic by default, real datasets behind an explicit flag.

Synthetic is the default and the only thing tests and CI ever touch. Real data is
optional and entirely manual: no Kaggle credentials are requested, no download is
attempted, nothing is cached. If the file is not on disk the loader prints exactly
how to obtain it and stops.

Both supported real datasets are *adapted* into the canonical schema of
:class:`~src.data.schema.TransactionSchema`, and both adaptations lose something.
Those losses are documented on each loader rather than hidden, because they change
what the sequence models are able to learn.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import RAW_DIR, SYNTHETIC_DIR
from src.data.schema import FraudType, TransactionSchema

LOGGER = logging.getLogger(__name__)

S = TransactionSchema

DATASETS: tuple[str, ...] = ("synthetic", "ieee-cis", "ulb")

DOWNLOAD_INSTRUCTIONS: dict[str, str] = {
    "ieee-cis": """
IEEE-CIS Fraud Detection is not downloaded automatically and needs no credentials
here -- fetch it manually and drop the files in place.

  1. Open https://www.kaggle.com/competitions/ieee-fraud-detection/data
  2. Accept the competition rules (a free Kaggle account is required).
  3. Download `train_transaction.csv` and `train_identity.csv`.
  4. Put both files in: {raw}

Then re-run with `--dataset ieee-cis`.
""",
    "ulb": """
The ULB credit-card fraud dataset is not downloaded automatically.

  1. Open https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  2. Download `creditcard.csv`.
  3. Put the file in: {raw}

Then re-run with `--dataset ulb`.

NOTE: this dataset has no cardholder, merchant or device identifiers -- only 28
anonymised PCA components, an elapsed-time column and an amount. Per-customer
sequences therefore cannot be constructed from it, so it exercises the tabular
baseline honestly and the sequence models only degenerately. Prefer `synthetic`
or `ieee-cis` for the architecture comparison.
""",
}


class DatasetNotAvailableError(FileNotFoundError):
    """Raised when a real dataset has not been placed in ``data/raw``."""


def _require(paths: list[Path], dataset: str) -> None:
    """Check that every required file exists, otherwise explain how to get it.

    Args:
        paths: Files the loader needs.
        dataset: Dataset key, used to select the instruction text.

    Raises:
        DatasetNotAvailableError: If any file is missing.
    """
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return
    instructions = DOWNLOAD_INSTRUCTIONS[dataset].format(raw=paths[0].parent.resolve())
    listing = "\n".join(f"  - {p}" for p in missing)
    raise DatasetNotAvailableError(
        f"Missing files for dataset '{dataset}':\n{listing}\n{instructions}"
    )


# --------------------------------------------------------------------------------------
# Synthetic
# --------------------------------------------------------------------------------------


def load_synthetic(data_dir: Path = SYNTHETIC_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the generated synthetic dataset, generating it if absent.

    Args:
        data_dir: Directory containing ``transactions.parquet``.

    Returns:
        Tuple of (transactions, metadata).
    """
    path = data_dir / "transactions.parquet"
    meta_path = data_dir / "metadata.json"
    if not path.exists():
        from src.simulate import simulate, write_dataset

        LOGGER.info("synthetic dataset not found at %s; generating", path)
        df, metadata = simulate()
        write_dataset(df, metadata, data_dir)
        return df, metadata

    df = pd.read_parquet(path)
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return df, metadata


# --------------------------------------------------------------------------------------
# IEEE-CIS
# --------------------------------------------------------------------------------------


def load_ieee_cis(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Adapt the IEEE-CIS competition data into the canonical schema.

    What the adaptation assumes, and why it is a proxy rather than the real thing:

    * **Customer.** The data has no cardholder key. The community-standard proxy
      is a composite of ``card1``/``card2``/``addr1``, which groups transactions
      that almost certainly share a card. It over-merges (two cards with the same
      BIN and postcode collide) and under-merges (a re-issued card looks new), so
      per-customer histories are noisier than the synthetic ones.
    * **Merchant.** There is no merchant column. ``ProductCD`` combined with the
      purchaser email domain is the closest available high-cardinality entity;
      it is a weaker signal than a true merchant ID.
    * **Timestamp.** ``TransactionDT`` is an offset in seconds from an undisclosed
      reference. An arbitrary epoch is applied, which is fine because every
      downstream operation is relative.
    * **Fraud mechanism.** Unknown, so every positive is labelled
      ``FraudType.UNKNOWN`` and the per-mechanism breakdown is suppressed.

    Args:
        raw_dir: Directory holding ``train_transaction.csv`` and
            ``train_identity.csv``.

    Returns:
        Tuple of (transactions, metadata).
    """
    txn_path = raw_dir / "train_transaction.csv"
    id_path = raw_dir / "train_identity.csv"
    _require([txn_path, id_path], "ieee-cis")

    txn = pd.read_csv(txn_path)
    ident = pd.read_csv(id_path)
    df = txn.merge(ident, on="TransactionID", how="left")

    epoch = pd.Timestamp("2017-11-30")
    card1 = df.get("card1", pd.Series(0, index=df.index)).fillna(-1).astype(int).astype(str)
    card2 = df.get("card2", pd.Series(0, index=df.index)).fillna(-1).astype(int).astype(str)
    addr1 = df.get("addr1", pd.Series(0, index=df.index)).fillna(-1).astype(int).astype(str)

    out = pd.DataFrame(
        {
            S.TXN_ID: df["TransactionID"].astype(str),
            S.CUSTOMER_ID: "c" + card1 + "_" + card2 + "_" + addr1,
            S.TIMESTAMP: epoch + pd.to_timedelta(df["TransactionDT"], unit="s"),
            S.AMOUNT: df["TransactionAmt"].astype(float),
            S.MERCHANT_ID: (
                df["ProductCD"].astype(str) + "|" + df["P_emaildomain"].fillna("na").astype(str)
            ),
            S.MCC: df["ProductCD"].fillna("na").astype(str),
            S.DEVICE_ID: df.get("DeviceInfo", pd.Series("na", index=df.index))
            .fillna("na")
            .astype(str),
            S.DEVICE_TYPE: df.get("DeviceType", pd.Series("na", index=df.index))
            .fillna("na")
            .astype(str),
            S.COUNTRY: df.get("addr2", pd.Series(-1, index=df.index))
            .fillna(-1)
            .astype(int)
            .astype(str),
            S.CHANNEL: np.where(df["ProductCD"].astype(str) == "W", "in_store", "online"),
            S.ENTRY_MODE: df.get("card6", pd.Series("na", index=df.index))
            .fillna("na")
            .astype(str),
            S.IS_FRAUD: df["isFraud"].astype(np.int8),
        }
    )
    out[S.FRAUD_TYPE] = np.where(
        out[S.IS_FRAUD] == 1, FraudType.UNKNOWN.value, FraudType.NONE.value
    )
    out = out.sort_values(S.TIMESTAMP, kind="mergesort").reset_index(drop=True)

    metadata: dict[str, Any] = {
        "source": "ieee-cis",
        "n_transactions": int(len(out)),
        "n_fraud": int(out[S.IS_FRAUD].sum()),
        "fraud_rate": float(out[S.IS_FRAUD].mean()),
        "start": str(out[S.TIMESTAMP].min()),
        "end": str(out[S.TIMESTAMP].max()),
        "drift_start": None,
        "fraud_mechanisms_known": False,
        "caveats": [
            "customer_id is a card1/card2/addr1 proxy, not a true cardholder key",
            "merchant_id is a ProductCD|email-domain proxy; no merchant column exists",
            "fraud mechanism labels are unavailable, so the per-mechanism breakdown is omitted",
        ],
    }
    return out, metadata


# --------------------------------------------------------------------------------------
# ULB
# --------------------------------------------------------------------------------------


def load_ulb(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Adapt the ULB credit-card dataset into the canonical schema.

    This dataset is included because it is the one everybody reaches for, and it
    is worth being explicit about why it cannot support this project's thesis: it
    has no cardholder, merchant or device identifiers, only 28 anonymised PCA
    components. There are no sequences to model and no high-cardinality entities
    to embed — the two things the deep models are supposed to exploit. Loading it
    maps every row to a single pseudo-customer so the pipeline runs end to end,
    and the PCA components are carried through as ``v1..v28`` extra columns for
    the tabular baseline.

    Args:
        raw_dir: Directory holding ``creditcard.csv``.

    Returns:
        Tuple of (transactions, metadata).
    """
    path = raw_dir / "creditcard.csv"
    _require([path], "ulb")

    df = pd.read_csv(path)
    epoch = pd.Timestamp("2013-09-01")
    out = pd.DataFrame(
        {
            S.TXN_ID: [f"T{i:08d}" for i in range(len(df))],
            S.CUSTOMER_ID: "c0",
            S.TIMESTAMP: epoch + pd.to_timedelta(df["Time"], unit="s"),
            S.AMOUNT: df["Amount"].astype(float),
            S.MERCHANT_ID: "unknown",
            S.MCC: "unknown",
            S.DEVICE_ID: "unknown",
            S.DEVICE_TYPE: "unknown",
            S.COUNTRY: "unknown",
            S.CHANNEL: "unknown",
            S.ENTRY_MODE: "unknown",
            S.IS_FRAUD: df["Class"].astype(np.int8),
        }
    )
    out[S.FRAUD_TYPE] = np.where(
        out[S.IS_FRAUD] == 1, FraudType.UNKNOWN.value, FraudType.NONE.value
    )
    for i in range(1, 29):
        col = f"V{i}"
        if col in df.columns:
            out[f"v{i}"] = df[col].astype(np.float32)
    out = out.sort_values(S.TIMESTAMP, kind="mergesort").reset_index(drop=True)

    LOGGER.warning(
        "ULB has no entity identifiers: sequence models degenerate to a single "
        "global stream. Use it to sanity-check the baseline, not to compare "
        "architectures."
    )
    metadata: dict[str, Any] = {
        "source": "ulb",
        "n_transactions": int(len(out)),
        "n_fraud": int(out[S.IS_FRAUD].sum()),
        "fraud_rate": float(out[S.IS_FRAUD].mean()),
        "start": str(out[S.TIMESTAMP].min()),
        "end": str(out[S.TIMESTAMP].max()),
        "drift_start": None,
        "fraud_mechanisms_known": False,
        "caveats": [
            "no cardholder/merchant/device keys; per-customer sequences are not constructible",
            "all rows share one pseudo-customer, so sequence models see a global stream",
        ],
    }
    return out, metadata


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------


def load_dataset(
    name: str = "synthetic",
    *,
    data_dir: Path = SYNTHETIC_DIR,
    raw_dir: Path = RAW_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a dataset by name.

    Args:
        name: One of :data:`DATASETS`.
        data_dir: Where the synthetic dataset lives.
        raw_dir: Where manually downloaded real datasets live.

    Returns:
        Tuple of (transactions in time order, metadata).

    Raises:
        ValueError: If ``name`` is not a known dataset.
    """
    if name == "synthetic":
        return load_synthetic(data_dir)
    if name == "ieee-cis":
        return load_ieee_cis(raw_dir)
    if name == "ulb":
        return load_ulb(raw_dir)
    raise ValueError(f"unknown dataset '{name}'; choose one of {DATASETS}")
