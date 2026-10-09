"""Append-only audit log of every scoring decision.

A fraud system declines people's payments. When a customer complains, or a
regulator asks, or the model is retrained and behaves differently, someone has to
be able to reconstruct exactly what was decided and on what basis. That record
has to be written at decision time — it cannot be recovered later by re-scoring,
because the model, the threshold and the calibration will all have moved.

Append-only in the operational sense: the file is opened in append mode, each
record is a single line of JSON flushed immediately, and nothing in this codebase
rewrites or deletes an existing line. A tamper-evident chain would need a hash
link per record and off-box storage; the ``record_hash`` field here provides the
first half of that, so records can be checked for individual modification even
though a full chain is out of scope.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """One scoring decision, as written to the log.

    Attributes:
        timestamp: When the decision was made (server time, UTC).
        txn_id: Transaction scored.
        customer_id: Customer the transaction belongs to.
        fraud_probability: Calibrated probability returned.
        raw_score: Uncalibrated model output.
        decision: The action taken.
        threshold: Threshold in force at decision time.
        model_version: Deployed model version.
        model_name: Deployed model name.
        runtime: Inference runtime used.
        latency_ms: Server-side scoring latency.
        history_used: Number of prior transactions in the window.
        amount: Transaction amount.
        merchant_id: Merchant on the transaction.
    """

    timestamp: str
    txn_id: str
    customer_id: str
    fraud_probability: float
    raw_score: float
    decision: str
    threshold: float
    model_version: str
    model_name: str
    runtime: str
    latency_ms: float
    history_used: int
    amount: float
    merchant_id: str
    record_hash: str = field(default="")

    def finalise(self) -> AuditRecord:
        """Compute the record hash over the record's own content.

        Returns:
            ``self``, with ``record_hash`` populated.
        """
        payload = {k: v for k, v in asdict(self).items() if k != "record_hash"}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.record_hash = digest
        return self


class AuditLog:
    """Thread-safe append-only JSON-lines writer."""

    def __init__(self, path: Path) -> None:
        """Open the log for appending.

        Args:
            path: Destination file; parent directories are created.
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: AuditRecord) -> AuditRecord:
        """Append one record and flush it to disk.

        Flushing per record rather than buffering: a crash must not lose the
        decisions taken in the seconds before it, since those are exactly the
        ones anyone will ask about.

        Args:
            record: The record to write.

        Returns:
            The finalised record.
        """
        record.finalise()
        line = json.dumps(asdict(record), separators=(",", ":"), default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        return record

    def read_all(self) -> list[dict[str, Any]]:
        """Read every record back.

        Returns:
            List of record dicts, oldest first; empty if the log does not exist.
        """
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def verify(self) -> list[int]:
        """Check every record's hash against its content.

        Returns:
            Line numbers (0-based) whose hash does not match, i.e. records that
            were modified after being written.
        """
        bad: list[int] = []
        for i, record in enumerate(self.read_all()):
            stored = record.pop("record_hash", "")
            digest = hashlib.sha256(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if digest != stored:
                bad.append(i)
        return bad


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()
