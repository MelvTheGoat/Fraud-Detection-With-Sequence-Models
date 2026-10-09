"""Request and response models for the scoring API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """One authorisation request.

    Only fields available at authorisation time appear here. There is
    deliberately no ``is_fraud`` field, on prior transactions either: a chargeback
    is confirmed weeks later, so a scorer that accepted historical labels would be
    consuming information the live system does not have.
    """

    txn_id: str = Field(description="Unique transaction identifier.")
    timestamp: datetime = Field(description="Authorisation time, UTC.")
    amount: float = Field(ge=0, description="Transaction amount.")
    merchant_id: str = Field(description="Merchant identifier.")
    mcc: str = Field(description="Merchant category code.")
    device_id: str = Field(description="Device or token identifier.")
    device_type: str = Field(description="Device type / fingerprint family.")
    country: str = Field(description="Country of the transaction.")
    channel: Literal["online", "in_store", "unknown"] = Field(description="Acceptance channel.")
    entry_mode: str = Field(description="Entry mode, e.g. chip, contactless, ecom.")


class ScoreRequest(BaseModel):
    """A transaction to score together with the customer's recent history."""

    customer_id: str = Field(description="Customer identifier.")
    transaction: Transaction = Field(description="The transaction being authorised.")
    history: list[Transaction] = Field(
        default_factory=list,
        description=(
            "The customer's prior transactions in ascending time order. Only the "
            "most recent window is used; older entries are ignored. An empty "
            "history is valid — a customer's first transaction still has to be "
            "scored."
        ),
    )


class ContributingTransaction(BaseModel):
    """A prior transaction the model read from when scoring."""

    txn_id: str
    timestamp: datetime
    amount: float
    merchant_id: str
    device_id: str
    offset: int = Field(description="Positions back from the scored transaction; 0 is itself.")
    attention: float | None = Field(
        default=None, description="Attention weight, if the model has attention."
    )


class ScoreResponse(BaseModel):
    """Scoring decision returned to the caller."""

    txn_id: str
    customer_id: str
    fraud_probability: float = Field(description="Calibrated probability of fraud.")
    raw_score: float = Field(description="Model output before calibration.")
    decision: Literal["approve", "review", "block"]
    threshold: float = Field(description="Decision threshold in force.")
    model_version: str
    model_family: str
    runtime: str = Field(description="Inference runtime that produced the score.")
    latency_ms: float
    history_used: int = Field(description="Prior transactions actually used.")
    top_contributors: list[ContributingTransaction] = Field(default_factory=list)
    attention_caveat: str | None = Field(
        default=None,
        description="Present when contributors are attention-derived; states their limits.",
    )


class HealthResponse(BaseModel):
    """Liveness and readiness."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    runtime: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Description of the deployed model."""

    model_version: str
    model_family: str
    model_name: str
    runtime: str
    n_parameters: int
    window_length: int
    n_numeric_features: int
    embedding_dims: dict[str, int]
    decision_threshold: float
    review_threshold: float
    calibration: str
    trained_on: str
    latency_budget_p99_ms: float


class MetricsResponse(BaseModel):
    """Operational counters since process start."""

    requests_total: int
    decisions: dict[str, int]
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    latency_budget_p99_ms: float
    within_budget: bool
    mean_fraud_probability: float
    uptime_seconds: float
