"""FastAPI scoring service.

Endpoints:

``POST /score``
    Transaction plus customer history in; fraud probability, decision, the prior
    transactions that drove the flag, and the model version out. Every call is
    written to the append-only audit log before the response is returned.

``GET /health``
    Liveness and whether a model is actually loaded. A service that is up but has
    no model is not healthy, and returning 200 for it is how outages get missed.

``GET /model-info``
    What is deployed: version, family, parameter count, window length, embedding
    widths, thresholds, calibration method, and the latency budget it is held to.

``GET /metrics``
    Live latency percentiles against the budget, decision mix, and request
    counts. Plain JSON rather than a Prometheus exposition format, to keep the
    dependency list free of a metrics client for something this small.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from src.api.audit import AuditLog, AuditRecord, now_iso
from src.api.schemas import (
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    ScoreRequest,
    ScoreResponse,
)
from src.api.service import ScoringService
from src.config import ARTIFACT_DIR, AUDIT_DIR, ServingConfig

LOGGER = logging.getLogger(__name__)


class _State:
    """Process-local service state and rolling operational counters."""

    def __init__(self) -> None:
        """Initialise empty state."""
        self.service: ScoringService | None = None
        self.audit: AuditLog | None = None
        self.serving = ServingConfig()
        self.started = time.perf_counter()
        self.latencies: deque[float] = deque(maxlen=10000)
        self.decisions: Counter[str] = Counter()
        self.probabilities: deque[float] = deque(maxlen=10000)
        self.requests = 0
        self.lock = threading.Lock()
        self.load_error: str | None = None


state = _State()


def _artifact_dir() -> Path:
    """Directory holding the deployment bundle, overridable for tests."""
    return Path(os.environ.get("FRAUD_ARTIFACT_DIR", str(ARTIFACT_DIR)))


def _audit_path() -> Path:
    """Path of the audit log, overridable for tests."""
    return Path(os.environ.get("FRAUD_AUDIT_PATH", str(AUDIT_DIR / "decisions.jsonl")))


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the deployment bundle at startup.

    A missing bundle is not a startup failure: the process comes up, ``/health``
    reports ``degraded`` and ``/score`` returns 503. That is deliberate, so an
    orchestrator sees an unhealthy-but-diagnosable service rather than a crash
    loop with the reason buried in logs.

    Args:
        _: The application, unused.

    Yields:
        Control to the running application.
    """
    state.audit = AuditLog(_audit_path())
    try:
        state.service = ScoringService.load(_artifact_dir(), serving=state.serving)
        LOGGER.info(
            "loaded %s (%s) on %s",
            state.service.deployment.model_name,
            state.service.deployment.model_version,
            state.service.runtime,
        )
    except Exception as exc:  # pragma: no cover - exercised by the degraded-path test
        state.load_error = str(exc)
        LOGGER.warning("no model loaded: %s", exc)
    yield


app = FastAPI(
    title="Transaction fraud scoring",
    version="0.1.0",
    description="Real-time fraud scoring with sequence models. Synthetic data only.",
    lifespan=lifespan,
)


def _require_service() -> ScoringService:
    """Return the loaded service or raise a 503.

    Returns:
        The loaded :class:`~src.api.service.ScoringService`.

    Raises:
        HTTPException: 503 when no model is loaded.
    """
    if state.service is None:
        raise HTTPException(
            status_code=503,
            detail=f"no model loaded: {state.load_error or 'run the experiment first'}",
        )
    return state.service


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    """Score a transaction and record the decision.

    Args:
        request: Transaction and customer history.

    Returns:
        The scoring decision.

    Raises:
        HTTPException: 503 if no model is loaded, 400 on malformed history.
    """
    service = _require_service()
    try:
        response = service.score(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if state.audit is not None:
        state.audit.write(
            AuditRecord(
                timestamp=now_iso(),
                txn_id=response.txn_id,
                customer_id=response.customer_id,
                fraud_probability=response.fraud_probability,
                raw_score=response.raw_score,
                decision=response.decision,
                threshold=response.threshold,
                model_version=response.model_version,
                model_name=service.deployment.model_name,
                runtime=response.runtime,
                latency_ms=response.latency_ms,
                history_used=response.history_used,
                amount=request.transaction.amount,
                merchant_id=request.transaction.merchant_id,
            )
        )

    with state.lock:
        state.requests += 1
        state.latencies.append(response.latency_ms)
        state.decisions[response.decision] += 1
        state.probabilities.append(response.fraud_probability)
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report liveness and whether a model is loaded."""
    loaded = state.service is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        runtime=state.service.runtime if state.service else "none",
        uptime_seconds=time.perf_counter() - state.started,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Describe the deployed model."""
    service = _require_service()
    deployment = service.deployment
    return ModelInfoResponse(
        model_version=deployment.model_version,
        model_family=deployment.model_family,
        model_name=deployment.model_name,
        runtime=service.runtime,
        n_parameters=deployment.n_parameters,
        window_length=service.config.max_len,
        n_numeric_features=service.n_numeric_features(),
        embedding_dims=service.embedding_dims(),
        decision_threshold=deployment.block_threshold,
        review_threshold=deployment.review_threshold,
        calibration=deployment.calibration,
        trained_on=deployment.trained_on,
        latency_budget_p99_ms=state.serving.latency_budget_p99_ms,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Report latency percentiles, decision mix and request counts."""
    with state.lock:
        samples = np.array(state.latencies, dtype=np.float64)
        probabilities = np.array(state.probabilities, dtype=np.float64)
        decisions = dict(state.decisions)
        requests = state.requests

    def percentile(q: float) -> float:
        return float(np.percentile(samples, q)) if samples.size else 0.0

    p99 = percentile(99)
    return MetricsResponse(
        requests_total=requests,
        decisions=decisions,
        latency_ms_p50=percentile(50),
        latency_ms_p95=percentile(95),
        latency_ms_p99=p99,
        latency_budget_p99_ms=state.serving.latency_budget_p99_ms,
        within_budget=bool(samples.size == 0 or p99 <= state.serving.latency_budget_p99_ms),
        mean_fraud_probability=float(probabilities.mean()) if probabilities.size else 0.0,
        uptime_seconds=time.perf_counter() - state.started,
    )
