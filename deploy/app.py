"""Public demo wrapper around the scoring API.

This module adds a browser front end to the *unmodified* scoring service. It
imports :data:`src.api.app.app` and attaches routes to it rather than mounting a
second application, so ``/score``, ``/health``, ``/model-info``, ``/metrics`` and
the generated ``/docs`` keep their exact production paths and the page you click
around in is talking to the same endpoint a payment switch would call.

Routes added here:

``GET /``
    The demo page.

``GET /demo/data``
    The case pool: customer histories sampled from the test period, the
    vocabulary for the form's dropdowns, and the ground-truth labels the page
    keeps hidden until you ask for them.

``GET /demo/notice``
    What this is, what it is not, and the headline numbers, so the disclaimer
    lives in one place instead of being retyped into the HTML.

Entry point for the container::

    uvicorn deploy.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from deploy import episodes
from src.api.app import app, state

LOGGER = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DEMO_DATA = Path(os.environ.get("FRAUD_DEMO_DATA", str(HERE / "demo_data.json")))
INDEX = HERE / "static" / "index.html"

# No numbers here on purpose. Every figure the page shows comes from the
# ``measured`` block of the case file, which is built from the same run that
# produced the bundle being served. Hardcoding the README's three-seed figures
# would state an error bar nobody measured whenever the bundle came from a
# shorter run, which is the exact failure this project spends a chapter on.
NOTICE: dict[str, Any] = {
    "what_this_is": (
        "A fraud scoring model served the way it would be served in production: "
        "ONNX Runtime, single transaction, calibrated probability, a cost-chosen "
        "threshold, and an append-only audit record per decision."
    ),
    "what_this_is_not": (
        "Trained on synthetic data from a simulator in this repository, not on "
        "real card transactions. The numbers are honest about that simulator and "
        "say nothing about performance on a real portfolio."
    ),
    "source": "https://github.com/MelvTheGoat/Fraud-Detection-With-Sequence-Models",
}


@lru_cache(maxsize=1)
def _demo_payload() -> dict[str, Any]:
    """Read and cache the case file.

    Cached because it is a few hundred kilobytes of static JSON read on every
    page load, and the file cannot change without a redeploy.

    Raises:
        FileNotFoundError: If the case file was not built into the image.
    """
    return json.loads(DEMO_DATA.read_text())


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the demo page."""
    if not INDEX.exists():  # pragma: no cover - packaging error, not a runtime path
        raise HTTPException(status_code=500, detail="demo page missing from the image")
    return FileResponse(INDEX, media_type="text/html")


@app.get("/demo/notice", include_in_schema=False)
def notice() -> JSONResponse:
    """What this demo is and is not, plus the headline numbers."""
    deployment = state.service.deployment if state.service is not None else None
    payload: dict[str, Any] = dict(NOTICE)
    payload["model_loaded"] = state.service is not None
    if deployment is not None:
        payload["deployed"] = {
            "model_name": deployment.model_name,
            "selected_because": deployment.selected_because,
            "calibration": deployment.calibration,
            "block_threshold": deployment.block_threshold,
            "review_threshold": deployment.review_threshold,
        }
    elif state.load_error:
        payload["load_error"] = state.load_error
    return JSONResponse(payload)


@app.get("/demo/data", include_in_schema=False)
def demo_data() -> JSONResponse:
    """Case pool and form vocabulary."""
    try:
        return JSONResponse(_demo_payload())
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"demo case file not found at {DEMO_DATA}; run deploy.make_demo_data",
        ) from None


class EpisodeRequest(BaseModel):
    """Which synthetic burst to graft onto which case."""

    case_id: str = Field(description="Id of a case from /demo/data.")
    mechanism: str = Field(description="card_testing or account_takeover.")
    seed: int = Field(default=0, description="So a given case and preset repeat.")


@app.post("/demo/episode", include_in_schema=False)
def demo_episode(request: EpisodeRequest) -> JSONResponse:
    """Build a synthetic fraud episode on top of a case's real history.

    The episode is built here rather than in the browser so that the shape of
    each mechanism lives next to the simulator semantics it mirrors, and so the
    smoke test can assert the model responds to it.
    """
    payload = _demo_payload()
    case = next((c for c in payload["cases"] if c["id"] == request.case_id), None)
    if case is None:
        raise HTTPException(status_code=404, detail=f"no case {request.case_id!r}")
    if request.mechanism not in episodes.BUILDERS:
        raise HTTPException(
            status_code=400,
            detail=f"mechanism must be one of {sorted(episodes.BUILDERS)}",
        )
    episode = episodes.build(
        request.mechanism, case, payload["attack_pools"], seed=request.seed
    )
    return JSONResponse(episode.to_payload())
