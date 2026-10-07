"""End-to-end smoke test for the demo deployment.

Starts nothing and mocks nothing: it drives the real application through
FastAPI's test client, against the real artefact bundle, with the real demo
cases. If this passes, the container will serve.

What it checks, and why each one is here rather than being assumed:

* ``/health`` reports a loaded model. A service that is up without a model is
  the failure mode this project explicitly refuses to report as healthy.
* ``/model-info`` agrees with ``deployment.json``. A mismatch means the page
  would draw its threshold markers in the wrong place.
* every demo case scores, and no request body contains a label field. The
  second half is the one that matters: the demo holds ground truth in order to
  reveal it, and a copy-paste slip that put it in the request would be invisible
  in the output and would invalidate every score on the page.
* fraud cases score above legitimate ones *on average*. Not case by case —
  with 0.87 PR-AUC some individual fraud will score below some legitimate
  transaction, and a test that demanded otherwise would be a test that fails
  honestly-good models.
* the takeover preset raises the score on a legitimate customer. This is the
  interactive promise of the page; if editing the form cannot move the score,
  the demo is a static screenshot.

Usage::

    python -m deploy.smoke_test
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from deploy.app import app

LABEL_FIELDS = ("is_fraud", "fraud_type", "truth")


def _assert(condition: bool, message: str) -> None:
    """Fail loudly with a message, rather than an opaque AssertionError."""
    if not condition:
        print(f"FAIL  {message}")
        raise SystemExit(1)
    print(f"ok    {message}")


def main() -> int:
    """Run the smoke test."""
    with TestClient(app) as client:
        health = client.get("/health").json()
        _assert(health["model_loaded"], f"model loaded ({health['runtime']})")
        _assert(health["status"] == "ok", f"health status is ok, got {health['status']}")

        info = client.get("/model-info").json()
        descriptor = json.loads(Path("artifacts/deployment.json").read_text())
        _assert(
            abs(info["decision_threshold"] - descriptor["block_threshold"]) < 1e-12,
            f"block threshold matches the descriptor ({info['decision_threshold']:.6f})",
        )
        _assert(
            info["review_threshold"] < info["decision_threshold"],
            "review threshold sits below the block threshold",
        )

        data = client.get("/demo/data").json()
        _assert(len(data["cases"]) >= 10, f"{len(data['cases'])} demo cases")

        fraud_scores: list[float] = []
        legit_scores: list[float] = []
        decisions: dict[str, int] = {}

        for case in data["cases"]:
            body = {
                "customer_id": case["customer_id"],
                "transaction": case["candidate"],
                "history": case["history"],
            }
            blob = json.dumps(body)
            for field in LABEL_FIELDS:
                _assert(
                    f'"{field}"' not in blob,
                    f"{case['id']}: request carries no {field}",
                )

            response = client.post("/score", json=body)
            _assert(response.status_code == 200, f"{case['id']}: HTTP 200")
            payload = response.json()
            decisions[payload["decision"]] = decisions.get(payload["decision"], 0) + 1
            (fraud_scores if case["truth"]["is_fraud"] else legit_scores).append(
                payload["fraud_probability"]
            )
            _assert(
                payload["history_used"] == len(case["history"]),
                f"{case['id']}: all {len(case['history'])} prior transactions used",
            )

        mean_fraud = statistics.mean(fraud_scores)
        mean_legit = statistics.mean(legit_scores)
        print(f"      mean score: fraud {mean_fraud:.4f}, legitimate {mean_legit:.4f}")
        print(f"      decisions: {decisions}")
        _assert(
            mean_fraud > mean_legit,
            f"fraud scores higher than legitimate on average ({mean_fraud:.4f} > {mean_legit:.4f})",
        )

        # The interactive promise: editing the transaction has to move the score.
        legitimate = next(c for c in data["cases"] if not c["truth"]["is_fraud"])
        base = client.post(
            "/score",
            json={
                "customer_id": legitimate["customer_id"],
                "transaction": legitimate["candidate"],
                "history": legitimate["history"],
            },
        ).json()["fraud_probability"]

        tampered = dict(legitimate["candidate"])
        tampered["device_id"] = "D999999_9"
        tampered["amount"] = round(legitimate["customer_median_amount"] * 9, 2)
        tampered["channel"] = "online"
        tampered["entry_mode"] = "ecom"
        moved = client.post(
            "/score",
            json={
                "customer_id": legitimate["customer_id"],
                "transaction": tampered,
                "history": legitimate["history"],
            },
        ).json()["fraud_probability"]
        print(f"      takeover preset: {base:.4f} -> {moved:.4f}")
        _assert(moved > base, "the takeover preset raises the score")

        page = client.get("/")
        _assert(page.status_code == 200, "the demo page is served")
        _assert(b"Score this transaction" in page.content, "the page carries its form")

        metrics = client.get("/metrics").json()
        _assert(
            metrics["requests_total"] >= len(data["cases"]),
            f"metrics counted {metrics['requests_total']} requests",
        )
        print(
            f"      latency p50 {metrics['latency_ms_p50']:.2f} ms, "
            f"p99 {metrics['latency_ms_p99']:.2f} ms, "
            f"budget {metrics['latency_budget_p99_ms']:.0f} ms, "
            f"within budget: {metrics['within_budget']}"
        )

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
