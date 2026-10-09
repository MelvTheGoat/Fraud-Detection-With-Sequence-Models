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

        # The interactive promise of the page. This is asserted on episodes, not
        # on single-field edits, because the first version of this test asserted
        # the latter and failed -- correctly. Both mechanisms in the simulator are
        # bursts of 3-15 transactions, and the features the model reads are
        # properties of a sequence, so a single edited transaction on a clean
        # history should move the score barely at all. The test now checks the
        # thing the model is supposed to detect.
        legitimate = next(c for c in data["cases"] if not c["truth"]["is_fraud"])
        base = client.post(
            "/score",
            json={
                "customer_id": legitimate["customer_id"],
                "transaction": legitimate["candidate"],
                "history": legitimate["history"],
            },
        ).json()

        for mechanism in ("card_testing", "account_takeover"):
            episode = client.post(
                "/demo/episode",
                json={"case_id": legitimate["id"], "mechanism": mechanism},
            )
            _assert(episode.status_code == 200, f"{mechanism}: episode built")
            burst = episode.json()
            scored = client.post(
                "/score",
                json={
                    "customer_id": legitimate["customer_id"],
                    "transaction": burst["candidate"],
                    "history": legitimate["history"] + burst["injected"],
                },
            ).json()
            print(
                f"      {mechanism}: {base['raw_score']:.2e} -> "
                f"{scored['raw_score']:.2e} raw, "
                f"{base['fraud_probability']:.4f} -> {scored['fraud_probability']:.4f} "
                f"calibrated, {scored['decision']}"
            )
            # On the raw score, because isotonic pins the bottom of its range to
            # exactly zero: a calibrated comparison would be comparing 0.0 to 0.0
            # on a model whose raw output moved by four orders of magnitude.
            _assert(
                scored["raw_score"] > base["raw_score"],
                f"{mechanism}: the injected burst raises the raw score",
            )
            _assert(
                scored["decision"] != "approve",
                f"{mechanism}: the injected burst is not approved",
            )
            _assert(
                len(scored["top_contributors"]) > 0,
                f"{mechanism}: contributors returned for a non-approved decision",
            )

        bad = client.post(
            "/demo/episode", json={"case_id": legitimate["id"], "mechanism": "nonsense"}
        )
        _assert(bad.status_code == 400, "an unknown mechanism is a 400, not a crash")
        missing = client.post(
            "/demo/episode", json={"case_id": "no-such-case", "mechanism": "card_testing"}
        )
        _assert(missing.status_code == 404, "an unknown case is a 404")

        page = client.get("/")
        _assert(page.status_code == 200, "the demo page is served")
        _assert(b"Check this payment" in page.content, "the page carries its form")
        # The country field has to accept values outside the training vocabulary.
        # A visitor from Lagos typing NG and getting a 422 would be the page
        # refusing the one thing it invites them to do.
        _assert(b"\"NG\"" in page.content, "the page offers countries beyond the training set")

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
