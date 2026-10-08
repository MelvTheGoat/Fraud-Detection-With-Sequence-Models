"""Export a browser-runnable demo: the real ONNX graph plus precomputed inputs.

Why this exists alongside the container
---------------------------------------
``deploy/app.py`` serves the whole system and accepts any transaction, which is
the honest deployment. It also needs somewhere to run. This exporter produces a
static page that needs nothing: ONNX Runtime Web executes the *same exported
graph* in the visitor's browser, and the isotonic calibrator is replayed exactly
from its 42 breakpoints.

What it cannot do, stated plainly
---------------------------------
The feature builder is pandas and scikit-learn — a fitted scaler, six category
encoders and out-of-time target-encoding maps. Porting it to JavaScript would
mean reimplementing the one part of this project whose whole point is that the
online and offline paths agree bit for bit, and a parity bug there is invisible
and flattering. So the features are **not** ported: they are computed here, by
the real :class:`~src.data.features.FeatureBuilder`, for a fixed grid of
scenarios, and shipped as tensors.

The consequence is a page where model inference is live and real but the input
space is a menu rather than a form. Every scenario also carries the score the
Python service produced for it, so the page can display both numbers and let a
visitor see that the browser agrees with the server rather than taking it on
faith.

Grid
----
A true cross product, so the two controls on the page are independent and no
combination is offered that was not actually computed:

    15 cases  x  10 variants  =  150 scenarios

Usage::

    python -m deploy.export_browser_demo --out build/browser
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deploy import episodes
from src.api.schemas import ScoreRequest, Transaction
from src.api.service import ScoringService

LOGGER = logging.getLogger(__name__)

#: Amount multipliers applied to the real transaction. Chosen to straddle the
#: customer's own distribution rather than to be round: 0.25x and 2x are
#: ordinary variation, 5x and above is where an amount z-score starts to carry
#: real signal, and 25x is well outside anything the customer has done.
AMOUNT_MULTIPLIERS = (0.25, 0.5, 2.0, 5.0, 10.0, 25.0)

#: A device token nothing in the dataset has ever seen, so it encodes to the
#: reserved unknown index rather than to a learned embedding row.
UNSEEN_DEVICE = "D999999_9"


@dataclass(frozen=True)
class Variant:
    """One column of the scenario grid.

    Attributes:
        key: Identifier used by the page.
        label: What the control shows.
        blurb: One line explaining what is being asked of the model.
    """

    key: str
    label: str
    blurb: str


VARIANTS: tuple[Variant, ...] = (
    Variant(
        "actual",
        "As it happened",
        "The transaction exactly as the simulator generated it, with the "
        "customer's real prior history behind it.",
    ),
    *(
        Variant(
            f"amount_{str(m).replace('.', '_')}",
            f"Amount x{m:g}",
            f"The same transaction with the amount multiplied by {m:g} and "
            "nothing else touched. One field, one transaction — expect the "
            "score to move far less than intuition says it should.",
        )
        for m in AMOUNT_MULTIPLIERS
    ),
    Variant(
        "unseen_device",
        "From a device never seen",
        "The same transaction from a device token that appears nowhere in the "
        "dataset, so it encodes to the reserved unknown index.",
    ),
    Variant(
        "card_testing",
        "Card-testing burst injected",
        "Six authorisations inside about twenty minutes, £0.50-£9, at six "
        "merchants this customer has never used, all from one unseen device.",
    ),
    Variant(
        "account_takeover",
        "Account takeover injected",
        "Five transactions over a few hours at 3-10x the customer's median, at "
        "unfamiliar merchants, from an unseen device — and in the customer's "
        "own country, because takeover does not need to look foreign.",
    ),
)


def _transaction(payload: dict[str, Any]) -> Transaction:
    """Build a request transaction from a demo-case dict."""
    return Transaction(**payload)


def _scenario_inputs(
    service: ScoringService,
    case: dict[str, Any],
    variant: Variant,
    pools: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build one scenario's model inputs and its display metadata.

    Args:
        service: The loaded scoring service, used for both features and score.
        case: A demo case from ``deploy/demo_data.json``.
        variant: Which column of the grid.
        pools: Merchant pools for the injected episodes.

    Returns:
        Tuple of (arrays keyed ``numeric``/``categorical``/``mask``, metadata).
    """
    history = list(case["history"])
    candidate = dict(case["candidate"])
    injected: list[dict[str, Any]] = []

    if variant.key.startswith("amount_"):
        multiplier = float(variant.key.removeprefix("amount_").replace("_", "."))
        candidate["amount"] = round(candidate["amount"] * multiplier, 2)
    elif variant.key == "unseen_device":
        candidate["device_id"] = UNSEEN_DEVICE
    elif variant.key in episodes.BUILDERS:
        episode = episodes.build(variant.key, case, pools)
        injected = list(episode.injected)
        history = history + injected
        candidate = dict(episode.candidate)

    request = ScoreRequest(
        customer_id=case["customer_id"],
        transaction=_transaction(candidate),
        history=[_transaction(t) for t in history],
    )
    numeric, categorical, mask, history_used = service.build_window(request)

    # The service's own answer, so the page can show that the browser agrees
    # rather than asserting it. explain=False keeps this to the scoring path.
    response = service.score(request, explain=False)

    return (
        {"numeric": numeric, "categorical": categorical, "mask": mask},
        {
            "variant": variant.key,
            "candidate": candidate,
            "injected": injected,
            "history_used": history_used,
            "server_raw": response.raw_score,
            "server_probability": response.fraud_probability,
            "server_decision": response.decision,
        },
    )


def _isotonic_breakpoints(path: Path) -> dict[str, list[float]]:
    """Export the fitted isotonic map as breakpoints.

    ``sklearn``'s :class:`IsotonicRegression` predicts by linear interpolation
    between ``X_thresholds_`` and ``y_thresholds_``, clamped to the end values.
    That is exactly ``np.interp``, and exactly what a dozen lines of JavaScript
    can replay — no approximation and nothing refitted.

    Args:
        path: Path to ``calibrator.pkl``.

    Returns:
        Dict with ``x`` and ``y`` breakpoint lists.
    """
    with path.open("rb") as handle:
        calibrator = pickle.load(handle)
    inner = calibrator.model
    return {
        "x": [float(v) for v in np.asarray(inner.X_thresholds_)],
        "y": [float(v) for v in np.asarray(inner.y_thresholds_)],
    }


def build(
    artifact_dir: Path, demo_data: Path, out_dir: Path
) -> dict[str, Any]:
    """Export the model, the scenario tensors and the manifest.

    Args:
        artifact_dir: Deployment bundle directory.
        demo_data: Path to ``demo_data.json``.
        out_dir: Destination directory.

    Returns:
        A summary of what was written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(demo_data.read_text())
    service = ScoringService.load(artifact_dir)
    pools = payload["attack_pools"]

    numeric_rows: list[np.ndarray] = []
    categorical_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    scenarios: list[dict[str, Any]] = []

    for case in payload["cases"]:
        for variant in VARIANTS:
            arrays, meta = _scenario_inputs(service, case, variant, pools)
            numeric_rows.append(arrays["numeric"].astype(np.float32).reshape(-1))
            categorical_rows.append(arrays["categorical"].astype(np.int32).reshape(-1))
            mask_rows.append(arrays["mask"].astype(np.float32).reshape(-1))
            meta["case_id"] = case["id"]
            meta["index"] = len(scenarios)
            scenarios.append(meta)
        LOGGER.info("%s: %d variants", case["id"], len(VARIANTS))

    numeric = np.concatenate(numeric_rows)
    categorical = np.concatenate(categorical_rows)
    mask = np.concatenate(mask_rows)

    # One flat buffer per input, indexed by scenario. Three typed arrays beat 150
    # JSON objects by about a factor of four on the wire and avoid parsing a
    # million numbers in the main thread on a phone.
    blobs = {
        "numeric": numeric.tobytes(),
        "categorical": categorical.tobytes(),
        "mask": mask.tobytes(),
    }
    model_bytes = (artifact_dir / "model.onnx").read_bytes()

    js = out_dir / "demo-data.js"
    js.write_text(
        "window.__FRAUD_DEMO = "
        + json.dumps(
            {
                "model_b64": base64.b64encode(model_bytes).decode(),
                "numeric_b64": base64.b64encode(blobs["numeric"]).decode(),
                "categorical_b64": base64.b64encode(blobs["categorical"]).decode(),
                "mask_b64": base64.b64encode(blobs["mask"]).decode(),
                "shape": {
                    "n_scenarios": len(scenarios),
                    "max_len": service.config.max_len,
                    "n_numeric": service.config.n_numeric,
                    "n_categorical": len(service.config.categorical),
                },
                "isotonic": _isotonic_breakpoints(artifact_dir / "calibrator.pkl"),
                "deployment": json.loads((artifact_dir / "deployment.json").read_text()),
                "measured": payload.get("measured", {}),
                "dataset": payload.get("dataset", {}),
                "variants": [v.__dict__ for v in VARIANTS],
                "cases": [
                    {
                        "id": c["id"],
                        "customer_id": c["customer_id"],
                        "note": c["note"],
                        "truth": c["truth"],
                        "median_amount": c["customer_median_amount"],
                        "devices": c["customer_devices"],
                        "countries": c["customer_countries"],
                        "n_history": len(c["history"]),
                        "history_tail": c["history"][-10:],
                    }
                    for c in payload["cases"]
                ],
                "scenarios": scenarios,
            },
            separators=(",", ":"),
        )
        + ";\n",
        encoding="utf-8",
    )

    summary = {
        "scenarios": len(scenarios),
        "cases": len(payload["cases"]),
        "variants": len(VARIANTS),
        "model_kb": round(len(model_bytes) / 1024, 1),
        "tensors_kb": round(sum(len(b) for b in blobs.values()) / 1024, 1),
        "demo_data_js_kb": round(js.stat().st_size / 1024, 1),
        "mean_server_probability": float(
            np.mean([s["server_probability"] for s in scenarios])
        ),
        "blocked": sum(1 for s in scenarios if s["server_decision"] == "block"),
    }
    LOGGER.info("wrote %s", js)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Export the browser demo payload.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--demo-data", type=Path, default=Path("deploy/demo_data.json"))
    parser.add_argument("--out", type=Path, default=Path("build/browser"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = build(args.artifacts, args.demo_data, args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
