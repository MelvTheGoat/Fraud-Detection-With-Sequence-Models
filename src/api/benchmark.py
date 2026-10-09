"""Latency benchmark: ONNX Runtime against eager PyTorch, plus the full request path.

Three things are measured separately, because they answer different questions:

``model``
    Forward pass only, batch size 1, on identical inputs. This is the number that
    tells you whether ONNX export was worth doing.

``features``
    Building the padded window from a request: the causal feature computation and
    the vocabulary and risk lookups. On models this small the feature path is
    often the larger half of the budget, which is exactly the sort of thing that
    gets missed when only the model is timed.

``end_to_end``
    Everything the service does per request, minus HTTP. This is what the p99
    budget in :class:`~src.config.ServingConfig` is written against.

Percentiles rather than means throughout. A mean latency hides the tail, and the
tail is what breaches an authorisation timeout.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.api.schemas import ScoreRequest, Transaction
from src.api.service import ScoringService
from src.config import ARTIFACT_DIR, RESULTS_DIR, ServingConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """Latency distribution for one measured path.

    Attributes:
        name: What was measured.
        n: Number of samples.
        mean_ms: Mean latency.
        p50_ms: Median.
        p95_ms: 95th percentile.
        p99_ms: 99th percentile.
        max_ms: Worst observed.
    """

    name: str
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    def to_row(self) -> dict[str, float | str | int]:
        """Flatten to a dict."""
        return {
            "path": self.name,
            "n": self.n,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
        }


def measure(name: str, fn: Callable[[], Any], n: int = 500, warmup: int = 50) -> LatencyStats:
    """Time a callable and summarise the distribution.

    Args:
        name: Label for the measured path.
        fn: Zero-argument callable to time.
        n: Number of timed iterations.
        warmup: Untimed iterations first, so lazy initialisation and any
            first-call graph setup do not land in the percentiles.

    Returns:
        Populated :class:`LatencyStats`.
    """
    for _ in range(warmup):
        fn()
    samples = np.empty(n, dtype=np.float64)
    for i in range(n):
        started = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - started) * 1000.0
    return LatencyStats(
        name=name,
        n=n,
        mean_ms=float(samples.mean()),
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        p99_ms=float(np.percentile(samples, 99)),
        max_ms=float(samples.max()),
    )


def synthetic_request(service: ScoringService, history_length: int | None = None) -> ScoreRequest:
    """Build a plausible request for benchmarking.

    Args:
        service: The loaded service, used for the window length.
        history_length: How many prior transactions to include. Defaults to the
            full depth the service needs for exact parity with training, which
            is also the most expensive case to build features for.

    Returns:
        A request with a full history, which is the expensive case.
    """
    import pandas as pd

    base = pd.Timestamp("2024-01-15T09:00:00")
    length = service.required_history if history_length is None else history_length

    def make(i: int, txn_id: str) -> Transaction:
        return Transaction(
            txn_id=txn_id,
            timestamp=(base + pd.Timedelta(hours=6 * i)).to_pydatetime(),
            amount=float(10 + (i * 7) % 90),
            merchant_id=f"M{(i * 37) % 500:05d}",
            mcc="5411" if i % 3 else "5812",
            device_id=f"D000001_{i % 3}",
            device_type="ios_safari" if i % 2 else "android_chrome",
            country="GB",
            channel="online" if i % 2 else "in_store",
            entry_mode="ecom" if i % 2 else "chip",
        )

    history = [make(i, f"T{i:08d}") for i in range(length)]
    return ScoreRequest(
        customer_id="C000001",
        transaction=make(length, "T99999999"),
        history=history,
    )


def benchmark(
    directory: Path = ARTIFACT_DIR, n: int = 500, history_length: int | None = None
) -> dict[str, Any]:
    """Benchmark both runtimes and the full request path.

    Args:
        directory: Deployment bundle directory.
        n: Timed iterations per measurement.
        history_length: History length in the synthetic request.

    Returns:
        Dict with per-path stats, the speedup, and whether the budget holds.
    """
    serving = ServingConfig()
    onnx_service = ScoringService.load(directory, prefer_onnx=True, serving=serving)
    torch_service = ScoringService.load(directory, prefer_onnx=False, serving=serving)
    request = synthetic_request(onnx_service, history_length)

    numeric, categorical, mask, _ = onnx_service.build_window(request)
    torch_inputs = (
        torch.from_numpy(numeric),
        torch.from_numpy(categorical),
        torch.from_numpy(mask),
    )

    stats: list[LatencyStats] = []

    def torch_forward() -> Any:
        with torch.no_grad():
            return torch_service.model(*torch_inputs)

    stats.append(measure("model:pytorch", torch_forward, n=n))

    if onnx_service.session is not None:
        session = onnx_service.session

        def onnx_forward() -> Any:
            return session.run(
                ["logit"], {"numeric": numeric, "categorical": categorical, "mask": mask}
            )

        stats.append(measure("model:onnxruntime", onnx_forward, n=n))
    else:
        LOGGER.warning("no ONNX graph found; skipping the ONNX Runtime measurement")

    stats.append(measure("features", lambda: onnx_service.build_window(request), n=n))
    stats.append(
        measure("end_to_end:onnxruntime", lambda: onnx_service.score(request), n=n)
        if onnx_service.session is not None
        else measure("end_to_end:pytorch", lambda: onnx_service.score(request), n=n)
    )
    stats.append(measure("end_to_end:pytorch", lambda: torch_service.score(request), n=n))

    by_name = {s.name: s for s in stats}
    speedup = None
    if "model:onnxruntime" in by_name:
        speedup = by_name["model:pytorch"].p50_ms / max(by_name["model:onnxruntime"].p50_ms, 1e-9)

    deployed = by_name.get("end_to_end:onnxruntime") or by_name["end_to_end:pytorch"]
    return {
        "model_name": onnx_service.deployment.model_name,
        "runtime": onnx_service.runtime,
        "history_length": len(request.history),
        "stats": [s.to_row() for s in stats],
        "onnx_speedup_p50": speedup,
        "latency_budget_p99_ms": serving.latency_budget_p99_ms,
        "deployed_p99_ms": deployed.p99_ms,
        "within_budget": bool(deployed.p99_ms <= serving.latency_budget_p99_ms),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the benchmark and write the results."""
    parser = argparse.ArgumentParser(description="Benchmark scoring latency.")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--history", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = benchmark(args.artifacts, n=args.n, history_length=args.history)

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "latency.json").write_text(json.dumps(report, indent=2))

    print(f"model: {report['model_name']}  runtime: {report['runtime']}")
    print(f"{'path':<28} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    for row in report["stats"]:
        print(
            f"{row['path']:<28} {row['p50_ms']:>8.3f} {row['p95_ms']:>8.3f} "
            f"{row['p99_ms']:>8.3f} {row['max_ms']:>8.3f}"
        )
    if report["onnx_speedup_p50"]:
        print(f"\nONNX Runtime vs PyTorch (model only, p50): {report['onnx_speedup_p50']:.2f}x")
    verdict = "MET" if report["within_budget"] else "BREACHED"
    print(
        f"budget p99 <= {report['latency_budget_p99_ms']:.0f} ms: "
        f"{verdict} (deployed p99 {report['deployed_p99_ms']:.2f} ms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
