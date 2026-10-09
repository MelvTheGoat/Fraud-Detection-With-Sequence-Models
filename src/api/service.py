"""Scoring service: ONNX export, runtime selection, and the online feature path.

Latency budget
--------------
A card authorisation has an end-to-end budget of a few hundred milliseconds
before the acquirer times out, and the fraud model is one hop inside it, sharing
that budget with network, decisioning and the issuer's own core. This service
claims **p99 <= 50 ms** for scoring, measured server-side from request parse to
response. :mod:`src.api.benchmark` measures whether it holds, and ``/metrics``
reports the live distribution against the budget rather than asserting it.

Runtime
-------
ONNX Runtime is preferred over eager PyTorch for single-transaction scoring: the
graph is fixed, batch size is 1, and there is no Python-level module dispatch per
layer. Both are benchmarked rather than assumed — on models this small the gap is
not always large, and the honest number is the measured one.

ONNX Runtime is configured with one intra-op thread on purpose. At batch size 1
there is nothing to parallelise across, and thread hand-off shows up directly in
the tail latency that the budget is written against.

Explanations
------------
Contributing prior transactions are computed from the PyTorch model, since ONNX
gives no access to attention or gradients. They are therefore computed only when
the decision is not ``approve`` — about 1% of traffic — so the common path stays
on the fast runtime. Approved transactions return an empty contributor list.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.api.schemas import (
    ContributingTransaction,
    ScoreRequest,
    ScoreResponse,
    Transaction,
)
from src.config import ARTIFACT_DIR, ServingConfig
from src.data.features import NUMERIC_FEATURES, FeatureBuilder
from src.data.schema import PAD_INDEX, CategoricalSpec, TransactionSchema
from src.explain import ATTENTION_CAVEAT
from src.models import ModelConfig, SequenceClassifier, build_model

LOGGER = logging.getLogger(__name__)

S = TransactionSchema

FEATURE_BUILDER_FILE = "feature_builder.pkl"
DEPLOYMENT_FILE = "deployment.json"
CALIBRATOR_FILE = "calibrator.pkl"
ONNX_FILE = "model.onnx"


# --------------------------------------------------------------------------------------
# Checkpoints and ONNX export
# --------------------------------------------------------------------------------------


def load_checkpoint(path: Path) -> tuple[SequenceClassifier, ModelConfig, dict[str, Any]]:
    """Rebuild a trained model from a checkpoint written by :mod:`src.train`.

    Args:
        path: Path to the ``.pt`` checkpoint.

    Returns:
        Tuple of (model in eval mode, its config, the checkpoint payload).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = payload["model_config"]
    config = ModelConfig(
        n_numeric=raw["n_numeric"],
        categorical=[CategoricalSpec(**c) for c in raw["categorical"]],
        max_len=raw["max_len"],
        d_model=raw["d_model"],
        dropout=raw["dropout"],
        emb_max_dim=raw["emb_max_dim"],
        hidden=raw["hidden"],
        n_layers=raw["n_layers"],
        n_heads=raw["n_heads"],
        kernel_size=raw["kernel_size"],
        cell=raw["cell"],
        embedding_dims=raw["embedding_dims"],
    )
    model = build_model(payload["family"], config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, payload


def export_onnx(model: SequenceClassifier, config: ModelConfig, path: Path) -> Path:
    """Export a trained model to ONNX with a dynamic batch axis.

    Args:
        model: A trained model in eval mode.
        config: Its configuration.
        path: Destination ``.onnx`` file.

    Returns:
        The written path.
    """
    model.eval()
    dummy_numeric = torch.zeros(1, config.max_len, config.n_numeric, dtype=torch.float32)
    dummy_categorical = torch.zeros(
        1, config.max_len, len(config.categorical), dtype=torch.int64
    )
    dummy_mask = torch.ones(1, config.max_len, dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_numeric, dummy_categorical, dummy_mask),
        str(path),
        input_names=["numeric", "categorical", "mask"],
        output_names=["logit"],
        dynamic_axes={
            "numeric": {0: "batch"},
            "categorical": {0: "batch"},
            "mask": {0: "batch"},
            "logit": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    return path


# --------------------------------------------------------------------------------------
# Deployment bundle
# --------------------------------------------------------------------------------------


@dataclass
class Deployment:
    """Everything about the deployed model that is not weights.

    Attributes:
        model_name: Name of the selected arm, e.g. ``"gru_focal"``.
        model_family: Architecture family.
        model_version: Version string reported to callers.
        block_threshold: Calibrated probability at or above which to block.
        review_threshold: Probability at or above which to queue for review.
        calibration: Which calibrator is applied — ``"raw"``, ``"temperature"``
            or ``"isotonic"``.
        temperature: Fitted temperature, when applicable.
        n_parameters: Parameter count.
        trained_on: Description of the training data.
        selected_because: Why this arm was deployed.
    """

    model_name: str
    model_family: str
    model_version: str
    block_threshold: float
    review_threshold: float
    calibration: str = "raw"
    temperature: float | None = None
    n_parameters: int = 0
    trained_on: str = "synthetic"
    selected_because: str = ""

    def save(self, directory: Path) -> Path:
        """Write the deployment descriptor as JSON."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / DEPLOYMENT_FILE
        path.write_text(json.dumps(self.__dict__, indent=2))
        return path

    @staticmethod
    def load(directory: Path) -> Deployment:
        """Read a deployment descriptor written by :meth:`save`."""
        payload = json.loads((directory / DEPLOYMENT_FILE).read_text())
        return Deployment(**payload)


# --------------------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------------------


@dataclass
class ScoringService:
    """Loads the deployed model and scores single transactions.

    Attributes:
        builder: The fitted feature builder from training.
        model: The PyTorch model, used for explanations and as a runtime fallback.
        config: Model configuration.
        deployment: Deployment descriptor.
        serving: Serving configuration.
        session: ONNX Runtime session when available.
        calibrator: Fitted calibrator object, when the deployment uses one.
    """

    builder: FeatureBuilder
    model: SequenceClassifier
    config: ModelConfig
    deployment: Deployment
    serving: ServingConfig = field(default_factory=ServingConfig)
    session: Any | None = None
    calibrator: Any | None = None

    @property
    def runtime(self) -> str:
        """Name of the inference runtime in use."""
        return "onnxruntime" if self.session is not None else "pytorch"

    @classmethod
    def load(
        cls,
        directory: Path = ARTIFACT_DIR,
        *,
        prefer_onnx: bool = True,
        serving: ServingConfig | None = None,
    ) -> ScoringService:
        """Load a deployment bundle from disk.

        Args:
            directory: Directory holding the bundle.
            prefer_onnx: Use ONNX Runtime when the exported graph is present.
            serving: Serving configuration override.

        Returns:
            A ready service.

        Raises:
            FileNotFoundError: If the bundle is incomplete.
        """
        deployment = Deployment.load(directory)
        builder_path = directory / FEATURE_BUILDER_FILE
        if not builder_path.exists():
            raise FileNotFoundError(f"feature builder not found at {builder_path}")
        builder = FeatureBuilder.load(builder_path)

        checkpoint = directory / f"{deployment.model_name}_seed0.pt"
        if not checkpoint.exists():
            candidates = sorted(directory.glob(f"{deployment.model_name}_seed*.pt"))
            if not candidates:
                raise FileNotFoundError(f"no checkpoint for {deployment.model_name} in {directory}")
            checkpoint = candidates[0]
        model, config, _ = load_checkpoint(checkpoint)

        session = None
        onnx_path = directory / ONNX_FILE
        if prefer_onnx and onnx_path.exists():
            try:
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads = (serving or ServingConfig()).onnx_intra_op_threads
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
            except Exception:  # pragma: no cover - depends on optional runtime
                LOGGER.exception("failed to start ONNX Runtime; falling back to PyTorch")
                session = None

        calibrator = None
        calibrator_path = directory / CALIBRATOR_FILE
        if deployment.calibration != "raw" and calibrator_path.exists():
            import pickle

            with calibrator_path.open("rb") as handle:
                calibrator = pickle.load(handle)

        return cls(
            builder=builder,
            model=model,
            config=config,
            deployment=deployment,
            serving=serving or ServingConfig(),
            session=session,
            calibrator=calibrator,
        )

    # -- feature assembly ------------------------------------------------------------

    @property
    def required_history(self) -> int:
        """Prior transactions a request should carry for exact parity with training.

        The model reads a window of ``max_len`` positions, and *every* position
        in that window carries features built from its own ``lookback`` prior
        transactions. So reproducing the training-time inputs exactly needs
        ``max_len - 1`` transactions to fill the window plus ``lookback`` more
        behind the oldest of them.

        A shorter history still scores: the oldest window positions simply see
        less history than they did at training time, which degrades those
        positions' features rather than failing. The service does not reject
        such requests, because a real caller will sometimes have less history
        than the ideal, but the benchmark and the parity test both use the full
        depth.
        """
        return self.config.max_len - 1 + self.builder.lookback

    def build_window(self, request: ScoreRequest) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Turn a request into padded model inputs.

        Features are computed over the whole supplied history and the last
        ``max_len`` rows become the model window, so the scored transaction lands
        on the final position exactly as in training.

        Args:
            request: The scoring request.

        Returns:
            Tuple of (numeric ``(1, L, F)``, categorical ``(1, L, C)``, mask
            ``(1, L)``, number of prior transactions used).
        """
        max_len = self.config.max_len
        history = sorted(request.history, key=lambda t: t.timestamp)[-self.required_history :]
        records = [t.model_dump() for t in [*history, request.transaction]]
        numeric, categorical = self.builder.transform_window(records)
        # Features needed the deeper history; the model only sees the last L rows.
        numeric = numeric[-max_len:]
        categorical = categorical[-max_len:]

        n = numeric.shape[0]
        padded_numeric = np.zeros((1, max_len, numeric.shape[1]), dtype=np.float32)
        padded_categorical = np.full(
            (1, max_len, categorical.shape[1]), PAD_INDEX, dtype=np.int64
        )
        mask = np.zeros((1, max_len), dtype=np.float32)
        padded_numeric[0, max_len - n :] = numeric
        padded_categorical[0, max_len - n :] = categorical
        mask[0, max_len - n :] = 1.0
        return padded_numeric, padded_categorical, mask, len(history)

    # -- inference --------------------------------------------------------------------

    def _raw_score(
        self, numeric: np.ndarray, categorical: np.ndarray, mask: np.ndarray
    ) -> float:
        """Run the model and return a probability.

        Args:
            numeric: Padded numeric inputs.
            categorical: Padded categorical codes.
            mask: Padding mask.

        Returns:
            Sigmoid of the model logit.
        """
        if self.session is not None:
            logit = self.session.run(
                ["logit"], {"numeric": numeric, "categorical": categorical, "mask": mask}
            )[0]
            return float(1.0 / (1.0 + np.exp(-np.asarray(logit).reshape(-1)[0])))
        with torch.no_grad():
            logit = self.model(
                torch.from_numpy(numeric), torch.from_numpy(categorical), torch.from_numpy(mask)
            )
        return float(torch.sigmoid(logit).item())

    def calibrate(self, raw: float) -> float:
        """Apply the deployed calibration map to a raw score.

        Args:
            raw: Uncalibrated probability.

        Returns:
            Calibrated probability.
        """
        if self.calibrator is None:
            return raw
        return float(np.asarray(self.calibrator.transform(np.asarray([raw]))).reshape(-1)[0])

    def decide(self, probability: float) -> str:
        """Map a probability to an action.

        Args:
            probability: Calibrated fraud probability.

        Returns:
            ``"block"``, ``"review"`` or ``"approve"``.
        """
        if probability >= self.deployment.block_threshold:
            return "block"
        if probability >= self.deployment.review_threshold:
            return "review"
        return "approve"

    def contributors(
        self,
        request: ScoreRequest,
        numeric: np.ndarray,
        categorical: np.ndarray,
        mask: np.ndarray,
        limit: int = 3,
    ) -> tuple[list[ContributingTransaction], str | None]:
        """Prior transactions the model read from, most relevant first.

        Args:
            request: The original request, for rendering the transactions.
            numeric: Padded numeric inputs.
            categorical: Padded categorical codes.
            mask: Padding mask.
            limit: How many to return.

        Returns:
            Tuple of (contributors, caveat text when attention-derived).
        """
        max_len = self.config.max_len
        history = sorted(request.history, key=lambda t: t.timestamp)[-(max_len - 1) :]
        window: list[Transaction] = [*history, request.transaction]

        with torch.no_grad():
            weights = self.model.attention_weights(
                torch.from_numpy(numeric), torch.from_numpy(categorical), torch.from_numpy(mask)
            )
        if weights is None:
            # No attention: fall back to gradient x input over positions, which
            # answers "how much does the logit move with this position's inputs".
            tensor = torch.from_numpy(numeric).requires_grad_(True)
            logit = self.model(
                tensor, torch.from_numpy(categorical), torch.from_numpy(mask)
            )
            self.model.zero_grad(set_to_none=True)
            logit.backward()
            grad = tensor.grad
            scores = (grad * tensor).detach()[0].sum(dim=1).numpy() if grad is not None else np.zeros(max_len)
            caveat = None
        else:
            scores = weights[0].numpy()
            caveat = ATTENTION_CAVEAT

        used = scores[max_len - len(window) :]
        order = np.argsort(-np.abs(used))[:limit]
        out = []
        for position in sorted(order.tolist()):
            transaction = window[position]
            out.append(
                ContributingTransaction(
                    txn_id=transaction.txn_id,
                    timestamp=transaction.timestamp,
                    amount=transaction.amount,
                    merchant_id=transaction.merchant_id,
                    device_id=transaction.device_id,
                    offset=position - (len(window) - 1),
                    attention=float(used[position]) if weights is not None else None,
                )
            )
        out.sort(key=lambda c: abs(c.attention if c.attention is not None else 0.0), reverse=True)
        return out, caveat

    def score(self, request: ScoreRequest, *, explain: bool = True) -> ScoreResponse:
        """Score one transaction end to end.

        Args:
            request: The scoring request.
            explain: Compute contributors for non-approved decisions.

        Returns:
            A populated :class:`~src.api.schemas.ScoreResponse`.
        """
        started = time.perf_counter()
        numeric, categorical, mask, history_used = self.build_window(request)
        raw = self._raw_score(numeric, categorical, mask)
        probability = self.calibrate(raw)
        decision = self.decide(probability)

        contributors: list[ContributingTransaction] = []
        caveat: str | None = None
        if explain and decision != "approve":
            contributors, caveat = self.contributors(request, numeric, categorical, mask)

        latency_ms = (time.perf_counter() - started) * 1000.0
        return ScoreResponse(
            txn_id=request.transaction.txn_id,
            customer_id=request.customer_id,
            fraud_probability=probability,
            raw_score=raw,
            decision=decision,  # type: ignore[arg-type]
            threshold=self.deployment.block_threshold,
            model_version=self.deployment.model_version,
            model_family=self.deployment.model_family,
            runtime=self.runtime,
            latency_ms=latency_ms,
            history_used=history_used,
            top_contributors=contributors,
            attention_caveat=caveat,
        )

    def embedding_dims(self) -> dict[str, int]:
        """Embedding width per categorical column, for ``/model-info``."""
        return {
            spec.column: dim
            for spec, dim in zip(self.config.categorical, self.config.embedding_dims, strict=True)
        }

    def n_numeric_features(self) -> int:
        """Number of numeric features per position."""
        return len(NUMERIC_FEATURES)
