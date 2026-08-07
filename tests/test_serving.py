"""Serving: online/offline feature parity, ONNX equivalence, the API, and the audit log.

The parity test is the important one. The classic production failure in a fraud
system is not a bad model, it is an online feature pipeline that disagrees with
the offline one — a different window boundary, a different unknown-category
fallback, a different scaling constant — so the deployed model is fed inputs it
was never trained on and quietly degrades. Here both paths call the same
function, and this asserts it stays that way.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.audit import AuditLog, AuditRecord, now_iso
from src.api.schemas import ScoreRequest, Transaction
from src.api.service import Deployment, ScoringService, export_onnx, load_checkpoint
from src.config import SequenceConfig, TrainConfig
from src.data.schema import PAD_INDEX, TransactionSchema
from src.data.sequences import gather
from src.train import train_one

S = TransactionSchema


@pytest.fixture(scope="session")
def deployment_dir(
    tmp_path_factory, sequence_data, split, builder_and_frame
) -> Path:
    """Build a minimal but complete deployment bundle."""
    builder, _ = builder_and_frame
    directory = tmp_path_factory.mktemp("bundle")

    result = train_one(
        "transformer",
        sequence_data,
        split,
        strategy="focal",
        seed=0,
        train_config=TrainConfig(max_epochs=1, batch_size=256),
    )
    result.scores.name = "transformer_focal"
    result.save(directory)
    export_onnx(result.model, result.config, directory / "model.onnx")
    builder.save(directory / "feature_builder.pkl")
    Deployment(
        model_name="transformer_focal",
        model_family="transformer",
        model_version="test-0.1",
        block_threshold=0.5,
        review_threshold=0.1,
        calibration="raw",
        n_parameters=result.scores.n_params,
        trained_on="synthetic (tests)",
        selected_because="fixture",
    ).save(directory)
    return directory


@pytest.fixture(scope="session")
def service(deployment_dir: Path) -> ScoringService:
    return ScoringService.load(deployment_dir)


def _request_from_frame(frame: pd.DataFrame, row: int, history_depth: int) -> ScoreRequest:
    """Rebuild a scoring request from a feature frame row and its true history.

    Args:
        frame: The offline feature frame.
        row: Row to score.
        history_depth: How many prior transactions to include. Use
            :attr:`ScoringService.required_history` for exact parity.
    """
    target = frame.iloc[row]
    customer = target[S.CUSTOMER_ID]
    history = frame[
        (frame[S.CUSTOMER_ID] == customer) & (frame.index < row)
    ].tail(history_depth)

    def to_transaction(source: pd.Series) -> Transaction:
        return Transaction(
            txn_id=str(source[S.TXN_ID]),
            timestamp=pd.Timestamp(source[S.TIMESTAMP]).to_pydatetime(),
            amount=float(source[S.AMOUNT]),
            merchant_id=str(source[S.MERCHANT_ID]),
            mcc=str(source[S.MCC]),
            device_id=str(source[S.DEVICE_ID]),
            device_type=str(source[S.DEVICE_TYPE]),
            country=str(source[S.COUNTRY]),
            channel=str(source[S.CHANNEL]),
            entry_mode=str(source[S.ENTRY_MODE]),
        )

    return ScoreRequest(
        customer_id=str(customer),
        transaction=to_transaction(target),
        history=[to_transaction(r) for _, r in history.iterrows()],
    )


# --------------------------------------------------------------------------------------
# Online / offline parity
# --------------------------------------------------------------------------------------


def _deployment_regime_rows(
    frame: pd.DataFrame, split, depth: int, limit: int = 20
) -> list[int]:
    """Rows whose entire feature window lies after the training period.

    This is the regime a deployed model actually operates in, and the one where
    parity is exact. Windows overlapping the training period contain rows whose
    four label-derived encodings were computed out-of-fold at training time and
    cannot be reproduced online: an out-of-fold encoding is a property of the
    training procedure, not of the transaction. See DECISIONS.md.

    Args:
        frame: The offline feature frame.
        split: The temporal split.
        depth: Required history depth.
        limit: Maximum rows to return.

    Returns:
        Row indices with a full, in-regime window.
    """
    after_training = ~split.train
    out: list[int] = []
    for positions in frame.groupby(S.CUSTOMER_ID).indices.values():
        for p in range(depth, len(positions)):
            window = positions[p - depth : p + 1]
            if after_training[window].all():
                out.append(int(positions[p]))
                break
        if len(out) >= limit:
            break
    return out


def test_serving_features_match_the_training_pipeline(
    service: ScoringService, builder_and_frame, sequence_data, split
) -> None:
    """The online window must equal the offline one, feature for feature.

    Any disagreement here means the deployed model is being fed something other
    than what it was trained on — the single most common way a fraud model that
    validated well underperforms in production.
    """
    _, frame = builder_and_frame
    depth = service.required_history
    candidates = _deployment_regime_rows(frame, split, depth)
    assert candidates, "no rows with a full window inside the deployment regime"

    for row in candidates:
        request = _request_from_frame(frame, row, depth)
        online_numeric, online_categorical, online_mask, _ = service.build_window(request)
        offline = gather(sequence_data, np.asarray([row]))

        np.testing.assert_array_equal(online_mask, offline["mask"].numpy())
        np.testing.assert_array_equal(online_categorical, offline["categorical"].numpy())
        np.testing.assert_allclose(
            online_numeric, offline["numeric"].numpy(), rtol=1e-4, atol=1e-4
        )


def test_causal_features_match_even_across_the_training_boundary(
    service: ScoringService, builder_and_frame, sequence_data
) -> None:
    """The 37 causal features match everywhere, not only in the deployment regime.

    Only the four label-derived risk encodings can differ, and only for windows
    overlapping the training period. Pinning that down stops a future change
    from widening the gap unnoticed.
    """
    from src.data.features import CAUSAL_FEATURES, NUMERIC_FEATURES

    _, frame = builder_and_frame
    depth = service.required_history
    causal_index = [list(NUMERIC_FEATURES).index(name) for name in CAUSAL_FEATURES]
    candidates = np.flatnonzero(frame["log_hist_count"].to_numpy() > np.log1p(20))[:15]

    for row in candidates:
        request = _request_from_frame(frame, int(row), depth)
        online_numeric, _, _, _ = service.build_window(request)
        offline = gather(sequence_data, np.asarray([row]))["numeric"].numpy()
        np.testing.assert_allclose(
            online_numeric[..., causal_index],
            offline[..., causal_index],
            rtol=1e-4,
            atol=1e-4,
        )


def test_serving_scores_match_the_batch_scores(
    service: ScoringService, builder_and_frame, sequence_data, split
) -> None:
    """Same inputs, same weights, same score — through the whole serving path."""
    _, frame = builder_and_frame
    depth = service.required_history
    candidates = _deployment_regime_rows(frame, split, depth, limit=10)
    assert candidates

    for row in candidates:
        request = _request_from_frame(frame, row, depth)
        online = service.score(request, explain=False)
        batch = gather(sequence_data, np.asarray([row]))
        with torch.no_grad():
            expected = float(
                torch.sigmoid(
                    service.model(batch["numeric"], batch["categorical"], batch["mask"])
                ).item()
            )
        assert online.raw_score == pytest.approx(expected, abs=2e-4)


def test_onnx_matches_pytorch(service: ScoringService, deployment_dir: Path) -> None:
    """The exported graph must compute the same function as the eager model."""
    if service.session is None:
        pytest.skip("onnxruntime is unavailable")
    torch_service = ScoringService.load(deployment_dir, prefer_onnx=False)
    assert torch_service.session is None

    rng = np.random.default_rng(0)
    max_len = service.config.max_len
    for _ in range(20):
        length = int(rng.integers(1, max_len + 1))
        numeric = np.zeros((1, max_len, service.config.n_numeric), dtype=np.float32)
        categorical = np.full(
            (1, max_len, len(service.config.categorical)), PAD_INDEX, dtype=np.int64
        )
        mask = np.zeros((1, max_len), dtype=np.float32)
        numeric[0, max_len - length :] = rng.normal(size=(length, service.config.n_numeric))
        for j, spec in enumerate(service.config.categorical):
            categorical[0, max_len - length :, j] = rng.integers(1, spec.cardinality, size=length)
        mask[0, max_len - length :] = 1.0

        onnx_score = service._raw_score(numeric, categorical, mask)
        torch_score = torch_service._raw_score(numeric, categorical, mask)
        assert onnx_score == pytest.approx(torch_score, abs=1e-5)


def test_history_longer_than_the_window_is_truncated(
    service: ScoringService, builder_and_frame
) -> None:
    _, frame = builder_and_frame
    request = _request_from_frame(frame, 500, 500)
    _, _, mask, used = service.build_window(request)
    assert mask.sum() <= service.config.max_len
    assert used <= service.required_history


def test_a_customers_first_transaction_is_scorable(
    service: ScoringService, builder_and_frame
) -> None:
    """No history is a valid state, not an error."""
    _, frame = builder_and_frame
    request = _request_from_frame(frame, 0, service.required_history)
    request.history = []
    response = service.score(request)
    assert 0.0 <= response.fraud_probability <= 1.0
    assert response.history_used == 0


def test_unknown_entities_fall_back_to_the_unknown_code(
    service: ScoringService, builder_and_frame
) -> None:
    """A merchant nobody has ever seen must not crash the scorer."""
    _, frame = builder_and_frame
    request = _request_from_frame(frame, 800, service.required_history)
    request.transaction.merchant_id = "M_NEVER_SEEN_BEFORE"
    request.transaction.device_type = "quantum_toaster"
    response = service.score(request)
    assert 0.0 <= response.fraud_probability <= 1.0


# --------------------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------------------


def test_audit_log_appends_and_verifies(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "decisions.jsonl")
    for i in range(5):
        log.write(
            AuditRecord(
                timestamp=now_iso(),
                txn_id=f"T{i}",
                customer_id="C1",
                fraud_probability=0.1 * i,
                raw_score=0.1 * i,
                decision="approve",
                threshold=0.5,
                model_version="v1",
                model_name="gru_focal",
                runtime="onnxruntime",
                latency_ms=1.0,
                history_used=3,
                amount=10.0,
                merchant_id="M1",
            )
        )
    records = log.read_all()
    assert len(records) == 5
    assert [r["txn_id"] for r in records] == [f"T{i}" for i in range(5)]
    assert all(r["record_hash"] for r in records)
    assert log.verify() == []


def test_audit_log_detects_tampering(tmp_path: Path) -> None:
    """A rewritten decision must be detectable after the fact."""
    path = tmp_path / "decisions.jsonl"
    log = AuditLog(path)
    log.write(
        AuditRecord(
            timestamp=now_iso(),
            txn_id="T1",
            customer_id="C1",
            fraud_probability=0.9,
            raw_score=0.9,
            decision="block",
            threshold=0.5,
            model_version="v1",
            model_name="gru_focal",
            runtime="pytorch",
            latency_ms=1.0,
            history_used=3,
            amount=10.0,
            merchant_id="M1",
        )
    )
    record = json.loads(path.read_text())
    record["decision"] = "approve"
    path.write_text(json.dumps(record) + "\n")
    assert log.verify() == [0]


def test_audit_log_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert AuditLog(tmp_path / "nothing.jsonl").read_all() == []


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------


@pytest.fixture
def client(deployment_dir: Path, tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("FRAUD_ARTIFACT_DIR", str(deployment_dir))
    monkeypatch.setenv("FRAUD_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    from src.api import app as app_module

    app_module.state = app_module._State()
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_health_reports_a_loaded_model(client: TestClient) -> None:
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_model_info_describes_the_deployment(client: TestClient) -> None:
    payload = client.get("/model-info").json()
    assert payload["model_family"] == "transformer"
    assert payload["n_parameters"] > 0
    assert payload["window_length"] == SequenceConfig().max_len
    assert payload["n_numeric_features"] > 0
    assert "merchant_code" in payload["embedding_dims"]
    assert payload["latency_budget_p99_ms"] > 0


def test_score_endpoint_returns_a_decision(client: TestClient, builder_and_frame) -> None:
    _, frame = builder_and_frame
    request = _request_from_frame(frame, 900, 60)
    response = client.post("/score", json=json.loads(request.model_dump_json()))
    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["fraud_probability"] <= 1.0
    assert payload["decision"] in {"approve", "review", "block"}
    assert payload["model_version"] == "test-0.1"
    assert payload["latency_ms"] > 0
    assert payload["history_used"] > 0


def test_score_writes_an_audit_record(client: TestClient, builder_and_frame, tmp_path: Path) -> None:
    _, frame = builder_and_frame
    request = _request_from_frame(frame, 901, 60)
    client.post("/score", json=json.loads(request.model_dump_json()))
    records = AuditLog(tmp_path / "audit.jsonl").read_all()
    assert len(records) == 1
    assert records[0]["txn_id"] == request.transaction.txn_id
    assert records[0]["model_name"] == "transformer_focal"


def test_flagged_transactions_carry_contributors_and_the_caveat(
    client: TestClient, builder_and_frame, deployment_dir: Path
) -> None:
    """Blocked decisions must explain themselves, with attention's limits stated."""
    service = ScoringService.load(deployment_dir)
    service.deployment.block_threshold = 0.0  # force a block
    _, frame = builder_and_frame
    request = _request_from_frame(frame, 902, service.required_history)
    response = service.score(request)
    assert response.decision == "block"
    assert response.top_contributors
    assert response.attention_caveat is not None
    assert "not" in response.attention_caveat.lower()
    offsets = [c.offset for c in response.top_contributors]
    assert all(o <= 0 for o in offsets), "a contributor cannot be in the future"


def test_metrics_track_latency_and_decisions(client: TestClient, builder_and_frame) -> None:
    _, frame = builder_and_frame
    for row in (910, 911, 912):
        request = _request_from_frame(frame, row, 60)
        client.post("/score", json=json.loads(request.model_dump_json()))
    payload = client.get("/metrics").json()
    assert payload["requests_total"] == 3
    assert payload["latency_ms_p50"] > 0
    assert payload["latency_ms_p99"] >= payload["latency_ms_p50"]
    assert sum(payload["decisions"].values()) == 3
    assert payload["latency_budget_p99_ms"] > 0


def test_malformed_request_is_rejected(client: TestClient) -> None:
    response = client.post("/score", json={"customer_id": "C1"})
    assert response.status_code == 422


def test_service_reports_degraded_without_a_bundle(tmp_path: Path, monkeypatch) -> None:
    """An up-but-modelless service must not report itself healthy."""
    monkeypatch.setenv("FRAUD_ARTIFACT_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("FRAUD_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    from src.api import app as app_module

    app_module.state = app_module._State()
    with TestClient(app_module.app) as test_client:
        health = test_client.get("/health").json()
        assert health["status"] == "degraded"
        assert health["model_loaded"] is False
        assert test_client.post("/score", json={}).status_code in {422, 503}
        assert test_client.get("/model-info").status_code == 503


def test_checkpoint_round_trip(deployment_dir: Path) -> None:
    model, config, payload = load_checkpoint(deployment_dir / "transformer_focal_seed0.pt")
    assert payload["family"] == "transformer"
    assert config.max_len == SequenceConfig().max_len
    assert not model.training
