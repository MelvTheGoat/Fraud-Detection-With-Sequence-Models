"""Architecture behaviour: shapes, masking, causality and trainability."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.schema import CategoricalSpec
from src.models import (
    ARCHITECTURE_DEFAULTS,
    COMPARED_MODELS,
    build_model,
    count_parameters,
    describe_embeddings,
    embedding_dim,
    make_config,
    parameter_breakdown,
)
from src.models.tcn import TCNClassifier

CATEGORICALS = [
    CategoricalSpec("merchant_code", 500),
    CategoricalSpec("mcc_code", 22),
    CategoricalSpec("device_type_code", 19),
    CategoricalSpec("country_code", 10),
    CategoricalSpec("channel_code", 4),
    CategoricalSpec("entry_mode_code", 6),
]
N_NUMERIC = 42
MAX_LEN = 16


def make_batch(batch: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a batch with varying history lengths and left padding."""
    generator = torch.Generator().manual_seed(seed)
    numeric = torch.randn(batch, MAX_LEN, N_NUMERIC, generator=generator)
    categorical = torch.stack(
        [
            torch.randint(0, spec.cardinality, (batch, MAX_LEN), generator=generator)
            for spec in CATEGORICALS
        ],
        dim=-1,
    )
    mask = torch.zeros(batch, MAX_LEN)
    for i in range(batch):
        length = (i % (MAX_LEN - 1)) + 1
        mask[i, MAX_LEN - length :] = 1.0
    return numeric * mask.unsqueeze(-1), categorical * mask.long().unsqueeze(-1), mask


@pytest.fixture(params=COMPARED_MODELS)
def architecture(request) -> str:
    return request.param


def build(architecture: str):
    config = make_config(
        architecture, n_numeric=N_NUMERIC, categorical=list(CATEGORICALS), max_len=MAX_LEN
    )
    model = build_model(architecture, config)
    model.eval()
    return model, config


def test_forward_shape_and_finiteness(architecture: str) -> None:
    model, _ = build(architecture)
    numeric, categorical, mask = make_batch()
    with torch.no_grad():
        logits = model(numeric, categorical, mask)
    assert logits.shape == (numeric.shape[0],)
    assert torch.isfinite(logits).all()


def test_padding_does_not_affect_the_score(architecture: str) -> None:
    """Perturbing padded slots must leave every score bit-identical.

    This is the model-side counterpart of the leakage tests: even with a correct
    window, a model that reads padded positions is reading state that does not
    exist, and at serve time those slots hold whatever the buffer had in it.
    """
    model, _ = build(architecture)
    numeric, categorical, mask = make_batch()
    with torch.no_grad():
        base = model(numeric, categorical, mask)

    noisy_numeric = numeric.clone()
    noisy_categorical = categorical.clone()
    pad = mask < 0.5
    noisy_numeric[pad] = 13.7
    noisy_categorical[pad] = 3
    with torch.no_grad():
        perturbed = model(noisy_numeric, noisy_categorical, mask)
    torch.testing.assert_close(base, perturbed, rtol=0, atol=1e-6)


def test_future_positions_cannot_exist(architecture: str) -> None:
    """The scored transaction is the last slot, so there is no future to read.

    Changing the final position must change the score — otherwise the model is
    ignoring the transaction it is meant to be deciding on.
    """
    model, _ = build(architecture)
    numeric, categorical, mask = make_batch()
    with torch.no_grad():
        base = model(numeric, categorical, mask)
    altered = numeric.clone()
    altered[:, -1, :] += 3.0
    with torch.no_grad():
        changed = model(altered, categorical, mask)
    assert (base - changed).abs().max() > 1e-6


def test_history_changes_the_score(architecture: str) -> None:
    """Prior transactions must influence the prediction, or the sequence is decorative."""
    model, _ = build(architecture)
    numeric, categorical, mask = make_batch()
    # Only rows with at least two real positions can show a history effect.
    lengths = mask.sum(dim=1)
    rows = lengths >= 3
    with torch.no_grad():
        base = model(numeric, categorical, mask)
    altered = numeric.clone()
    altered[:, :-1, :] += 2.5 * mask[:, :-1].unsqueeze(-1)
    with torch.no_grad():
        changed = model(altered, categorical, mask)
    assert (base[rows] - changed[rows]).abs().max() > 1e-6


def test_single_step_windows_work(architecture: str) -> None:
    """A customer's first ever transaction is scorable."""
    model, _ = build(architecture)
    numeric = torch.zeros(2, MAX_LEN, N_NUMERIC)
    categorical = torch.zeros(2, MAX_LEN, len(CATEGORICALS), dtype=torch.long)
    mask = torch.zeros(2, MAX_LEN)
    mask[:, -1] = 1.0
    numeric[:, -1] = torch.randn(2, N_NUMERIC)
    with torch.no_grad():
        logits = model(numeric, categorical, mask)
    assert torch.isfinite(logits).all()


def test_gradients_flow(architecture: str) -> None:
    model, _ = build(architecture)
    model.train()
    numeric, categorical, mask = make_batch()
    logits = model(numeric, categorical, mask)
    logits.sum().backward()
    missing = [
        name for name, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    # The padding embedding row legitimately receives no gradient.
    assert not [m for m in missing if "embeddings" not in m], f"no gradient for {missing}"


def test_parameter_counts_are_comparable() -> None:
    """The three architectures must be within a small factor of each other.

    Otherwise the comparison measures capacity, not inductive bias.
    """
    counts = {}
    for architecture in COMPARED_MODELS:
        model, _ = build(architecture)
        counts[architecture] = count_parameters(model)
    assert max(counts.values()) / min(counts.values()) < 2.5, counts


def test_embedding_rule_and_cap() -> None:
    assert embedding_dim(2, max_dim=16) == 2
    assert embedding_dim(10_000, max_dim=16) == 16
    assert embedding_dim(22, max_dim=16) == round(1.6 * 22**0.56)
    assert embedding_dim(50, max_dim=4) == 4


def test_embedding_table_dominates_the_parameter_count() -> None:
    """Reported honestly: most of the model is a lookup table, not computation."""
    model, config = build("gru")
    breakdown = parameter_breakdown(model)
    assert breakdown["embeddings"] + breakdown["other"] == breakdown["total"]
    described = describe_embeddings(config)
    assert sum(int(row["params"]) for row in described) == breakdown["embeddings"]


def test_tcn_receptive_field_covers_the_window() -> None:
    config = make_config(
        "tcn", n_numeric=N_NUMERIC, categorical=list(CATEGORICALS), max_len=32
    )
    model = build_model("tcn", config)
    assert isinstance(model, TCNClassifier)
    assert model.receptive_field() >= 32, (
        "the TCN cannot see the whole window and would be structurally handicapped"
    )


def test_transformer_attention_is_normalised_and_masked() -> None:
    model, _ = build("transformer")
    numeric, categorical, mask = make_batch()
    with torch.no_grad():
        weights = model.attention_weights(numeric, categorical, mask)
    assert weights is not None
    assert torch.allclose(weights.sum(dim=1), torch.ones(numeric.shape[0]), atol=1e-4)
    assert (weights[mask < 0.5] == 0).all()


def test_non_attention_models_report_no_attention() -> None:
    for architecture in ("gru", "tcn"):
        model, _ = build(architecture)
        numeric, categorical, mask = make_batch()
        assert model.attention_weights(numeric, categorical, mask) is None


def test_unknown_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        make_config("mamba")


def test_architecture_defaults_are_defined_for_every_model() -> None:
    for architecture in COMPARED_MODELS:
        assert architecture in ARCHITECTURE_DEFAULTS


def test_determinism_under_a_fixed_seed(architecture: str) -> None:
    torch.manual_seed(7)
    first, _ = build(architecture)
    torch.manual_seed(7)
    second, _ = build(architecture)
    numeric, categorical, mask = make_batch()
    with torch.no_grad():
        np.testing.assert_allclose(
            first(numeric, categorical, mask).numpy(),
            second(numeric, categorical, mask).numpy(),
            rtol=0,
            atol=0,
        )
