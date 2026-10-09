"""Sequence architectures sharing one input encoder and one training interface."""

from __future__ import annotations

from src.models.base import (
    ModelConfig,
    SequenceClassifier,
    count_parameters,
    describe_embeddings,
    embedding_dim,
    parameter_breakdown,
)
from src.models.rnn import RecurrentClassifier
from src.models.tcn import TCNClassifier
from src.models.transformer import TransformerClassifier

#: Architecture registry. Keys are the names used on the command line, in
#: results tables and in score filenames.
MODEL_REGISTRY: dict[str, type[SequenceClassifier]] = {
    "gru": RecurrentClassifier,
    "lstm": RecurrentClassifier,
    "tcn": TCNClassifier,
    "transformer": TransformerClassifier,
}

#: The three architectures compared in the write-up.
COMPARED_MODELS: tuple[str, ...] = ("gru", "tcn", "transformer")

#: Per-architecture defaults, chosen to keep the three models within a factor of
#: about two of each other in parameter count so the comparison is about
#: inductive bias rather than capacity. The TCN needs four blocks because
#: dilations 1/2/4/8 with kernel 3 give a receptive field of 61, which covers the
#: 32-position window; two blocks would reach only 13 and the model would be
#: structurally unable to see most of the history the others get.
ARCHITECTURE_DEFAULTS: dict[str, dict[str, int]] = {
    "gru": {"n_layers": 1, "hidden": 64},
    "lstm": {"n_layers": 1, "hidden": 64},
    "tcn": {"n_layers": 4, "hidden": 48, "kernel_size": 3},
    "transformer": {"n_layers": 2, "hidden": 64, "n_heads": 4},
}


def make_config(name: str, **kwargs: object) -> ModelConfig:
    """Build a :class:`ModelConfig` with the architecture's defaults applied.

    Args:
        name: Architecture key.
        **kwargs: Fields for :class:`ModelConfig`; explicit values win over the
            architecture defaults.

    Returns:
        A configuration ready to pass to :func:`build_model`.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{name}'; choose from {sorted(MODEL_REGISTRY)}")
    merged: dict[str, object] = dict(ARCHITECTURE_DEFAULTS[name])
    merged.update(kwargs)
    if name in {"gru", "lstm"}:
        merged["cell"] = name
    return ModelConfig(**merged)  # type: ignore[arg-type]


def build_model(name: str, config: ModelConfig) -> SequenceClassifier:
    """Instantiate an architecture by name.

    Args:
        name: Key from :data:`MODEL_REGISTRY`.
        config: Model configuration. For ``"lstm"`` the cell is overridden.

    Returns:
        An untrained model.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{name}'; choose from {sorted(MODEL_REGISTRY)}")
    if name in {"gru", "lstm"}:
        config.cell = name
    return MODEL_REGISTRY[name](config)


__all__ = [
    "ARCHITECTURE_DEFAULTS",
    "COMPARED_MODELS",
    "MODEL_REGISTRY",
    "ModelConfig",
    "RecurrentClassifier",
    "SequenceClassifier",
    "TCNClassifier",
    "TransformerClassifier",
    "build_model",
    "count_parameters",
    "describe_embeddings",
    "embedding_dim",
    "make_config",
    "parameter_breakdown",
]
