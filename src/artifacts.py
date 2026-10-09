"""Run artefacts: the scores every downstream module consumes.

Calibration, policy, monitoring and the write-up all need the same thing — the
scores a model assigned to the validation and test rows — and they should never
retrain a model to get them. Every trainer returns a :class:`ModelScores`, and
everything after training reads only that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModelScores:
    """Scores and metadata from a single training run.

    Attributes:
        name: Model identifier, e.g. ``"gru_focal"``.
        family: Architecture family, e.g. ``"gru"``, ``"lightgbm"``.
        seed: Seed the run used.
        val_scores: Scores for the validation rows, in validation row order.
        test_scores: Scores for the test rows, in test row order.
        train_seconds: Wall-clock training time.
        n_params: Parameter count (trainable tensor elements, or tree leaves).
        meta: Anything else worth recording — epochs run, best epoch, config.
    """

    name: str
    family: str
    seed: int
    val_scores: np.ndarray
    test_scores: np.ndarray
    train_seconds: float
    n_params: int
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: Path) -> Path:
        """Write scores to ``<directory>/<name>_seed<seed>.npz``.

        Args:
            directory: Destination directory, created if needed.

        Returns:
            Path to the written file.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}_seed{self.seed}.npz"
        np.savez_compressed(
            path,
            val_scores=self.val_scores,
            test_scores=self.test_scores,
            header=json.dumps(
                {
                    "name": self.name,
                    "family": self.family,
                    "seed": self.seed,
                    "train_seconds": self.train_seconds,
                    "n_params": self.n_params,
                    "meta": self.meta,
                },
                default=str,
            ),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> ModelScores:
        """Read a :class:`ModelScores` written by :meth:`save`.

        Args:
            path: Path to the ``.npz`` file.

        Returns:
            The reconstructed object.
        """
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
            return cls(
                name=header["name"],
                family=header["family"],
                seed=int(header["seed"]),
                val_scores=data["val_scores"],
                test_scores=data["test_scores"],
                train_seconds=float(header["train_seconds"]),
                n_params=int(header["n_params"]),
                meta=header.get("meta", {}),
            )


def load_all_scores(directory: Path) -> list[ModelScores]:
    """Load every score file in a directory, sorted by name and seed.

    Args:
        directory: Directory containing ``*.npz`` score files.

    Returns:
        List of :class:`ModelScores`; empty if the directory does not exist.
    """
    if not directory.exists():
        return []
    return sorted(
        (ModelScores.load(p) for p in directory.glob("*.npz")),
        key=lambda s: (s.name, s.seed),
    )


def write_json(obj: Any, path: Path) -> Path:
    """Write ``obj`` as indented JSON, creating parent directories.

    Args:
        obj: Any JSON-serialisable object.
        path: Destination path.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path
