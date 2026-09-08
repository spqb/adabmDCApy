"""Result objects returned by the high-level adabmDCA API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adabmDCA.alignment import Alignment
from adabmDCA.api.serialization import (
    resolve_format,
    result_document,
    write_dataframe,
    write_json,
    write_numpy,
    write_text,
)
from adabmDCA.training_config import TrainingConfig


@dataclass(frozen=True)
class ModelMetadata:
    """Portable description of a loaded DCA model."""

    length: int
    alphabet: str
    tokens: str
    device: str
    dtype: str
    source: str | None = None
    package_version: str | None = None
    schema_version: str = "1.0"

    @property
    def num_states(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"num_states": self.num_states}

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write model metadata as a versioned JSON document."""
        return write_json(path, result_document("model_metadata", self.to_dict()), indent=indent)


@dataclass(frozen=True)
class EnergyResult:
    """Energies and associated sequence/model metadata."""

    sequences: tuple[str, ...]
    energies: np.ndarray
    model: ModelMetadata
    names: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a portable representation of the result."""
        return result_document(
            "energy",
            {
                "names": self.names,
                "sequences": self.sequences,
                "energies": self.energies,
                "model": self.model.to_dict(),
                "warnings": self.warnings,
            },
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        return write_json(path, self.to_dict(), indent=indent)

    def to_dataframe(self):
        """Return the result as a pandas DataFrame."""
        import pandas as pd

        data: dict[str, Any] = {
            "sequence": self.sequences,
            "energy": self.energies,
        }
        if self.names:
            data = {"name": self.names, **data}
        return pd.DataFrame(data)

    def to_fasta(self, path: str | Path) -> Path:
        """Write sequences and energies to a FASTA file."""
        names = self.names or tuple(f"sequence_{i + 1}" for i in range(len(self.sequences)))
        content = "".join(
            f">{name} | DCAenergy: {energy:.3f}\n{sequence}\n"
            for name, sequence, energy in zip(names, self.sequences, self.energies)
        )
        return write_text(path, content)

    def to_csv(self, path: str | Path) -> Path:
        return write_dataframe(path, self.to_dataframe(), index=False)

    def save(self, path: str | Path, *, format: str | None = None) -> Path:
        selected = resolve_format(path, format, {"csv", "fasta", "json"})
        return getattr(self, f"to_{selected}")(path)

    def save_bundle(self, directory: str | Path, *, stem: str = "energies") -> dict[str, Path]:
        folder = Path(directory)
        return {
            "fasta": self.to_fasta(folder / f"{stem}.fasta"),
            "csv": self.to_csv(folder / f"{stem}.csv"),
            "summary": self.to_json(folder / f"{stem}.json"),
        }


@dataclass(frozen=True)
class ContactMapResult:
    """Contact scores computed from a model or alignment."""

    scores: np.ndarray
    method: str
    alphabet: str
    tokens: str
    model: ModelMetadata | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return result_document(
            "contact_map",
            {
                "scores": self.scores,
                "method": self.method,
                "alphabet": self.alphabet,
                "tokens": self.tokens,
                "model": None if self.model is None else self.model.to_dict(),
                "warnings": self.warnings,
            },
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        return write_json(path, self.to_dict(), indent=indent)

    def to_long_dataframe(self):
        """Return one row per matrix entry."""
        import pandas as pd

        rows, columns = np.indices(self.scores.shape)
        return pd.DataFrame(
            {
                "position_i": rows.ravel(),
                "position_j": columns.ravel(),
                "score": self.scores.ravel(),
            }
        )

    def save_matrix(self, path: str | Path) -> Path:
        """Write the contact map using the historical ``i,j,score`` format."""
        return write_dataframe(path, self.to_long_dataframe(), index=False, header=False)

    def to_csv(self, path: str | Path) -> Path:
        """Write a labelled long-form contact table."""
        return write_dataframe(path, self.to_long_dataframe(), index=False)

    def to_npy(self, path: str | Path) -> Path:
        return write_numpy(path, self.scores)

    def save(self, path: str | Path, *, format: str | None = None) -> Path:
        selected = resolve_format(path, format, {"csv", "json", "npy"})
        return getattr(self, f"to_{selected}")(path)

    def save_bundle(self, directory: str | Path, *, label: str | None = None) -> dict[str, Path]:
        folder = Path(directory)
        stem = f"{label}_contact_map" if label else "contact_map"
        return {
            "matrix": self.save_matrix(folder / f"{stem}.txt"),
            "csv": self.to_csv(folder / f"{stem}.csv"),
            "npy": self.to_npy(folder / f"{stem}.npy"),
            "summary": self.to_json(folder / f"{stem}.json"),
        }


@dataclass(frozen=True)
class MutationRecord:
    """One single-residue mutation and its DCA score."""

    position: int
    wild_type: str
    mutant: str
    sequence: str
    delta_energy: float

    @property
    def position_1based(self) -> int:
        return self.position + 1

    @property
    def label(self) -> str:
        """Historical, zero-based mutation label used by the CLI."""
        return f"{self.wild_type}{self.position}{self.mutant}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation": self.label,
            "position": self.position,
            "position_1based": self.position_1based,
            "wild_type": self.wild_type,
            "mutant": self.mutant,
            "delta_energy": self.delta_energy,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class MutationScanResult:
    """Single-mutant scores for a wild-type sequence."""

    wild_type: str
    wild_type_energy: float
    mutations: tuple[MutationRecord, ...]
    model: ModelMetadata
    name: str = "wild_type"
    warnings: tuple[str, ...] = ()

    @property
    def delta_energies(self) -> np.ndarray:
        return np.asarray([record.delta_energy for record in self.mutations])

    def to_dict(self) -> dict[str, Any]:
        return result_document(
            "mutation_scan",
            {
                "name": self.name,
                "wild_type": self.wild_type,
                "wild_type_energy": self.wild_type_energy,
                "mutations": [record.to_dict() for record in self.mutations],
                "model": self.model.to_dict(),
                "warnings": self.warnings,
            },
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        return write_json(path, self.to_dict(), indent=indent)

    def to_dataframe(self):
        """Return one row per mutation, with zero- and one-based positions."""
        import pandas as pd

        return pd.DataFrame([record.to_dict() for record in self.mutations])

    def to_fasta(self, path: str | Path) -> Path:
        """Write the mutation library using the historical CLI format."""
        content = "".join(
            f">{record.label} | DCAscore: {record.delta_energy:.3f}\n{record.sequence}\n" for record in self.mutations
        )
        return write_text(path, content)

    def to_csv(self, path: str | Path) -> Path:
        return write_dataframe(path, self.to_dataframe(), index=False)

    def save(self, path: str | Path, *, format: str | None = None) -> Path:
        selected = resolve_format(path, format, {"csv", "fasta", "json"})
        return getattr(self, f"to_{selected}")(path)

    def save_bundle(self, directory: str | Path, *, stem: str | None = None) -> dict[str, Path]:
        folder = Path(directory)
        selected_stem = stem or f"{self.name}_DMS"
        return {
            "fasta": self.to_fasta(folder / f"{selected_stem}.fasta"),
            "csv": self.to_csv(folder / f"{selected_stem}.csv"),
            "summary": self.to_json(folder / f"{selected_stem}.json"),
        }


@dataclass(frozen=True)
class SamplingProgress:
    """Progress event emitted during sequence generation."""

    stage: str
    completed: int
    total: int
    pearson: float | None = None
    slope: float | None = None


@dataclass(frozen=True)
class ProfileSplitResult:
    """Training/test alignment split produced by the Cobalt algorithm."""

    training: Alignment
    test: Alignment
    score: int
    attempts: int
    tokens: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return result_document(
            "profile_split",
            {
                "training": {
                    "names": self.training.names,
                    "sequences": self.training.sequences,
                },
                "test": {"names": self.test.names, "sequences": self.test.sequences},
                "score": self.score,
                "attempts": self.attempts,
                "tokens": self.tokens,
                "seed": self.seed,
            },
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        return write_json(path, self.to_dict(), indent=indent)

    def save_bundle(self, output_prefix: str | Path) -> dict[str, Path]:
        prefix = Path(output_prefix)
        training_path = prefix.with_suffix(prefix.suffix + ".train.fasta")
        test_path = prefix.with_suffix(prefix.suffix + ".test.fasta")
        summary_path = prefix.with_suffix(prefix.suffix + ".split.json")
        return {
            "training": _write_alignment(self.training, training_path),
            "test": _write_alignment(self.test, test_path),
            "summary": self.to_json(summary_path),
        }


@dataclass(frozen=True)
class SamplingResult:
    """Generated sequences, energies, and sampling diagnostics."""

    sequences: tuple[str, ...]
    energies: np.ndarray
    num_sweeps: int
    sampler: str
    beta: float
    seed: int
    model: ModelMetadata
    mixing_history: dict[str, Sequence[float]] = field(default_factory=dict)
    sampling_history: dict[str, Sequence[float]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return result_document(
            "sampling",
            {
                "sequences": self.sequences,
                "energies": self.energies,
                "num_sweeps": self.num_sweeps,
                "sampler": self.sampler,
                "beta": self.beta,
                "seed": self.seed,
                "model": self.model.to_dict(),
                "mixing_history": self.mixing_history,
                "sampling_history": self.sampling_history,
                "warnings": self.warnings,
            },
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        return write_json(path, self.to_dict(), indent=indent)

    def to_dataframe(self):
        """Return generated sequences and energies as a DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            {
                "name": [f"sequence_{i + 1}" for i in range(len(self.sequences))],
                "sequence": self.sequences,
                "energy": self.energies,
            }
        )

    def to_fasta(self, path: str | Path) -> Path:
        """Write generated sequences and energies to FASTA."""
        content = "".join(
            f">sequence {index} | DCAenergy: {energy:.3f}\n{sequence}\n"
            for index, (sequence, energy) in enumerate(zip(self.sequences, self.energies), 1)
        )
        return write_text(path, content)

    def to_csv(self, path: str | Path) -> Path:
        return write_dataframe(path, self.to_dataframe(), index=False)

    def save(self, path: str | Path, *, format: str | None = None) -> Path:
        selected = resolve_format(path, format, {"csv", "fasta", "json"})
        return getattr(self, f"to_{selected}")(path)

    def save_bundle(self, directory: str | Path, *, label: str | None = None) -> dict[str, Path]:
        """Save samples, diagnostics, and metadata using predictable filenames."""
        folder = Path(directory)
        prefix = f"{label}_" if label else ""
        artifacts = {
            "samples": self.to_fasta(folder / f"{prefix}samples.fasta"),
            "samples_csv": self.to_csv(folder / f"{prefix}samples.csv"),
            "summary": self.to_json(folder / f"{prefix}sampling.json"),
            "mixing_log": write_dataframe(
                folder / f"{prefix}mix.log",
                _history_dataframe(self.mixing_history),
                index=False,
            ),
            "sampling_log": write_dataframe(
                folder / f"{prefix}sampling.log",
                _history_dataframe(self.sampling_history),
                index=False,
            ),
        }
        return artifacts


@dataclass(frozen=True)
class TrainingProgress:
    """One metrics update emitted by a training routine."""

    epoch: int
    metrics: dict[str, float]
    stage: str = "optimization"
    gradient_steps: int = 0
    structure_steps: int = 0
    sweeps: int = 0


@dataclass(frozen=True)
class TrainingResult:
    """Trained model and the state required to inspect or resume it."""

    model: Any
    history: dict[str, Sequence[float]]
    chains: Any
    log_weights: Any
    pseudocount: float
    num_sequences: int
    effective_sequences: float
    artifacts: dict[str, Path] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    converged: bool = False
    stop_reason: str | None = None
    gradient_steps: int = 0
    structure_steps: int = 0
    sweeps: int = 0
    config: TrainingConfig | None = None
    input_report: dict[str, Any] = field(default_factory=dict)

    def history_dataframe(self):
        """Return the training history as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame.from_dict(self.history)

    def to_dict(self) -> dict[str, Any]:
        """Return a portable summary without embedding model or chain tensors."""
        return result_document(
            "training",
            {
                "model": self.model.metadata.to_dict(),
                "history": self.history,
                "pseudocount": self.pseudocount,
                "num_sequences": self.num_sequences,
                "effective_sequences": self.effective_sequences,
                "artifacts": self.artifacts,
                "warnings": self.warnings,
                "converged": self.converged,
                "stop_reason": self.stop_reason,
                "gradient_steps": self.gradient_steps,
                "structure_steps": self.structure_steps,
                "sweeps": self.sweeps,
                "config": self.config,
                "input_report": self.input_report,
                "final_metrics": self.final_metrics,
            },
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        return write_json(path, self.to_dict(), indent=indent)

    def to_csv(self, path: str | Path) -> Path:
        return write_dataframe(path, self.history_dataframe(), index=False)

    def save_bundle(self, directory: str | Path, *, label: str | None = None) -> dict[str, Path]:
        """Save a portable training summary and normalized history table."""
        folder = Path(directory)
        prefix = f"{label}_" if label else ""
        return {
            "summary": self.to_json(folder / f"{prefix}training.json"),
            "history": self.to_csv(folder / f"{prefix}history.csv"),
        }

    @property
    def final_metrics(self) -> dict[str, Any]:
        """Return the last emitted metric values, or an empty mapping."""
        if not self.history.get("Epochs"):
            return {}
        return {key: values[-1] for key, values in self.history.items() if values}


@dataclass(frozen=True)
class ReintegrationResult:
    """Prepared reintegration dataset and its completed training result."""

    training: TrainingResult
    alignment: Alignment
    weights: np.ndarray
    lambda_value: float
    scaling_factor: float
    label: str
    artifacts: dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return result_document(
            "reintegration",
            {
                "lambda": self.lambda_value,
                "scaling_factor": self.scaling_factor,
                "label": self.label,
                "num_sequences": self.alignment.num_sequences,
                "sequence_length": self.alignment.sequence_length,
                "weights": self.weights,
                "training": self.training.to_dict()["data"],
                "artifacts": self.artifacts,
            },
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        return write_json(path, self.to_dict(), indent=indent)

    def save_bundle(self, directory: str | Path) -> dict[str, Path]:
        from adabmDCA.api.serialization import write_numpy_text

        folder = Path(directory)
        return {
            "alignment": _write_alignment(self.alignment, folder / f"{self.label}_msa.fasta"),
            "weights": write_numpy_text(folder / f"{self.label}_weights.dat", self.weights),
            "summary": self.to_json(folder / f"{self.label}_reintegration.json"),
        }


@dataclass(frozen=True)
class ThermodynamicIntegrationProgress:
    """One progress event from thermodynamic integration."""

    stage: str
    completed: int
    total: int
    theta: float
    entropy: float | None = None
    mean_sequence_identity: float | None = None


@dataclass(frozen=True)
class ThermodynamicIntegrationResult:
    """Entropy estimate and integration trajectory."""

    entropy: float
    free_energy: float
    theta_max: float
    target_fraction: float
    history: dict[str, Sequence[float]]
    model: ModelMetadata
    artifacts: dict[str, Path] = field(default_factory=dict)

    def history_dataframe(self):
        return _history_dataframe(self.history)

    def to_dict(self) -> dict[str, Any]:
        return result_document(
            "thermodynamic_integration",
            {
                "entropy": self.entropy,
                "free_energy": self.free_energy,
                "theta_max": self.theta_max,
                "target_fraction": self.target_fraction,
                "history": self.history,
                "model": self.model.to_dict(),
                "artifacts": self.artifacts,
            },
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        return write_json(path, self.to_dict(), indent=indent)

    def to_csv(self, path: str | Path) -> Path:
        return write_dataframe(path, self.history_dataframe(), index=False)

    def to_log(self, path: str | Path) -> Path:
        columns = tuple(self.history)
        header = " ".join(f"{column:<15}" for column in columns)
        rows = zip(*(self.history[column] for column in columns))
        content = header + "\n" + "".join(" ".join(f"{value:<15.6g}" for value in row) + "\n" for row in rows)
        return write_text(path, content)

    def save_bundle(self, directory: str | Path, *, label: str = "entropy") -> dict[str, Path]:
        folder = Path(directory)
        return {
            "log": self.to_log(folder / f"{label}.log"),
            "csv": self.to_csv(folder / f"{label}.csv"),
            "summary": self.to_json(folder / f"{label}.json"),
        }


def _history_dataframe(history: dict[str, Sequence[float]]):
    import pandas as pd

    return pd.DataFrame.from_dict(history)


def _write_alignment(alignment: Alignment, path: str | Path) -> Path:
    content = "".join(f">{name}\n{sequence}\n" for name, sequence in zip(alignment.names, alignment.sequences))
    return write_text(path, content)
