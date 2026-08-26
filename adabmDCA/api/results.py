"""Result objects returned by the high-level adabmDCA API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


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


@dataclass(frozen=True)
class EnergyResult:
    """Energies and associated sequence/model metadata."""

    sequences: tuple[str, ...]
    energies: np.ndarray
    model: ModelMetadata
    names: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

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
        output = Path(path)
        names = self.names or tuple(f"sequence_{i + 1}" for i in range(len(self.sequences)))
        with output.open("w") as handle:
            for name, sequence, energy in zip(names, self.sequences, self.energies):
                handle.write(f">{name} | DCAenergy: {energy:.3f}\n{sequence}\n")
        return output


@dataclass(frozen=True)
class ContactMapResult:
    """Contact scores computed from a model or alignment."""

    scores: np.ndarray
    method: str
    alphabet: str
    tokens: str
    model: ModelMetadata | None = None
    warnings: tuple[str, ...] = ()

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
        output = Path(path)
        self.to_long_dataframe().to_csv(output, index=False, header=False)
        return output


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

    def to_dataframe(self):
        """Return one row per mutation, with zero- and one-based positions."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "mutation": record.label,
                    "position": record.position,
                    "position_1based": record.position_1based,
                    "wild_type": record.wild_type,
                    "mutant": record.mutant,
                    "delta_energy": record.delta_energy,
                    "sequence": record.sequence,
                }
                for record in self.mutations
            ]
        )

    def to_fasta(self, path: str | Path) -> Path:
        """Write the mutation library using the historical CLI format."""
        output = Path(path)
        with output.open("w") as handle:
            for record in self.mutations:
                handle.write(
                    f">{record.label} | DCAscore: {record.delta_energy:.3f}\n"
                    f"{record.sequence}\n"
                )
        return output


@dataclass(frozen=True)
class SamplingProgress:
    """Progress event emitted during sequence generation."""

    stage: str
    completed: int
    total: int
    pearson: float | None = None
    slope: float | None = None


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
        output = Path(path)
        with output.open("w") as handle:
            for index, (sequence, energy) in enumerate(zip(self.sequences, self.energies), 1):
                handle.write(f">sequence {index} | DCAenergy: {energy:.3f}\n{sequence}\n")
        return output


@dataclass(frozen=True)
class TrainingProgress:
    """One metrics update emitted by a training routine."""

    epoch: int
    metrics: dict[str, float]


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

    def history_dataframe(self):
        """Return the training history as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame.from_dict(self.history)
