"""Result objects returned by the high-level adabmDCA API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from adabmDCA.alignment import Alignment
from adabmDCA.exceptions import OutputSerializationError
from adabmDCA.serialization import (
    _atomic_write,
    resolve_format,
    result_document,
    write_dataframe,
    write_json,
    write_numpy,
    write_text,
)
from adabmDCA.training_config import TrainingConfig

if TYPE_CHECKING:
    import pandas as pd
    import torch

    from adabmDCA.api.model import DCAModel
    from adabmDCA.ptt.sampler import PartitionEstimate, PTTSampler


@dataclass(frozen=True)
class ModelMetadata:
    """Portable description of a loaded DCA model.

    Attributes:
        length: Number of sites ``L``.
        alphabet: Alphabet name or custom token string given when loading.
        tokens: Ordered token string of length ``q``.
        device: Device of the parameters, e.g. ``"cpu"`` or ``"cuda:0"``.
        dtype: Precision of the parameters, e.g. ``"float32"``.
        source: Path of the file the model was loaded from, if any.
        package_version: adabmDCA version that loaded the model.
        schema_version: Version of this metadata format.
    """

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
        """Number of states per site, ``q``."""
        return len(self.tokens)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible document with the result type and schema version."""
        return asdict(self) | {"num_states": self.num_states}

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the metadata as a versioned JSON document.

        Args:
            path: Output file.
            indent: JSON indentation.

        Returns:
            The written path.
        """
        return write_json(path, result_document("model_metadata", self.to_dict()), indent=indent)


@dataclass(frozen=True)
class EnergyResult:
    """DCA energies of scored sequences, returned by :func:`score_sequences`.

    Attributes:
        sequences: Scored sequences, in input order.
        energies: Energy of each sequence (lower is more probable).
        model: Metadata of the scoring model.
        names: Sequence names when read from an alignment, else empty.
        warnings: Non-fatal issues found while scoring.
        cde_sum: Summed context-dependent entropy of each sequence, or ``None``.
        local_free_energies: ``energies - local_lambda * cde_sum``, or ``None``.
        local_lambda: Weight used for the local free energies, or ``None``.
    """

    sequences: tuple[str, ...]
    energies: np.ndarray
    model: ModelMetadata
    names: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    cde_sum: np.ndarray | None = None
    local_free_energies: np.ndarray | None = None
    local_lambda: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a portable representation of the result."""
        return result_document(
            "energy",
            {
                "names": self.names,
                "sequences": self.sequences,
                "energies": self.energies,
                "cde_sum": self.cde_sum,
                "local_free_energies": self.local_free_energies,
                "local_lambda": self.local_lambda,
                "model": self.model.to_dict(),
                "warnings": self.warnings,
            },
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write :meth:`to_dict` as JSON.

        Args:
            path: Output file.
            indent: JSON indentation.

        Returns:
            The written path.
        """
        return write_json(path, self.to_dict(), indent=indent)

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per sequence: name (if known), sequence, energy and CDE columns."""
        import pandas as pd

        data: dict[str, Any] = {
            "sequence": self.sequences,
            "energy": self.energies,
        }
        if self.cde_sum is not None:
            data["cde_sum"] = self.cde_sum
        if self.local_free_energies is not None:
            data["local_free_energy"] = self.local_free_energies
        if self.names:
            data = {"name": self.names, **data}
        return pd.DataFrame(data)

    def to_fasta(self, path: str | Path) -> Path:
        """Write the sequences to FASTA, with energies in the headers.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        names = self.names or tuple(f"sequence_{i + 1}" for i in range(len(self.sequences)))
        content = "".join(
            f">{name} | DCAenergy: {energy:.3f}"
            + (f" | local_free_energy: {self.local_free_energies[i]:.3f}" if self.local_free_energies is not None else "")
            + f"\n{sequence}\n"
            for i, (name, sequence, energy) in enumerate(zip(names, self.sequences, self.energies))
        )
        return write_text(path, content)

    def to_csv(self, path: str | Path) -> Path:
        """Write :meth:`to_dataframe` as CSV.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        return write_dataframe(path, self.to_dataframe(), index=False)

    def save(self, path: str | Path, *, format: str | None = None) -> Path:
        """Write the result in the format given by ``format`` or by the file extension.

        Args:
            path: Output file.
            format: One of ``"csv"``, ``"fasta"`` or ``"json"``; inferred from the extension when ``None``.

        Returns:
            The written path.
        """
        selected = resolve_format(path, format, {"csv", "fasta", "json"})
        return getattr(self, f"to_{selected}")(path)

    def save_bundle(self, directory: str | Path, *, stem: str = "energies") -> dict[str, Path]:
        """Write ``<stem>.fasta``, ``<stem>.csv`` and ``<stem>.json`` to ``directory``.

        Args:
            directory: Output folder.
            stem: File-name stem.

        Returns:
            The written paths by kind: ``fasta``, ``csv`` and ``summary``.
        """
        folder = Path(directory)
        return {
            "fasta": self.to_fasta(folder / f"{stem}.fasta"),
            "csv": self.to_csv(folder / f"{stem}.csv"),
            "summary": self.to_json(folder / f"{stem}.json"),
        }


@dataclass(frozen=True)
class ContactMapResult:
    """Contact scores returned by :func:`predict_contacts`.

    Attributes:
        scores: Symmetric ``(L, L)`` array of APC-corrected scores, zero diagonal.
        method: ``"model"`` for a trained model, ``"mean_field"`` for an alignment.
        alphabet: Alphabet used.
        tokens: Ordered token string.
        model: Metadata of the model, for ``method="model"``.
        warnings: Non-fatal issues found while scoring.
    """

    scores: np.ndarray
    method: str
    alphabet: str
    tokens: str
    model: ModelMetadata | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible document with the result type and schema version."""
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
        """Write :meth:`to_dict` as JSON.

        Args:
            path: Output file.
            indent: JSON indentation.

        Returns:
            The written path.
        """
        return write_json(path, self.to_dict(), indent=indent)

    def to_long_dataframe(self) -> pd.DataFrame:
        """Return one row per matrix entry: ``position_i``, ``position_j`` (0-based) and ``score``."""
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
        """Write the scores as headerless ``i,j,score`` lines, the format of ``adabmDCA contacts``.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        return write_dataframe(path, self.to_long_dataframe(), index=False, header=False)

    def to_csv(self, path: str | Path) -> Path:
        """Write :meth:`to_long_dataframe` as CSV with a header.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        return write_dataframe(path, self.to_long_dataframe(), index=False)

    def to_npy(self, path: str | Path) -> Path:
        """Write the ``(L, L)`` score matrix as a NumPy ``.npy`` file.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        return write_numpy(path, self.scores)

    def save(self, path: str | Path, *, format: str | None = None) -> Path:
        """Write the result in the format given by ``format`` or by the file extension.

        Args:
            path: Output file.
            format: One of ``"csv"``, ``"json"`` or ``"npy"``; inferred from the extension when ``None``.

        Returns:
            The written path.
        """
        selected = resolve_format(path, format, {"csv", "json", "npy"})
        return getattr(self, f"to_{selected}")(path)

    def save_bundle(self, directory: str | Path, *, label: str | None = None) -> dict[str, Path]:
        """Write the matrix (``.txt``, ``.csv``, ``.npy``) and a JSON summary to ``directory``.

        Args:
            directory: Output folder.
            label: Optional file-name prefix, giving ``<label>_contact_map.*``.

        Returns:
            The written paths by kind: ``matrix``, ``csv``, ``npy`` and ``summary``.
        """
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
    """One single-site mutant from :func:`scan_mutations`.

    Attributes:
        position: 0-based site of the substitution.
        wild_type: Wild-type token at that site.
        mutant: Substituted token.
        sequence: Full mutant sequence.
        delta_energy: ``E(mutant) - E(wild type)``; negative values favour the mutant.
    """

    position: int
    wild_type: str
    mutant: str
    sequence: str
    delta_energy: float

    @property
    def position_1based(self) -> int:
        """1-based site of the substitution."""
        return self.position + 1

    @property
    def label(self) -> str:
        """Historical, zero-based mutation label used by the CLI."""
        return f"{self.wild_type}{self.position}{self.mutant}"

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a flat dictionary, one row of :meth:`MutationScanResult.to_dataframe`."""
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
    """All single-site mutants of a sequence, returned by :func:`scan_mutations`.

    Attributes:
        wild_type: The scanned sequence.
        wild_type_energy: Its DCA energy.
        mutations: One :class:`MutationRecord` per mutant, by site then token.
        model: Metadata of the scoring model.
        name: Label of the scan, used in file names.
        warnings: Non-fatal issues found while scoring.
    """

    wild_type: str
    wild_type_energy: float
    mutations: tuple[MutationRecord, ...]
    model: ModelMetadata
    name: str = "wild_type"
    warnings: tuple[str, ...] = ()

    @property
    def delta_energies(self) -> np.ndarray:
        """Energy differences of all mutants, in the order of ``mutations``."""
        return np.asarray([record.delta_energy for record in self.mutations])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible document with the result type and schema version."""
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
        """Write :meth:`to_dict` as JSON.

        Args:
            path: Output file.
            indent: JSON indentation.

        Returns:
            The written path.
        """
        return write_json(path, self.to_dict(), indent=indent)

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per mutant, with 0- and 1-based positions and ``delta_energy``."""
        import pandas as pd

        return pd.DataFrame([record.to_dict() for record in self.mutations])

    def to_fasta(self, path: str | Path) -> Path:
        """Write the mutant sequences as FASTA, headers ``<wt><pos><mut> | DCAscore: <delta>``.

        Positions in the headers are 0-based, as in ``adabmDCA dms``.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        content = "".join(
            f">{record.label} | DCAscore: {record.delta_energy:.3f}\n{record.sequence}\n" for record in self.mutations
        )
        return write_text(path, content)

    def to_csv(self, path: str | Path) -> Path:
        """Write :meth:`to_dataframe` as CSV.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        return write_dataframe(path, self.to_dataframe(), index=False)

    def save(self, path: str | Path, *, format: str | None = None) -> Path:
        """Write the result in the format given by ``format`` or by the file extension.

        Args:
            path: Output file.
            format: One of ``"csv"``, ``"fasta"`` or ``"json"``; inferred from the extension when ``None``.

        Returns:
            The written path.
        """
        selected = resolve_format(path, format, {"csv", "fasta", "json"})
        return getattr(self, f"to_{selected}")(path)

    def save_bundle(self, directory: str | Path, *, stem: str | None = None) -> dict[str, Path]:
        """Write the FASTA library, CSV table and JSON summary to ``directory``.

        Args:
            directory: Output folder.
            stem: File-name stem; defaults to ``<name>_DMS``.

        Returns:
            The written paths by kind: ``fasta``, ``csv`` and ``summary``.
        """
        folder = Path(directory)
        selected_stem = stem or f"{self.name}_DMS"
        return {
            "fasta": self.to_fasta(folder / f"{selected_stem}.fasta"),
            "csv": self.to_csv(folder / f"{selected_stem}.csv"),
            "summary": self.to_json(folder / f"{selected_stem}.json"),
        }


@dataclass(frozen=True)
class SamplingProgress:
    """Progress event passed to the ``progress`` callback of :func:`sample_sequences`.

    Attributes:
        stage: ``"sampling"``, or a ``"ptt_*"`` stage when sampling with PTT.
        completed: Work done in this phase (sweeps or exchange rounds).
        total: Work planned for this phase.
        pearson: Current Cij Pearson correlation with the reference, if measured.
        slope: Current Cij regression slope, if measured.
        details: Extra phase-specific values (PTT renewal fractions, ...).
    """

    stage: str
    completed: int
    total: int
    pearson: float | None = None
    slope: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileSplitResult:
    """Training/test split of an alignment, returned by :func:`split_alignment`.

    Attributes:
        training: Training alignment.
        test: Test alignment.
        score: ``len(training) * len(test)``; Cobalt keeps the attempt maximizing it.
        attempts: Number of attempts made.
        tokens: Alphabet tokens.
        seed: Random seed used.
        method: ``"cobalt"`` or ``"clustering"``.
        identity: Sequence-identity threshold of the clustering method.
        train_fraction: Requested training fraction of the clustering method.
    """

    training: Alignment
    test: Alignment
    score: int
    attempts: int
    tokens: str
    seed: int
    method: str = "cobalt"
    identity: float | None = None
    train_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible document with the result type and schema version."""
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
                "method": self.method,
                "identity": self.identity,
                "train_fraction": self.train_fraction,
            },
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write :meth:`to_dict` as JSON.

        Args:
            path: Output file.
            indent: JSON indentation.

        Returns:
            The written path.
        """
        return write_json(path, self.to_dict(), indent=indent)

    def save_bundle(self, output_prefix: str | Path) -> dict[str, Path]:
        """Write ``<prefix>.train.fasta``, ``<prefix>.test.fasta`` and ``<prefix>.split.json``.

        Args:
            output_prefix: Path prefix of the three files.

        Returns:
            The written paths by kind: ``training``, ``test`` and ``summary``.
        """
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
    """Generated sequences and diagnostics, returned by :func:`sample_sequences`.

    Attributes:
        sequences: Generated sequences.
        energies: DCA energy of each sequence.
        num_sweeps: Monte Carlo sweeps performed (all replicas, for PTT).
        sampler: Local sampler used, or ``"ptt"``.
        beta: Inverse temperature.
        seed: Random seed.
        model: Metadata of the sampled model.
        mixing_history: Mixing-time measurements, by quantity.
        sampling_history: Pearson and slope of the Cij against the reference during
            generation, when a reference was given.
        warnings: Non-fatal issues, e.g. an unconverged mixing estimate.
        sampling_dtype: Precision used for sampling.
        ptt_diagnostics: PTT only: mixing, renewal and ladder-health diagnostics.
        cde_sum: Summed context-dependent entropy of each sequence.
        local_lambda_fit: Least-squares fit of energy against ``cde_sum``, if identifiable.
        cij_reference: Reference connected correlations, when diagnostics were collected.
        cij_generated: Generated connected correlations, when diagnostics were collected.
        pca_reference: Reference sequences projected on their principal components.
        pca_generated: Generated sequences projected on the same components.
        data_comparison: With diagnostics and a reference: the share of data and
            samples in each cluster of the data (k-means on the principal
            components) and their energy distributions under the model.
        distance_comparison: With diagnostics and a reference: Hamming distances
            (fraction of sites) within and between natural and generated
            sequences, all pairs and nearest neighbours, plus held-out ->
            reference and generated -> held-out nearest distances when a test
            alignment was given; ``privet`` holds the PRIVET fit and per-sample
            scores (also columns of :meth:`to_dataframe`).
        pca_explained_variance_ratio: Variance explained by each component.
        steering_potentials: Steered sampling only: ``V(x, steering_strength)`` of
            each sequence.
        log_importance_weights: Steered sampling only: log weights that turn
            averages over these sequences into averages under the unsteered model,
            ``<g>_p = mean(exp(log_w) * g)``. PTT normalizes them with its estimate
            of ``log Z_s - log Z_0``; ordinary sampling self-normalizes them to mean 1.
        steering: Steered sampling only: ``strength``, ``input``, proposal block
            length and acceptance, ``effective_sample_size`` of the weights, and for
            PTT ``log_z_ratio`` and the rung ``strengths``.
    """

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
    sampling_dtype: str = "float32"
    ptt_diagnostics: dict[str, Any] = field(default_factory=dict)
    cde_sum: np.ndarray | None = None
    local_lambda_fit: dict[str, float | int] | None = None
    cij_reference: np.ndarray | None = field(default=None, repr=False, compare=False)
    cij_generated: np.ndarray | None = field(default=None, repr=False, compare=False)
    pca_reference: np.ndarray | None = field(default=None, repr=False, compare=False)
    pca_generated: np.ndarray | None = field(default=None, repr=False, compare=False)
    pca_explained_variance_ratio: np.ndarray | None = field(default=None, repr=False, compare=False)
    data_comparison: dict[str, Any] = field(default_factory=dict)
    distance_comparison: dict[str, Any] = field(default_factory=dict)
    steering_potentials: np.ndarray | None = None
    log_importance_weights: np.ndarray | None = None
    steering: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible document with the result type and schema version."""
        return result_document(
            "sampling",
            {
                "sequences": self.sequences,
                "energies": self.energies,
                "num_sweeps": self.num_sweeps,
                "sampler": self.sampler,
                "beta": self.beta,
                "seed": self.seed,
                "sampling_dtype": self.sampling_dtype,
                "model": self.model.to_dict(),
                "mixing_history": self.mixing_history,
                "sampling_history": self.sampling_history,
                "ptt_diagnostics": self.ptt_diagnostics,
                "data_comparison": self.data_comparison,
                "distance_comparison": self.distance_comparison,
                "cde_sum": self.cde_sum,
                "local_lambda_fit": self.local_lambda_fit,
                "steering": self.steering,
                "steering_potentials": self.steering_potentials,
                "log_importance_weights": self.log_importance_weights,
                "warnings": self.warnings,
            },
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write :meth:`to_dict` as JSON.

        Args:
            path: Output file.
            indent: JSON indentation.

        Returns:
            The written path.
        """
        return write_json(path, self.to_dict(), indent=indent)

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per sequence: name, sequence, energy, and ``cde_sum``, steering and PRIVET columns when present.

        PRIVET columns are ``NaN`` for sequences left out of the comparison (beyond ``n_measure``).
        """
        import pandas as pd

        data = {
            "name": [f"sequence_{i + 1}" for i in range(len(self.sequences))],
            "sequence": self.sequences,
            "energy": self.energies,
        }
        if self.cde_sum is not None:
            data["cde_sum"] = self.cde_sum
        if self.steering_potentials is not None:
            data["steering_potential"] = self.steering_potentials
            data["log_importance_weight"] = self.log_importance_weights
        privet = self.distance_comparison.get("privet")
        if privet:
            for key in ("log10_p_train", "log10_p_test", "delta_p"):
                if key in privet:
                    column = np.full(len(self.sequences), np.nan)
                    column[privet["sample_index"]] = privet[key]
                    data[f"privet_{key}"] = column
        return pd.DataFrame(data)

    def to_fasta(self, path: str | Path) -> Path:
        """Write the sequences as FASTA, with energies in the headers.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        content = "".join(
            f">sequence {index} | DCAenergy: {energy:.3f}\n{sequence}\n"
            for index, (sequence, energy) in enumerate(zip(self.sequences, self.energies), 1)
        )
        return write_text(path, content)

    def to_csv(self, path: str | Path) -> Path:
        """Write :meth:`to_dataframe` as CSV.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        return write_dataframe(path, self.to_dataframe(), index=False)

    def save(self, path: str | Path, *, format: str | None = None) -> Path:
        """Write the result in the format given by ``format`` or by the file extension.

        Args:
            path: Output file.
            format: One of ``"csv"``, ``"fasta"`` or ``"json"``; inferred from the extension when ``None``.

        Returns:
            The written path.
        """
        selected = resolve_format(path, format, {"csv", "fasta", "json"})
        return getattr(self, f"to_{selected}")(path)

    def save_bundle(self, directory: str | Path, *, label: str | None = None) -> dict[str, Path]:
        """Write samples (FASTA and CSV), a JSON summary and, in ``logs/``, the mixing/sampling logs.

        PTT results also get ``logs/ptt.log`` and, when available, ladder-health
        and renewal logs; diagnostics add the data-comparison and distance logs.

        Args:
            directory: Output folder.
            label: Optional file-name prefix, giving ``<label>_samples.fasta`` etc.

        Returns:
            The written paths by kind.
        """
        folder = Path(directory)
        logs = folder / "logs"
        prefix = f"{label}_" if label else ""
        artifacts = {
            "samples": self.to_fasta(folder / f"{prefix}samples.fasta"),
            "samples_csv": self.to_csv(folder / f"{prefix}samples.csv"),
            "summary": self.to_json(folder / f"{prefix}sampling.json"),
            "mixing_log": write_dataframe(
                logs / f"{prefix}mix.log",
                _history_dataframe(self.mixing_history),
                index=False,
            ),
            "sampling_log": write_dataframe(
                logs / f"{prefix}sampling.log",
                _history_dataframe(self.sampling_history),
                index=False,
            ),
        }
        if self.data_comparison:
            import pandas as pd

            clusters = self.data_comparison["clusters"]
            artifacts["data_vs_samples_log"] = write_dataframe(
                logs / f"{prefix}data_vs_samples.log",
                pd.DataFrame({"cluster": range(1, len(clusters["data_fraction"]) + 1),
                              "data_fraction": clusters["data_fraction"],
                              "sample_fraction": clusters["sample_fraction"]}),
                index=False,
            )
        if self.distance_comparison:
            import pandas as pd

            distances = self.distance_comparison
            edges = np.asarray(distances["bin_edges"])
            columns = {"distance": 0.5 * (edges[1:] + edges[:-1])}
            columns.update({f"all_pairs_{key}": value for key, value in distances["all_pairs"].items()})
            columns.update({f"nearest_{key}": value for key, value in distances["nearest"].items()})
            artifacts["distances_log"] = write_dataframe(
                logs / f"{prefix}distances.log", pd.DataFrame(columns), index=False,
            )
        if self.ptt_diagnostics:
            import pandas as pd

            artifacts["ptt_log"] = write_dataframe(
                logs / f"{prefix}ptt.log", pd.DataFrame(self.ptt_diagnostics["models"]), index=False,
            )
            health = self.ptt_diagnostics.get("ladder_health")
            if health:
                artifacts["ptt_ladder_health_log"] = write_dataframe(
                    logs / f"{prefix}ptt_ladder_health.log", pd.DataFrame(health["pairs"]), index=False,
                )
                if health.get("replicas"):
                    artifacts["ptt_replicas_log"] = write_dataframe(
                        logs / f"{prefix}ptt_replicas.log", pd.DataFrame(health["replicas"]), index=False,
                    )
            history = self.ptt_diagnostics.get("renewal_history")
            if history is not None:
                rows = [
                    {"phase": phase, "round": index + 1, "ladder_old": old, "endpoint_fresh": fresh}
                    for phase in ("warmup", "stationary")
                    for index, (old, fresh) in enumerate(
                        zip(history.get(f"{phase}_ladder_old", ()), history.get(f"{phase}_endpoint_fresh", ()))
                    )
                ]
                artifacts["ptt_renewal_log"] = write_dataframe(
                    logs / f"{prefix}ptt_renewal.log",
                    pd.DataFrame(rows, columns=["phase", "round", "ladder_old", "endpoint_fresh"]), index=False,
                )
        return artifacts

    def _save_energy_cde_plot(self, folder: Path, prefix: str) -> Path | None:
        if self.cde_sum is None:
            return None
        import matplotlib.pyplot as plt

        from adabmDCA.plot import plot_energy_cde_scatter

        path = folder / f"{prefix}energy_vs_cde.png"
        figure, axis = plt.subplots(dpi=192, figsize=(7, 5))
        plot_energy_cde_scatter(axis, self.cde_sum, self.energies, self.local_lambda_fit)
        figure.tight_layout()
        figure.savefig(path, dpi=192, facecolor="white")
        plt.close(figure)
        return path

    def _save_data_comparison_plot(self, folder: Path, prefix: str) -> Path | None:
        if not self.data_comparison:
            return None
        import matplotlib.pyplot as plt

        from adabmDCA.plot import plot_data_comparison

        path = folder / f"{prefix}data_vs_samples.png"
        figure = plt.figure(dpi=192, figsize=(15, 4.8))
        plot_data_comparison(figure, self.data_comparison, self.pca_reference, self.pca_generated)
        figure.tight_layout()
        figure.savefig(path, dpi=192, facecolor="white")
        plt.close(figure)
        return path

    def _save_privet_plot(self, folder: Path, prefix: str) -> Path | None:
        privet = self.distance_comparison.get("privet")
        if not privet:
            return None
        import matplotlib.pyplot as plt

        from adabmDCA.plot import plot_privet

        path = folder / f"{prefix}privet.png"
        figure = plt.figure(dpi=192, figsize=(15, 4.8))
        plot_privet(figure, privet)
        figure.tight_layout()
        figure.savefig(path, dpi=192, facecolor="white")
        plt.close(figure)
        return path

    def _save_distance_plot(self, folder: Path, prefix: str) -> Path | None:
        if not self.distance_comparison:
            return None
        import matplotlib.pyplot as plt

        from adabmDCA.plot import plot_distance_comparison

        path = folder / f"{prefix}distances.png"
        figure = plt.figure(dpi=192, figsize=(12, 4.8))
        plot_distance_comparison(figure, self.distance_comparison)
        figure.tight_layout()
        figure.savefig(path, dpi=192, facecolor="white")
        plt.close(figure)
        return path

    def save_diagnostic_plots(self, directory: str | Path, *, label: str | None = None) -> dict[str, Path]:
        """Save diagnostic plots as PNG files: mixing, Cij scatter, PCA and energy against CDE.

        Plots whose data were not collected (see ``collect_diagnostics`` of
        :func:`sample_sequences`) are skipped.

        Args:
            directory: Output folder.
            label: Optional file-name prefix.

        Returns:
            The written paths by plot kind.
        """
        if self.ptt_diagnostics:
            import matplotlib.pyplot as plt

            from adabmDCA.plot import (
                plot_cij_scatter,
                plot_PCA,
                plot_ptt_autocorrelation,
                plot_ptt_ladder_health,
                plot_ptt_renewal,
            )

            folder = Path(directory)
            folder.mkdir(parents=True, exist_ok=True)
            prefix = f"{label}_" if label else ""
            mixing = self.ptt_diagnostics["mixing"]
            renewal = mixing.get("method") == "renewal"
            paths = {
                **({"ptt_renewal_plot": folder / f"{prefix}ptt_renewal.png"} if renewal
                   else {"ptt_mixing_plot": folder / f"{prefix}ptt_mixing.png"}),
                **({"ptt_ladder_health_plot": folder / f"{prefix}ptt_ladder_health.png"}
                   if self.ptt_diagnostics.get("ladder_health") else {}),
                "cij_scatter_plot": folder / f"{prefix}cij_scatter.png",
                "pca_1_2_plot": folder / f"{prefix}pca_1_2.png",
                "pca_3_4_plot": folder / f"{prefix}pca_3_4.png",
            }
            if self.cij_reference is None or self.cij_generated is None:
                raise OutputSerializationError(
                    "Cij plot data were not collected; request PTT sampling diagnostics."
                )
            if self.pca_reference is None or self.pca_generated is None or self.pca_explained_variance_ratio is None:
                raise OutputSerializationError(
                    "PCA plot data were not collected; request PTT sampling diagnostics."
                )
            plot_dpi = 192
            if renewal:
                stationary = mixing.get("stationary", True)
                figure = plt.figure(dpi=plot_dpi, figsize=(11 if stationary else 6.5, 7.5))
                plot_ptt_renewal(
                    figure, self.ptt_diagnostics["renewal_history"],
                    tolerance=float(mixing["tolerance"]), n_models=int(mixing["replicas"]),
                    warmup_rounds=mixing["warmup_rounds"], renewal_rounds=mixing["renewal_rounds"],
                    chunk_rounds=int(mixing.get("chunk_rounds") or 0), stationary=stationary,
                )
                figure.tight_layout()
                figure.savefig(paths["ptt_renewal_plot"], dpi=plot_dpi)
                plt.close(figure)
            else:
                tau_int = mixing.get("tau_int")
                tau_exp = mixing.get("tau_exp")
                figure, axis = plt.subplots(dpi=plot_dpi, figsize=(7, 5))
                plot_ptt_autocorrelation(
                    axis,
                    np.asarray(self.ptt_diagnostics["mixing_correlation"]),
                    None if tau_int is None else float(tau_int),
                    None if tau_exp is None else float(tau_exp),
                )
                figure.tight_layout()
                figure.savefig(paths["ptt_mixing_plot"], dpi=plot_dpi)
                plt.close(figure)

            if "ptt_ladder_health_plot" in paths:
                figure = plt.figure(dpi=plot_dpi, figsize=(12, 8.5))
                plot_ptt_ladder_health(figure, self.ptt_diagnostics["ladder_health"],
                                       mixing=self.ptt_diagnostics.get("mixing"))
                figure.tight_layout()
                figure.savefig(paths["ptt_ladder_health_plot"], dpi=plot_dpi)
                plt.close(figure)

            figure, axis = plt.subplots(dpi=plot_dpi, figsize=(6, 6))
            plot_cij_scatter(
                axis, self.cij_reference, self.cij_generated,
                pearson=self.ptt_diagnostics.get("final_pearson"),
            )
            figure.tight_layout()
            figure.savefig(paths["cij_scatter_plot"], dpi=plot_dpi)
            plt.close(figure)

            for pc1, pc2, path_key in ((0, 1, "pca_1_2_plot"), (2, 3, "pca_3_4_plot")):
                figure = plt.figure(dpi=plot_dpi, figsize=(7, 6.5))
                plot_PCA(
                    figure, self.pca_reference, pc1=pc1, pc2=pc2,
                    data2=self.pca_generated, labels=["Natural", "Generated"],
                    colors=["#31688E", "#E76F51"],
                    title=f"Natural and generated sequences: PC{pc1 + 1} vs PC{pc2 + 1}",
                    explained_variance_ratio=self.pca_explained_variance_ratio,
                )
                figure.savefig(paths[path_key], dpi=plot_dpi, facecolor="white", bbox_inches="tight")
                plt.close(figure)
            energy_cde_path = self._save_energy_cde_plot(folder, prefix)
            if energy_cde_path is not None:
                paths["energy_cde_plot"] = energy_cde_path
            comparison_path = self._save_data_comparison_plot(folder, prefix)
            if comparison_path is not None:
                paths["data_vs_samples_plot"] = comparison_path
            distance_path = self._save_distance_plot(folder, prefix)
            if distance_path is not None:
                paths["distances_plot"] = distance_path
            privet_path = self._save_privet_plot(folder, prefix)
            if privet_path is not None:
                paths["privet_plot"] = privet_path
            return paths
        required_mixing = {"t_half", "seqid_t", "std_seqid_t", "seqid_t_t_half", "std_seqid_t_t_half"}
        required_sampling = {"nsweeps", "pearson"}
        if not required_mixing.issubset(self.mixing_history) or not required_sampling.issubset(
            self.sampling_history
        ):
            raise OutputSerializationError("Sampling plots require mixing and sampling histories.")
        if self.cij_reference is None or self.cij_generated is None:
            raise OutputSerializationError(
                "Cij plot data were not collected; call sample_sequences with collect_diagnostics=True."
            )
        if self.pca_reference is None or self.pca_generated is None or self.pca_explained_variance_ratio is None:
            raise OutputSerializationError(
                "PCA plot data were not collected; call sample_sequences with collect_diagnostics=True."
            )

        import matplotlib.pyplot as plt

        from adabmDCA.plot import plot_autocorrelation, plot_cij_scatter, plot_PCA, plot_pearson_sampling

        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f"{label}_" if label else ""
        paths = {
            "autocorrelation_plot": folder / f"{prefix}autocorrelation.png",
            "pearson_plot": folder / f"{prefix}pearson_sampling.png",
            "cij_scatter_plot": folder / f"{prefix}cij_scatter.png",
            "pca_1_2_plot": folder / f"{prefix}pca_1_2.png",
            "pca_3_4_plot": folder / f"{prefix}pca_3_4.png",
        }
        plot_dpi = 192

        figure, axis = plt.subplots(dpi=plot_dpi, figsize=(7, 5))
        plot_autocorrelation(
            axis,
            np.asarray(self.mixing_history["t_half"]),
            np.asarray(self.mixing_history["seqid_t_t_half"]),
            np.asarray(self.mixing_history["seqid_t"]),
            autocorr_std=np.asarray(self.mixing_history["std_seqid_t_t_half"]),
            independent_std=np.asarray(self.mixing_history["std_seqid_t"]),
        )
        figure.tight_layout()
        figure.savefig(paths["autocorrelation_plot"], dpi=plot_dpi)
        plt.close(figure)

        figure, axis = plt.subplots(dpi=plot_dpi, figsize=(7, 5))
        plot_pearson_sampling(
            axis,
            np.asarray(self.sampling_history["nsweeps"]),
            np.asarray(self.sampling_history["pearson"]),
        )
        figure.tight_layout()
        figure.savefig(paths["pearson_plot"], dpi=plot_dpi)
        plt.close(figure)

        figure, axis = plt.subplots(dpi=plot_dpi, figsize=(6, 6))
        plot_cij_scatter(axis, self.cij_reference, self.cij_generated)
        figure.tight_layout()
        figure.savefig(paths["cij_scatter_plot"], dpi=plot_dpi)
        plt.close(figure)

        for pc1, pc2, path_key in ((0, 1, "pca_1_2_plot"), (2, 3, "pca_3_4_plot")):
            figure = plt.figure(dpi=plot_dpi, figsize=(7, 6.5))
            plot_PCA(
                figure,
                self.pca_reference,
                pc1=pc1,
                pc2=pc2,
                data2=self.pca_generated,
                labels=["Natural", "Generated"],
                colors=["#31688E", "#E76F51"],
                title=f"Natural and generated sequences: PC{pc1 + 1} vs PC{pc2 + 1}",
                explained_variance_ratio=self.pca_explained_variance_ratio,
            )
            figure.savefig(paths[path_key], dpi=plot_dpi, facecolor="white", bbox_inches="tight")
            plt.close(figure)
        energy_cde_path = self._save_energy_cde_plot(folder, prefix)
        if energy_cde_path is not None:
            paths["energy_cde_plot"] = energy_cde_path
        comparison_path = self._save_data_comparison_plot(folder, prefix)
        if comparison_path is not None:
            paths["data_vs_samples_plot"] = comparison_path
        distance_path = self._save_distance_plot(folder, prefix)
        if distance_path is not None:
            paths["distances_plot"] = distance_path
        privet_path = self._save_privet_plot(folder, prefix)
        if privet_path is not None:
            paths["privet_plot"] = privet_path
        return paths


@dataclass(frozen=True)
class TrainingProgress:
    """One accepted update, passed to the ``progress`` callback of :func:`train_model`.

    Attributes:
        epoch: Step number of the record (gradient steps, or graph steps for PCD eaDCA/edDCA).
        metrics: Numeric values of the history row: ``Pearson``, ``LL_val``, ``Density``, ...
        stage: Current training phase.
        gradient_steps: Accepted parameter updates so far.
        structure_steps: Graph activations or decimations so far.
        sweeps: Monte Carlo sweeps so far.
        partition_estimate: PTT only: log Z and its provenance.
    """

    epoch: int
    metrics: dict[str, Any]
    stage: str = "optimization"
    gradient_steps: int = 0
    structure_steps: int = 0
    sweeps: int = 0
    partition_estimate: dict[str, Any] | None = None


@dataclass(frozen=True)
class TrainingDatasetSummary:
    """Size and filtering of one training or validation alignment.

    Attributes:
        source: Path of the alignment, if read from a file.
        original_sequences: Sequences in the file.
        retained_sequences: Sequences kept after filtering.
        removed_invalid: Sequences dropped for unknown tokens or wrong length.
        removed_duplicates: Duplicate sequences dropped.
        sequence_length: Number of sites ``L``.
        num_states: Number of states ``q``.
        effective_sequences: Sum of the sequence weights (``Meff``).
    """

    source: str | None
    original_sequences: int
    retained_sequences: int
    removed_invalid: int
    removed_duplicates: int
    sequence_length: int
    num_states: int
    effective_sequences: float


@dataclass(frozen=True)
class TrainingInitialization:
    """Resolved training setup, passed to ``on_initialized`` of :func:`train_model`.

    Attributes:
        training: Summary of the training alignment.
        validation: Summary of the validation alignment, if any.
        device: Device used for training.
        dtype: Precision of the parameters.
        n_chains: Number of Markov chains.
        effective_pseudocount: Pseudocount applied to the statistics.
        config: The validated :class:`TrainingConfig`.
    """

    training: TrainingDatasetSummary
    validation: TrainingDatasetSummary | None
    device: str
    dtype: str
    n_chains: int
    effective_pseudocount: float
    config: TrainingConfig


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of :func:`train_model`.

    Attributes:
        model: The trained :class:`DCAModel`.
        history: One list per quantity, one entry per recorded update (see
            :meth:`history_dataframe`).
        chains: Final Markov chains, one-hot, shape ``(n_chains, L, q)``.
        pseudocount: Pseudocount applied to the training statistics.
        num_sequences: Training sequences after filtering.
        effective_sequences: Effective number of training sequences (``Meff``).
        artifacts: Written files by kind (``params``, ``history``, ``ptt_archive``, ...).
        warnings: Non-fatal issues found during training.
        converged: Whether training stopped at its target (Pearson, density,
            validation plateau or converged graph) rather than at a step limit.
        stop_reason: Why training stopped, e.g. ``"target_pearson"``.
        gradient_steps: Accepted parameter updates.
        structure_steps: Graph activations or decimations.
        sweeps: Monte Carlo sweeps performed.
        config: The validated :class:`TrainingConfig`.
        input_report: Filtering report of the training alignment.
        initialization: Resolved setup, see :class:`TrainingInitialization`.
        partition_estimate: PTT only: estimate of log Z for the final model.
        ptt_sampler: PTT only: the sampler, usable to draw more sequences.
    """

    model: DCAModel
    history: dict[str, Sequence[float]]
    chains: torch.Tensor
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
    initialization: TrainingInitialization | None = None
    partition_estimate: PartitionEstimate | None = None
    ptt_sampler: PTTSampler | None = field(default=None, repr=False, compare=False)

    def history_dataframe(self) -> pd.DataFrame:
        """Return the history as a DataFrame, one row per recorded update."""
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
                "initialization": self.initialization,
                "partition_estimate": self.partition_estimate,
                "final_metrics": self.final_metrics,
            },
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write :meth:`to_dict` as JSON.

        Args:
            path: Output file.
            indent: JSON indentation.

        Returns:
            The written path.
        """
        return write_json(path, self.to_dict(), indent=indent)

    def to_csv(self, path: str | Path) -> Path:
        """Write the history with the columns and names of ``history.csv``.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        from adabmDCA.training_log import columns_for_config, history_columns, history_rows

        validation = self.initialization is not None and self.initialization.validation is not None
        columns = (columns_for_config(self.config, validation=validation) if self.config is not None
                   else history_columns(model_type="bmDCA", ptt_optimizer=None, validation=validation))
        rows = history_rows(dict(self.history), columns)

        def write(temporary):
            import csv

            with open(temporary, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(column.name for column in columns)
                writer.writerows(rows)

        return _atomic_write(path, write)

    def save_bundle(self, directory: str | Path, *, label: str | None = None) -> dict[str, Path]:
        """Write a JSON summary, and the history table unless training already wrote it.

        Args:
            directory: Output folder.
            label: Optional file-name prefix.

        Returns:
            The written paths by kind: ``summary`` and possibly ``history``.
        """
        folder = Path(directory)
        prefix = f"{label}_" if label else ""
        saved = {"summary": self.to_json(folder / f"{prefix}training.json")}
        if "history" not in self.artifacts:
            saved["history"] = self.to_csv(folder / f"{prefix}history.csv")
        return saved

    @property
    def final_metrics(self) -> dict[str, Any]:
        """Last value of every history quantity, or an empty mapping before any update."""
        if not self.history.get("Epochs"):
            return {}
        return {key: values[-1] for key, values in self.history.items() if values}


@dataclass(frozen=True)
class ReintegrationResult:
    """Outcome of :func:`reintegrate_model`.

    Attributes:
        training: Result of training on the combined alignment.
        alignment: Natural plus experimental sequences used for training.
        weights: Signed weight of each sequence of ``alignment``.
        lambda_value: Weight of the experimental data.
        scaling_factor: Factor applied to the experimental weights.
        label: File-name prefix of the written files.
        artifacts: Written files by kind.
    """

    training: TrainingResult
    alignment: Alignment
    weights: np.ndarray
    lambda_value: float
    scaling_factor: float
    label: str
    artifacts: dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible document with the result type and schema version."""
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
        """Write :meth:`to_dict` as JSON.

        Args:
            path: Output file.
            indent: JSON indentation.

        Returns:
            The written path.
        """
        return write_json(path, self.to_dict(), indent=indent)

    def save_bundle(self, directory: str | Path) -> dict[str, Path]:
        """Write the combined alignment, its weights and a JSON summary to ``directory``.

        Files are named ``<label>_msa.fasta``, ``<label>_weights.dat`` and
        ``<label>_reintegration.json``.

        Args:
            directory: Output folder.

        Returns:
            The written paths by kind: ``alignment``, ``weights`` and ``summary``.
        """
        from adabmDCA.serialization import write_numpy_text

        folder = Path(directory)
        return {
            "alignment": _write_alignment(self.alignment, folder / f"{self.label}_msa.fasta"),
            "weights": write_numpy_text(folder / f"{self.label}_weights.dat", self.weights),
            "summary": self.to_json(folder / f"{self.label}_reintegration.json"),
        }


@dataclass(frozen=True)
class ThermodynamicIntegrationProgress:
    """Progress event passed to the ``progress`` callback of :func:`estimate_entropy`.

    Attributes:
        stage: ``"theta_search"`` while raising ``theta_max``, then ``"integration"``.
        completed: Steps done in this stage.
        total: Largest number of steps of this stage.
        theta: Current bias strength.
        entropy: Current entropy estimate (integration stage only).
        mean_sequence_identity: Mean identity of the chains with the target.
    """

    stage: str
    completed: int
    total: int
    theta: float
    entropy: float | None = None
    mean_sequence_identity: float | None = None


@dataclass(frozen=True)
class ThermodynamicIntegrationResult:
    """Outcome of :func:`estimate_entropy`.

    Attributes:
        entropy: Estimated model entropy, in nats.
        free_energy: Free energy at ``theta = 0`` from the integration.
        theta_max: Largest bias strength reached.
        target_fraction: Fraction of chains matching the target at ``theta_max``.
        history: Per integration step: ``theta``, ``free_energy``, ``entropy``,
            ``mean_sequence_identity`` and ``elapsed_seconds``.
        model: Metadata of the model.
        artifacts: Written files by kind, when ``output_dir`` was given.
    """

    entropy: float
    free_energy: float
    theta_max: float
    target_fraction: float
    history: dict[str, Sequence[float]]
    model: ModelMetadata
    artifacts: dict[str, Path] = field(default_factory=dict)

    def history_dataframe(self) -> pd.DataFrame:
        """Return the integration history as a DataFrame, one row per step."""
        return _history_dataframe(self.history)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible document with the result type and schema version."""
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
        """Write :meth:`to_dict` as JSON.

        Args:
            path: Output file.
            indent: JSON indentation.

        Returns:
            The written path.
        """
        return write_json(path, self.to_dict(), indent=indent)

    def to_csv(self, path: str | Path) -> Path:
        """Write the integration history as CSV.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        return write_dataframe(path, self.history_dataframe(), index=False)

    def to_log(self, path: str | Path) -> Path:
        """Write the integration history as an aligned, space-separated text table.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        columns = tuple(self.history)
        header = " ".join(f"{column:<15}" for column in columns)
        rows = zip(*(self.history[column] for column in columns))
        content = header + "\n" + "".join(" ".join(f"{value:<15.6g}" for value in row) + "\n" for row in rows)
        return write_text(path, content)

    def save_bundle(self, directory: str | Path, *, label: str = "entropy") -> dict[str, Path]:
        """Write ``<label>.log``, ``<label>.csv`` and ``<label>.json`` to ``directory``.

        Args:
            directory: Output folder.
            label: File-name stem.

        Returns:
            The written paths by kind: ``log``, ``csv`` and ``summary``.
        """
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
