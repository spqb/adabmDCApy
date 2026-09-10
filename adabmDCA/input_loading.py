"""Shared, side-effect-free loading of alignments and sequence weights."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from adabmDCA.alignment import (
    Alignment,
    AlignmentFormat,
    normalize_gap_symbols,
    read_alignment,
)
from adabmDCA.api.exceptions import (
    InputValidationError,
    ModelCompatibilityError,
    WeightLoadError,
)
from adabmDCA.fasta import compute_weights, encode_sequence, get_tokens

AlignmentInput = str | Path | Alignment
InvalidSequencePolicy = Literal["error", "drop"]
WeightInput = str | Path | Sequence[float] | np.ndarray | torch.Tensor


@dataclass(frozen=True)
class AlignmentLoadConfig:
    """Policy applied after parsing an alignment."""

    alphabet: str = "protein"
    invalid_sequences: InvalidSequencePolicy = "error"
    remove_duplicates: bool = False
    expected_length: int | None = None
    format: AlignmentFormat = "auto"
    alignment_index: int | None = None
    normalize_dots: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.alphabet, str) or not self.alphabet:
            raise InputValidationError("alphabet must be a non-empty string.")
        if self.invalid_sequences not in {"error", "drop"}:
            raise InputValidationError("invalid_sequences must be either 'error' or 'drop'.")
        if self.format not in {"auto", "fasta", "stockholm"}:
            raise InputValidationError(
                "format must be 'auto', 'fasta', or 'stockholm'.",
                details={"format": self.format},
            )
        if self.alignment_index is not None and self.alignment_index < 0:
            raise InputValidationError("alignment_index cannot be negative.")
        if self.expected_length is not None and self.expected_length < 1:
            raise InputValidationError("expected_length must be positive.")


@dataclass(frozen=True)
class LoadedAlignment:
    """Validated alignment plus provenance of filtering transformations."""

    alignment: Alignment
    tokens: str
    retained_indices: tuple[int, ...]
    dropped_indices: tuple[int, ...] = ()
    duplicate_indices: tuple[int, ...] = ()
    original_size: int = 0

    @property
    def encoded_sequences(self) -> np.ndarray:
        return encode_sequence(list(self.alignment.sequences), tokens=self.tokens)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable filtering and provenance report."""
        return {
            "source": (None if self.alignment.source is None else str(self.alignment.source)),
            "original_sequences": self.original_size,
            "retained_sequences": len(self.alignment),
            "retained_indices": list(self.retained_indices),
            "dropped_indices": list(self.dropped_indices),
            "duplicate_indices": list(self.duplicate_indices),
            "sequence_length": self.alignment.sequence_length,
            "tokens": self.tokens,
        }


def load_alignment(
    source: AlignmentInput,
    *,
    config: AlignmentLoadConfig | None = None,
) -> LoadedAlignment:
    """Parse, validate, filter, and deduplicate one alignment."""
    policy = config or AlignmentLoadConfig()
    if isinstance(source, Alignment):
        alignment = source
    else:
        alignment = read_alignment(
            source,
            format=policy.format,
            alignment_index=policy.alignment_index,
        )
    if policy.normalize_dots:
        alignment = normalize_gap_symbols(alignment)

    tokens = get_tokens(policy.alphabet)
    if policy.expected_length is not None and alignment.sequence_length != policy.expected_length:
        raise ModelCompatibilityError(
            f"Alignment length ({alignment.sequence_length}) does not match the expected length "
            f"({policy.expected_length}).",
            details={
                "alignment_length": alignment.sequence_length,
                "expected_length": policy.expected_length,
            },
        )

    token_set = set(tokens)
    invalid = tuple(index for index, sequence in enumerate(alignment.sequences) if not set(sequence) <= token_set)
    if invalid and policy.invalid_sequences == "error":
        unexpected = sorted(set().union(*(set(alignment.sequences[index]) - token_set for index in invalid)))
        raise InputValidationError(
            "Alignment contains sequences with tokens outside the selected alphabet.",
            details={
                "invalid_indices": list(invalid),
                "invalid_names": [alignment.names[index] for index in invalid],
                "unexpected_tokens": unexpected,
                "tokens": tokens,
            },
        )

    invalid_set = set(invalid)
    candidates = [index for index in range(len(alignment)) if index not in invalid_set]
    duplicate_indices: list[int] = []
    if policy.remove_duplicates:
        seen: set[str] = set()
        unique: list[int] = []
        for index in candidates:
            sequence = alignment.sequences[index]
            if sequence in seen:
                duplicate_indices.append(index)
            else:
                seen.add(sequence)
                unique.append(index)
        candidates = unique

    if not candidates:
        details: dict[str, object] = {
            "original_sequences": len(alignment),
            "selected_alphabet": policy.alphabet,
            "allowed_tokens": tokens,
        }
        if alignment and len(invalid) == len(alignment):
            unexpected = sorted(set().union(*(set(sequence) - token_set for sequence in alignment.sequences)))
            details["unexpected_tokens"] = unexpected
            raise InputValidationError(
                f"All sequences were removed because they contain symbols outside the selected alphabet "
                f"'{policy.alphabet}'. Use the alphabet that matches the alignment, for example "
                f"'--alphabet protein', '--alphabet dna', '--alphabet rna', or an explicit custom alphabet.",
                details=details,
            )
        raise InputValidationError(
            "The alignment is empty after applying the loading policy.",
            details=details,
        )

    retained = tuple(candidates)
    selected = Alignment(
        names=tuple(alignment.names[index] for index in retained),
        sequences=tuple(alignment.sequences[index] for index in retained),
        source=alignment.source,
    )
    return LoadedAlignment(
        alignment=selected,
        tokens=tokens,
        retained_indices=retained,
        dropped_indices=invalid if policy.invalid_sequences == "drop" else (),
        duplicate_indices=tuple(duplicate_indices),
        original_size=len(alignment),
    )


def load_sequence_weights(
    source: WeightInput | None,
    *,
    loaded_alignment: LoadedAlignment,
    no_reweighting: bool,
    clustering_seqid: float,
    device: torch.device,
    dtype: torch.dtype,
    allow_negative: bool = False,
    require_positive_sum: bool = True,
) -> torch.Tensor:
    """Load or calculate weights and align them with retained sequences.

    ``allow_negative`` and ``require_positive_sum`` are intended for signed
    experimental adjustment vectors; ordinary statistical weights should keep
    their safe defaults.
    """
    encoded = torch.as_tensor(
        loaded_alignment.encoded_sequences,
        dtype=torch.int64,
        device=device,
    )
    if no_reweighting:
        weights = torch.ones(len(encoded), device=device, dtype=dtype)
    elif source is None:
        weights = compute_weights(
            data=encoded,
            th=clustering_seqid,
            device=device,
            dtype=dtype,
        )
    else:
        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                if not path.is_file():
                    raise WeightLoadError(
                        f"Weights file '{path}' was not found.",
                        details={"path": str(path)},
                    )
                values = np.loadtxt(path, dtype=float, ndmin=1)
            elif isinstance(source, torch.Tensor):
                values = source.detach().cpu().numpy()
            else:
                values = np.asarray(source, dtype=float)
        except WeightLoadError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise WeightLoadError(f"Could not load sequence weights: {exc}") from exc

        values = np.asarray(values, dtype=float)
        if values.ndim == 0:
            values = values.reshape(1)
        elif values.ndim != 1:
            raise WeightLoadError(
                "Sequence weights must be one-dimensional.",
                details={"shape": tuple(values.shape)},
            )
        retained_size = len(loaded_alignment.alignment)
        if len(values) == loaded_alignment.original_size:
            values = values[list(loaded_alignment.retained_indices)]
        elif len(values) != retained_size:
            raise WeightLoadError(
                "The number of weights does not match either the original or retained alignment.",
                details={
                    "weights": len(values),
                    "original_sequences": loaded_alignment.original_size,
                    "retained_sequences": retained_size,
                },
            )
        weights = torch.as_tensor(values, device=device, dtype=dtype)

    if not torch.isfinite(weights).all():
        raise WeightLoadError("Sequence weights must all be finite.")
    if not allow_negative and (weights < 0).any():
        raise WeightLoadError("Sequence weights cannot be negative.")
    if require_positive_sum and weights.sum() <= 0:
        raise WeightLoadError("Sequence weights must have a positive sum.")
    return weights
