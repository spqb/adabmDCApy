"""Input normalization shared by the high-level API."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch

from adabmDCA.api.exceptions import InputValidationError, ModelCompatibilityError
from adabmDCA.fasta import get_tokens, import_from_fasta
from adabmDCA.utils import get_device, get_dtype


def resolve_runtime(device: str, dtype: str) -> tuple[torch.device, torch.dtype]:
    """Resolve friendly runtime names without printing to stdout."""
    try:
        resolved_device = get_device(device, message=False)
        resolved_dtype = get_dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(str(exc)) from exc
    return resolved_device, resolved_dtype


def normalize_sequences(
    sequences: str | Iterable[str],
    *,
    tokens: str,
    expected_length: int | None = None,
) -> tuple[str, ...]:
    """Validate and normalize one or more aligned biological sequences."""
    values = (sequences,) if isinstance(sequences, str) else tuple(str(s) for s in sequences)
    if not values:
        raise InputValidationError("At least one sequence is required.")

    lengths = {len(sequence) for sequence in values}
    if len(lengths) != 1:
        raise InputValidationError(
            "All sequences must have the same length.",
            details={"lengths": sorted(lengths)},
        )

    unexpected = sorted(set("".join(values)) - set(tokens))
    if unexpected:
        raise InputValidationError(
            "Sequences contain tokens that are not present in the selected alphabet.",
            details={"unexpected_tokens": unexpected, "tokens": tokens},
        )

    sequence_length = len(values[0])
    if expected_length is not None and sequence_length != expected_length:
        raise ModelCompatibilityError(
            f"Sequence length ({sequence_length}) does not match model length ({expected_length}).",
            details={"sequence_length": sequence_length, "model_length": expected_length},
        )
    return values


def load_fasta_sequences(
    path: str | Path,
    *,
    alphabet: str,
    expected_length: int | None = None,
    remove_duplicates: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load names and validated sequences from a FASTA file."""
    fasta_path = Path(path)
    if not fasta_path.is_file():
        raise InputValidationError(
            f"FASTA file '{fasta_path}' was not found.",
            details={"path": str(fasta_path)},
        )
    try:
        names, sequences = import_from_fasta(
            str(fasta_path),
            remove_duplicates=remove_duplicates,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise InputValidationError(f"Could not read FASTA file '{fasta_path}': {exc}") from exc
    normalized = normalize_sequences(
        sequences.tolist(),
        tokens=get_tokens(alphabet),
        expected_length=expected_length,
    )
    return tuple(str(name) for name in names), normalized
