"""Input normalization shared by the high-level API."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from adabmDCA.api.exceptions import InputValidationError, ModelCompatibilityError
from adabmDCA.input_loading import AlignmentInput, AlignmentLoadConfig, load_alignment
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
    try:
        values = (sequences,) if isinstance(sequences, str) else tuple(str(s) for s in sequences)
    except TypeError as exc:
        raise InputValidationError("sequences must be a string or an iterable of strings.") from exc
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
    path: AlignmentInput,
    *,
    alphabet: str,
    expected_length: int | None = None,
    remove_duplicates: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load names and validated sequences from a FASTA file."""
    loaded = load_alignment(
        path,
        config=AlignmentLoadConfig(
            alphabet=alphabet,
            invalid_sequences="error",
            remove_duplicates=remove_duplicates,
            expected_length=expected_length,
        ),
    )
    return loaded.alignment.names, loaded.alignment.sequences
