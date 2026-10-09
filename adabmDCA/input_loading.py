"""Shared, side-effect-free loading of alignments and sequence weights."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
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
from adabmDCA.alphabet import detect_alphabet
from adabmDCA.exceptions import (
    InputValidationError,
    ModelCompatibilityError,
    WeightLoadError,
)
from adabmDCA.fasta import compute_weights, get_tokens

AlignmentInput = str | Path | Alignment
InvalidSequencePolicy = Literal["error", "drop"]
WeightInput = str | Path | Sequence[float] | np.ndarray | torch.Tensor


@dataclass(frozen=True)
class AlignmentLoadConfig:
    """How :func:`load_alignment` reads, validates and filters an alignment.

    Attributes:
        alphabet: ``"auto"`` (detect DNA, RNA or protein), a built-in name, or a
            custom token string.
        invalid_sequences: ``"error"`` to reject, or ``"drop"`` to remove,
            sequences with tokens outside the alphabet.
        remove_duplicates: Keep only the first copy of identical sequences.
        expected_length: Required alignment length, or ``None``.
        format: ``"auto"``, ``"fasta"`` or ``"stockholm"``.
        alignment_index: Alignment to read from a multi-alignment Stockholm file.
        normalize_dots: Turn ``"."`` gaps into ``"-"``.
    """

    alphabet: str = "auto"
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


def load_alignment(
    source: AlignmentInput,
    *,
    config: AlignmentLoadConfig | None = None,
    alphabet: str | None = None,
    invalid_sequences: InvalidSequencePolicy | None = None,
    remove_duplicates: bool | None = None,
    expected_length: int | None = None,
    format: AlignmentFormat | None = None,
    alignment_index: int | None = None,
    normalize_dots: bool | None = None,
) -> Alignment:
    """Parse, validate, filter, and deduplicate an alignment.

    The alphabet is detected from standard DNA, RNA, or protein tokens by
    default; pass ``alphabet`` for ambiguous or custom alignments. Direct
    keyword options override the corresponding fields in ``config``.
    The returned :class:`Alignment` contains its selected encoding tokens and
    filtering provenance, so callers do not need a separate wrapper type.

    Args:
        source: Path, :class:`Alignment` or sequences.
        config: Loading policy; see :class:`AlignmentLoadConfig`.
        alphabet: Overrides ``config.alphabet``.
        invalid_sequences: Overrides ``config.invalid_sequences``.
        remove_duplicates: Overrides ``config.remove_duplicates``.
        expected_length: Overrides ``config.expected_length``.
        format: Overrides ``config.format``.
        alignment_index: Overrides ``config.alignment_index``.
        normalize_dots: Overrides ``config.normalize_dots``.

    Returns:
        The filtered :class:`Alignment`, with ``tokens``, ``encoded_sequences``
        and the indices of the retained sequences.

    Raises:
        AlignmentLoadError: If the file cannot be read.
        InputValidationError: If sequences are invalid under ``"error"``, the
            length is wrong, or no sequence remains.

    Example:
        >>> alignment = load_alignment("family.fasta", invalid_sequences="drop")
        >>> alignment.tokens, alignment.num_sequences
    """
    policy = config or AlignmentLoadConfig()
    overrides = {
        name: value
        for name, value in {
            "alphabet": alphabet,
            "invalid_sequences": invalid_sequences,
            "remove_duplicates": remove_duplicates,
            "expected_length": expected_length,
            "format": format,
            "alignment_index": alignment_index,
            "normalize_dots": normalize_dots,
        }.items()
        if value is not None
    }
    if overrides:
        policy = replace(policy, **overrides)
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

    if policy.alphabet == "auto":
        resolved_alphabet = alignment.tokens or detect_alphabet(alignment.sequences)
    else:
        resolved_alphabet = policy.alphabet
    tokens = get_tokens(resolved_alphabet)
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
    return Alignment(
        names=selected.names,
        sequences=selected.sequences,
        source=selected.source,
        tokens=tokens,
        retained_indices=retained,
        dropped_indices=invalid if policy.invalid_sequences == "drop" else (),
        duplicate_indices=tuple(duplicate_indices),
        original_size=len(alignment),
    )


def load_sequence_weights(
    source: WeightInput | None,
    *,
    loaded_alignment: Alignment,
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

    Args:
        source: Weights file (one number per line), sequence, array or tensor, or
            ``None`` to compute them. Their count may match the original or the
            retained sequences.
        loaded_alignment: Alignment returned by :func:`load_alignment`.
        no_reweighting: Give every sequence weight 1 (``source`` is ignored).
        clustering_seqid: Computed weights: each sequence gets
            ``1 / (number of sequences more than this identical to it)``,
            itself included.
        device: Device of the returned tensor.
        dtype: Precision of the returned tensor.
        allow_negative: Accept negative weights.
        require_positive_sum: Reject weights whose sum is not positive.

    Returns:
        One weight per retained sequence.

    Raises:
        WeightLoadError: If the weights cannot be read, have the wrong count, or
            are not finite, negative, or sum to zero when disallowed.
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
        retained_size = len(loaded_alignment)
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
