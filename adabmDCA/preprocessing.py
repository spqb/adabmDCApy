"""Pure and file-oriented multiple-sequence-alignment preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from adabmDCA.alignment import (
    Alignment,
    AlignmentFormat,
    normalize_gap_symbols,
    read_alignment,
)
from adabmDCA.api.exceptions import InputValidationError
from adabmDCA.alphabet import get_tokens


@dataclass(frozen=True)
class AlignmentProcessingConfig:
    """Explicit transformations applied by :func:`preprocess_alignment`."""

    remove_insertions: bool = False
    max_gap_fraction: float | None = None
    remove_duplicates: bool = False
    alphabet: str | None = None
    gap_token: str = "-"


@dataclass(frozen=True)
class AlignmentProcessingReport:
    """Auditable summary of an alignment-processing operation."""

    input_sequences: int
    output_sequences: int
    input_length: int
    output_length: int
    removed_insertion_characters: int = 0
    normalized_gap_characters: int = 0
    removed_for_gap_fraction: int = 0
    removed_as_duplicates: int = 0
    max_gap_fraction: float | None = None
    removed_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        output = Path(path)
        output.write_text(json.dumps(self.to_dict(), indent=indent) + "\n", encoding="utf-8")
        return output


@dataclass(frozen=True)
class AlignmentFilterResult:
    """Filtered alignment plus masks and per-sequence gap fractions."""

    alignment: Alignment
    keep_mask: tuple[bool, ...]
    gap_fractions: tuple[float, ...]
    removed_names: tuple[str, ...]
    report: AlignmentProcessingReport


@dataclass(frozen=True)
class AlignmentProcessingResult:
    """Processed alignment, provenance mask, report, and optional output."""

    alignment: Alignment
    keep_mask: tuple[bool, ...]
    report: AlignmentProcessingReport
    output_path: Path | None = None


def _remove_insertions_from_alignment(alignment: Alignment) -> Alignment:
    sequences = tuple(
        "".join(character for character in sequence if character != "." and not character.islower())
        for sequence in alignment.sequences
    )
    return Alignment(names=alignment.names, sequences=sequences, source=alignment.source)


def remove_insertions(alignment: Alignment) -> Alignment:
    """Remove dots and lowercase insertion residues from every sequence."""
    return _remove_insertions_from_alignment(alignment)


def filter_gap_fraction(
    alignment: Alignment,
    *,
    max_gap_fraction: float = 0.2,
    gap_token: str = "-",
) -> AlignmentFilterResult:
    """Remove sequences whose gap fraction is greater than the threshold.

    A sequence with a gap fraction exactly equal to the threshold is retained.
    """
    if not 0.0 <= max_gap_fraction <= 1.0:
        raise InputValidationError("max_gap_fraction must be between 0 and 1.")
    if len(gap_token) != 1:
        raise InputValidationError("gap_token must be exactly one character.")
    fractions = tuple(sequence.count(gap_token) / len(sequence) for sequence in alignment.sequences)
    keep_mask = tuple(fraction <= max_gap_fraction for fraction in fractions)
    if not any(keep_mask):
        raise InputValidationError(
            "Gap filtering removed every sequence.",
            details={"max_gap_fraction": max_gap_fraction},
        )
    names = tuple(name for name, keep in zip(alignment.names, keep_mask) if keep)
    sequences = tuple(sequence for sequence, keep in zip(alignment.sequences, keep_mask) if keep)
    removed = tuple(name for name, keep in zip(alignment.names, keep_mask) if not keep)
    filtered = Alignment(names, sequences, source=alignment.source)
    report = AlignmentProcessingReport(
        input_sequences=alignment.num_sequences,
        output_sequences=filtered.num_sequences,
        input_length=alignment.sequence_length,
        output_length=filtered.sequence_length,
        removed_for_gap_fraction=len(removed),
        max_gap_fraction=max_gap_fraction,
        removed_names=removed,
    )
    return AlignmentFilterResult(filtered, keep_mask, fractions, removed, report)


def _validate_alphabet(alignment: Alignment, alphabet: str) -> None:
    tokens = get_tokens(alphabet)
    unexpected = sorted(set("".join(alignment.sequences)) - set(tokens))
    if unexpected:
        raise InputValidationError(
            "The processed alignment contains tokens outside the selected alphabet.",
            details={"unexpected_tokens": unexpected, "tokens": tokens},
        )


def preprocess_alignment(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    input_format: AlignmentFormat = "auto",
    alignment_index: int | None = None,
    config: AlignmentProcessingConfig | None = None,
    remove_insertions: bool | None = None,
    max_gap_fraction: float | None = None,
    remove_duplicates: bool | None = None,
    alphabet: str | None = None,
    gap_token: str | None = None,
    line_width: int = 0,
) -> AlignmentProcessingResult:
    """Read, explicitly transform, optionally validate, and write an MSA."""
    selected = config or AlignmentProcessingConfig()
    should_remove_insertions = (
        selected.remove_insertions if remove_insertions is None else remove_insertions
    )
    should_remove_duplicates = (
        selected.remove_duplicates if remove_duplicates is None else remove_duplicates
    )
    selected_gap_fraction = (
        selected.max_gap_fraction if max_gap_fraction is None else max_gap_fraction
    )
    selected_alphabet = selected.alphabet if alphabet is None else alphabet
    selected_gap_token = selected.gap_token if gap_token is None else gap_token

    original = read_alignment(input_path, format=input_format, alignment_index=alignment_index)
    current = original
    removed_insertions = 0
    normalized_gaps = 0
    if should_remove_insertions:
        current = _remove_insertions_from_alignment(current)
        removed_insertions = sum(map(len, original.sequences)) - sum(map(len, current.sequences))
    else:
        normalized_gaps = sum(sequence.count(".") for sequence in current.sequences)
        current = normalize_gap_symbols(current)

    original_indices = list(range(original.num_sequences))
    removed_names: list[str] = []
    removed_for_gaps = 0
    if selected_gap_fraction is not None:
        filter_gap_token = "-" if selected_gap_token == "." else selected_gap_token
        filtered = filter_gap_fraction(
            current,
            max_gap_fraction=selected_gap_fraction,
            gap_token=filter_gap_token,
        )
        retained_indices = [index for index, keep in zip(original_indices, filtered.keep_mask) if keep]
        original_indices = retained_indices
        removed_names.extend(filtered.removed_names)
        removed_for_gaps = len(filtered.removed_names)
        current = filtered.alignment

    removed_duplicates_count = 0
    if should_remove_duplicates:
        seen: set[str] = set()
        duplicate_keep = []
        for sequence in current.sequences:
            keep = sequence not in seen
            duplicate_keep.append(keep)
            seen.add(sequence)
        duplicate_names = [name for name, keep in zip(current.names, duplicate_keep) if not keep]
        removed_names.extend(duplicate_names)
        removed_duplicates_count = len(duplicate_names)
        original_indices = [index for index, keep in zip(original_indices, duplicate_keep) if keep]
        current = Alignment(
            names=tuple(name for name, keep in zip(current.names, duplicate_keep) if keep),
            sequences=tuple(seq for seq, keep in zip(current.sequences, duplicate_keep) if keep),
            source=current.source,
        )

    if selected_alphabet is not None:
        _validate_alphabet(current, selected_alphabet)

    keep_set = set(original_indices)
    keep_mask = tuple(index in keep_set for index in range(original.num_sequences))
    written = current.write_fasta(output_path, line_width=line_width) if output_path else None
    report = AlignmentProcessingReport(
        input_sequences=original.num_sequences,
        output_sequences=current.num_sequences,
        input_length=original.sequence_length,
        output_length=current.sequence_length,
        removed_insertion_characters=removed_insertions,
        normalized_gap_characters=normalized_gaps,
        removed_for_gap_fraction=removed_for_gaps,
        removed_as_duplicates=removed_duplicates_count,
        max_gap_fraction=selected_gap_fraction,
        removed_names=tuple(removed_names),
    )
    return AlignmentProcessingResult(current, keep_mask, report, written)
