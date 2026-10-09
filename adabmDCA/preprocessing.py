"""Pure and file-oriented multiple-sequence-alignment preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from adabmDCA.alignment import (
    Alignment,
    AlignmentFormat,
    normalize_gap_symbols,
    read_alignment,
)
from adabmDCA.alphabet import detect_alphabet, get_tokens
from adabmDCA.exceptions import InputValidationError
from adabmDCA.serialization import RESULT_SCHEMA_VERSION, write_json


@dataclass(frozen=True)
class AlignmentProcessingConfig:
    """Transformations applied by :func:`preprocess_alignment`, in this order.

    Attributes:
        remove_insertions: Delete lowercase residues and dots (insertion states
            of profile alignments); otherwise dots become ``"-"``.
        max_gap_fraction: Remove sequences with a larger fraction of gaps, or ``None``.
        remove_duplicates: Keep only the first copy of identical sequences.
        alphabet: Check tokens against this alphabet (``"auto"`` detects it), or
            ``None`` to skip the check.
        gap_token: Gap symbol counted by ``max_gap_fraction``.
        unknown_tokens: With ``alphabet``: ``"gap"`` replaces unknown tokens with
            ``"-"``, ``"remove"`` drops their sequences, ``"error"`` raises.
    """

    remove_insertions: bool = False
    max_gap_fraction: float | None = None
    remove_duplicates: bool = False
    alphabet: str | None = None
    gap_token: str = "-"
    unknown_tokens: str = "gap"


@dataclass(frozen=True)
class AlignmentProcessingReport:
    """What :func:`preprocess_alignment` changed.

    Attributes:
        input_sequences: Sequences read.
        output_sequences: Sequences kept.
        input_length: Alignment length before processing.
        output_length: Alignment length after processing.
        removed_insertion_characters: Insertion residues deleted.
        normalized_gap_characters: Dots turned into ``"-"``.
        removed_for_gap_fraction: Sequences dropped for too many gaps.
        removed_as_duplicates: Duplicate sequences dropped.
        replaced_unknown_characters: Unknown tokens replaced with ``"-"``.
        removed_for_unknown_tokens: Sequences dropped for unknown tokens.
        max_gap_fraction: Gap threshold used, if any.
        removed_names: Names of all dropped sequences.
    """

    input_sequences: int
    output_sequences: int
    input_length: int
    output_length: int
    removed_insertion_characters: int = 0
    normalized_gap_characters: int = 0
    removed_for_gap_fraction: int = 0
    removed_as_duplicates: int = 0
    replaced_unknown_characters: int = 0
    removed_for_unknown_tokens: int = 0
    max_gap_fraction: float | None = None
    removed_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return the report as a dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the report as a versioned JSON document.

        Args:
            path: Output file.
            indent: JSON indentation.

        Returns:
            The written path.
        """
        return write_json(
            path,
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "result_type": "alignment_processing",
                **self.to_dict(),
            },
            indent=indent,
        )


@dataclass(frozen=True)
class AlignmentFilterResult:
    """Outcome of :func:`filter_gap_fraction`.

    Attributes:
        alignment: Kept sequences.
        keep_mask: For each input sequence, whether it was kept.
        gap_fractions: Gap fraction of each input sequence.
        removed_names: Names of the removed sequences.
        report: Summary of the filtering.
    """

    alignment: Alignment
    keep_mask: tuple[bool, ...]
    gap_fractions: tuple[float, ...]
    removed_names: tuple[str, ...]
    report: AlignmentProcessingReport


@dataclass(frozen=True)
class AlignmentProcessingResult:
    """Outcome of :func:`preprocess_alignment`.

    Attributes:
        alignment: Processed alignment.
        keep_mask: For each input sequence, whether it was kept.
        report: What was changed, see :class:`AlignmentProcessingReport`.
        output_path: Written FASTA file, if ``output_path`` was given.
    """

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
    """Remove dots and lowercase insertion residues from every sequence.

    Args:
        alignment: Profile alignment with insertion states in lowercase.

    Returns:
        A new :class:`Alignment` containing only the match columns.
    """
    return _remove_insertions_from_alignment(alignment)


def filter_gap_fraction(
    alignment: Alignment,
    *,
    max_gap_fraction: float = 0.2,
    gap_token: str = "-",
) -> AlignmentFilterResult:
    """Remove sequences whose gap fraction is greater than the threshold.

    A sequence with a gap fraction exactly equal to the threshold is retained.

    Args:
        alignment: Alignment to filter.
        max_gap_fraction: Largest allowed fraction of gaps, in ``[0, 1]``.
        gap_token: Gap symbol.

    Returns:
        An :class:`AlignmentFilterResult`.
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


def _selected_tokens(alignment: Alignment, alphabet: str) -> str:
    if alphabet != "auto":
        return get_tokens(alphabet)
    # Ignore only symbols absent from every built-in alphabet while detecting.
    standard_tokens = set().union(*(set(get_tokens(name)) for name in ("dna", "rna", "protein")))
    known_sequences = tuple(
        "".join(character for character in sequence if character in standard_tokens)
        for sequence in alignment.sequences
    )
    return get_tokens(detect_alphabet(known_sequences))


def _handle_unknown_tokens(
    alignment: Alignment, alphabet: str, policy: str
) -> tuple[Alignment, tuple[bool, ...], int, tuple[str, ...]]:
    tokens = _selected_tokens(alignment, alphabet)
    allowed = set(tokens) | {"-"}
    invalid = tuple(index for index, sequence in enumerate(alignment.sequences)
                    if set(sequence) - allowed)
    if policy == "error" and invalid:
        unexpected = sorted(set().union(*(set(alignment.sequences[index]) - allowed for index in invalid)))
        raise InputValidationError(
            "The processed alignment contains tokens outside the selected alphabet.",
            details={"unexpected_tokens": unexpected, "invalid_names": [alignment.names[index] for index in invalid],
                     "tokens": tokens},
        )
    if policy == "remove":
        invalid_set = set(invalid)
        keep = tuple(index not in invalid_set for index in range(alignment.num_sequences))
        if not any(keep):
            raise InputValidationError("Removing sequences with unknown tokens would empty the alignment.")
        cleaned = Alignment(
            names=tuple(name for name, retained in zip(alignment.names, keep) if retained),
            sequences=tuple(sequence for sequence, retained in zip(alignment.sequences, keep) if retained),
            source=alignment.source,
        )
        return cleaned, keep, 0, tuple(alignment.names[index] for index in invalid)
    replaced = sum(character not in allowed for sequence in alignment.sequences for character in sequence)
    cleaned = Alignment(
        names=alignment.names,
        sequences=tuple("".join(character if character in allowed else "-" for character in sequence)
                        for sequence in alignment.sequences),
        source=alignment.source,
    )
    return cleaned, (True,) * alignment.num_sequences, replaced, ()


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
    unknown_tokens: str | None = None,
    line_width: int = 0,
) -> AlignmentProcessingResult:
    """Clean a raw alignment for DCA: read, transform, filter and optionally write it.

    Steps, each optional: remove insertions (or normalize dots to ``"-"``),
    check tokens against an alphabet, drop gappy sequences, drop duplicates.
    Keyword options override the corresponding :class:`AlignmentProcessingConfig`
    fields.

    Args:
        input_path: FASTA or Stockholm file.
        output_path: FASTA file to write, or ``None``.
        input_format: ``"auto"``, ``"fasta"`` or ``"stockholm"``.
        alignment_index: Alignment to read from a multi-alignment Stockholm file.
        config: Base settings; see :class:`AlignmentProcessingConfig`.
        remove_insertions: Overrides ``config.remove_insertions``.
        max_gap_fraction: Overrides ``config.max_gap_fraction``.
        remove_duplicates: Overrides ``config.remove_duplicates``.
        alphabet: Overrides ``config.alphabet``.
        gap_token: Overrides ``config.gap_token``.
        unknown_tokens: Overrides ``config.unknown_tokens``.
        line_width: Largest sequence-line length of the output; ``0`` means no wrapping.

    Returns:
        An :class:`AlignmentProcessingResult`.

    Raises:
        InputValidationError: If an option is invalid, or unknown tokens are
            found with ``unknown_tokens="error"``.

    Example:
        >>> result = preprocess_alignment("RF00379.sto", output_path="clean.fasta",
        ...                               remove_insertions=True, max_gap_fraction=0.2)
        >>> result.report.output_sequences
    """
    selected = config or AlignmentProcessingConfig()
    should_remove_insertions = selected.remove_insertions if remove_insertions is None else remove_insertions
    should_remove_duplicates = selected.remove_duplicates if remove_duplicates is None else remove_duplicates
    selected_gap_fraction = selected.max_gap_fraction if max_gap_fraction is None else max_gap_fraction
    selected_alphabet = selected.alphabet if alphabet is None else alphabet
    selected_gap_token = selected.gap_token if gap_token is None else gap_token
    selected_unknown_tokens = selected.unknown_tokens if unknown_tokens is None else unknown_tokens
    if selected_unknown_tokens not in {"gap", "remove", "error"}:
        raise InputValidationError("unknown_tokens must be 'gap', 'remove', or 'error'.")

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
    replaced_unknown = 0
    removed_for_unknown = 0
    if selected_alphabet is not None:
        current, unknown_keep, replaced_unknown, unknown_names = _handle_unknown_tokens(
            current, selected_alphabet, selected_unknown_tokens
        )
        original_indices = [index for index, keep in zip(original_indices, unknown_keep) if keep]
        removed_names.extend(unknown_names)
        removed_for_unknown = len(unknown_names)
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
        replaced_unknown_characters=replaced_unknown,
        removed_for_unknown_tokens=removed_for_unknown,
        max_gap_fraction=selected_gap_fraction,
        removed_names=tuple(removed_names),
    )
    return AlignmentProcessingResult(current, keep_mask, report, written)
