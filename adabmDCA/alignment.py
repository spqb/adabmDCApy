"""Alignment containers and FASTA/Stockholm input-output helpers."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

import numpy as np
from Bio import SeqIO

if TYPE_CHECKING:
    import pandas as pd
    import torch

from adabmDCA.exceptions import (
    AdabmDCAError,
    AlignmentFormatError,
    AlignmentLengthError,
    AlignmentLoadError,
    InputValidationError,
)

AlignmentFormat = Literal["auto", "fasta", "stockholm"]


@dataclass(frozen=True)
class Alignment:
    """Aligned sequences with optional alphabet and filtering provenance.

    Names and sequences retain their input order. The alignment must contain
    at least one nonempty sequence, and every sequence must have the same
    length. Constructing an alignment checks these structural constraints;
    :func:`adabmDCA.input_loading.load_alignment` additionally resolves the
    alphabet, validates symbols, and optionally filters rows.

    Args:
        names: Sequence names, one per row.
        sequences: Aligned sequence strings in the same order as ``names``.
        source: Path of the source file, if known.
        tokens: Ordered alphabet used for encoding. ``None`` until resolved;
            index ``i`` in an encoding corresponds to ``tokens[i]``.
        retained_indices: Positions of the current rows in the original
            alignment. Defaults to ``0, 1, ..., M - 1``.
        dropped_indices: Original positions removed for invalid symbols.
        duplicate_indices: Original positions removed as duplicates.
        original_size: Number of rows before filtering. Defaults to ``M``.

    Note:
        The provenance fields are populated by ``load_alignment``. Plain
        ``Alignment`` construction does not check sequences against ``tokens``.
    """

    names: tuple[str, ...]
    sequences: tuple[str, ...]
    source: Path | None = None
    tokens: str | None = None
    retained_indices: tuple[int, ...] = ()
    dropped_indices: tuple[int, ...] = ()
    duplicate_indices: tuple[int, ...] = ()
    original_size: int = 0

    def __post_init__(self) -> None:
        """Normalize input fields and check alignment dimensions."""
        object.__setattr__(self, "names", tuple(str(name) for name in self.names))
        object.__setattr__(self, "sequences", tuple(str(seq) for seq in self.sequences))
        if self.source is not None:
            object.__setattr__(self, "source", Path(self.source))
        if self.tokens is not None and (not isinstance(self.tokens, str) or not self.tokens):
            raise InputValidationError("Alignment tokens must be a non-empty string.")
        object.__setattr__(self, "retained_indices", tuple(self.retained_indices))
        object.__setattr__(self, "dropped_indices", tuple(self.dropped_indices))
        object.__setattr__(self, "duplicate_indices", tuple(self.duplicate_indices))
        if len(self.names) != len(self.sequences):
            raise InputValidationError(
                "Alignment names and sequences must have the same size.",
                details={"names": len(self.names), "sequences": len(self.sequences)},
            )
        if not self.sequences:
            raise InputValidationError("An alignment must contain at least one sequence.")
        lengths = sorted({len(sequence) for sequence in self.sequences})
        if len(lengths) != 1:
            raise AlignmentLengthError(
                "Alignment sequences must have a common length.",
                details={"lengths": lengths},
            )
        if lengths[0] == 0:
            raise AlignmentLengthError("Alignment sequences cannot be empty.")
        if not self.retained_indices:
            object.__setattr__(self, "retained_indices", tuple(range(len(self.sequences))))
        if self.original_size == 0:
            object.__setattr__(self, "original_size", len(self.sequences))
        if self.original_size < len(self.sequences):
            raise InputValidationError("Alignment original_size cannot be smaller than its retained size.")
        if any(index < 0 or index >= self.original_size for index in self.retained_indices):
            raise InputValidationError("Alignment retained_indices are outside original_size.")

    def __len__(self) -> int:
        """Return the number of sequences in the alignment."""
        return len(self.sequences)

    def __repr__(self) -> str:
        """Show sequence count, aligned length, and tokens without row data."""
        return (
            f"Alignment(num_sequences={self.num_sequences}, "
            f"sequence_length={self.sequence_length}, tokens={self.tokens!r})"
        )

    @property
    def num_sequences(self) -> int:
        """Number of sequences (``M``) currently in the alignment."""
        return len(self.sequences)

    @property
    def sequence_length(self) -> int:
        """Number of aligned positions (``L``) in each sequence."""
        return len(self.sequences[0])

    def to_dataframe(self) -> pd.DataFrame:
        """Return a pandas DataFrame with ``name`` and ``sequence`` columns.

        The DataFrame has one row per sequence and preserves alignment order.
        """
        import pandas as pd

        return pd.DataFrame({"name": self.names, "sequence": self.sequences})

    @property
    def encoded_sequences(self) -> np.ndarray:
        """Return integer token indices as a NumPy array of shape ``(M, L)``.

        Each element is the position of its residue in ``self.tokens``.

        Raises:
            InputValidationError: If no alphabet has been resolved. Pass the
                alignment through ``load_alignment`` first.
        """
        if self.tokens is None:
            raise InputValidationError(
                "Alignment has no encoding tokens; load it with load_alignment first."
            )
        from adabmDCA.fasta import encode_sequence

        return encode_sequence(list(self.sequences), tokens=self.tokens)

    def to_onehot(self, *, flatten: bool = False) -> torch.Tensor:
        """Encode the alignment as a CPU ``torch.float32`` one-hot tensor.

        Args:
            flatten: If ``False``, return shape ``(M, L, q)``. If ``True``,
                combine the position and token dimensions into shape
                ``(M, L * q)``, useful for PCA.

        Returns:
            A tensor whose state features follow the order of ``self.tokens``
            at each aligned position.

        Raises:
            InputValidationError: If ``tokens`` is unavailable. Load the
                alignment with ``load_alignment`` first.
        """
        import torch

        from adabmDCA.functional import one_hot

        if self.tokens is None:
            raise InputValidationError(
                "Alignment has no encoding tokens; load it with load_alignment first."
            )
        encoded = torch.as_tensor(self.encoded_sequences, dtype=torch.int64)
        representation = one_hot(encoded, num_classes=len(self.tokens))
        return representation.flatten(start_dim=1) if flatten else representation

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible loading report without sequence data.

        The report includes the source, original and retained row counts,
        filtering indices, aligned length, and tokens. For an alignment that
        has not been loaded, filtering indices describe its unchanged rows.
        """
        return {
            "source": None if self.source is None else str(self.source),
            "original_sequences": self.original_size,
            "retained_sequences": len(self),
            "retained_indices": list(self.retained_indices),
            "dropped_indices": list(self.dropped_indices),
            "duplicate_indices": list(self.duplicate_indices),
            "sequence_length": self.sequence_length,
            "tokens": self.tokens,
        }

    def write_fasta(self, path: str | Path, *, line_width: int = 0) -> Path:
        """Write names and sequences to an aligned FASTA file.

        Args:
            path: Output file path.
            line_width: Maximum sequence line length. ``0`` writes each
                sequence on one line.

        Returns:
            The output path as a :class:`pathlib.Path`.
        """
        return write_alignment(self, path, format="fasta", line_width=line_width)


@dataclass(frozen=True)
class AlignmentConversionResult:
    """Outcome of :func:`convert_alignment`.

    Attributes:
        alignment: The converted alignment.
        input_format: Detected or given input format.
        output_format: Written format (``"fasta"``).
        output_path: Written file.
    """

    alignment: Alignment
    input_format: str
    output_format: str
    output_path: Path

    def to_dict(self) -> dict[str, object]:
        """Return the formats, output path and alignment size as a dictionary."""
        return {
            "input_format": self.input_format,
            "output_format": self.output_format,
            "output_path": str(self.output_path),
            "num_sequences": self.alignment.num_sequences,
            "sequence_length": self.alignment.sequence_length,
        }


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt")
    return path.open("r", encoding="utf-8")


def detect_alignment_format(path: str | Path) -> Literal["fasta", "stockholm"]:
    """Detect whether a file is FASTA or Stockholm from its first non-empty line.

    Args:
        path: Alignment file, optionally gzip-compressed.

    Returns:
        ``"fasta"`` or ``"stockholm"``.

    Raises:
        AlignmentFormatError: If the format cannot be recognized.
    """
    input_path = Path(path)
    if not input_path.is_file():
        raise AlignmentLoadError(
            f"Alignment file '{input_path}' was not found.",
            details={"path": str(input_path)},
        )
    with _open_text(input_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                return "fasta"
            if line.upper().startswith("# STOCKHOLM"):
                return "stockholm"
            break
    raise AlignmentFormatError(
        f"Could not detect the alignment format of '{input_path}'.",
        details={"supported_formats": ["fasta", "stockholm"]},
    )


def read_alignment(
    path: str | Path,
    *,
    format: AlignmentFormat = "auto",
    alignment_index: int | None = None,
) -> Alignment:
    """Read one aligned FASTA or Stockholm alignment, without filtering.

    For validation, alphabet detection and filtering use :func:`load_alignment`.

    Args:
        path: Alignment file, optionally gzip-compressed.
        format: ``"auto"`` (detected), ``"fasta"`` or ``"stockholm"``.
        alignment_index: Which alignment of a multi-alignment Stockholm file to
            read (0-based); required when there is more than one.

    Returns:
        The :class:`Alignment`.

    Raises:
        AlignmentLoadError: If the file cannot be read.
        AlignmentFormatError: If the content is not a valid alignment.
        AlignmentLengthError: If the sequences have different lengths.
    """
    input_path = Path(path)
    selected_format = detect_alignment_format(input_path) if format == "auto" else format
    if selected_format not in {"fasta", "stockholm"}:
        raise AlignmentFormatError(
            f"Unsupported alignment format '{selected_format}'.",
            details={"supported_formats": ["fasta", "stockholm"]},
        )
    if not input_path.is_file():
        raise AlignmentLoadError(
            f"Alignment file '{input_path}' was not found.",
            details={"path": str(input_path)},
        )

    try:
        with _open_text(input_path) as handle:
            if selected_format == "stockholm":
                parsed = _parse_stockholm(handle, source=input_path)
            else:
                records = list(SeqIO.parse(handle, "fasta"))
                parsed = (
                    [
                        Alignment(
                            names=tuple(record.description or record.id for record in records),
                            sequences=tuple(str(record.seq) for record in records),
                            source=input_path,
                        )
                    ]
                    if records
                    else []
                )
    except AdabmDCAError:
        raise
    except (OSError, UnicodeError) as exc:
        raise AlignmentLoadError(
            f"Could not read alignment '{input_path}': {exc}",
            details={"path": str(input_path), "format": selected_format},
        ) from exc
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise AlignmentFormatError(
            f"Could not parse '{input_path}' as {selected_format}: {exc}",
            details={"path": str(input_path), "format": selected_format},
        ) from exc
    if not parsed:
        raise AlignmentFormatError(f"No alignment was found in '{input_path}'.")
    if alignment_index is None:
        if len(parsed) != 1:
            raise AlignmentFormatError(
                "The input contains multiple alignments; select one with alignment_index.",
                details={"num_alignments": len(parsed)},
            )
        alignment_index = 0
    if not 0 <= alignment_index < len(parsed):
        raise InputValidationError(
            f"alignment_index {alignment_index} is out of range.",
            details={"num_alignments": len(parsed)},
        )

    return parsed[alignment_index]


def _parse_stockholm(handle: TextIO, *, source: Path) -> list[Alignment]:
    """Parse Stockholm sequence rows while preserving dots and letter case."""
    alignments: list[Alignment] = []
    order: list[str] = []
    fragments: dict[str, list[str]] = {}
    inside_alignment = False

    def finish_alignment() -> None:
        nonlocal order, fragments, inside_alignment
        if not inside_alignment:
            return
        if not order:
            raise AlignmentFormatError("A Stockholm block contains no sequences.")
        alignments.append(
            Alignment(
                names=tuple(order),
                sequences=tuple("".join(fragments[name]) for name in order),
                source=source,
            )
        )
        order = []
        fragments = {}
        inside_alignment = False

    for line_number, raw_line in enumerate(handle, 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.upper().startswith("# STOCKHOLM"):
            if inside_alignment:
                raise AlignmentFormatError(
                    "A new Stockholm header appeared before the previous block ended.",
                    details={"line": line_number},
                )
            inside_alignment = True
            continue
        if line == "//":
            finish_alignment()
            continue
        if line.startswith("#"):
            continue
        if not inside_alignment:
            raise AlignmentFormatError(
                "Sequence data appeared outside a Stockholm block.",
                details={"line": line_number},
            )
        fields = line.split()
        if len(fields) < 2:
            raise AlignmentFormatError(
                "Invalid Stockholm sequence row.",
                details={"line": line_number, "content": line},
            )
        name, fragment = fields[0], fields[1]
        if name not in fragments:
            order.append(name)
            fragments[name] = []
        fragments[name].append(fragment)

    if inside_alignment:
        raise AlignmentFormatError("The final Stockholm block is missing its '//' terminator.")
    return alignments


def write_alignment(
    alignment: Alignment,
    path: str | Path,
    *,
    format: Literal["fasta"] = "fasta",
    line_width: int = 0,
) -> Path:
    """Write an alignment to a FASTA file.

    Args:
        alignment: Alignment to write.
        path: Output file.
        format: Output format; only ``"fasta"`` is supported.
        line_width: Largest sequence-line length; ``0`` writes each sequence on one line.

    Returns:
        The written path.
    """
    if format != "fasta":
        raise AlignmentFormatError(
            f"Unsupported output format '{format}'.",
            details={"supported_formats": ["fasta"]},
        )
    if line_width < 0:
        raise InputValidationError("line_width cannot be negative.")
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for name, sequence in zip(alignment.names, alignment.sequences):
            handle.write(f">{name}\n")
            if line_width:
                for start in range(0, len(sequence), line_width):
                    handle.write(sequence[start : start + line_width] + "\n")
            else:
                handle.write(sequence + "\n")
    return output_path


def normalize_gap_symbols(
    alignment: Alignment,
    *,
    source_gap: str = ".",
    target_gap: str = "-",
) -> Alignment:
    """Replace one alignment gap symbol with another.

    Stockholm permits dots as gap symbols, while adabmDCA and aligned FASTA
    files conventionally use hyphens. This transformation does not remove
    alignment columns.

    Args:
        alignment: Alignment to transform.
        source_gap: Gap symbol to replace.
        target_gap: Replacement gap symbol.

    Returns:
        A new :class:`Alignment`.
    """
    if len(source_gap) != 1 or len(target_gap) != 1:
        raise InputValidationError("Gap symbols must be exactly one character.")
    return Alignment(
        names=alignment.names,
        sequences=tuple(sequence.replace(source_gap, target_gap) for sequence in alignment.sequences),
        source=alignment.source,
        tokens=alignment.tokens,
        retained_indices=alignment.retained_indices,
        dropped_indices=alignment.dropped_indices,
        duplicate_indices=alignment.duplicate_indices,
        original_size=alignment.original_size,
    )


def convert_alignment(
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_format: AlignmentFormat = "auto",
    output_format: Literal["fasta"] = "fasta",
    alignment_index: int | None = None,
    line_width: int = 0,
) -> AlignmentConversionResult:
    """Convert a FASTA or Stockholm alignment to canonical aligned FASTA.

    Dots accepted as gaps by Stockholm are written as hyphens, the gap token
    used throughout adabmDCA. Lowercase residues are preserved; callers that
    want to delete insertion residues should use :func:`preprocess_alignment`.

    Args:
        input_path: FASTA or Stockholm file.
        output_path: FASTA file to write.
        input_format: ``"auto"``, ``"fasta"`` or ``"stockholm"``.
        output_format: Only ``"fasta"`` is supported.
        alignment_index: Alignment to convert in a multi-alignment Stockholm file.
        line_width: Largest sequence-line length; ``0`` means no wrapping.

    Returns:
        An :class:`AlignmentConversionResult`.
    """
    detected = detect_alignment_format(input_path) if input_format == "auto" else input_format
    alignment = read_alignment(
        input_path,
        format=detected,
        alignment_index=alignment_index,
    )
    alignment = normalize_gap_symbols(alignment)
    written = write_alignment(alignment, output_path, format=output_format, line_width=line_width)
    return AlignmentConversionResult(alignment, detected, output_format, written)


def convert_stockholm_to_fasta(
    input_path: str | Path,
    output_path: str | Path,
    *,
    alignment_index: int | None = None,
    line_width: int = 0,
) -> AlignmentConversionResult:
    """Convert one Stockholm alignment to FASTA; see :func:`convert_alignment`.

    Args:
        input_path: Stockholm file.
        output_path: FASTA file to write.
        alignment_index: Alignment to convert when the file contains several.
        line_width: Largest sequence-line length; ``0`` means no wrapping.

    Returns:
        An :class:`AlignmentConversionResult`.
    """
    return convert_alignment(
        input_path,
        output_path,
        input_format="stockholm",
        output_format="fasta",
        alignment_index=alignment_index,
        line_width=line_width,
    )
