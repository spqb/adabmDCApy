"""Alignment containers and FASTA/Stockholm input-output helpers."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Literal, TextIO

from Bio import SeqIO

from adabmDCA.api.exceptions import (
    AlignmentFormatError,
    AlignmentLengthError,
    InputValidationError,
)


AlignmentFormat = Literal["auto", "fasta", "stockholm"]


@dataclass(frozen=True)
class Alignment:
    """An immutable multiple-sequence alignment.

    Names and sequences retain their input order. Every sequence must have the
    same aligned length.
    """

    names: tuple[str, ...]
    sequences: tuple[str, ...]
    source: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(str(name) for name in self.names))
        object.__setattr__(self, "sequences", tuple(str(seq) for seq in self.sequences))
        if self.source is not None:
            object.__setattr__(self, "source", Path(self.source))
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

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    @property
    def sequence_length(self) -> int:
        return len(self.sequences[0])

    def to_dataframe(self):
        """Return names and aligned sequences as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame({"name": self.names, "sequence": self.sequences})

    def write_fasta(self, path: str | Path, *, line_width: int = 0) -> Path:
        """Write the alignment to FASTA and return the output path."""
        return write_alignment(self, path, format="fasta", line_width=line_width)


@dataclass(frozen=True)
class AlignmentConversionResult:
    """Result of reading and converting an alignment file."""

    alignment: Alignment
    input_format: str
    output_format: str
    output_path: Path

    def to_dict(self) -> dict[str, object]:
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
    """Detect FASTA or Stockholm from the first meaningful input line."""
    input_path = Path(path)
    if not input_path.is_file():
        raise InputValidationError(
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
    """Read one aligned FASTA or Stockholm alignment.

    If a Stockholm file contains multiple alignments, callers must explicitly
    select one with ``alignment_index``.
    """
    input_path = Path(path)
    selected_format = detect_alignment_format(input_path) if format == "auto" else format
    if selected_format not in {"fasta", "stockholm"}:
        raise AlignmentFormatError(
            f"Unsupported alignment format '{selected_format}'.",
            details={"supported_formats": ["fasta", "stockholm"]},
        )
    if not input_path.is_file():
        raise InputValidationError(f"Alignment file '{input_path}' was not found.")

    try:
        with _open_text(input_path) as handle:
            if selected_format == "stockholm":
                parsed = _parse_stockholm(handle, source=input_path)
            else:
                records = list(SeqIO.parse(handle, "fasta"))
                parsed = [
                    Alignment(
                        names=tuple(record.description or record.id for record in records),
                        sequences=tuple(str(record.seq) for record in records),
                        source=input_path,
                    )
                ] if records else []
    except Exception as exc:
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
    """Write an alignment. FASTA is currently the supported output format."""
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
    """
    if len(source_gap) != 1 or len(target_gap) != 1:
        raise InputValidationError("Gap symbols must be exactly one character.")
    return Alignment(
        names=alignment.names,
        sequences=tuple(sequence.replace(source_gap, target_gap) for sequence in alignment.sequences),
        source=alignment.source,
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
    """Convert one Stockholm alignment to FASTA."""
    return convert_alignment(
        input_path,
        output_path,
        input_format="stockholm",
        output_format="fasta",
        alignment_index=alignment_index,
        line_width=line_width,
    )
