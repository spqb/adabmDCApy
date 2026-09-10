"""Small presentation helpers shared by command-line adapters."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def print_header(title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))


def print_configuration(values: Mapping[str, Any]) -> None:
    print("\nConfiguration:")
    for name, value in values.items():
        if value is not None:
            print(f"  {name}: {value}")


def print_completion(
    message: str,
    *,
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Path] | None = None,
) -> None:
    print(f"\n{message}")
    for name, value in (metrics or {}).items():
        print(f"  {name}: {value}")
    if artifacts:
        print("\nOutputs:")
        for name, path in artifacts.items():
            print(f"  {name}: {path}")


def input_stem(path: str | Path) -> str:
    """Return a useful stem for plain or gzip-compressed sequence files."""
    value = Path(path)
    if value.suffix.lower() == ".gz":
        value = value.with_suffix("")
    return value.stem


def resolve_alphabet(args) -> None:
    """Resolve CLI auto once, before constructing any workflow configuration."""
    if args.alphabet != "auto":
        return
    from adabmDCA.alignment import normalize_gap_symbols, read_alignment
    from adabmDCA.alphabet import detect_alphabet

    symbols = set()
    # Model symbols disambiguate short/subset alignments and allow sampling
    # or contact prediction without an input alignment.
    params = getattr(args, "path_params", None)
    if params is not None:
        from adabmDCA.api.exceptions import ModelLoadError

        if not Path(params).is_file():
            raise ModelLoadError(
                f"Model parameter file '{params}' was not found.",
                details={"path": str(params)},
            )
        with open(params, encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if parts and parts[0] == "h" and len(parts) == 4:
                    symbols.update(parts[2])
                elif parts and parts[0] == "J" and len(parts) == 6:
                    symbols.update(parts[3])
                    symbols.update(parts[4])
    source = getattr(args, "data", None) or getattr(args, "input_msa", None)
    if source is not None:
        alignment = normalize_gap_symbols(read_alignment(source))
        for sequence in alignment.sequences:
            symbols.update(sequence)
    args.alphabet = detect_alphabet(symbols)
