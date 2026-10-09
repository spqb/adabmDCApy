"""Small presentation helpers shared by command-line adapters."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ExplicitOptionParser(argparse.ArgumentParser):
    """Retain explicit options so archive workflows can inherit stored defaults."""

    def parse_known_args(self, args=None, namespace=None):
        import sys
        arguments = list(sys.argv[1:] if args is None else args)
        result, unknown = super().parse_known_args(arguments, namespace)
        result._explicit_options = {
            self._option_string_actions[option].dest
            for arg in arguments
            if (option := arg.split("=", 1)[0]) in self._option_string_actions
        }
        return result, unknown


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
    archive = getattr(args, "ptt_resume", None)
    params_path = getattr(args, "path_params", None)
    if archive is None and params_path is not None:
        if getattr(args, "strategy", None) == "ptt":
            archive = params_path
        elif Path(params_path).is_file():
            with open(params_path, "rb") as handle:
                if handle.read(8) == b"\x89HDF\r\n\x1a\n":
                    archive = params_path
    if archive is not None:
        import json

        import h5py

        from adabmDCA.exceptions import InputValidationError
        from adabmDCA.fasta import get_tokens
        try:
            with h5py.File(archive, "r") as handle:
                tokens = json.loads(handle.attrs["metadata"])["tokens"]
        except (OSError, KeyError, ValueError) as exc:
            raise InputValidationError("PTT requires a structured HDF5 ladder archive.") from exc
        if args.alphabet != "auto" and get_tokens(args.alphabet) != tokens:
            raise InputValidationError("Explicit alphabet conflicts with the PTT archive.")
        args.alphabet = tokens
        return
    if args.alphabet != "auto":
        return
    from adabmDCA.alignment import normalize_gap_symbols, read_alignment
    from adabmDCA.alphabet import detect_alphabet

    symbols = set()
    # Model symbols disambiguate short/subset alignments and allow sampling
    # or contact prediction without an input alignment.
    params = getattr(args, "path_params", None)
    if params is not None:
        from adabmDCA.exceptions import ModelLoadError

        if not Path(params).is_file():
            raise ModelLoadError(
                f"Model parameter file '{params}' was not found.",
                details={"path": str(params)},
            )
        from adabmDCA.io import open_params

        with open_params(params) as handle:
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
