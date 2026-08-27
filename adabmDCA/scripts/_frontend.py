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
        print("  Outputs:")
        for name, path in artifacts.items():
            print(f"    {name}: {path}")


def input_stem(path: str | Path) -> str:
    """Return a useful stem for plain or gzip-compressed sequence files."""
    value = Path(path)
    if value.suffix.lower() == ".gz":
        value = value.with_suffix("")
    return value.stem
