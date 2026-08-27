"""Portable and atomic serialization helpers for high-level API results."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adabmDCA.api.exceptions import AdabmDCAError, OutputSerializationError

RESULT_SCHEMA_VERSION = "1.0"


def to_jsonable(value: Any) -> Any:
    """Recursively convert scientific Python values to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return to_jsonable(value.detach().cpu().tolist())
    raise OutputSerializationError(
        f"Values of type '{type(value).__name__}' cannot be serialized.",
        details={"type": f"{type(value).__module__}.{type(value).__qualname__}"},
    )


def result_document(result_type: str, data: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap result data in a stable, versioned document envelope."""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_type": result_type,
        "data": to_jsonable(data),
    }


def _atomic_write(path: str | Path, writer: Callable[[Path], None]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        writer(temporary)
        temporary.replace(output)
    except AdabmDCAError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise OutputSerializationError(
            f"Could not serialize output to '{output}': {exc}",
            details={"path": str(output)},
        ) from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def write_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    indent: int = 2,
) -> Path:
    """Write strict UTF-8 JSON atomically."""
    serializable = to_jsonable(payload)

    def writer(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(serializable, indent=indent, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    return _atomic_write(path, writer)


def write_dataframe(path: str | Path, dataframe: Any, **kwargs: Any) -> Path:
    """Write a pandas-compatible dataframe atomically."""
    return _atomic_write(path, lambda temporary: dataframe.to_csv(temporary, **kwargs))


def write_text(path: str | Path, content: str) -> Path:
    """Write UTF-8 text atomically."""
    return _atomic_write(path, lambda temporary: temporary.write_text(content, encoding="utf-8"))


def write_numpy(path: str | Path, array: np.ndarray) -> Path:
    """Write a NumPy array atomically without altering the requested suffix."""

    def writer(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)

    return _atomic_write(path, writer)


def write_numpy_text(path: str | Path, array: np.ndarray, **kwargs: Any) -> Path:
    """Write a NumPy array as an atomic text artifact."""
    return _atomic_write(path, lambda temporary: np.savetxt(temporary, array, **kwargs))


def resolve_format(path: str | Path, format: str | None, supported: set[str]) -> str:
    """Resolve an explicit format or infer it from a filename suffix."""
    selected = format.lower().lstrip(".") if format else Path(path).suffix.lower().lstrip(".")
    aliases = {"fa": "fasta", "fas": "fasta", "txt": "csv"}
    selected = aliases.get(selected, selected)
    if selected not in supported:
        raise OutputSerializationError(
            f"Unsupported output format '{selected or '<none>'}'.",
            details={"supported_formats": sorted(supported), "path": str(path)},
        )
    return selected
