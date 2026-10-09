"""Per-tensor caches for data derived from model parameters (kernel layouts and the like)."""

from __future__ import annotations

import os
import weakref
from collections.abc import Callable
from typing import Any

import torch

_CACHE: dict[tuple, tuple] = {}


def cached(source: torch.Tensor, kind: str, build: Callable[[], Any]) -> Any:
    """``build()``, cached per (``source`` tensor, ``kind``) while the tensor is alive and unmodified.

    Entries are dropped when the tensor is garbage-collected; an in-place change
    of the tensor (a new ``_version``) rebuilds the value.
    """
    key = (id(source), kind)
    entry = _CACHE.get(key)
    if entry is not None and entry[0]() is source and entry[1] == source._version:
        return entry[2]
    value = build()
    reference = weakref.ref(source, lambda _, key=key: _CACHE.pop(key, None))
    _CACHE[key] = (reference, source._version, value)
    return value


def sparse_kernels_enabled() -> bool:
    """Whether kernels may use sparse coupling layouts (``ADABMDCA_SPARSE=0`` disables them)."""
    return os.environ.get("ADABMDCA_SPARSE", "1") != "0"
