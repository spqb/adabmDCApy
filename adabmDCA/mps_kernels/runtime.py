"""Availability checks and cached Metal shader compilation.

This module never compiles shaders or initializes the GPU at import time.
"""

import os
from functools import lru_cache

import torch


def is_mps_available() -> bool:
    """Whether Metal shaders are available and enabled for this process."""
    return (os.environ.get("ADABMDCA_MPS", "1") != "0"
            and hasattr(torch.mps, "compile_shader") and torch.backends.mps.is_available())


@lru_cache(maxsize=32)
def compile_shader(source: str):
    """Compile source lazily and reuse the resulting shader library."""
    return torch.mps.compile_shader(source)
