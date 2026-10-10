"""Multithreaded CPU kernels compiled with Numba (optional: ``pip install adabmDCA[cpu]``).

On CPU, :mod:`adabmDCA.sampling` and the PTT sampler use these kernels when
Numba is installed; otherwise they run PyTorch code. Both sample the same
distributions. The number of threads follows ``torch.get_num_threads()``, and
the environment variable ``ADABMDCA_NUMBA=0`` disables the kernels (see
:mod:`adabmDCA.numba_kernels.threads` for the opt-in ``ADABMDCA_NUMBA_PIN=1``).

Modules:
    sampling: Gibbs, Metropolis and Metropolized Gibbs updates of integer
        states, for one model or a stack of replicas (PTT), and exact samples
        of the independent-site (profile) model.
    exchange: Acceptance of PTT exchanges between two models.
    swaps: Fused swaps, permutations and metadata for seven-model generation.
    random: The counter-based random numbers the kernels use.
    threads: Thread count and placement.
"""

from __future__ import annotations

import os
from importlib import import_module

try:
    import numba
except ImportError:  # pragma: no cover - optional dependency
    numba = None
else:
    if "NUMBA_THREADING_LAYER" not in os.environ:
        # Numba's OpenMP layer runs a second OpenMP runtime next to PyTorch's; its
        # idle threads keep spinning and slow every later PyTorch operation several
        # times. The workqueue layer is a plain thread pool that sleeps when idle.
        numba.config.THREADING_LAYER = "workqueue"

_EXPORTS = {
    "exchange_log_acceptance": "adabmDCA.numba_kernels.exchange",
    "METHODS": "adabmDCA.numba_kernels.sampling",
    "sample_categorical": "adabmDCA.numba_kernels.sampling",
    "sample_replicas": "adabmDCA.numba_kernels.sampling",
    "sample_profile_states": "adabmDCA.numba_kernels.sampling",
    "kernel_couplings": "adabmDCA.numba_kernels.sampling",
    "categorical_sampler": "adabmDCA.numba_kernels.sampling",
    "replica_sampler": "adabmDCA.numba_kernels.sampling",
    "onehot_sampler": "adabmDCA.numba_kernels.sampling",
}

__all__ = ["is_numba_available", *_EXPORTS]


def is_numba_available() -> bool:
    """Whether the Numba kernels are installed and enabled (``ADABMDCA_NUMBA`` is not ``0``)."""
    return numba is not None and os.environ.get("ADABMDCA_NUMBA", "1") != "0"


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'adabmDCA.numba_kernels' has no attribute {name!r}")
    if numba is None:
        raise ImportError("The Numba kernels need Numba: pip install 'adabmDCA[cpu]'.")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
