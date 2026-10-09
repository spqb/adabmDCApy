"""Thread count and placement of the Numba kernels.

The kernels use ``torch.get_num_threads()`` threads of Numba's workqueue pool.
Its workers sleep between calls, and Linux wakes them next to the thread that
wakes them: on CPUs with several core complexes (AMD Zen, for instance) the
whole pool can end up on one complex, two threads per core, which nearly
halves the speed of the kernels. ``ADABMDCA_NUMBA_PIN=1`` therefore pins
worker ``i`` to the ``i``-th core (then to the second hardware thread of each
core) when the kernels use at least as many threads as the process has physical
cores; with fewer threads the workers stay free, since several jobs sharing a
machine would otherwise all be pinned to the same cores. Pinning is off by
default: measured on a 16-core Threadripper, it was 1.2-1.7 times faster in some
sessions and up to 1.5 times slower in others, because a pinned worker cannot
leave a core that another thread is using and every call waits for its slowest
worker.
"""

from __future__ import annotations

import ctypes
import os
from functools import cache

import numba
import numpy as np
import torch
from numba import njit, prange


def _core_of(cpu: int) -> tuple[str, str]:
    base = f"/sys/devices/system/cpu/cpu{cpu}/topology/"
    try:
        with open(base + "core_id") as core, open(base + "physical_package_id") as package:
            return core.read().strip(), package.read().strip()
    except OSError:
        return str(cpu), "0"


@cache
def _cpus_by_core(allowed: frozenset[int]) -> tuple[tuple[int, ...], int]:
    """The allowed CPUs, one per physical core first, then the other hardware threads; and the core count."""
    first, rest, seen = [], [], set()
    for cpu in sorted(allowed):
        core = _core_of(cpu)
        (rest if core in seen else first).append(cpu)
        seen.add(core)
    return tuple(first + rest), len(first)


def _affinity_function():
    try:
        function = ctypes.CDLL(None, use_errno=True).sched_setaffinity
    except (AttributeError, OSError):  # not Linux
        return None
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t]
    return function


_set_affinity = _affinity_function()
_placed_threads = 0  # thread count of the last placement, pinned or not

if _set_affinity is not None:

    @njit(parallel=True)
    def _pin_workers(masks, pinned):
        for _ in prange(masks.shape[0]):
            worker = numba.get_thread_id()
            # pid 0 is the calling thread.
            if _set_affinity(0, masks.shape[1] * 8, masks[worker].ctypes.data) == 0:
                pinned[worker] = 1


def _place_workers(threads: int) -> None:
    cpus, cores = _cpus_by_core(frozenset(os.sched_getaffinity(0)))
    if threads < cores:
        return
    masks = np.zeros((threads, max(cpus) // 64 + 1), dtype=np.uint64)
    for worker in range(threads):
        cpu = cpus[worker % len(cpus)]
        masks[worker, cpu // 64] = np.uint64(1) << np.uint64(cpu % 64)
    _pin_workers(masks, np.zeros(threads, dtype=np.int8))


def set_kernel_threads() -> int:
    """Use ``torch.get_num_threads()`` threads (at most Numba's pool size) and return the count."""
    global _placed_threads
    threads = max(1, min(torch.get_num_threads(), numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(threads)
    if threads != _placed_threads:
        _placed_threads = threads
        if (_set_affinity is not None and os.environ.get("ADABMDCA_NUMBA_PIN", "0") == "1"
                and numba.config.THREADING_LAYER == "workqueue"):
            _place_workers(threads)
    return threads
