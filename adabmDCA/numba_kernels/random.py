"""Counter-based random numbers for the Numba kernels.

Every number is a fixed hash (SplitMix64) of ``(seed, step, chain, slot)``.
Threads need no shared generator state, so results do not depend on the
number of threads or on how chains are split between them; each kernel call
draws its ``seed`` from PyTorch, which keeps ``torch.manual_seed`` (and the PTT
sampler's own random state) in control.
"""

from __future__ import annotations

import numpy as np
import torch
from numba import njit

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
_TO_UNIT = 1.0 / 9007199254740992.0  # 2**-53

# Slots: independent streams for the different random numbers of one update.
SLOT_UNIFORM = 0
SLOT_SECOND_UNIFORM = 1
SLOT_PROPOSAL = 2
SLOT_SITE = 3
SLOT_PROFILE = 4


@njit(cache=True, inline="always")
def _mix(x):
    x = (x ^ (x >> np.uint64(30))) * _MIX_1
    x = (x ^ (x >> np.uint64(27))) * _MIX_2
    return x ^ (x >> np.uint64(31))


@njit(cache=True, inline="always")
def uniform(seed, step, chain, slot):
    """A float64 in [0, 1) determined by its four coordinates."""
    x = _mix(seed + np.uint64(step) * _GOLDEN)
    x = _mix(x + np.uint64(chain) * _GOLDEN)
    x = _mix(x + np.uint64(slot) * _GOLDEN)
    return (x >> np.uint64(11)) * _TO_UNIT


@njit(cache=True, inline="always")
def randint(seed, step, chain, slot, high):
    """An integer in [0, high), from :func:`uniform`."""
    value = int(uniform(seed, step, chain, slot) * high)
    return value if value < high else high - 1


def draw_seed() -> np.uint64:
    """A 63-bit seed from PyTorch's current random state."""
    return np.uint64(int(torch.randint(0, 2**63 - 1, (1,), dtype=torch.int64)))
