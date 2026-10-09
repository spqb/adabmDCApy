"""Acceptance of PTT exchanges between two models, from integer states.

Swapping configurations ``x`` (from the lower model) and ``y`` (from the
upper one) is accepted with probability ``min(1, exp(D(y) - D(x)))``, where
``D = E_upper - E_lower``. The kernel gathers the couplings of the occupied
states instead of multiplying one-hot matrices, which costs ``L**2`` reads per
configuration instead of ``(L * q)**2`` operations, half of them when the
couplings are symmetric (each pair is then summed once). Like the sampling
kernels, a block of chains visits the sites together, so the couplings of each
site are read from cache by every chain of the block.
"""

from __future__ import annotations

import numpy as np
import torch
from numba import njit, prange

from adabmDCA.numba_kernels.threads import set_kernel_threads
from adabmDCA.statmech import couplings_are_symmetric


@njit(cache=True, fastmath=True, parallel=True)
def _exchange_kernel(lower_bias, lower_couplings, upper_bias, upper_couplings, lower_states, upper_states,
                     symmetric, block, output):
    n, length = lower_states.shape
    for b in prange((n + block - 1) // block):
        end = min(n, (b + 1) * block)
        for chain in range(b * block, end):
            output[chain] = 0.0
        for i in range(length):
            for chain in range(b * block, end):
                x = lower_states[chain]
                y = upper_states[chain]
                # D(y) - D(x) restricted to the terms of site i: all pairs (i, j) with
                # weight 1/2, or, for symmetric couplings, pairs j > i with weight 1
                # plus the diagonal block with weight 1/2.
                field = (upper_couplings[i, x[i], i, x[i]] - lower_couplings[i, x[i], i, x[i]]
                         - upper_couplings[i, y[i], i, y[i]] + lower_couplings[i, y[i], i, y[i]])
                pairs = 0.0
                for j in range(i + 1 if symmetric else 0, length):
                    if j != i:
                        pairs += (upper_couplings[i, x[i], j, x[j]] - lower_couplings[i, x[i], j, x[j]]
                                  - upper_couplings[i, y[i], j, y[j]] + lower_couplings[i, y[i], j, y[j]])
                weight = 1.0 if symmetric else 0.5
                output[chain] += (upper_bias[i, x[i]] - lower_bias[i, x[i]]
                                  - upper_bias[i, y[i]] + lower_bias[i, y[i]] + 0.5 * field + weight * pairs)


@njit(cache=True, fastmath=True, parallel=True)
def _sparse_energies(states, bias, nbr_ptr, nbrs, blocks, symmetric, output):
    """Energies ``-sum h - 1/2 sum J`` from the sparse layout of the sampling kernels.

    ``blocks[s, x_j, x_i] = J[i, x_i, j, x_j]`` for neighbour slot ``s`` of site ``i``.
    Symmetric couplings sum each pair once.
    """
    n, length = states.shape
    for chain in prange(n):
        x = states[chain]
        energy = 0.0
        for i in range(length):
            energy -= bias[i, x[i]]
            for s in range(nbr_ptr[i], nbr_ptr[i + 1]):
                j = nbrs[s]
                if not symmetric:
                    energy -= 0.5 * blocks[s, x[j], x[i]]
                elif j > i:
                    energy -= blocks[s, x[j], x[i]]
                elif j == i:
                    energy -= 0.5 * blocks[s, x[j], x[i]]
        output[chain] = energy


def _sparse_layout_of(couplings):
    """The cached sparse layout ``(nbr_ptr, nbrs, blocks)`` of ``couplings``, or ``None`` if they are dense."""
    from adabmDCA.numba_kernels.sampling import sparse_layout

    return sparse_layout(couplings.detach()[None], source=couplings)


def _sparse_exchange(lower, upper, lower_states, upper_states, lower_layout, upper_layout, symmetric):
    states = [lower_states.detach().to(torch.int32).contiguous().numpy(),
              upper_states.detach().to(torch.int32).contiguous().numpy()]
    energies = {}
    for name, model, layout in (("lower", lower, lower_layout), ("upper", upper, upper_layout)):
        bias = model["bias"].detach().contiguous().numpy()
        nbr_ptr, nbrs, blocks = layout
        for which, x in (("x", states[0]), ("y", states[1])):
            output = np.empty(len(x), dtype=np.float64)
            _sparse_energies(x, bias, nbr_ptr[0], nbrs, blocks, symmetric, output)
            energies[name, which] = output
    # log acceptance D(y) - D(x), with D = E_upper - E_lower.
    return ((energies["upper", "y"] - energies["lower", "y"]) - (energies["upper", "x"] - energies["lower", "x"]))


def exchange_log_acceptance(
    lower: dict[str, torch.Tensor],
    upper: dict[str, torch.Tensor],
    lower_states: torch.Tensor,
    upper_states: torch.Tensor,
) -> torch.Tensor:
    """Log acceptance ``min(0, D(y) - D(x))`` of swapping ``lower_states[n]`` and ``upper_states[n]``.

    Args:
        lower, upper: Models with ``bias`` ``(L, q)`` and ``coupling_matrix``
            ``(L, q, L, q)``, float32 or float64 CPU tensors of the same dtype.
        lower_states, upper_states: Integer states ``(N, L)`` of the two models.

    Returns:
        A float64 tensor of shape ``(N,)``.
    """
    dtype = lower["bias"].dtype
    if (lower["bias"].device.type != "cpu" or dtype not in (torch.float32, torch.float64)
            or upper["bias"].dtype != dtype):
        raise ValueError("Exchanges need float32 or float64 CPU models of the same dtype.")
    if lower_states.shape != upper_states.shape:
        raise ValueError("Exchange populations must have matching shapes.")

    def array(tensor, dtype):
        return tensor.detach().to(device="cpu", dtype=dtype).contiguous().numpy()

    num_chains = len(lower_states)
    output = np.zeros(num_chains, dtype=np.float64)
    if num_chains:
        block = max(1, min(32, num_chains // (4 * set_kernel_threads())))
        symmetric = couplings_are_symmetric(lower["coupling_matrix"]) and couplings_are_symmetric(
            upper["coupling_matrix"])
        lower_layout = _sparse_layout_of(lower["coupling_matrix"])
        upper_layout = None if lower_layout is None else _sparse_layout_of(upper["coupling_matrix"])
        if upper_layout is not None:
            # Both graphs are sparse: energies from the coupled pairs only.
            output = _sparse_exchange(lower, upper, lower_states, upper_states, lower_layout, upper_layout,
                                      symmetric)
            return torch.from_numpy(output).clamp_max_(0.0)
        _exchange_kernel(array(lower["bias"], dtype), array(lower["coupling_matrix"], dtype),
                         array(upper["bias"], dtype), array(upper["coupling_matrix"], dtype),
                         array(lower_states, torch.int32), array(upper_states, torch.int32), symmetric, block, output)
    return torch.from_numpy(output).clamp_max_(0.0)
