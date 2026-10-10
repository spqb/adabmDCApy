"""Fused chain and metadata movement for fixed seven-model PTT sampling.

PyTorch supplies uniforms and permutations in the reference order. Separate
outputs allow each row to read either input population without write races.
Energy evaluation and its accumulation policy are unchanged.
"""

import os

import numpy as np
import torch
from numba import njit, prange

from adabmDCA.numba_kernels import is_numba_available
from adabmDCA.numba_kernels.threads import set_kernel_threads


def swaps_available(sampler, birth, reached, memory):
    """Restrict dispatch to unsteered, full, seven-model generation ladders."""
    if (sampler.device.type != 'cpu' or not is_numba_available()
            or os.environ.get('ADABMDCA_NUMBA_SWAPS', '1') == '0'
            or sampler.mode != 'generate' or len(sampler.models) != 7
            or sampler.active_start != 0 or sampler.reservoir is not None
            or sampler.steering is not None):
        return False
    n, length = sampler.chains[-1].shape
    if (not n or not length or len(sampler.chains) != 7
            or sampler.lineage.device.type != 'cpu' or sampler.lineage.dtype != torch.int64
            or sampler.lineage.shape != (7, n)
            or any(x.device.type != 'cpu' or x.dtype != torch.int32 or x.shape != (n, length)
                   for x in sampler.chains)):
        return False
    return all(value is None or (value.device.type == 'cpu' and value.shape == (7, n) and value.dtype in dtypes)
               for value, dtypes in ((birth, (torch.int64,)), (reached, (torch.bool,)),
                                     (memory, (torch.float32, torch.float64))))


@njit(cache=True, parallel=True)
def _move_pair(x0, x1, x, log_a, log_u, order0, order1, ids0, ids1, ids,
               birth0, birth1, birth, reached0, reached1, reached, memory0, memory1, memory, flags):
    accepted = 0
    for row in prange(x0.shape[0]):
        a, b = order0[row], order1[row]
        swap_a, swap_b = log_u[a] < log_a[a], log_u[b] < log_a[b]
        accepted += int(swap_a)
        for site in range(x0.shape[1]):
            x[0, row, site] = x1[a, site] if swap_a else x0[a, site]
            x[1, row, site] = x0[b, site] if swap_b else x1[b, site]
        ids[0, row] = ids1[a] if swap_a else ids0[a]
        ids[1, row] = ids0[b] if swap_b else ids1[b]
        if flags & 1:
            birth[0, row] = birth1[a] if swap_a else birth0[a]
            birth[1, row] = birth0[b] if swap_b else birth1[b]
        if flags & 2:
            reached[0, row] = reached1[a] if swap_a else reached0[a]
            reached[1, row] = reached0[b] if swap_b else reached1[b]
        if flags & 4:
            memory[0, row] = memory1[a] if swap_a else memory0[a]
            memory[1, row] = memory0[b] if swap_b else memory1[b]
    return accepted


@torch.no_grad()
def swap_and_permute(chains, lineage, log_a, log_u, orders, counts=None, index=0, *, birth=None, reached=None, memory=None):
    """Return permuted rows and an integer acceptance count; inputs stay intact.

    ``counts`` and ``index`` are unused, for the shared PTT movement interface.
    CPU acceptance means are accumulated directly in the sampler's float64
    statistics, without changing their summation order.
    """
    n, length = chains[0].shape
    output = {'chains': torch.empty((2, n, length), dtype=torch.int32),
              'lineage': torch.empty((2, n), dtype=torch.int64)}
    buffers, flags = [], 0
    for bit, key, rows, dtype in ((1, 'birth', birth, torch.int64), (2, 'reached', reached, torch.bool),
                                 (4, 'memory', memory, torch.float64)):
        if rows is None:
            output[key] = None
            dummy = np.empty(0, dtype={torch.int64: np.int64, torch.bool: np.bool_, torch.float64: np.float64}[dtype])
            buffers.extend((dummy, dummy, dummy.reshape(2, 0)))
        else:
            flags |= bit
            output[key] = torch.empty((2, n), dtype=rows[0].dtype)
            buffers.extend((rows[0].detach().numpy(), rows[1].detach().numpy(), output[key].numpy()))
    set_kernel_threads()
    output['accepted'] = _move_pair(
        chains[0].detach().numpy(), chains[1].detach().numpy(), output['chains'].numpy(),
        log_a.detach().numpy(), log_u.detach().numpy(), orders[0].numpy(), orders[1].numpy(),
        lineage[0].numpy(), lineage[1].numpy(), output['lineage'].numpy(), *buffers, flags,
    )
    return output
