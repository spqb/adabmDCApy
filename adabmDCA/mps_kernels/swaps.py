"""Apply PTT acceptance masks and permutations together with chain metadata.

Adjacent pairs remain sequential. Random uniforms and both permutations come
from PyTorch in the reference order; energy probabilities use exchange.py.
Separate output buffers prevent races between differently permuted rows.
"""

import os

import torch

from adabmDCA.mps_kernels.runtime import compile_shader, is_mps_available

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;
kernel void move_pair(
    const device int* x0, const device int* x1, device int* x,
    const device float* log_a, const device float* log_u,
    const device long* order0, const device long* order1,
    const device long* ids0, const device long* ids1, device long* ids,
    const device long* birth0, const device long* birth1, device long* birth,
    const device bool* reached0, const device bool* reached1, device bool* reached,
    const device float* memory0, const device float* memory1, device float* memory,
    device atomic_uint* counts,
    constant uint& n, constant uint& l, constant uint& flags, constant uint& index,
    uint tid [[thread_position_in_grid]]) {
    if (tid >= n * l) return;
    const uint row = tid / l, site = tid % l;
    const uint a = uint(order0[row]), b = uint(order1[row]);
    const bool swap_a = log_u[a] < log_a[a], swap_b = log_u[b] < log_a[b];
    x[tid] = swap_a ? x1[a * l + site] : x0[a * l + site];
    x[n * l + tid] = swap_b ? x0[b * l + site] : x1[b * l + site];
    if (site == 0) {
        ids[row] = swap_a ? ids1[a] : ids0[a];
        ids[n + row] = swap_b ? ids0[b] : ids1[b];
        if (flags & 1) {
            birth[row] = swap_a ? birth1[a] : birth0[a];
            birth[n + row] = swap_b ? birth0[b] : birth1[b];
        }
        if (flags & 2) {
            reached[row] = swap_a ? reached1[a] : reached0[a];
            reached[n + row] = swap_b ? reached0[b] : reached1[b];
        }
        if (flags & 4) {
            memory[row] = swap_a ? memory1[a] : memory0[a];
            memory[n + row] = swap_b ? memory0[b] : memory1[b];
        }
        // order0 is a permutation: count each original chain exactly once.
        if (swap_a) atomic_fetch_add_explicit(counts + index, 1u, memory_order_relaxed);
    }
}
"""


def swaps_available(sampler, birth, reached, memory):
    """Select only the native categorical, unsteered MPS path."""
    if (sampler.device.type != 'mps' or not is_mps_available()
            or os.environ.get('ADABMDCA_MPS_SWAPS', '1') == '0'
            or sampler.steering is not None):
        return False
    chains = sampler.chains
    n, length = chains[-1].shape
    if (not n or n * length >= 2**31 or sampler.lineage.dtype != torch.int64
            or sampler.lineage.device != sampler.device or sampler.lineage.shape != (len(chains), n)
            or any(x.shape != (n, length) or x.dtype != torch.int32 or x.device != sampler.device for x in chains)
            or any(p['bias'].dtype != torch.float32 or p['coupling_matrix'].dtype != torch.float32 for p in sampler.models)):
        return False
    return all(value is None or (value.device == sampler.device and value.shape == sampler.lineage.shape and value.dtype == dtype)
               for value, dtype in ((birth, torch.int64), (reached, torch.bool), (memory, torch.float32)))


@torch.no_grad()
def swap_and_permute(chains, lineage, log_a, log_u, orders, counts, index, *, birth=None, reached=None, memory=None):
    """Return two output rows for chains and each enabled metadata tensor.

    Input rows are not modified. ``counts[index]`` accumulates accepted chains
    on the device, for a single transfer at the end of ``advance``.
    """
    n, length = chains[0].shape
    output = {'chains': torch.empty((2, n, length), device=chains[0].device, dtype=torch.int32),
              'lineage': torch.empty((2, n), device=chains[0].device, dtype=torch.int64)}
    buffers = []
    flags = 0
    for bit, key, rows in ((1, 'birth', birth), (2, 'reached', reached), (4, 'memory', memory)):
        if rows is None:
            output[key] = None
            buffers.extend((log_a, log_a, log_a))  # Unused, type-independent dummy buffers.
        else:
            flags |= bit
            output[key] = torch.empty((2, n), device=rows[0].device, dtype=rows[0].dtype)
            buffers.extend((rows[0].contiguous(), rows[1].contiguous(), output[key]))
    compile_shader(_SOURCE).move_pair(
        chains[0].contiguous(), chains[1].contiguous(), output['chains'], log_a.contiguous(), log_u.contiguous(),
        orders[0], orders[1], lineage[0].contiguous(), lineage[1].contiguous(), output['lineage'],
        *buffers, counts, n, length, flags, index, threads=n * length, group_size=256,
    )
    return output


def acceptance_rates(counts, completed, n):
    """Round means in MPS float32, accumulated in CPU float64 like the reference."""
    return counts[:completed].float().div(n).cpu().double().sum(0).div(completed).tolist()
