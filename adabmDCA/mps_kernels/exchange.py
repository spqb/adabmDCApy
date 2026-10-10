"""Fused categorical replica exchange on Apple GPUs.

Compute energy differences directly from both models, avoiding one-hot tensors
and cancellation between large absolute energies. One SIMD group owns a pair
of chains and reduces the full (possibly non-gauge-fixed) symmetric Hamiltonian.
"""

import torch

from adabmDCA.mps_kernels.runtime import compile_shader
from adabmDCA.mps_kernels.sparse import sparse_coupling_layout

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;
kernel void exchange(
    const device float* h0, const device float* h1,
    const device float* j0, const device float* j1,
    const device int* x0, const device int* x1,
    device float* out, constant uint& n, constant uint& l, constant uint& q,
    uint tid [[thread_position_in_grid]]) {
    uint chain = tid / 32, lane = tid % 32;
    if (chain >= n) return;
    float difference = 0;
    for (uint i = lane; i < l; i += 32) {
        uint a = x0[chain * l + i], b = x1[chain * l + i];
        uint ha = i * q + a, hb = i * q + b;
        difference += (h1[ha] - h0[ha]) - (h1[hb] - h0[hb]);
        for (uint j = 0; j < l; ++j) {
            uint ia = (ha * l + j) * q + x0[chain * l + j];
            uint ib = (hb * l + j) * q + x1[chain * l + j];
            difference += 0.5f * ((j1[ia] - j0[ia]) - (j1[ib] - j0[ib]));
        }
    }
    difference = simd_sum(difference);
    if (lane == 0) out[chain] = min(difference, 0.0f);
}
"""


@torch.no_grad()
def exchange_log_acceptance(lower, upper, lower_chains, upper_chains):
    """Return min(0, E_delta(upper_chains) - E_delta(lower_chains)), float32."""
    h0, h1 = lower['bias'], upper['bias']
    j0, j1 = lower['coupling_matrix'], upper['coupling_matrix']
    n, length = lower_chains.shape
    q = h0.shape[1]
    if (upper_chains.shape != lower_chains.shape or h0.shape != (length, q)
            or h1.shape != h0.shape or j0.shape != (length, q, length, q) or j1.shape != j0.shape
            or any(t.device.type != 'mps' for t in (h0, h1, j0, j1, lower_chains, upper_chains))
            or any(t.dtype != torch.float32 for t in (h0, h1, j0, j1))
            or max(j0.numel(), lower_chains.numel()) >= 2**31):
        raise ValueError('Metal exchange requires matching categorical chains and MPS float32 models.')
    out = torch.empty(n, device=h0.device, dtype=torch.float32)
    if n:
        layouts = [sparse_coupling_layout(j[None], source=j) for j in (j0, j1)]
        if all(layout is not None for layout in layouts):
            _sparse_exchange(lower, upper, lower_chains, upper_chains, out, layouts)
            return out
        compile_shader(_SOURCE).exchange(
            h0.contiguous(), h1.contiguous(), j0.contiguous(), j1.contiguous(),
            lower_chains.to(torch.int32).contiguous(), upper_chains.to(torch.int32).contiguous(),
            out, n, length, q, threads=n * 32, group_size=128,
        )
    return out


_SPARSE_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;
kernel void sparse_exchange(
    const device float* h0, const device float* h1,
    const device int* neighbours0, const device float* blocks0,
    const device int* neighbours1, const device float* blocks1,
    const device int* x0, const device int* x1,
    device float* out, constant uint& n, constant uint& l, constant uint& q,
    constant uint& width0, constant uint& width1,
    uint tid [[thread_position_in_grid]]) {
    uint chain = tid / 32, lane = tid % 32;
    if (chain >= n) return;
    float difference = 0;
    for (uint i = lane; i < l; i += 32) {
        uint a = x0[chain * l + i], b = x1[chain * l + i];
        uint ha = i * q + a, hb = i * q + b;
        difference += (h1[ha] - h0[ha]) - (h1[hb] - h0[hb]);
        for (uint k = 0; k < width0; ++k) {
            uint slot = i * width0 + k;
            int j = neighbours0[slot];
            if (j >= 0) {
                uint ia = (slot * q + x0[chain * l + j]) * q + a;
                uint ib = (slot * q + x1[chain * l + j]) * q + b;
                difference -= 0.5f * (blocks0[ia] - blocks0[ib]);
            }
        }
        for (uint k = 0; k < width1; ++k) {
            uint slot = i * width1 + k;
            int j = neighbours1[slot];
            if (j >= 0) {
                uint ia = (slot * q + x0[chain * l + j]) * q + a;
                uint ib = (slot * q + x1[chain * l + j]) * q + b;
                difference += 0.5f * (blocks1[ia] - blocks1[ib]);
            }
        }
    }
    difference = simd_sum(difference);
    if (lane == 0) out[chain] = min(difference, 0.0f);
}
"""


def _sparse_exchange(lower, upper, x0, x1, out, layouts):
    neighbours0, blocks0, width0 = layouts[0]
    neighbours1, blocks1, width1 = layouts[1]
    n, length = x0.shape
    q = lower['bias'].shape[1]
    compile_shader(_SPARSE_SOURCE).sparse_exchange(
        lower['bias'].contiguous(), upper['bias'].contiguous(), neighbours0, blocks0,
        neighbours1, blocks1, x0.to(torch.int32).contiguous(), x1.to(torch.int32).contiguous(),
        out, n, length, q, width0, width1, threads=n * 32, group_size=128,
    )
