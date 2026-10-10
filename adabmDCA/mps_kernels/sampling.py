"""Fused Metal random-site samplers for Apple GPUs.

One SIMD group owns a chain, retains its integer states in threadgroup memory,
then performs many sequential updates in one dispatch. Candidate fields are
loaded from a cached (site, source site, source state, candidate) layout for
Gibbs-type updates; Metropolis uses the original layout for its two states.
Small alphabets also parallelize the sum over source sites across SIMD lanes.

Requires ``torch.mps.compile_shader`` (PyTorch 2.7+). Public dispatch falls back
on PyTorch for unsupported dtypes/shapes or when ``ADABMDCA_MPS=0`` in the
default precision. Explicit BF16 mode requires supported kernels. Random sites
and the key for the on-GPU Philox generator come from PyTorch's MPS RNG, so
``torch.manual_seed`` and
``torch.mps.get_rng_state/set_rng_state`` control reproducibility. Streams are
not expected to match CPU or CUDA. Inputs are never modified.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from adabmDCA._tensor_cache import cached
from adabmDCA.mps_kernels.random import SOURCE as RANDOM_SOURCE
from adabmDCA.mps_kernels.random import draw_seed
from adabmDCA.mps_kernels.runtime import compile_shader
from adabmDCA.mps_kernels.sparse import sparse_coupling_layout

METHODS = ("gibbs", "metropolis", "metropolized_gibbs")
# Bound random-site buffers and the maximum GPU dispatch duration.
_STEPS_PER_LAUNCH = 512


def _steps_per_launch(method, length, q, n, sparse):
    """Conservative M4 tuning; small populations amortize longer dispatches."""
    if n < 1024 or not 64 <= length <= 1024:
        return _STEPS_PER_LAUNCH
    if sparse is None:
        if q > 16 or method == "metropolis":
            return 64
        if 4 <= q <= 8 and method == "metropolized_gibbs":
            return 128
    elif method == "metropolis" and q > 16 and sparse[2] >= q:
        return 128
    return _STEPS_PER_LAUNCH

_SOURCE = r"""
kernel void sample(
    device int* states,
    const device float* h,
    const device COUPLING_TYPE* J,
    const device int* sites,
    const device float* uniforms,
    const device int* seed,
    const device int* neighbours,
    constant uint& width,
    constant uint& n,
    constant uint& steps,
    constant uint& first_step,
    constant float& beta,
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid % 32, local_chain = tid / 32;
    const uint chain = group.x * CHAINS_PER_GROUP + local_chain, replica = group.y;
    if (chain >= n) return;
    const uint row = (replica * n + chain) * L;
    const uint model = replica * L * Q;
    const uint coupling_model = model * L * Q;
    threadgroup int storage[CHAINS_PER_GROUP][L];
    threadgroup int* x = storage[local_chain];
    for (uint j = lane; j < L; j += 32) x[j] = states[row + j];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < steps; ++t) {
        const uint site = sites[replica * steps + t];
        const int old = x[site];
        const uint random_index = ((replica * steps + t) * n + chain) * 2;
#if EXPLICIT_RANDOM
        const float u = uniforms[random_index], v = uniforms[random_index + 1];
#else
        uint4 random = 0;
        if (lane == 0) random = philox(uint4(chain, first_step + t, replica, 0), uint2(seed[0], seed[1]));
        const float u = float(simd_broadcast(random.x, 0) >> 8) * 0x1.0p-24f;
        const float v = float(simd_broadcast(random.y, 0) >> 8) * 0x1.0p-24f;
#endif
#if METHOD == 1
        const uint proposed = min(uint(u * Q), uint(Q - 1));
        float difference = lane == 0 ? h[model + site * Q + proposed] - h[model + site * Q + old] : 0;
#if SPARSE
        for (uint k = lane; k < width; k += 32) {
            const uint slot = (replica * L + site) * width + k;
            const int j = neighbours[slot];
            if (j >= 0) {
                const uint offset = (slot * Q + x[j]) * Q;
                difference += float(J[offset + proposed]) - float(J[offset + old]);
            }
        }
#else
        for (uint j = lane; j < L; j += 32) {
            const uint offset = coupling_model + site * Q * L * Q + j * Q + x[j];
            difference += float(J[offset + proposed * L * Q]) - float(J[offset + old * L * Q]);
        }
#endif
        difference = beta * simd_sum(difference);
        const bool accepted = v < exp(min(difference, 0.0f));
#else
        const uint candidate = lane % PAD, part = lane / PAD;
        float field = 0;
        if (candidate < Q) {
            field = part == 0 ? h[model + site * Q + candidate] : 0;
#if SPARSE
            for (uint k = part; k < width; k += 32 / PAD) {
                const uint slot = (replica * L + site) * width + k;
                const int j = neighbours[slot];
                if (j >= 0) field += float(J[(slot * Q + x[j]) * Q + candidate]);
            }
#else
            for (uint j = part; j < L; j += 32 / PAD) {
                const uint offset = coupling_model + ((site * L + j) * Q + x[j]) * Q;
                field += float(J[offset + candidate]);
            }
#endif
        }
        for (uint offset = 16; offset >= PAD; offset /= 2)
            field += simd_shuffle_down(field, offset);
        const float logit = lane < Q ? beta * field : -INFINITY;
        const float weight = exp(logit - simd_max(logit));
#if METHOD == 2
        const float proposal_weight = lane == uint(old) ? 0 : weight;
#else
        const float proposal_weight = weight;
#endif
        const float total = simd_sum(proposal_weight);
        const float cdf = simd_prefix_inclusive_sum(proposal_weight);
        const uint proposed = simd_min(cdf > u * total ? lane : uint(Q - 1));
#if METHOD == 2
        // Direct sum: total_weight - weight[proposed] loses small probabilities
        // when a single state dominates. An underflowed proposal stays put.
        const float excluded = simd_sum(lane == proposed ? 0 : weight);
        const bool accepted = total > 0 && v * excluded < total;
#else
        const bool accepted = true;
#endif
#endif
        if (lane == 0) x[site] = accepted ? proposed : old;
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = lane; j < L; j += 32) states[row + j] = x[j];
}
"""


def _supported(states, bias, couplings):
    return (states.device.type == "mps" and bias.device == states.device == couplings.device
            and bias.dtype == torch.float32 and couplings.dtype in (torch.float32, torch.bfloat16)
            and states.numel() < 2**31 and couplings.numel() < 2**31
            and 1 <= bias.shape[-1] <= 32 and 1 <= bias.shape[-2] <= 4096)


def _library(method: str, length: int, q: int, explicit_random: bool = False, sparse: bool = False,
             coupling_dtype: torch.dtype = torch.float32):
    coupling_type = "bfloat" if coupling_dtype == torch.bfloat16 else "float"
    pad = max(2, 1 << (q - 1).bit_length())
    defines = (f"#define COUPLING_TYPE {coupling_type}\n#define L {length}\n#define Q {q}\n#define PAD {pad}\n#define METHOD {METHODS.index(method)}\n"
               f"#define SPARSE {int(sparse)}\n#define EXPLICIT_RANDOM {int(explicit_random)}\n#define CHAINS_PER_GROUP {4 if length <= 1024 else 1}\n")
    return compile_shader(defines + RANDOM_SOURCE + _SOURCE)


def _layout(couplings, method="metropolized_gibbs", coupling_dtype=None):
    dtype = coupling_dtype or couplings.dtype
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Metal coupling storage must be float32 or bfloat16.")
    def build():
        source = couplings.detach()
        layout = source if method == "metropolis" else source.movedim(-3, -1)
        return (layout.contiguous() if dtype == source.dtype else layout.to(
            dtype=dtype, memory_format=torch.contiguous_format, copy=True))

    # Inference tensors have no version counter; don't cache their derived layout.
    if torch.is_inference(couplings):
        return build()
    return cached(couplings, f"mps_sampling_{method}_{dtype}", build)


def _run(method, states, biases, couplings, steps, beta, sites=None, uniforms=None, sparse=None):
    """Update contiguous int32 (R,N,L) states; explicit randomness is for kernel validation."""
    replicas, n, length = states.shape
    if not n or not replicas or not steps:
        return states
    neighbours, width = states, 0
    if sparse is not None:
        neighbours, couplings, width = sparse
    library = _library(method, length, biases.shape[-1], uniforms is not None, sparse is not None,
                       coupling_dtype=couplings.dtype)
    seed = draw_seed(states.device)
    per_launch = _steps_per_launch(method, length, biases.shape[-1], n, sparse)
    chains_per_group = 4 if length <= 1024 else 1
    group_threads = 32 * chains_per_group
    # Draw sites in fixed-size banks independently of the launch policy. This
    # keeps RNG consumption unchanged when tuning or switching graph layouts.
    for bank_start in range(0, steps, _STEPS_PER_LAUNCH):
        bank_count = min(_STEPS_PER_LAUNCH, steps - bank_start)
        bank = (torch.randint(length, (replicas, bank_count), device=states.device, dtype=torch.int32)
                if sites is None else sites[:, bank_start:bank_start + bank_count])
        for offset in range(0, bank_count, per_launch):
            start = bank_start + offset
            count = min(per_launch, bank_count - offset)
            selected = bank[:, offset:offset + count].contiguous()
            random = (biases  # Unused buffer in the counter-based variant.
                      if uniforms is None else uniforms[:, start:start + count].contiguous())
            library.sample(states, biases, couplings, selected, random, seed, neighbours, width, n, count, start, float(beta),
                           threads=(((n + chains_per_group - 1) // chains_per_group) * group_threads, replicas),
                           group_size=(group_threads, 1))
    return states


@torch.no_grad()
def sample_categorical(
    method: str,
    states: torch.Tensor,
    params: dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
    *,
    sites: torch.Tensor | None = None,
    coupling_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Sample int32 states (N,L) on MPS. ``sites`` can specify the update sequence.

    Supports float32 fields and float32/BF16 coupling storage, alphabets up to
    32 states and lengths up to 4096. Couplings have shape (L,q,L,q), with zero within-site blocks, as in
    the other Potts samplers. States must be valid indices in [0,q). Chains and
    couplings must have fewer than 2**31 elements. Model parameters stay on the device.
    ``coupling_dtype`` optionally casts the cached layout without modifying masters.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown sampling method: {method}")
    h, j = params["bias"], params["coupling_matrix"]
    if states.ndim != 2 or h.ndim != 2 or states.shape[1] != h.shape[0]:
        raise ValueError("States must have shape (N,L) and biases (L,q).")
    length, q = h.shape
    if j.shape != (length, q, length, q) or not _supported(states, h, j):
        raise ValueError("Metal sampling requires MPS float32 biases and float32/BF16 couplings, 1 <= q <= 32 and 1 <= L <= 4096.")
    if coupling_dtype not in (None, torch.float32, torch.bfloat16):
        raise ValueError("Metal coupling storage must be float32 or bfloat16.")
    if states.dtype not in (torch.int32, torch.int64):
        raise ValueError("Categorical states must be int32 or int64.")
    if nsweeps < 0:
        raise ValueError("nsweeps must be nonnegative.")
    result = states.to(torch.int32).contiguous().clone()
    steps = nsweeps * length
    if sites is not None:
        if sites.ndim != 1 or sites.dtype not in (torch.int32, torch.int64):
            raise ValueError("sites must be a one-dimensional integer tensor.")
        if sites.numel() and (int(sites.min()) < 0 or int(sites.max()) >= length):
            raise ValueError("sites must be within the sequence length.")
        sites = sites.to(device=states.device, dtype=torch.int32)[None].contiguous()
        steps = sites.numel()
    if steps and len(result):
        sparse = sparse_coupling_layout(j[None], source=j, coupling_dtype=coupling_dtype)
        layout = _layout(j, method, coupling_dtype)[None] if sparse is None else sparse[1]
        _run(method, result[None], h.contiguous()[None], layout, steps, beta, sites=sites, sparse=sparse)
    return result


def categorical_sampler(method: str) -> Callable:
    """Return an integer-state sampler with the usual (states, params, nsweeps, beta) signature."""
    def sampler(states, params, nsweeps, beta=1.0, *, sites=None, coupling_dtype=None):
        return sample_categorical(method, states, params, nsweeps, beta, sites=sites, coupling_dtype=coupling_dtype)
    sampler.__name__ = f"{method}_sampling_mps"
    return sampler


def onehot_sampler(method: str) -> Callable:
    """One-hot adapter with the standard sampler signature and PyTorch fallback."""
    from adabmDCA.sampling import get_sampler

    fallback = torch.jit.script(get_sampler(method))

    def sampler(chains, params, nsweeps, beta=1.0, *, coupling_dtype=None):
        if not _supported(chains, params["bias"], params["coupling_matrix"]) or chains.dtype != torch.float32:
            if coupling_dtype is not None or params["coupling_matrix"].dtype == torch.bfloat16:
                raise ValueError("BF16 Metal sampling requires float32 chains/biases and supported dimensions.")
            return fallback(chains, params, nsweeps, beta)
        states = sample_categorical(method, chains.argmax(-1), params, nsweeps, beta, coupling_dtype=coupling_dtype)
        return torch.nn.functional.one_hot(states.long(), chains.shape[-1]).to(chains.dtype)
    sampler.__name__ = f"{method}_sampling_mps"
    return sampler


@torch.no_grad()
def sample_replicas(method, states, biases, couplings, nsweeps, beta=1.0):
    """Sample stacked categorical (R,N,L) populations with one Metal dispatch per chunk."""
    if method not in METHODS:
        raise ValueError(f"Unknown sampling method: {method}")
    if states.ndim != 3 or biases.ndim != 3:
        raise ValueError("Replica states must be (R,N,L) and biases (R,L,q).")
    replicas, _, length = states.shape
    q = biases.shape[-1]
    if (biases.shape != (replicas, length, q)
            or couplings.shape != (replicas, length, q, length, q)
            or not _supported(states, biases, couplings)):
        raise ValueError("Metal replicas require matching float32 MPS models and supported dimensions.")
    if states.dtype not in (torch.int32, torch.int64) or nsweeps < 0:
        raise ValueError("Replica states must be integers and nsweeps nonnegative.")
    result = states.to(torch.int32).contiguous().clone()
    if not nsweeps or not result.numel():
        return result
    sparse = sparse_coupling_layout(couplings)
    layout = _layout(couplings, method) if sparse is None else sparse[1]
    return _run(method, result, biases.contiguous(), layout, nsweeps * length, beta, sparse=sparse)


def replica_sampler(method):
    """Return a stacked-replica sampler with the PTT kernel signature."""
    def sampler(states, biases, couplings, nsweeps, beta=1.0):
        return sample_replicas(method, states, biases, couplings, nsweeps, beta)
    return sampler
