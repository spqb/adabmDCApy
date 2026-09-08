"""Frozen benchmark reference from cb8dc8b0dbea84ac034d9aef97622f53bb8fa3e1.

Keep this pre-optimization implementation unchanged so controlled-random tests
and end-to-end timings compare against the actual original Triton samplers.
It intentionally retains the original beta=0 padding bug.
"""

from __future__ import annotations

from typing import Dict

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the PyTorch installation
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _gibbs_step_kernel(
        states_ptr,
        bias_ptr,
        couplings_ptr,
        sites_ptr,
        uniforms_ptr,
        step,
        num_chains: tl.constexpr,
        length: tl.constexpr,
        num_states: tl.constexpr,
        beta,
        BLOCK_L: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        INDEPENDENT: tl.constexpr,
    ):
        chain = tl.program_id(0)
        position = tl.arange(0, BLOCK_L)
        candidate = tl.arange(0, BLOCK_Q)
        position_mask = position < length
        candidate_mask = candidate < num_states
        if INDEPENDENT:
            site = tl.load(sites_ptr + chain)
            uniform_offset = chain
        else:
            site = tl.load(sites_ptr + step)
            uniform_offset = step * num_chains + chain
        background = tl.load(
            states_ptr + chain * length + position,
            mask=position_mask,
            other=0,
        ).to(tl.int64)
        coupling_offset = (
            ((site * num_states + candidate[:, None]) * length + position[None, :]) * num_states
            + background[None, :]
        )
        coupling = tl.load(
            couplings_ptr + coupling_offset,
            mask=candidate_mask[:, None] & position_mask[None, :],
            other=0.0,
        )
        field = tl.load(
            bias_ptr + site * num_states + candidate,
            mask=candidate_mask,
            other=-float("inf"),
        ) + tl.sum(coupling, axis=1)
        logits = beta * field
        weights = tl.exp(logits - tl.max(logits, axis=0))
        weights = tl.where(candidate_mask, weights, 0.0)
        cumulative = tl.cumsum(weights, axis=0)
        threshold = tl.load(uniforms_ptr + uniform_offset) * tl.sum(weights, axis=0)
        new_state = tl.sum(threshold > cumulative, axis=0)
        tl.store(states_ptr + chain * length + site, new_state)

    @triton.jit
    def _metropolis_step_kernel(
        states_ptr,
        bias_ptr,
        couplings_ptr,
        sites_ptr,
        proposals_ptr,
        uniforms_ptr,
        step,
        num_chains: tl.constexpr,
        length: tl.constexpr,
        num_states: tl.constexpr,
        beta,
        BLOCK_N: tl.constexpr,
        BLOCK_L: tl.constexpr,
        INDEPENDENT: tl.constexpr,
    ):
        chain = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        position = tl.arange(0, BLOCK_L)
        chain_mask = chain < num_chains
        position_mask = position < length
        if INDEPENDENT:
            site = tl.load(sites_ptr + chain, mask=chain_mask, other=0)
            coupling_site = site[:, None]
            random_offset = chain
        else:
            site = tl.load(sites_ptr + step)
            coupling_site = site
            random_offset = step * num_chains + chain

        old_state = tl.load(states_ptr + chain * length + site, mask=chain_mask, other=0).to(tl.int64)
        proposed = tl.load(
            proposals_ptr + random_offset,
            mask=chain_mask,
            other=0,
        ).to(tl.int64)
        background = tl.load(
            states_ptr + chain[:, None] * length + position[None, :],
            mask=chain_mask[:, None] & position_mask[None, :],
            other=0,
        ).to(tl.int64)

        # Flattened layout is (target_site, target_state, source_site,
        # source_state). Only the old and proposed target-state rows are read.
        old_offset = (
            ((coupling_site * num_states + old_state[:, None]) * length + position[None, :]) * num_states
            + background
        )
        proposed_offset = (
            ((coupling_site * num_states + proposed[:, None]) * length + position[None, :]) * num_states
            + background
        )
        mask = chain_mask[:, None] & position_mask[None, :]
        old_coupling = tl.load(couplings_ptr + old_offset, mask=mask, other=0.0)
        proposed_coupling = tl.load(couplings_ptr + proposed_offset, mask=mask, other=0.0)
        delta_energy = (
            tl.load(bias_ptr + site * num_states + old_state, mask=chain_mask, other=0.0)
            - tl.load(bias_ptr + site * num_states + proposed, mask=chain_mask, other=0.0)
            + tl.sum(old_coupling - proposed_coupling, axis=1)
        )
        uniform = tl.load(
            uniforms_ptr + random_offset,
            mask=chain_mask,
            other=1.0,
        )
        accepted = uniform < tl.exp(-beta * delta_energy)
        final_state = tl.where(accepted, proposed, old_state)
        tl.store(states_ptr + chain * length + site, final_state, mask=chain_mask)


def is_triton_available() -> bool:
    """Return whether Triton was imported successfully."""
    return triton is not None


def gibbs_sampling_triton(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Run Gibbs sampling with fused categorical field evaluation."""
    kernel_params, num_chains, length, num_states, num_steps = _prepare_inputs(chains, params, nsweeps)
    if num_chains == 0 or num_steps == 0:
        return chains.clone()
    states = chains.argmax(dim=-1).to(torch.int32)
    steps_per_chunk = max(1, min(num_steps, 16_000_000 // num_chains))
    for chunk_start in range(0, num_steps, steps_per_chunk):
        chunk_steps = min(steps_per_chunk, num_steps - chunk_start)
        sites = torch.randint(0, length, (chunk_steps,), device=chains.device, dtype=torch.int32)
        uniforms = torch.rand((chunk_steps, num_chains), device=chains.device, dtype=chains.dtype)
        _gibbs_steps_triton(states, kernel_params, sites, uniforms, beta)
    return torch.nn.functional.one_hot(states.to(torch.int64), num_classes=num_states).to(chains.dtype)


def gibbs_step_independent_sites_triton(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    """Apply one fused Gibbs update at an independently drawn site per chain."""
    kernel_params, num_chains, length, num_states, _ = _prepare_inputs(chains, params, 1)
    if num_chains == 0:
        return chains
    states = chains.argmax(dim=-1).to(torch.int32)
    sites = torch.randint(0, length, (num_chains,), device=chains.device, dtype=torch.int32)
    uniforms = torch.rand(num_chains, device=chains.device, dtype=chains.dtype)
    _gibbs_step_independent_triton(states, kernel_params, sites, uniforms, beta)
    updated = torch.nn.functional.one_hot(states.to(torch.int64), num_classes=num_states).to(chains.dtype)
    chains.copy_(updated)
    return chains


def metropolis_sampling_triton(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Run Metropolis sampling with a fused categorical Triton update."""
    kernel_params, num_chains, length, num_states, num_steps = _prepare_inputs(chains, params, nsweeps)
    if num_chains == 0 or num_steps == 0:
        return chains.clone()
    states = chains.argmax(dim=-1).to(torch.int32)
    # Bound temporary random-number storage for long sampling runs.
    steps_per_chunk = max(1, min(num_steps, 16_000_000 // num_chains))
    for chunk_start in range(0, num_steps, steps_per_chunk):
        chunk_steps = min(steps_per_chunk, num_steps - chunk_start)
        sites = torch.randint(0, length, (chunk_steps,), device=chains.device, dtype=torch.int32)
        proposals = torch.randint(
            0,
            num_states,
            (chunk_steps, num_chains),
            device=chains.device,
            dtype=torch.int32,
        )
        uniforms = torch.rand((chunk_steps, num_chains), device=chains.device, dtype=chains.dtype)
        _metropolis_steps_triton(states, kernel_params, sites, proposals, uniforms, beta)
    return torch.nn.functional.one_hot(states.to(torch.int64), num_classes=num_states).to(chains.dtype)


def metropolis_step_independent_sites_triton(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    """Apply one fused Metropolis update at an independently drawn site per chain."""
    kernel_params, num_chains, length, num_states, _ = _prepare_inputs(chains, params, 1)
    if num_chains == 0:
        return chains
    states = chains.argmax(dim=-1).to(torch.int32)
    sites = torch.randint(0, length, (num_chains,), device=chains.device, dtype=torch.int32)
    proposals = torch.randint(0, num_states, (num_chains,), device=chains.device, dtype=torch.int32)
    uniforms = torch.rand(num_chains, device=chains.device, dtype=chains.dtype)
    _metropolis_step_independent_triton(states, kernel_params, sites, proposals, uniforms, beta)
    updated = torch.nn.functional.one_hot(states.to(torch.int64), num_classes=num_states).to(chains.dtype)
    chains.copy_(updated)
    return chains


def _prepare_inputs(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
) -> tuple[Dict[str, torch.Tensor], int, int, int, int]:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not chains.is_cuda:
        raise ValueError("The Triton sampler requires CUDA tensors")
    if chains.dtype not in (torch.float32, torch.float64):
        raise ValueError("The Triton sampler supports float32 and float64 chains")
    num_chains, length, num_states = chains.shape
    bias = params["bias"]
    couplings = params["coupling_matrix"]
    if bias.device != chains.device or couplings.device != chains.device:
        raise ValueError("Chains and parameters must be on the same CUDA device")
    kernel_params = {
        "bias": bias.contiguous(),
        "coupling_matrix": couplings.contiguous(),
    }
    return kernel_params, num_chains, length, num_states, nsweeps * length


def _gibbs_steps_triton(
    states: torch.Tensor,
    params: Dict[str, torch.Tensor],
    sites: torch.Tensor,
    uniforms: torch.Tensor,
    beta: float,
) -> None:
    """Apply controlled Gibbs updates; used for validation and chunking."""
    num_chains, length = states.shape
    num_states = params["bias"].shape[1]
    block_l = triton.next_power_of_2(length)
    block_q = triton.next_power_of_2(num_states)
    for step in range(sites.shape[0]):
        _gibbs_step_kernel[(num_chains,)](
            states,
            params["bias"],
            params["coupling_matrix"],
            sites,
            uniforms,
            step,
            num_chains,
            length,
            num_states,
            beta,
            BLOCK_L=block_l,
            BLOCK_Q=block_q,
            INDEPENDENT=False,
            num_warps=4,
        )


def _gibbs_step_independent_triton(
    states: torch.Tensor,
    params: Dict[str, torch.Tensor],
    sites: torch.Tensor,
    uniforms: torch.Tensor,
    beta: float,
) -> None:
    """Apply one controlled independent-site Gibbs update."""
    num_chains, length = states.shape
    num_states = params["bias"].shape[1]
    _gibbs_step_kernel[(num_chains,)](
        states,
        params["bias"],
        params["coupling_matrix"],
        sites,
        uniforms,
        0,
        num_chains,
        length,
        num_states,
        beta,
        BLOCK_L=triton.next_power_of_2(length),
        BLOCK_Q=triton.next_power_of_2(num_states),
        INDEPENDENT=True,
        num_warps=4,
    )


def _metropolis_steps_triton(
    states: torch.Tensor,
    params: Dict[str, torch.Tensor],
    sites: torch.Tensor,
    proposals: torch.Tensor,
    uniforms: torch.Tensor,
    beta: float,
) -> None:
    """Apply controlled Metropolis updates; used for validation and chunking."""
    num_chains, length = states.shape
    num_states = params["bias"].shape[1]
    block_n = 16
    block_l = triton.next_power_of_2(length)
    grid = (triton.cdiv(num_chains, block_n),)
    for step in range(sites.shape[0]):
        _metropolis_step_kernel[grid](
            states,
            params["bias"],
            params["coupling_matrix"],
            sites,
            proposals,
            uniforms,
            step,
            num_chains,
            length,
            num_states,
            beta,
            BLOCK_N=block_n,
            BLOCK_L=block_l,
            INDEPENDENT=False,
            num_warps=4,
        )


def _metropolis_step_independent_triton(
    states: torch.Tensor,
    params: Dict[str, torch.Tensor],
    sites: torch.Tensor,
    proposals: torch.Tensor,
    uniforms: torch.Tensor,
    beta: float,
) -> None:
    """Apply one controlled independent-site Metropolis update."""
    num_chains, length = states.shape
    num_states = params["bias"].shape[1]
    block_n = 16
    _metropolis_step_kernel[(triton.cdiv(num_chains, block_n),)](
        states,
        params["bias"],
        params["coupling_matrix"],
        sites,
        proposals,
        uniforms,
        0,
        num_chains,
        length,
        num_states,
        beta,
        BLOCK_N=block_n,
        BLOCK_L=triton.next_power_of_2(length),
        INDEPENDENT=True,
        num_warps=4,
    )
