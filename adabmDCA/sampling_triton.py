"""Optional Triton kernels for categorical GPU sampling."""

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
    def _write_one_hot_kernel(
        states_ptr,
        output_ptr,
        sites_ptr,
        num_chains: tl.constexpr,
        length: tl.constexpr,
        num_states: tl.constexpr,
        INDEPENDENT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if INDEPENDENT:
            chain = offset // num_states
            candidate = offset % num_states
            mask = chain < num_chains
            site = tl.load(sites_ptr + chain, mask=mask, other=0)
            state_offset = chain * length + site
            output_offset = state_offset * num_states + candidate
        else:
            mask = offset < num_chains * length * num_states
            candidate = offset % num_states
            state_offset = offset // num_states
            output_offset = offset
        state = tl.load(states_ptr + state_offset, mask=mask, other=0)
        tl.store(output_ptr + output_offset, state == candidate, mask=mask)

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
        STEPS: tl.constexpr = 1,
        TRANSPOSED: tl.constexpr = False,
    ):
        chain = tl.program_id(0)
        position = tl.arange(0, BLOCK_L)
        candidate = tl.arange(0, BLOCK_Q)
        position_mask = position < length
        candidate_mask = candidate < num_states
        background = tl.load(
            states_ptr + chain * length + position,
            mask=position_mask,
            other=0,
        ).to(tl.int64)
        # Each program owns a complete chain. Sequential updates can stay in
        # registers without synchronizing with any other program.
        for local_step in range(STEPS):
            if INDEPENDENT:
                site = tl.load(sites_ptr + chain)
                uniform_offset = chain
            else:
                site = tl.load(sites_ptr + step + local_step)
                uniform_offset = (step + local_step) * num_chains + chain
            if TRANSPOSED:
                # (target_site, source_site, source_state, target_state):
                # a background residue exposes contiguous candidate fields.
                coupling_offset = (
                    (site * length + position[:, None]) * num_states + background[:, None]
                ) * num_states + candidate[None, :]
                coupling = tl.load(
                    couplings_ptr + coupling_offset,
                    mask=position_mask[:, None] & candidate_mask[None, :],
                    other=0.0,
                )
                if couplings_ptr.dtype.element_ty == tl.bfloat16:
                    coupling = coupling.to(tl.float32)
                interaction = tl.sum(coupling, axis=0)
            else:
                coupling_offset = (
                    (site * num_states + candidate[:, None]) * length + position[None, :]
                ) * num_states + background[None, :]
                coupling = tl.load(
                    couplings_ptr + coupling_offset,
                    mask=candidate_mask[:, None] & position_mask[None, :],
                    other=0.0,
                )
                if couplings_ptr.dtype.element_ty == tl.bfloat16:
                    coupling = coupling.to(tl.float32)
                interaction = tl.sum(coupling, axis=1)
            field = (
                tl.load(
                    bias_ptr + site * num_states + candidate,
                    mask=candidate_mask,
                    other=0.0,
                )
                + interaction
            )
            # Mask after scaling: beta=0 must not turn padded -inf into NaN.
            logits = tl.where(candidate_mask, beta * field, -float("inf"))
            weights = tl.exp(logits - tl.max(logits, axis=0))
            cumulative = tl.cumsum(weights, axis=0)
            threshold = tl.load(uniforms_ptr + uniform_offset) * tl.sum(weights, axis=0)
            new_state = tl.minimum(tl.sum(threshold > cumulative, axis=0), num_states - 1)
            background = tl.where(position == site, new_state.to(tl.int64), background)
            if STEPS == 1:
                tl.store(states_ptr + chain * length + site, new_state)
        if STEPS > 1:
            tl.store(states_ptr + chain * length + position, background, mask=position_mask)

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
        STEPS: tl.constexpr = 1,
    ):
        chain = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        position = tl.arange(0, BLOCK_L)
        chain_mask = chain < num_chains
        position_mask = position < length
        background = tl.load(
            states_ptr + chain[:, None] * length + position[None, :],
            mask=chain_mask[:, None] & position_mask[None, :],
            other=0,
        ).to(tl.int64)
        for local_step in range(STEPS):
            if INDEPENDENT:
                site = tl.load(sites_ptr + chain, mask=chain_mask, other=0)
                coupling_site = site[:, None]
                random_offset = chain
            else:
                site = tl.load(sites_ptr + step + local_step)
                coupling_site = site
                random_offset = (step + local_step) * num_chains + chain

            if STEPS == 1:
                old_state = tl.load(states_ptr + chain * length + site, mask=chain_mask, other=0).to(tl.int64)
            else:
                old_state = tl.sum(tl.where(position[None, :] == coupling_site, background, 0), axis=1)
            proposed = tl.load(
                proposals_ptr + random_offset,
                mask=chain_mask,
                other=0,
            ).to(tl.int64)
            # Flattened layout is (target_site, target_state, source_site,
            # source_state). Only the old and proposed target-state rows are read.
            old_offset = (
                (coupling_site * num_states + old_state[:, None]) * length + position[None, :]
            ) * num_states + background
            proposed_offset = (
                (coupling_site * num_states + proposed[:, None]) * length + position[None, :]
            ) * num_states + background
            mask = chain_mask[:, None] & position_mask[None, :]
            old_coupling = tl.load(couplings_ptr + old_offset, mask=mask, other=0.0)
            proposed_coupling = tl.load(couplings_ptr + proposed_offset, mask=mask, other=0.0)
            # Cast before subtraction, not just before the reduction.
            if couplings_ptr.dtype.element_ty == tl.bfloat16:
                old_coupling = old_coupling.to(tl.float32)
                proposed_coupling = proposed_coupling.to(tl.float32)
            old_bias = tl.load(bias_ptr + site * num_states + old_state, mask=chain_mask, other=0.0)
            proposed_bias = tl.load(bias_ptr + site * num_states + proposed, mask=chain_mask, other=0.0)
            if bias_ptr.dtype.element_ty == tl.bfloat16:
                old_bias = old_bias.to(tl.float32)
                proposed_bias = proposed_bias.to(tl.float32)
            delta_energy = old_bias - proposed_bias + tl.sum(old_coupling - proposed_coupling, axis=1)
            uniform = tl.load(
                uniforms_ptr + random_offset,
                mask=chain_mask,
                other=1.0,
            )
            accepted = uniform < tl.exp(-beta * delta_energy)
            final_state = tl.where(accepted, proposed, old_state)
            background = tl.where(position[None, :] == coupling_site, final_state[:, None], background)
            if STEPS == 1:
                tl.store(states_ptr + chain * length + site, final_state, mask=chain_mask)
        if STEPS > 1:
            tl.store(
                states_ptr + chain[:, None] * length + position[None, :],
                background,
                mask=chain_mask[:, None] & position_mask[None, :],
            )


def is_triton_available() -> bool:
    """Return whether Triton was imported successfully."""
    return triton is not None


def gibbs_sampling_triton(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
    *,
    steps_per_launch: int = 16,
    transpose_couplings: bool = True,
    coupling_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run Gibbs updates, keeping each chain in registers across short chunks.

    ``steps_per_launch`` controls sequential updates per kernel, independently
    of random-number chunking. ``transpose_couplings`` makes candidate fields
    contiguous at the cost of one extra coupling-sized allocation per call.
    Disable it to save memory or benchmark the original gather layout.
    ``coupling_dtype`` optionally converts sampling couplings without changing
    the caller's parameters. BF16 conversion and transposition use one copy.
    """
    kernel_params, num_chains, length, num_states, num_steps = _prepare_inputs(chains, params, nsweeps)
    if num_chains == 0 or num_steps == 0:
        return chains.clone()
    if steps_per_launch < 1:
        raise ValueError("steps_per_launch must be positive")
    states = chains.argmax(dim=-1).to(torch.int32)
    # Prepare once per sampler call, never cache across parameter updates.
    if transpose_couplings:
        kernel_params["coupling_gibbs"] = (
            kernel_params["coupling_matrix"]
            .permute(0, 2, 3, 1)
            .to(
            dtype=coupling_dtype or kernel_params["coupling_matrix"].dtype,
            memory_format=torch.contiguous_format,
            copy=True,
            )
        )
    elif coupling_dtype is not None:
        kernel_params["coupling_matrix"] = kernel_params["coupling_matrix"].to(coupling_dtype)
    steps_per_chunk = max(1, min(num_steps, 16_000_000 // num_chains))
    for chunk_start in range(0, num_steps, steps_per_chunk):
        chunk_steps = min(steps_per_chunk, num_steps - chunk_start)
        sites = torch.randint(0, length, (chunk_steps,), device=chains.device, dtype=torch.int32)
        uniforms = torch.rand((chunk_steps, num_chains), device=chains.device, dtype=_uniform_dtype(chains))
        _gibbs_steps_triton(
            states,
            kernel_params,
            sites,
            uniforms,
            beta,
            steps_per_launch=steps_per_launch,
            transpose_couplings=transpose_couplings,
        )
    return _write_one_hot(states, torch.empty_like(chains), num_states)


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
    uniforms = torch.rand(num_chains, device=chains.device, dtype=_uniform_dtype(chains))
    _gibbs_step_independent_triton(states, kernel_params, sites, uniforms, beta)
    return _write_one_hot(states, chains, num_states, sites)


def metropolis_sampling_triton(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
    *,
    steps_per_launch: int = 16,
    coupling_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run Metropolis updates with ``steps_per_launch`` updates per kernel."""
    kernel_params, num_chains, length, num_states, num_steps = _prepare_inputs(chains, params, nsweeps)
    if num_chains == 0 or num_steps == 0:
        return chains.clone()
    if steps_per_launch < 1:
        raise ValueError("steps_per_launch must be positive")
    states = chains.argmax(dim=-1).to(torch.int32)
    if coupling_dtype is not None:
        kernel_params["coupling_matrix"] = kernel_params["coupling_matrix"].to(coupling_dtype)
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
        uniforms = torch.rand((chunk_steps, num_chains), device=chains.device, dtype=_uniform_dtype(chains))
        _metropolis_steps_triton(
            states,
            kernel_params,
            sites,
            proposals,
            uniforms,
            beta,
            steps_per_launch=steps_per_launch,
        )
    return _write_one_hot(states, torch.empty_like(chains), num_states)


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
    uniforms = torch.rand(num_chains, device=chains.device, dtype=_uniform_dtype(chains))
    _metropolis_step_independent_triton(states, kernel_params, sites, proposals, uniforms, beta)
    return _write_one_hot(states, chains, num_states, sites)


def _uniform_dtype(chains: torch.Tensor) -> torch.dtype:
    return torch.float32 if chains.dtype == torch.bfloat16 else chains.dtype


def _write_one_hot(
    states: torch.Tensor,
    output: torch.Tensor,
    num_states: int,
    sites: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write directly in the output dtype, optionally touching only selected sites."""
    if not output.is_contiguous():
        # Public independent-site updates also accept strided one-hot inputs.
        updated = torch.nn.functional.one_hot(states.long(), num_classes=num_states).to(output.dtype)
        output.copy_(updated)
        return output
    num_chains, length = states.shape
    size = num_chains * num_states if sites is not None else output.numel()
    if size:
        _write_one_hot_kernel[(triton.cdiv(size, 256),)](
            states,
            output,
            sites,
            num_chains,
            length,
            num_states,
            INDEPENDENT=sites is not None,
            BLOCK=256,
        )
    return output


def _prepare_inputs(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
) -> tuple[Dict[str, torch.Tensor], int, int, int, int]:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not chains.is_cuda:
        raise ValueError("The Triton sampler requires CUDA tensors")
    if chains.dtype not in (torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("The Triton sampler supports bfloat16, float32 and float64 chains")
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
    *,
    steps_per_launch: int = 16,
    transpose_couplings: bool = True,
) -> None:
    """Apply controlled Gibbs updates; used for validation and chunking."""
    num_chains, length = states.shape
    num_states = params["bias"].shape[1]
    block_l = triton.next_power_of_2(length)
    block_q = triton.next_power_of_2(num_states)
    if steps_per_launch < 1:
        raise ValueError("steps_per_launch must be positive")
    if num_chains == 0 or sites.numel() == 0:
        return
    couplings = params["coupling_matrix"]
    if transpose_couplings:
        couplings = params.get("coupling_gibbs")
        if couplings is None:
            couplings = params["coupling_matrix"].permute(0, 2, 3, 1).contiguous()
    for step in range(0, sites.shape[0], steps_per_launch):
        _gibbs_step_kernel[(num_chains,)](
            states,
            params["bias"],
            couplings,
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
            STEPS=min(steps_per_launch, sites.shape[0] - step),
            TRANSPOSED=transpose_couplings,
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
    *,
    steps_per_launch: int = 16,
) -> None:
    """Apply controlled Metropolis updates; used for validation and chunking."""
    num_chains, length = states.shape
    num_states = params["bias"].shape[1]
    block_n = 16
    block_l = triton.next_power_of_2(length)
    grid = (triton.cdiv(num_chains, block_n),)
    if steps_per_launch < 1:
        raise ValueError("steps_per_launch must be positive")
    if num_chains == 0 or sites.numel() == 0:
        return
    for step in range(0, sites.shape[0], steps_per_launch):
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
            STEPS=min(steps_per_launch, sites.shape[0] - step),
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
