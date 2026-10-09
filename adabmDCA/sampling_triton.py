"""Optional Triton kernels for categorical GPU sampling."""

from __future__ import annotations

import torch

from adabmDCA._tensor_cache import cached, sparse_kernels_enabled
from adabmDCA.statmech import couplings_are_symmetric

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
        WIDE: tl.constexpr = False,
    ):
        chain = tl.program_id(0)
        position = tl.arange(0, BLOCK_L)
        candidate = tl.arange(0, BLOCK_Q)
        position_mask = position < length
        candidate_mask = candidate < num_states
        # States and offsets stay int32 unless L*q*L*q or N*L reach 2**31
        # (``WIDE``); int64 tiles exhaust registers.
        if WIDE:
            chain = chain.to(tl.int64)
            position_index = position.to(tl.int64)
        else:
            position_index = position
        state_row = chain * length
        background = tl.load(states_ptr + state_row + position, mask=position_mask, other=0)
        if WIDE:
            background = background.to(tl.int64)
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
                    (site * length + position_index[:, None]) * num_states + background[:, None]
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
                    (site * num_states + candidate[:, None]) * length + position_index[None, :]
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
            new_state = tl.minimum(tl.sum((threshold > cumulative).to(tl.int32), axis=0), num_states - 1)
            background = tl.where(position == site, new_state.to(background.dtype), background)
            if STEPS == 1:
                tl.store(states_ptr + state_row + site, new_state)
        if STEPS > 1:
            tl.store(states_ptr + state_row + position, background, mask=position_mask)

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
        WIDE: tl.constexpr = False,
    ):
        chain = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        position = tl.arange(0, BLOCK_L)
        chain_mask = chain < num_chains
        position_mask = position < length
        mask = chain_mask[:, None] & position_mask[None, :]
        # States and offsets stay int32 unless L*q*L*q or N*L reach 2**31
        # (``WIDE``); int64 tiles exhaust registers and spill.
        if WIDE:
            chain = chain.to(tl.int64)
        state_row = chain * length
        background = tl.load(states_ptr + state_row[:, None] + position[None, :], mask=mask, other=0)
        if WIDE:
            background = background.to(tl.int64)
        row_size = length * num_states
        # Flattened layout is (target_site, target_state, source_site,
        # source_state); ``column`` tracks the source part for every position.
        # Only the old and proposed target-state rows are read.
        column = position[None, :] * num_states + background
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
                old_state = tl.load(states_ptr + state_row + site, mask=chain_mask, other=0)
                if WIDE:
                    old_state = old_state.to(tl.int64)
            else:
                old_state = tl.sum(tl.where(position[None, :] == coupling_site, background, 0), axis=1)
            proposed = tl.load(proposals_ptr + random_offset, mask=chain_mask, other=0)
            if WIDE:
                proposed = proposed.to(tl.int64)
            old_row = (coupling_site * num_states + old_state[:, None]) * row_size
            proposed_row = (coupling_site * num_states + proposed[:, None]) * row_size
            old_coupling = tl.load(couplings_ptr + old_row + column, mask=mask, other=0.0)
            proposed_coupling = tl.load(couplings_ptr + proposed_row + column, mask=mask, other=0.0)
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
            if STEPS == 1:
                tl.store(states_ptr + state_row + site, final_state, mask=chain_mask)
            else:
                hit = position[None, :] == coupling_site
                background = tl.where(hit, final_state[:, None], background)
                column = tl.where(hit, coupling_site * num_states + final_state[:, None], column)
        if STEPS > 1:
            tl.store(states_ptr + state_row[:, None] + position[None, :], background, mask=mask)

    @triton.jit
    def _categorical_exchange_kernel(
        lower_bias_ptr,
        lower_couplings_ptr,
        upper_bias_ptr,
        upper_couplings_ptr,
        lower_states_ptr,
        upper_states_ptr,
        output_ptr,
        num_chains: tl.constexpr,
        length: tl.constexpr,
        num_states: tl.constexpr,
        BLOCK_L: tl.constexpr,
        SYMMETRIC: tl.constexpr = False,
    ):
        """Accumulate D(x_lower)-D(x_upper), where D=E_lower-E_upper.

        With ``SYMMETRIC`` couplings each pair is read once: sources above the
        target with weight 1 and the diagonal block with weight 1/2.
        """
        chain = tl.program_id(0)
        target = tl.program_id(1)
        source = tl.arange(0, BLOCK_L)
        source_mask = source < length
        lower_target_state = tl.load(lower_states_ptr + chain * length + target).to(tl.int64)
        upper_target_state = tl.load(upper_states_ptr + chain * length + target).to(tl.int64)
        lower_source_state = tl.load(
            lower_states_ptr + chain * length + source, mask=source_mask, other=0
        ).to(tl.int64)
        upper_source_state = tl.load(
            upper_states_ptr + chain * length + source, mask=source_mask, other=0
        ).to(tl.int64)

        lower_offset = (
            (target * num_states + lower_target_state) * length + source
        ) * num_states + lower_source_state
        upper_offset = (
            (target * num_states + upper_target_state) * length + source
        ) * num_states + upper_source_state
        if SYMMETRIC:
            pair_mask = source_mask & (source >= target)
            weight = tl.where(source == target, 0.5, 1.0)
        else:
            pair_mask = source_mask
            weight = 0.5
        lower_delta = (
            tl.load(upper_couplings_ptr + lower_offset, mask=pair_mask, other=0.0)
            - tl.load(lower_couplings_ptr + lower_offset, mask=pair_mask, other=0.0)
        )
        upper_delta = (
            tl.load(upper_couplings_ptr + upper_offset, mask=pair_mask, other=0.0)
            - tl.load(lower_couplings_ptr + upper_offset, mask=pair_mask, other=0.0)
        )
        lower_bias_delta = (
            tl.load(upper_bias_ptr + target * num_states + lower_target_state)
            - tl.load(lower_bias_ptr + target * num_states + lower_target_state)
        )
        upper_bias_delta = (
            tl.load(upper_bias_ptr + target * num_states + upper_target_state)
            - tl.load(lower_bias_ptr + target * num_states + upper_target_state)
        )
        contribution = (
            lower_bias_delta - upper_bias_delta
            + tl.sum(weight * (lower_delta - upper_delta), axis=0)
        )
        tl.atomic_add(output_ptr + chain, contribution)

    @triton.jit
    def _replica_metropolis_kernel(
        states_ptr,
        biases_ptr,
        couplings_ptr,
        sites_ptr,
        proposals_ptr,
        uniforms_ptr,
        step,
        beta,
        num_chains: tl.constexpr,
        length: tl.constexpr,
        num_states: tl.constexpr,
        num_replicas: tl.constexpr,
        blocks_per_replica: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_L: tl.constexpr,
        STEPS: tl.constexpr,
        WIDE: tl.constexpr = False,
    ):
        program = tl.program_id(0)
        replica = program // blocks_per_replica
        block = program % blocks_per_replica
        chain = block * BLOCK_N + tl.arange(0, BLOCK_N)
        position = tl.arange(0, BLOCK_L)
        chain_mask = chain < num_chains
        position_mask = position < length
        mask = chain_mask[:, None] & position_mask[None, :]
        # States and in-replica offsets stay int32 unless L*q*L*q or R*N*L
        # reach 2**31 (``WIDE``); the replica base pointer is always 64-bit.
        # int64 tiles exhaust registers and spill.
        if WIDE:
            chain = chain.to(tl.int64)
        state_base = (replica * num_chains + chain[:, None]) * length
        background = tl.load(states_ptr + state_base + position[None, :], mask=mask, other=0)
        if WIDE:
            background = background.to(tl.int64)
        replica_biases = biases_ptr + replica * (length * num_states)
        replica_couplings = couplings_ptr + replica.to(tl.int64) * (length * num_states * length * num_states)
        # Flattened layout is (target_site, target_state, source_site,
        # source_state); ``column`` tracks the source part for every position.
        column = position[None, :] * num_states + background
        for local_step in range(STEPS):
            random_step = step + local_step
            site = tl.load(sites_ptr + random_step * num_replicas + replica)
            random_offset = (random_step * num_replicas + replica) * num_chains + chain
            old_state = tl.sum(tl.where(position[None, :] == site, background, 0), axis=1)
            proposed = tl.load(proposals_ptr + random_offset, mask=chain_mask, other=0)
            if WIDE:
                proposed = proposed.to(tl.int64)
            old_row = (site * num_states + old_state[:, None]) * (length * num_states)
            proposed_row = (site * num_states + proposed[:, None]) * (length * num_states)
            old_coupling = tl.load(replica_couplings + old_row + column, mask=mask, other=0.0)
            proposed_coupling = tl.load(replica_couplings + proposed_row + column, mask=mask, other=0.0)
            if couplings_ptr.dtype.element_ty == tl.bfloat16:
                old_coupling = old_coupling.to(tl.float32)
                proposed_coupling = proposed_coupling.to(tl.float32)
            old_bias = tl.load(replica_biases + site * num_states + old_state, mask=chain_mask, other=0.0)
            proposed_bias = tl.load(replica_biases + site * num_states + proposed, mask=chain_mask, other=0.0)
            if biases_ptr.dtype.element_ty == tl.bfloat16:
                old_bias = old_bias.to(tl.float32)
                proposed_bias = proposed_bias.to(tl.float32)
            delta_energy = old_bias - proposed_bias + tl.sum(old_coupling - proposed_coupling, axis=1)
            uniform = tl.load(uniforms_ptr + random_offset, mask=chain_mask, other=1.0)
            final_state = tl.where(uniform < tl.exp(-beta * delta_energy), proposed, old_state)
            hit = position[None, :] == site
            background = tl.where(hit, final_state[:, None], background)
            column = tl.where(hit, site * num_states + final_state[:, None], column)
        tl.store(states_ptr + state_base + position[None, :], background, mask=mask)

    @triton.jit
    def _replica_metropolized_gibbs_kernel(
        states_ptr,
        biases_ptr,
        couplings_ptr,
        sites_ptr,
        uniforms_ptr,
        step,
        beta,
        num_chains: tl.constexpr,
        length: tl.constexpr,
        num_states: tl.constexpr,
        num_replicas: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        STEPS: tl.constexpr,
        WIDE: tl.constexpr = False,
    ):
        """One program per chain; couplings use the (site, source_site, source_state, state) layout.

        Metropolized Gibbs (Liu 1996) proposes b != a from the site conditional
        restricted to the other states and accepts with (1 - p_a) / (1 - p_b).
        Both normalizers are summed directly, never as 1 - p, so conditionals
        close to one do not cancel. If every other state underflows, the
        chain stays put, which is the conditional to working precision.
        """
        program = tl.program_id(0)
        replica = program // num_chains
        chain = program % num_chains
        position = tl.arange(0, BLOCK_L)
        candidate = tl.arange(0, BLOCK_Q)
        position_mask = position < length
        candidate_mask = candidate < num_states
        tile_mask = position_mask[:, None] & candidate_mask[None, :]
        # int32 offsets unless L*L*q*q or R*N*L reach 2**31 (``WIDE``).
        if WIDE:
            chain = chain.to(tl.int64)
            position_index = position.to(tl.int64)
        else:
            position_index = position
        state_base = (replica * num_chains + chain) * length
        background = tl.load(states_ptr + state_base + position, mask=position_mask, other=0)
        if WIDE:
            background = background.to(tl.int64)
        replica_biases = biases_ptr + replica * (length * num_states)
        replica_couplings = couplings_ptr + replica.to(tl.int64) * (length * length * num_states * num_states)
        for local_step in range(STEPS):
            random_step = step + local_step
            site = tl.load(sites_ptr + random_step * num_replicas + replica)
            random_offset = ((random_step * num_replicas + replica) * num_chains + chain) * 2
            # Every source position exposes a contiguous row of candidate couplings.
            offset = (
                (site * length + position_index[:, None]) * num_states + background[:, None]
            ) * num_states + candidate[None, :]
            coupling = tl.load(replica_couplings + offset, mask=tile_mask, other=0.0)
            if couplings_ptr.dtype.element_ty == tl.bfloat16:
                coupling = coupling.to(tl.float32)
            field = tl.load(replica_biases + site * num_states + candidate, mask=candidate_mask, other=0.0)
            if biases_ptr.dtype.element_ty == tl.bfloat16:
                field = field.to(tl.float32)
            field += tl.sum(coupling, axis=0)
            logits = tl.where(candidate_mask, beta * field, -float("inf"))
            weights = tl.exp(logits - tl.max(logits, axis=0))
            old_state = tl.sum(tl.where(position == site, background, 0), axis=0)
            others = tl.where(candidate == old_state, 0.0, weights)
            others_total = tl.sum(others, axis=0)
            threshold = tl.load(uniforms_ptr + random_offset) * others_total
            # The first candidate whose cumulative weight exceeds the threshold;
            # zero-weight entries (the current state, padding) are skipped.
            proposed = tl.sum((tl.cumsum(others, axis=0) <= threshold).to(tl.int32), axis=0)
            proposed = tl.minimum(proposed, num_states - 1)
            proposed_total = tl.sum(tl.where(candidate == proposed, 0.0, weights), axis=0)
            uniform = tl.load(uniforms_ptr + random_offset + 1)
            accepted = (others_total > 0) & (uniform * proposed_total < others_total)
            new_state = tl.where(accepted, proposed, old_state)
            background = tl.where(position == site, new_state, background)
        tl.store(states_ptr + state_base + position, background, mask=position_mask)

    @triton.jit
    def _replica_sparse_kernel(
        states_ptr,
        biases_ptr,
        neighbours_ptr,
        blocks_ptr,
        sites_ptr,
        proposals_ptr,
        uniforms_ptr,
        step,
        beta,
        num_chains: tl.constexpr,
        length: tl.constexpr,
        num_states: tl.constexpr,
        num_replicas: tl.constexpr,
        degree,
        BLOCK_Q: tl.constexpr,
        BLOCK_D: tl.constexpr,
        STEPS: tl.constexpr,
        METHOD: tl.constexpr,
    ):
        """Local updates on a sparse coupling graph; one program per chain of a replica stack.

        ``neighbours`` (R, L, D) lists each site's coupled sites (-1 pads) and
        ``blocks`` (R, L, D, q, q) holds ``blocks[r, i, k, b, a] = J[i, a, j_k, b]``.
        Each step reads the D neighbour states and their coupling rows instead
        of all L positions. States stay in global memory: the new state is
        stored after a barrier, so the next step's reads see it. ``degree`` is a
        runtime value (it grows during edgeDCA training); only its power-of-two
        block ``BLOCK_D`` is compiled in.

        ``METHOD`` selects the update, with the arithmetic and random inputs of
        the dense kernels: 0 Metropolized Gibbs (``_replica_metropolized_gibbs_kernel``,
        two uniforms per update), 1 Gibbs (``_gibbs_step_kernel``, one uniform),
        2 Metropolis (``_metropolis_step_kernel``, a proposal and a uniform).
        Random inputs are indexed by (step, replica, chain).
        """
        program = tl.program_id(0)
        replica = (program // num_chains).to(tl.int64)
        chain = (program % num_chains).to(tl.int64)
        candidate = tl.arange(0, BLOCK_Q)
        slot = tl.arange(0, BLOCK_D)
        candidate_mask = candidate < num_states
        slot_mask = slot < degree
        row = states_ptr + (replica * num_chains + chain) * length
        replica_biases = biases_ptr + replica * (length * num_states)
        replica_neighbours = neighbours_ptr + replica * (length * degree)
        replica_blocks = blocks_ptr + replica * (length * degree * num_states * num_states)
        for local_step in range(STEPS):
            random_step = step + local_step
            site = tl.load(sites_ptr + random_step * num_replicas + replica).to(tl.int64)
            random_index = (random_step * num_replicas + replica) * num_chains + chain
            neighbour = tl.load(replica_neighbours + site * degree + slot, mask=slot_mask, other=-1)
            valid = neighbour >= 0
            neighbour_state = tl.load(row + neighbour, mask=valid, other=0).to(tl.int64)
            old_state = tl.load(row + site)
            row_offset = (site * degree + slot) * num_states + neighbour_state
            if METHOD == 2:
                proposed = tl.load(proposals_ptr + random_index)
                old_coupling = tl.load(replica_blocks + row_offset * num_states + old_state, mask=valid, other=0.0)
                proposed_coupling = tl.load(replica_blocks + row_offset * num_states + proposed, mask=valid,
                                            other=0.0)
                if blocks_ptr.dtype.element_ty == tl.bfloat16:
                    old_coupling = old_coupling.to(tl.float32)
                    proposed_coupling = proposed_coupling.to(tl.float32)
                old_bias = tl.load(replica_biases + site * num_states + old_state)
                proposed_bias = tl.load(replica_biases + site * num_states + proposed)
                if biases_ptr.dtype.element_ty == tl.bfloat16:
                    old_bias = old_bias.to(tl.float32)
                    proposed_bias = proposed_bias.to(tl.float32)
                delta_energy = old_bias - proposed_bias + tl.sum(old_coupling - proposed_coupling, axis=0)
                accepted = tl.load(uniforms_ptr + random_index) < tl.exp(-beta * delta_energy)
                new_state = tl.where(accepted, proposed, old_state)
            else:
                offset = row_offset[:, None] * num_states + candidate[None, :]
                coupling = tl.load(replica_blocks + offset, mask=valid[:, None] & candidate_mask[None, :],
                                   other=0.0)
                if blocks_ptr.dtype.element_ty == tl.bfloat16:
                    coupling = coupling.to(tl.float32)
                field = tl.load(replica_biases + site * num_states + candidate, mask=candidate_mask, other=0.0)
                if biases_ptr.dtype.element_ty == tl.bfloat16:
                    field = field.to(tl.float32)
                field += tl.sum(coupling, axis=0)
                logits = tl.where(candidate_mask, beta * field, -float("inf"))
                weights = tl.exp(logits - tl.max(logits, axis=0))
                if METHOD == 1:
                    cumulative = tl.cumsum(weights, axis=0)
                    threshold = tl.load(uniforms_ptr + random_index) * tl.sum(weights, axis=0)
                    new_state = tl.minimum(tl.sum((threshold > cumulative).to(tl.int32), axis=0), num_states - 1)
                else:
                    others = tl.where(candidate == old_state, 0.0, weights)
                    others_total = tl.sum(others, axis=0)
                    threshold = tl.load(uniforms_ptr + random_index * 2) * others_total
                    proposed = tl.sum((tl.cumsum(others, axis=0) <= threshold).to(tl.int32), axis=0)
                    proposed = tl.minimum(proposed, num_states - 1)
                    proposed_total = tl.sum(tl.where(candidate == proposed, 0.0, weights), axis=0)
                    uniform = tl.load(uniforms_ptr + random_index * 2 + 1)
                    accepted = (others_total > 0) & (uniform * proposed_total < others_total)
                    new_state = tl.where(accepted, proposed, old_state)
            # Every thread has read this step's states before one of them writes.
            tl.debug_barrier()
            tl.store(row + site, new_state.to(old_state.dtype))
            tl.debug_barrier()

    @triton.jit
    def _sparse_energy_kernel(
        states_ptr,
        bias_ptr,
        neighbours_ptr,
        blocks_ptr,
        output_ptr,
        length: tl.constexpr,
        num_states: tl.constexpr,
        degree,
        SYMMETRIC: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Add the energy terms of one site to its chain's energy ``-sum h - 1/2 sum J`` (sparse layout).

        Symmetric couplings count each pair once: neighbours above the site with
        weight 1 and the site itself with weight 1/2.
        """
        chain = tl.program_id(0).to(tl.int64)
        site = tl.program_id(1).to(tl.int64)
        slot = tl.arange(0, BLOCK_D)
        row = states_ptr + chain * length
        state = tl.load(row + site).to(tl.int64)
        neighbour = tl.load(neighbours_ptr + site * degree + slot, mask=slot < degree, other=-1)
        if SYMMETRIC:
            valid = neighbour >= site
            weight = tl.where(neighbour == site, 0.5, 1.0)
        else:
            valid = neighbour >= 0
            weight = 0.5
        neighbour_state = tl.load(row + neighbour, mask=valid, other=0).to(tl.int64)
        offset = ((site * degree + slot) * num_states + neighbour_state) * num_states + state
        coupling = tl.load(blocks_ptr + offset, mask=valid, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + site * num_states + state).to(tl.float32)
        tl.atomic_add(output_ptr + chain, -bias - tl.sum(weight * coupling, axis=0))


def _needs_wide_indices(num_items: int, length: int, num_states: int) -> bool:
    """Whether coupling offsets ((L*q)**2) or state offsets reach the int32 range."""
    return max(num_items * length, (length * num_states) ** 2) >= 2**31


def is_triton_available() -> bool:
    """Return whether Triton was imported successfully."""
    return triton is not None


def gibbs_sampling_triton(
    chains: torch.Tensor,
    params: dict[str, torch.Tensor],
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
    states = gibbs_sampling_categorical_triton(
        chains.argmax(dim=-1).to(torch.int32), params, nsweeps, beta,
        steps_per_launch=steps_per_launch, transpose_couplings=transpose_couplings,
        coupling_dtype=coupling_dtype,
    )
    num_states = params["bias"].shape[1]
    return _write_one_hot(states, torch.empty_like(chains), num_states)


def gibbs_sampling_categorical_triton(
    states: torch.Tensor,
    params: dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
    *,
    steps_per_launch: int = 16,
    transpose_couplings: bool = True,
    coupling_dtype: torch.dtype | None = None,
    sites: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Gibbs sampling directly on contiguous int32 categorical states.

    Each step updates one uniformly drawn site, shared by all chains, for
    ``nsweeps * L`` steps. ``sites`` instead gives the site of every step (its
    length then sets the number of steps and ``nsweeps`` is ignored). Sparse
    graphs (see :func:`sparse_coupling_layout`) read only the coupled pairs.
    """
    kernel_params, num_chains, length, _, num_steps = _prepare_categorical_inputs(states, params, nsweeps)
    if sites is not None:
        sites = sites.to(device=states.device, dtype=torch.int32)
        num_steps = len(sites)
    result = states.clone(memory_format=torch.contiguous_format)
    if num_chains == 0 or num_steps == 0:
        return result
    if steps_per_launch < 1:
        raise ValueError("steps_per_launch must be positive")
    sparse = sparse_coupling_layout(kernel_params["coupling_matrix"].unsqueeze(0),
                                    coupling_dtype or kernel_params["coupling_matrix"].dtype,
                                    source=params["coupling_matrix"], method="gibbs")
    if sparse is not None:
        transpose_couplings = False
    elif transpose_couplings:
        kernel_params["coupling_gibbs"] = kernel_params["coupling_matrix"].permute(0, 2, 3, 1).to(
            dtype=coupling_dtype or kernel_params["coupling_matrix"].dtype,
            memory_format=torch.contiguous_format,
            copy=True,
        )
    elif coupling_dtype is not None:
        kernel_params["coupling_matrix"] = kernel_params["coupling_matrix"].to(coupling_dtype)
    steps_per_chunk = max(1, min(num_steps, 16_000_000 // num_chains))
    for chunk_start in range(0, num_steps, steps_per_chunk):
        chunk_steps = min(steps_per_chunk, num_steps - chunk_start)
        chunk_sites = (torch.randint(0, length, (chunk_steps,), device=states.device, dtype=torch.int32)
                       if sites is None else sites[chunk_start:chunk_start + chunk_steps])
        uniforms = torch.rand(
            (chunk_steps, num_chains), device=states.device, dtype=_uniform_dtype(params["bias"])
        )
        if sparse is not None:
            _sparse_steps_triton(result.unsqueeze(0), kernel_params["bias"].unsqueeze(0), sparse,
                                 chunk_sites.contiguous(), uniforms, beta, "gibbs")
            continue
        _gibbs_steps_triton(
            result, kernel_params, chunk_sites, uniforms, beta,
            steps_per_launch=steps_per_launch, transpose_couplings=transpose_couplings,
        )
    return result


def gibbs_step_independent_sites_triton(
    chains: torch.Tensor,
    params: dict[str, torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    """Apply one fused Gibbs update at an independently drawn site per chain (CUDA, Triton).

    Args:
        chains: One-hot chains, shape ``(n_chains, L, q)``, on a CUDA device.
        params: ``bias`` ``(L, q)`` and ``coupling_matrix`` ``(L, q, L, q)``.
        beta: Inverse temperature.

    Returns:
        The updated chains.
    """
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
    params: dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
    *,
    steps_per_launch: int = 16,
    coupling_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run Metropolis updates with ``steps_per_launch`` updates per kernel."""
    states = metropolis_sampling_categorical_triton(
        chains.argmax(dim=-1).to(torch.int32), params, nsweeps, beta,
        steps_per_launch=steps_per_launch, coupling_dtype=coupling_dtype,
    )
    num_states = params["bias"].shape[1]
    return _write_one_hot(states, torch.empty_like(chains), num_states)


def metropolis_sampling_categorical_triton(
    states: torch.Tensor,
    params: dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
    *,
    steps_per_launch: int = 16,
    coupling_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run Metropolis sampling directly on contiguous int32 categorical states.

    Sparse graphs (see :func:`sparse_coupling_layout`) read only the coupled pairs.
    """
    kernel_params, num_chains, length, num_states, num_steps = _prepare_categorical_inputs(
        states, params, nsweeps
    )
    result = states.clone(memory_format=torch.contiguous_format)
    if num_chains == 0 or num_steps == 0:
        return result
    if steps_per_launch < 1:
        raise ValueError("steps_per_launch must be positive")
    sparse = sparse_coupling_layout(kernel_params["coupling_matrix"].unsqueeze(0),
                                    coupling_dtype or kernel_params["coupling_matrix"].dtype,
                                    source=params["coupling_matrix"], method="metropolis")
    if coupling_dtype is not None and sparse is None:
        kernel_params["coupling_matrix"] = kernel_params["coupling_matrix"].to(coupling_dtype)
    steps_per_chunk = max(1, min(num_steps, 16_000_000 // num_chains))
    for chunk_start in range(0, num_steps, steps_per_chunk):
        chunk_steps = min(steps_per_chunk, num_steps - chunk_start)
        sites = torch.randint(0, length, (chunk_steps,), device=states.device, dtype=torch.int32)
        proposals = torch.randint(
            0, num_states, (chunk_steps, num_chains), device=states.device, dtype=torch.int32,
        )
        uniforms = torch.rand(
            (chunk_steps, num_chains), device=states.device, dtype=_uniform_dtype(params["bias"])
        )
        if sparse is not None:
            _sparse_steps_triton(result.unsqueeze(0), kernel_params["bias"].unsqueeze(0), sparse, sites, uniforms,
                                 beta, "metropolis", proposals=proposals)
            continue
        _metropolis_steps_triton(
            result, kernel_params, sites, proposals, uniforms, beta, steps_per_launch=steps_per_launch,
        )
    return result


def metropolis_sampling_replicas_categorical_triton(
    states: torch.Tensor,
    biases: torch.Tensor,
    couplings: torch.Tensor,
    nsweeps: int,
    beta: float = 1.0,
    *,
    steps_per_launch: int | None = None,
) -> torch.Tensor:
    """Sample a stack of PTT replicas in one sequence of Triton launches.

    ``steps_per_launch=None`` uses the tuned default for the alphabet size.
    Sparse graphs (see :func:`sparse_coupling_layout`) read only the coupled pairs.
    """
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not states.is_cuda or states.dtype != torch.int32 or states.ndim != 3 or not states.is_contiguous():
        raise ValueError("Replica sampling requires contiguous CUDA int32 states of shape (R, N, L)")
    num_replicas, num_chains, length = states.shape
    if biases.ndim != 3 or biases.shape[:2] != (num_replicas, length):
        raise ValueError("Replica biases must have shape (R, L, q)")
    num_states = biases.shape[2]
    if couplings.shape != (num_replicas, length, num_states, length, num_states):
        raise ValueError("Replica couplings must have shape (R, L, q, L, q)")
    if biases.device != states.device or couplings.device != states.device:
        raise ValueError("Replica states and parameters must be on the same CUDA device")
    if biases.dtype != couplings.dtype:
        raise ValueError("Replica biases and couplings must use the same dtype")
    block_n, num_warps, default_steps = _replica_metropolis_launch(num_states)
    steps_per_launch = default_steps if steps_per_launch is None else steps_per_launch
    if steps_per_launch < 1:
        raise ValueError("steps_per_launch must be positive")
    result = states.clone(memory_format=torch.contiguous_format)
    num_steps = nsweeps * length
    if num_replicas == 0 or num_chains == 0 or num_steps == 0:
        return result
    sparse = sparse_coupling_layout(couplings, method="metropolis")
    steps_per_chunk = max(1, min(num_steps, 16_000_000 // (num_replicas * num_chains)))
    for chunk_start in range(0, num_steps, steps_per_chunk):
        chunk_steps = min(steps_per_chunk, num_steps - chunk_start)
        sites = torch.randint(
            0, length, (chunk_steps, num_replicas), device=states.device, dtype=torch.int32
        )
        proposals = torch.randint(
            0, num_states, (chunk_steps, num_replicas, num_chains),
            device=states.device, dtype=torch.int32,
        )
        uniforms = torch.rand(
            (chunk_steps, num_replicas, num_chains),
            device=states.device, dtype=_uniform_dtype(biases),
        )
        if sparse is not None:
            _sparse_steps_triton(result, biases.contiguous(), sparse, sites, uniforms, beta, "metropolis",
                                 proposals=proposals)
            continue
        _replica_metropolis_steps_triton(
            result, biases, couplings, sites, proposals, uniforms, beta,
            steps_per_launch=steps_per_launch, block_n=block_n, num_warps=num_warps,
        )
    return result


def _replica_metropolis_launch(num_states: int) -> tuple[int, int, int]:
    """(chains per program, warps, steps per launch) tuned on an RTX A5000 at L=307.

    Small programs keep each chain's reduction inside one warp. Short launches
    keep all chains of a replica on the same site, so its coupling rows stay
    in L2. Nucleotide alphabets favor even smaller programs.
    """
    return (2, 1, 16) if num_states <= 8 else (4, 1, 8)


def transpose_couplings_for_metropolized_gibbs(couplings: torch.Tensor) -> torch.Tensor:
    """View (L, q, L, q) couplings as (site, source_site, source_state, state).

    Stacking such views materializes the layout expected by
    ``metropolized_gibbs_sampling_replicas_categorical_triton`` in one copy.
    """
    return couplings.permute(0, 2, 3, 1)


def metropolized_gibbs_sampling_replicas_categorical_triton(
    states: torch.Tensor,
    biases: torch.Tensor,
    couplings: torch.Tensor,
    nsweeps: int,
    beta: float = 1.0,
    *,
    steps_per_launch: int = 8,
) -> torch.Tensor:
    """Metropolized Gibbs sampling of a stack of replicas in one sequence of launches.

    ``couplings`` has shape (R, L, L, q, q) in the layout returned by
    ``transpose_couplings_for_metropolized_gibbs`` for each replica. Every
    replica updates one random site per step, shared by all its chains.
    """
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not states.is_cuda or states.dtype != torch.int32 or states.ndim != 3 or not states.is_contiguous():
        raise ValueError("Replica sampling requires contiguous CUDA int32 states of shape (R, N, L)")
    num_replicas, num_chains, length = states.shape
    if biases.ndim != 3 or biases.shape[:2] != (num_replicas, length):
        raise ValueError("Replica biases must have shape (R, L, q)")
    num_states = biases.shape[2]
    if couplings.shape != (num_replicas, length, length, num_states, num_states) or not couplings.is_contiguous():
        raise ValueError("Metropolized Gibbs couplings must be contiguous with shape (R, L, L, q, q)")
    if biases.device != states.device or couplings.device != states.device:
        raise ValueError("Replica states and parameters must be on the same CUDA device")
    if biases.dtype not in (torch.float32, torch.float64) or couplings.dtype not in (biases.dtype, torch.bfloat16):
        raise ValueError(
            "Metropolized Gibbs requires float32/float64 biases and couplings of the same dtype or bfloat16"
        )
    if steps_per_launch < 1:
        raise ValueError("steps_per_launch must be positive")
    result = states.clone(memory_format=torch.contiguous_format)
    _metropolized_gibbs_sweeps(result, biases.contiguous(), couplings, nsweeps, beta, steps_per_launch)
    return result


def _metropolized_gibbs_sweeps(states, biases, couplings, nsweeps, beta, steps_per_launch, sparse=None):
    """Run ``nsweeps`` Metropolized Gibbs sweeps in place on (R, N, L) states.

    ``couplings`` are in the transposed dense layout, unless ``sparse`` (from
    :func:`sparse_coupling_layout`) is given; both draw the same random numbers.
    """
    num_replicas, num_chains, length = states.shape
    num_states = biases.shape[2]
    num_steps = nsweeps * length
    if num_replicas == 0 or num_chains == 0 or num_steps == 0:
        return
    steps_per_chunk = max(1, min(num_steps, 8_000_000 // (num_replicas * num_chains)))
    block_l = triton.next_power_of_2(length)
    block_q = triton.next_power_of_2(num_states)
    num_warps = _per_chain_warps(block_l, block_q)
    for chunk_start in range(0, num_steps, steps_per_chunk):
        chunk_steps = min(steps_per_chunk, num_steps - chunk_start)
        sites = torch.randint(0, length, (chunk_steps, num_replicas), device=states.device, dtype=torch.int32)
        uniforms = torch.rand(
            (chunk_steps, num_replicas, num_chains, 2), device=states.device, dtype=_uniform_dtype(biases)
        )
        if sparse is not None:
            _sparse_steps_triton(states, biases, sparse, sites, uniforms, beta, "metropolized_gibbs")
            continue
        _replica_metropolized_gibbs_steps_triton(
            states, biases, couplings, sites, uniforms, beta,
            steps_per_launch=steps_per_launch, block_l=block_l, block_q=block_q, num_warps=num_warps,
        )


# Sparse kernels (disabled by ADABMDCA_SPARSE=0) are chosen per sampler from the largest
# number of sites any site is coupled to. Measured on an RTX A5000 (2000 chains):
# - Gibbs and Metropolized Gibbs load a (neighbour block x state block) tile per update:
#   1.5-2.2 times faster on RF00379 edgeDCA models (largest degree 17-36% of L) and
#   2-5 times faster on a 279-site protein model with 5-15% of the pairs coupled
#   (largest degree 15-34% of L), but 0.3-0.7 times as fast from 52% of L, and far
#   slower once the tile exceeds 4096 values (256 neighbours x 32 states spills).
# - Metropolis loads two values per neighbour: 1.1-7 times faster up to 82% of L on
#   the protein model, about as fast on its full graph, 0.6 times on RF00379 bmDCA.
SPARSE_MAX_DEGREE_FRACTION = {"gibbs": 0.4, "metropolized_gibbs": 0.4, "metropolis": 0.6}
_SPARSE_MAX_TILE = 4096
# Sparse launches: one warp per chain and 32 steps per launch were fastest.
_SPARSE_STEPS_PER_LAUNCH = 32


def _largest_degree(couplings: torch.Tensor) -> int:
    """The largest number of sites any site of stacked couplings ``(R, L, q, L, q)`` is coupled to."""
    pair = (couplings != 0).any(dim=4).any(dim=2)
    return int(pair.sum(-1).max()) if pair.numel() else 0


def _sparse_pays_off(method: str, degree: int, length: int, num_states: int) -> bool:
    if degree > SPARSE_MAX_DEGREE_FRACTION[method] * length:
        return False
    if method == "metropolis":
        return True
    return triton.next_power_of_2(max(degree, 1)) * triton.next_power_of_2(num_states) <= _SPARSE_MAX_TILE


def _sparse_coupling_layout(couplings: torch.Tensor, dtype: torch.dtype):
    """Padded neighbour table and coupling blocks of stacked couplings ``(R, L, q, L, q)``."""
    num_replicas, length, num_states = couplings.shape[:3]
    pair = (couplings != 0).any(dim=4).any(dim=2)
    degree = pair.sum(-1)
    width = max(int(degree.max()) if degree.numel() else 0, 1)
    # Coupled sites first, in increasing order.
    order = torch.argsort((~pair).to(torch.int8), dim=-1, stable=True)[..., :width]
    valid = torch.arange(width, device=couplings.device) < degree.unsqueeze(-1)
    neighbours = torch.where(valid, order, -1).to(torch.int32).contiguous()
    replica = torch.arange(num_replicas, device=couplings.device)[:, None, None]
    site = torch.arange(length, device=couplings.device)[None, :, None]
    blocks = couplings.permute(0, 1, 3, 4, 2)[replica, site, order]  # (R, L, D, b, a)
    blocks = (blocks * valid[..., None, None]).to(dtype).contiguous()
    return neighbours, blocks, width


def sparse_coupling_layout(couplings: torch.Tensor, dtype: torch.dtype | None = None, source=None,
                           method: str = "metropolized_gibbs"):
    """The sparse layout of stacked couplings ``(R, L, q, L, q)`` if it pays off for ``method``, else ``None``.

    Returns ``(neighbours, blocks, degree)``: ``neighbours`` ``(R, L, D)`` lists each
    site's coupled sites (padded with -1), ``blocks`` ``(R, L, D, q, q)`` holds
    ``blocks[r, i, k, b, a] = J[i, a, j_k, b]``. The degree and the layout are cached
    per ``source`` tensor (``couplings`` by default); the layout is built only when used.
    """
    if not sparse_kernels_enabled():
        return None
    source = couplings if source is None else source
    degree = cached(source, "triton_degree", lambda: _largest_degree(couplings))
    if not _sparse_pays_off(method, degree, couplings.shape[1], couplings.shape[2]):
        return None
    dtype = dtype or couplings.dtype
    return cached(source, f"triton_sparse_{dtype}", lambda: _sparse_coupling_layout(couplings, dtype))


_SPARSE_METHODS = {"metropolized_gibbs": 0, "gibbs": 1, "metropolis": 2}


def _sparse_steps_triton(states, biases, sparse, sites, uniforms, beta, method, proposals=None):
    """Apply controlled sparse updates on (R, N, L) states.

    ``sites`` is (steps, R); ``uniforms`` is (steps, R, N, 2) for Metropolized
    Gibbs and (steps, R, N) otherwise; ``proposals`` (steps, R, N) for Metropolis.
    A single model is a stack of one, with the random layouts of its dense kernel.
    """
    neighbours, blocks, degree = sparse
    num_replicas, num_chains, length = states.shape
    num_states = biases.shape[2]
    if num_replicas == 0 or num_chains == 0:
        return
    for step in range(0, sites.shape[0], _SPARSE_STEPS_PER_LAUNCH):
        _replica_sparse_kernel[(num_replicas * num_chains,)](
            states, biases, neighbours, blocks, sites, uniforms if proposals is None else proposals, uniforms,
            step, beta, num_chains, length, num_states, num_replicas, degree,
            BLOCK_Q=triton.next_power_of_2(num_states), BLOCK_D=triton.next_power_of_2(degree),
            STEPS=min(_SPARSE_STEPS_PER_LAUNCH, sites.shape[0] - step), METHOD=_SPARSE_METHODS[method],
            num_warps=1,
        )


def _replica_metropolized_gibbs_sparse_steps_triton(states, biases, sparse, sites, uniforms, beta):
    """Apply controlled sparse Metropolized Gibbs updates on (R, N, L) states."""
    _sparse_steps_triton(states, biases, sparse, sites, uniforms, beta, "metropolized_gibbs")


def metropolized_gibbs_sampling_replicas_triton(
    states: torch.Tensor,
    biases: torch.Tensor,
    couplings: torch.Tensor,
    nsweeps: int,
    beta: float = 1.0,
    *,
    steps_per_launch: int = 8,
) -> torch.Tensor:
    """Metropolized Gibbs sampling of a stack of replicas, from couplings in their original layout.

    ``couplings`` has shape (R, L, q, L, q). Sparse graphs use the sparse kernel;
    otherwise the transposed copy of
    :func:`metropolized_gibbs_sampling_replicas_categorical_triton` is made. Either
    layout is cached and reused while ``couplings`` is unchanged.
    """
    if couplings.ndim != 5:
        raise ValueError("Replica couplings must have shape (R, L, q, L, q)")
    sparse = sparse_coupling_layout(couplings)
    if sparse is None:
        # (R, i, a, j, b) -> (R, i, j, b, a): the per-replica layout of transpose_couplings_for_metropolized_gibbs.
        transposed = cached(couplings, "triton_transposed", lambda: couplings.permute(0, 1, 3, 4, 2).contiguous())
        return metropolized_gibbs_sampling_replicas_categorical_triton(
            states, biases, transposed, nsweeps, beta, steps_per_launch=steps_per_launch)
    if not states.is_cuda or states.dtype != torch.int32 or states.ndim != 3 or not states.is_contiguous():
        raise ValueError("Replica sampling requires contiguous CUDA int32 states of shape (R, N, L)")
    result = states.clone(memory_format=torch.contiguous_format)
    _metropolized_gibbs_sweeps(result, biases.contiguous(), None, nsweeps, beta, steps_per_launch, sparse=sparse)
    return result


def metropolized_gibbs_sampling_triton(
    chains: torch.Tensor,
    params: dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
    *,
    steps_per_launch: int = 8,
    coupling_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Metropolized Gibbs sampling of one-hot chains; see ``metropolized_gibbs_sampling_categorical_triton``."""
    if chains.dtype not in (torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Metropolized Gibbs sampling supports bfloat16, float32 and float64 chains")
    states = metropolized_gibbs_sampling_categorical_triton(
        chains.argmax(dim=-1).to(torch.int32), params, nsweeps, beta,
        steps_per_launch=steps_per_launch, coupling_dtype=coupling_dtype,
    )
    num_states = params["bias"].shape[1]
    return _write_one_hot(states, torch.empty_like(chains), num_states)


def metropolized_gibbs_sampling_categorical_triton(
    states: torch.Tensor,
    params: dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
    *,
    steps_per_launch: int = 8,
    coupling_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Metropolized Gibbs sampling of contiguous CUDA int32 states (N, L).

    Each update draws one site shared by all chains. A new state b != a is
    proposed from the site conditional restricted to the other states and
    accepted with min(1, (1 - p_a) / (1 - p_b)) (Liu 1996). One program owns
    a chain and reads, for every source position, a contiguous row of
    candidate couplings from a transposed copy made once per call.
    Sparse graphs (see :func:`sparse_coupling_layout`) read only each site's
    coupled neighbours instead.
    ``coupling_dtype=torch.bfloat16`` stores that copy in BF16. Fields and
    acceptance use the bias precision, or FP32 for BF16 biases, as do the
    uniform random numbers. Couplings must have zero within-site blocks.
    """
    kernel_params, num_chains, length, num_states, num_steps = _prepare_categorical_inputs(states, params, nsweeps)
    result = states.clone(memory_format=torch.contiguous_format)
    if num_chains == 0 or num_steps == 0:
        return result
    if steps_per_launch < 1:
        raise ValueError("steps_per_launch must be positive")
    bias = kernel_params["bias"]
    source = params["coupling_matrix"]
    sparse = sparse_coupling_layout(kernel_params["coupling_matrix"].unsqueeze(0),
                                    coupling_dtype or kernel_params["coupling_matrix"].dtype, source=source)
    replicas = result.unsqueeze(0)
    if sparse is not None:
        _metropolized_gibbs_sweeps(replicas, bias.unsqueeze(0), None, nsweeps, beta, steps_per_launch, sparse=sparse)
        return result
    couplings = transpose_couplings_for_metropolized_gibbs(kernel_params["coupling_matrix"]).to(
        dtype=coupling_dtype or kernel_params["coupling_matrix"].dtype,
        memory_format=torch.contiguous_format,
        copy=True,
    )
    _metropolized_gibbs_sweeps(replicas, bias.unsqueeze(0), couplings.unsqueeze(0), nsweeps, beta, steps_per_launch)
    return result


def _replica_metropolized_gibbs_steps_triton(
    states, biases, couplings, sites, uniforms, beta, *, steps_per_launch, block_l, block_q, num_warps,
):
    """Apply controlled Metropolized Gibbs updates; used for validation and chunking."""
    num_replicas, num_chains, length = states.shape
    num_states = biases.shape[2]
    for step in range(0, sites.shape[0], steps_per_launch):
        _replica_metropolized_gibbs_kernel[(num_replicas * num_chains,)](
            states, biases, couplings, sites, uniforms, step, beta,
            num_chains, length, num_states, num_replicas,
            BLOCK_L=block_l, BLOCK_Q=block_q, STEPS=min(steps_per_launch, sites.shape[0] - step),
            WIDE=_needs_wide_indices(num_replicas * num_chains, length, num_states), num_warps=num_warps,
        )


def metropolis_step_independent_sites_triton(
    chains: torch.Tensor,
    params: dict[str, torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    """Apply one fused Metropolis update at an independently drawn site per chain (CUDA, Triton).

    Args:
        chains: One-hot chains, shape ``(n_chains, L, q)``, on a CUDA device.
        params: ``bias`` ``(L, q)`` and ``coupling_matrix`` ``(L, q, L, q)``.
        beta: Inverse temperature.

    Returns:
        The updated chains.
    """
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
    params: dict[str, torch.Tensor],
    nsweeps: int,
) -> tuple[dict[str, torch.Tensor], int, int, int, int]:
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


def _prepare_categorical_inputs(
    states: torch.Tensor,
    params: dict[str, torch.Tensor],
    nsweeps: int,
) -> tuple[dict[str, torch.Tensor], int, int, int, int]:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not states.is_cuda or states.dtype != torch.int32 or states.ndim != 2 or not states.is_contiguous():
        raise ValueError("Categorical Triton sampling requires contiguous CUDA int32 states of shape (N, L)")
    num_chains, length = states.shape
    bias = params["bias"]
    couplings = params["coupling_matrix"]
    num_states = bias.shape[1]
    if bias.shape[0] != length or couplings.shape != (length, num_states, length, num_states):
        raise ValueError("Categorical states and Potts parameters have incompatible dimensions")
    if bias.device != states.device or couplings.device != states.device:
        raise ValueError("Categorical states and parameters must be on the same CUDA device")
    return {
        "bias": bias.contiguous(),
        "coupling_matrix": couplings.contiguous(),
    }, num_chains, length, num_states, nsweeps * length


def exchange_log_acceptance_sparse_triton(
    lower_params: dict[str, torch.Tensor],
    upper_params: dict[str, torch.Tensor],
    lower_states: torch.Tensor,
    upper_states: torch.Tensor,
) -> torch.Tensor | None:
    """Hamiltonian-exchange log acceptance from sparse coupling layouts, or ``None`` if a graph is dense.

    Computes the four energies ``E_m(z)`` for both models and both populations
    from each site's coupled neighbours only.
    """
    layouts = []
    for params in (lower_params, upper_params):
        couplings = params["coupling_matrix"]
        layout = sparse_coupling_layout(couplings.unsqueeze(0), source=couplings)
        if layout is None:
            return None
        layouts.append(layout)
    if lower_states.shape != upper_states.shape:
        raise ValueError("Exchange populations must have matching shapes")
    num_chains, length = lower_states.shape
    num_states = lower_params["bias"].shape[1]
    symmetric = (couplings_are_symmetric(lower_params["coupling_matrix"])
                 and couplings_are_symmetric(upper_params["coupling_matrix"]))
    energies = {}
    for name, params, (neighbours, blocks, degree) in zip(("lower", "upper"), (lower_params, upper_params), layouts):
        for which, states in (("x", lower_states), ("y", upper_states)):
            output = torch.zeros(num_chains, device=states.device, dtype=torch.float32)
            if num_chains:
                _sparse_energy_kernel[(num_chains, length)](
                    states.contiguous(), params["bias"].contiguous(), neighbours, blocks, output,
                    length, num_states, degree, SYMMETRIC=symmetric, BLOCK_D=triton.next_power_of_2(degree),
                    num_warps=1,
                )
            energies[name, which] = output.double()
    # log acceptance D(y) - D(x), with D = E_upper - E_lower.
    log_acceptance = (energies["upper", "y"] - energies["lower", "y"]) - (energies["upper", "x"] - energies["lower", "x"])
    return log_acceptance.clamp_max_(0.0)


def exchange_log_acceptance_categorical_triton(
    lower_params: dict[str, torch.Tensor],
    upper_params: dict[str, torch.Tensor],
    lower_states: torch.Tensor,
    upper_states: torch.Tensor,
) -> torch.Tensor:
    """Compute Hamiltonian-exchange log acceptance from categorical states."""
    lower_kernel_params, num_chains, length, num_states, _ = _prepare_categorical_inputs(
        lower_states, lower_params, 0
    )
    upper_kernel_params, _, _, _, _ = _prepare_categorical_inputs(upper_states, upper_params, 0)
    if upper_states.shape != lower_states.shape:
        raise ValueError("Exchange populations must have matching shapes")
    if lower_params["bias"].dtype != upper_params["bias"].dtype:
        raise ValueError("Exchange models must use the same dtype")
    output = torch.zeros(num_chains, device=lower_states.device, dtype=lower_params["bias"].dtype)
    if num_chains:
        _categorical_exchange_kernel[(num_chains, length)](
            lower_kernel_params["bias"],
            lower_kernel_params["coupling_matrix"],
            upper_kernel_params["bias"],
            upper_kernel_params["coupling_matrix"],
            lower_states,
            upper_states,
            output,
            num_chains,
            length,
            num_states,
            BLOCK_L=triton.next_power_of_2(length),
            SYMMETRIC=couplings_are_symmetric(lower_params["coupling_matrix"])
            and couplings_are_symmetric(upper_params["coupling_matrix"]),
            num_warps=4,
        )
    return output.double().clamp_max_(0.0)


def _gibbs_steps_triton(
    states: torch.Tensor,
    params: dict[str, torch.Tensor],
    sites: torch.Tensor,
    uniforms: torch.Tensor,
    beta: float,
    *,
    steps_per_launch: int = 16,
    transpose_couplings: bool = True,
    num_warps: int | None = None,
) -> None:
    """Apply controlled Gibbs updates; used for validation and chunking."""
    num_chains, length = states.shape
    num_states = params["bias"].shape[1]
    block_l = triton.next_power_of_2(length)
    block_q = triton.next_power_of_2(num_states)
    num_warps = _per_chain_warps(block_l, block_q) if num_warps is None else num_warps
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
            WIDE=_needs_wide_indices(num_chains, length, num_states),
            num_warps=num_warps,
        )


def _per_chain_warps(block_l: int, block_q: int) -> int:
    """Warps for kernels with one program per chain reducing an (L, q) tile.

    The fewest warps that avoid register spills minimize the synchronization
    of every per-update reduction (tuned on an RTX A5000).
    """
    return min(8, max(1, block_l * block_q // 8192))


def _gibbs_step_independent_triton(
    states: torch.Tensor,
    params: dict[str, torch.Tensor],
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
        WIDE=_needs_wide_indices(num_chains, length, num_states),
        num_warps=4,
    )


def _metropolis_steps_triton(
    states: torch.Tensor,
    params: dict[str, torch.Tensor],
    sites: torch.Tensor,
    proposals: torch.Tensor,
    uniforms: torch.Tensor,
    beta: float,
    *,
    steps_per_launch: int = 16,
    block_n: int | None = None,
    num_warps: int | None = None,
) -> None:
    """Apply controlled Metropolis updates; used for validation and chunking."""
    num_chains, length = states.shape
    num_states = params["bias"].shape[1]
    default_block_n, default_warps = _metropolis_launch(num_states)
    block_n = default_block_n if block_n is None else block_n
    num_warps = default_warps if num_warps is None else num_warps
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
            WIDE=_needs_wide_indices(num_chains, length, num_states),
            num_warps=num_warps,
        )


def _metropolis_launch(num_states: int) -> tuple[int, int]:
    """(chains per program, warps) for the single-model Metropolis kernel.

    Tuned on an RTX A5000 at L=307 with 16 steps per launch: two chains in
    one warp keep each reduction inside the warp. Against the former int64
    kernel (16 chains, 4 warps) this was 1.4-1.7x faster for q=21 and
    5-7x faster for q=5, with 2000 and 10000 chains.
    """
    return 2, 1


def _replica_metropolis_steps_triton(
    states: torch.Tensor,
    biases: torch.Tensor,
    couplings: torch.Tensor,
    sites: torch.Tensor,
    proposals: torch.Tensor,
    uniforms: torch.Tensor,
    beta: float,
    *,
    steps_per_launch: int = 16,
    block_n: int = 16,
    num_warps: int = 4,
) -> None:
    """Apply controlled Metropolis updates to a stacked replica population."""
    num_replicas, num_chains, length = states.shape
    num_states = biases.shape[2]
    blocks_per_replica = triton.cdiv(num_chains, block_n)
    grid = (num_replicas * blocks_per_replica,)
    if num_replicas == 0 or num_chains == 0 or sites.numel() == 0:
        return
    for step in range(0, sites.shape[0], steps_per_launch):
        _replica_metropolis_kernel[grid](
            states,
            biases,
            couplings,
            sites,
            proposals,
            uniforms,
            step,
            beta,
            num_chains,
            length,
            num_states,
            num_replicas,
            blocks_per_replica,
            BLOCK_N=block_n,
            BLOCK_L=triton.next_power_of_2(length),
            STEPS=min(steps_per_launch, sites.shape[0] - step),
            WIDE=_needs_wide_indices(num_replicas * num_chains, length, num_states),
            num_warps=num_warps,
        )


def _metropolis_step_independent_triton(
    states: torch.Tensor,
    params: dict[str, torch.Tensor],
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
        WIDE=_needs_wide_indices(num_chains, length, num_states),
        num_warps=4,
    )
