"""Random-site Potts updates of integer states, compiled with Numba.

The kernels perform the same updates as the CUDA kernels: each step draws one
site, shared by all chains, and every chain updates it with its own random
numbers. Chains never interact, so blocks of chains run in parallel threads
through all steps without synchronizing; a block updates the same site for
each of its chains in a row, which reuses that site's couplings while they
are in cache. Gibbs and Metropolized Gibbs read the couplings transposed to
``(site, j, x_j, a)`` so that the fields of all states ``a`` come from
contiguous memory (:func:`kernel_couplings`). The kernels take a stack of
replicas, each with its own model and random sites; a single model is a stack
of one, and a stack gives every thread larger blocks than separate calls per
replica would. Random numbers come from :mod:`adabmDCA.numba_kernels.random`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from numba import njit, prange

from adabmDCA.numba_kernels.random import (
    SLOT_PROFILE,
    SLOT_PROPOSAL,
    SLOT_SECOND_UNIFORM,
    SLOT_SITE,
    SLOT_UNIFORM,
    draw_seed,
    randint,
    uniform,
)
from adabmDCA._tensor_cache import cached, sparse_kernels_enabled
from adabmDCA.numba_kernels.threads import set_kernel_threads

METHODS = ("gibbs", "metropolis", "metropolized_gibbs")


@njit(cache=True, inline="always")
def _site(seed, step, replica, sites, length):
    """The site of ``step`` in ``replica``: explicit when ``sites`` is not empty, else drawn."""
    if sites.shape[0] > 0:
        return sites[step]
    return randint(seed, step, replica, SLOT_SITE, length)


@njit(cache=True, fastmath=True, inline="always")
def _field(x, chain, h, transposed, site, field):
    """field[a] = h[site, a] + sum_j J[site, a, j, x_j], with transposed[site, j, b, a] = J[site, a, j, b]."""
    q = h.shape[1]
    for a in range(q):
        field[a] = h[site, a]
    for j in range(x.shape[1]):
        row = transposed[site, j, x[chain, j]]
        for a in range(q):
            field[a] += row[a]


@njit(cache=True, fastmath=True, inline="always")
def _sparse_field(x, chain, h, nbr_ptr, nbrs, blocks, site, field):
    """field[a] = h[site, a] + sum over the neighbours j of site of blocks[s, x_j, a] (= J[site, a, j, x_j])."""
    q = h.shape[1]
    for a in range(q):
        field[a] = h[site, a]
    for s in range(nbr_ptr[site], nbr_ptr[site + 1]):
        row = blocks[s, x[chain, nbrs[s]]]
        for a in range(q):
            field[a] += row[a]


@njit(cache=True, fastmath=True, inline="always")
def _gibbs_choose(field, q, beta, seed, t, stream):
    """Draw a state from exp(beta * field); ``field`` is overwritten."""
    top = field[0]
    for a in range(1, q):
        top = max(top, field[a])
    total = 0.0
    for a in range(q):
        field[a] = math.exp(beta * (field[a] - top))
        total += field[a]
    threshold = uniform(seed, t, stream, SLOT_UNIFORM) * total
    cumulative = 0.0
    for a in range(q):
        cumulative += field[a]
        if threshold < cumulative:
            return a
    return q - 1


@njit(cache=True, fastmath=True, inline="always")
def _metropolized_gibbs_choose(field, q, beta, old, seed, t, stream):
    """The state after one Metropolized Gibbs update from ``old`` (Liu 1996); ``field`` is overwritten."""
    top = field[0]
    for a in range(1, q):
        top = max(top, field[a])
    for a in range(q):
        field[a] = math.exp(beta * (field[a] - top))
    others_total = 0.0
    for a in range(q):
        if a != old:
            others_total += field[a]
    # Propose b != old from the conditional restricted to the other states.
    threshold = uniform(seed, t, stream, SLOT_UNIFORM) * others_total
    cumulative = 0.0
    proposed = q - 1
    for a in range(q):
        if a != old:
            cumulative += field[a]
            if cumulative > threshold:
                proposed = a
                break
    # Accept with min(1, (1 - p_old) / (1 - p_proposed)); both sums are direct.
    excluded_total = 0.0
    for a in range(q):
        if a != proposed:
            excluded_total += field[a]
    if others_total > 0.0 and uniform(seed, t, stream, SLOT_SECOND_UNIFORM) * excluded_total < others_total:
        return proposed
    return old


# The kernels update a stack of replicas: states (R, N, L), biases (R, L, q) and
# couplings (R, ...). Each (replica, block of chains) pair is one parallel task;
# random numbers are indexed by the chain's position r * N + chain in the stack,
# so a single model (R = 1) is the same computation as before stacking. Dense
# kernels read full coupling arrays; sparse kernels read, for each site, only
# the coupling blocks of its neighbours (see ``_sparse_layout``). Both perform
# the same arithmetic on the non-zero couplings and draw the same random numbers.


@njit(cache=True, fastmath=True, parallel=True)
def _gibbs_kernel(states, biases, transposed_stack, seed, first_step, steps, sites, beta, block):
    num_replicas, n, length = states.shape
    q = biases.shape[2]
    blocks = (n + block - 1) // block
    for task in prange(num_replicas * blocks):
        r, b = task // blocks, task % blocks
        x, h, transposed = states[r], biases[r], transposed_stack[r]
        field = np.empty(q, dtype=h.dtype)
        for t in range(first_step, first_step + steps):
            site = _site(seed, t, r, sites, length)
            for chain in range(b * block, min(n, (b + 1) * block)):
                _field(x, chain, h, transposed, site, field)
                x[chain, site] = _gibbs_choose(field, q, beta, seed, t, r * n + chain)


@njit(cache=True, fastmath=True, parallel=True)
def _gibbs_sparse_kernel(states, biases, nbr_ptr, nbrs, coupling_blocks, seed, first_step, steps, sites, beta, block):
    num_replicas, n, length = states.shape
    q = biases.shape[2]
    blocks = (n + block - 1) // block
    for task in prange(num_replicas * blocks):
        r, b = task // blocks, task % blocks
        x, h, pointers = states[r], biases[r], nbr_ptr[r]
        field = np.empty(q, dtype=h.dtype)
        for t in range(first_step, first_step + steps):
            site = _site(seed, t, r, sites, length)
            for chain in range(b * block, min(n, (b + 1) * block)):
                _sparse_field(x, chain, h, pointers, nbrs, coupling_blocks, site, field)
                x[chain, site] = _gibbs_choose(field, q, beta, seed, t, r * n + chain)


@njit(cache=True, fastmath=True, parallel=True)
def _metropolis_kernel(states, biases, coupling_stack, seed, first_step, steps, sites, beta, block):
    num_replicas, n, length = states.shape
    q = biases.shape[2]
    blocks = (n + block - 1) // block
    for task in prange(num_replicas * blocks):
        r, b = task // blocks, task % blocks
        x, h, couplings = states[r], biases[r], coupling_stack[r]
        for t in range(first_step, first_step + steps):
            site = _site(seed, t, r, sites, length)
            for chain in range(b * block, min(n, (b + 1) * block)):
                old = x[chain, site]
                new = randint(seed, t, r * n + chain, SLOT_PROPOSAL, q)
                if new == old:
                    continue
                # Only the fields of the current and proposed states are needed.
                gain = h[site, new] - h[site, old]
                for j in range(length):
                    state = x[chain, j]
                    gain += couplings[site, new, j, state] - couplings[site, old, j, state]
                if math.exp(beta * gain) > uniform(seed, t, r * n + chain, SLOT_UNIFORM):
                    x[chain, site] = new


@njit(cache=True, fastmath=True, parallel=True)
def _metropolis_sparse_kernel(states, biases, nbr_ptr, nbrs, coupling_blocks, seed, first_step, steps, sites, beta,
                              block):
    num_replicas, n, length = states.shape
    q = biases.shape[2]
    blocks = (n + block - 1) // block
    for task in prange(num_replicas * blocks):
        r, b = task // blocks, task % blocks
        x, h, pointers = states[r], biases[r], nbr_ptr[r]
        for t in range(first_step, first_step + steps):
            site = _site(seed, t, r, sites, length)
            for chain in range(b * block, min(n, (b + 1) * block)):
                old = x[chain, site]
                new = randint(seed, t, r * n + chain, SLOT_PROPOSAL, q)
                if new == old:
                    continue
                gain = h[site, new] - h[site, old]
                for s in range(pointers[site], pointers[site + 1]):
                    row = coupling_blocks[s, x[chain, nbrs[s]]]
                    gain += row[new] - row[old]
                if math.exp(beta * gain) > uniform(seed, t, r * n + chain, SLOT_UNIFORM):
                    x[chain, site] = new


@njit(cache=True, fastmath=True, parallel=True)
def _metropolized_gibbs_kernel(states, biases, transposed_stack, seed, first_step, steps, sites, beta, block):
    num_replicas, n, length = states.shape
    q = biases.shape[2]
    blocks = (n + block - 1) // block
    for task in prange(num_replicas * blocks):
        r, b = task // blocks, task % blocks
        x, h, transposed = states[r], biases[r], transposed_stack[r]
        field = np.empty(q, dtype=h.dtype)
        for t in range(first_step, first_step + steps):
            site = _site(seed, t, r, sites, length)
            for chain in range(b * block, min(n, (b + 1) * block)):
                _field(x, chain, h, transposed, site, field)
                x[chain, site] = _metropolized_gibbs_choose(field, q, beta, x[chain, site], seed, t, r * n + chain)


@njit(cache=True, fastmath=True, parallel=True)
def _metropolized_gibbs_sparse_kernel(states, biases, nbr_ptr, nbrs, coupling_blocks, seed, first_step, steps, sites,
                                      beta, block):
    num_replicas, n, length = states.shape
    q = biases.shape[2]
    blocks = (n + block - 1) // block
    for task in prange(num_replicas * blocks):
        r, b = task // blocks, task % blocks
        x, h, pointers = states[r], biases[r], nbr_ptr[r]
        field = np.empty(q, dtype=h.dtype)
        for t in range(first_step, first_step + steps):
            site = _site(seed, t, r, sites, length)
            for chain in range(b * block, min(n, (b + 1) * block)):
                _sparse_field(x, chain, h, pointers, nbrs, coupling_blocks, site, field)
                x[chain, site] = _metropolized_gibbs_choose(field, q, beta, x[chain, site], seed, t, r * n + chain)


@njit(cache=True, parallel=True)
def _profile_kernel(cdf, n, seed):
    length, q = cdf.shape
    states = np.empty((n, length), dtype=np.int32)
    for chain in prange(n):
        for site in range(length):
            u = uniform(seed, chain, site, SLOT_PROFILE)
            chosen = q - 1
            for a in range(q):
                if u < cdf[site, a]:
                    chosen = a
                    break
            states[chain, site] = chosen
    return states


def _block_size(num_chains: int, threads: int) -> int:
    """Chains per block: enough blocks to keep every thread busy, at most 32 per block."""
    return max(1, min(32, num_chains // (4 * threads)))


def kernel_couplings(method: str, couplings: torch.Tensor) -> torch.Tensor:
    """Couplings ``(..., L, q, L, q)`` in the layout the kernel of ``method`` reads.

    Metropolis reads two fields per update and keeps the original layout; Gibbs
    and Metropolized Gibbs read all ``q`` fields, contiguous in the transposed
    layout ``(..., L, L, q, q)`` with ``[i, j, b, a] = J[i, a, j, b]``.
    """
    if method != "metropolis":
        couplings = couplings.movedim(-3, -1)
    return couplings.contiguous()


def _check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"Unknown sampling method {method!r}; choose from {METHODS}.")


def _check_parameters(biases: torch.Tensor) -> None:
    if biases.device.type != "cpu" or biases.dtype not in (torch.float32, torch.float64):
        raise ValueError("The Numba kernels need float32 or float64 CPU parameters.")


# Sparse kernels (disabled by ADABMDCA_SPARSE=0) pay off while at most this
# fraction of the site pairs is coupled: on RF00379 and LBD models they were 3-7
# times faster than the dense kernels at 5-15% of the pairs, as fast at 70%, and
# 0.8 times as fast on full graphs.
SPARSE_MAX_PAIR_DENSITY = 0.5

def _sparse_layout(couplings: torch.Tensor):
    """Neighbour lists and coupling blocks of stacked couplings ``(R, L, q, L, q)``, or ``None`` if too dense.

    Returns ``nbr_ptr`` ``(R, L + 1)`` (offsets of each site's neighbours into the
    global arrays), ``nbrs`` ``(E,)`` (the neighbour sites) and ``blocks``
    ``(E, q, q)`` with ``blocks[s, b, a] = J[i, a, j, b]`` for neighbour slot ``s``
    of site ``i`` with neighbour ``j``.
    """
    if not sparse_kernels_enabled():
        return None
    num_replicas, length = couplings.shape[0], couplings.shape[1]
    pair = (couplings != 0).any(dim=4).any(dim=2)
    # The densest replica decides, so that no replica of a stack runs slower than with dense kernels.
    if num_replicas and float(pair.float().mean(dim=(1, 2)).max()) > SPARSE_MAX_PAIR_DENSITY:
        return None
    replica, site, neighbour = torch.nonzero(pair, as_tuple=True)
    offsets = torch.zeros(num_replicas * length + 1, dtype=torch.int64)
    offsets[1:] = pair.sum(-1).reshape(-1).cumsum(0)
    nbr_ptr = torch.stack([offsets[r * length:(r + 1) * length + 1] for r in range(num_replicas)])
    blocks = couplings[replica, site, :, neighbour, :].transpose(1, 2).contiguous()
    return nbr_ptr.numpy(), neighbour.to(torch.int32).numpy(), blocks.numpy()


def sparse_layout(couplings: torch.Tensor, source: torch.Tensor | None = None):
    """The sparse layout of stacked couplings ``(R, L, q, L, q)`` (see ``_sparse_layout``), or ``None`` if dense.

    Cached per ``source`` tensor (``couplings`` by default); shared by sampling and exchanges.
    """
    return cached(couplings if source is None else source, "numba_sparse", lambda: _sparse_layout(couplings))


def _prepared_couplings(method, couplings, source=None):
    """Dense kernel-layout or sparse arrays for stacked couplings ``(R, L, q, L, q)``, cached per tensor.

    ``source`` is the caller's tensor that identifies the couplings (``couplings``
    itself by default); the preparation is reused while it is alive and unmodified.
    """
    source = couplings if source is None else source
    layout = sparse_layout(couplings, source)
    if layout is not None:
        return ("sparse", *layout)
    if method == "metropolis":
        return ("dense", couplings.contiguous().numpy())
    return ("dense", cached(source, "numba_transposed", lambda: kernel_couplings(method, couplings).numpy()))


def _run(method, states, biases, prepared, num_steps, sites, beta) -> None:
    """Update stacked int32 ``states`` (R, N, L) in place with couplings from :func:`_prepared_couplings`."""
    num_replicas, num_chains, _ = states.shape
    if num_replicas == 0 or num_chains == 0 or num_steps == 0:
        return
    block = min(num_chains, _block_size(num_replicas * num_chains, set_kernel_threads()))
    arguments = (draw_seed(), 0, num_steps, sites, beta, block)
    if prepared[0] == "dense":
        kernel = {"gibbs": _gibbs_kernel, "metropolis": _metropolis_kernel,
                  "metropolized_gibbs": _metropolized_gibbs_kernel}[method]
        kernel(states.numpy(), biases.numpy(), prepared[1], *arguments)
    else:
        kernel = {"gibbs": _gibbs_sparse_kernel, "metropolis": _metropolis_sparse_kernel,
                  "metropolized_gibbs": _metropolized_gibbs_sparse_kernel}[method]
        kernel(states.numpy(), biases.numpy(), *prepared[1:], *arguments)


def sample_categorical(
    method: str,
    states: torch.Tensor,
    params: dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
    *,
    sites: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``nsweeps * L`` random-site updates of integer states.

    Args:
        method: ``"gibbs"``, ``"metropolis"`` or ``"metropolized_gibbs"``.
        states: CPU integer tensor of shape ``(N, L)``; it is not modified.
        params: ``bias`` ``(L, q)`` and ``coupling_matrix`` ``(L, q, L, q)``, float32
            or float64, with zero within-site coupling blocks. Sparse graphs (at
            most ``SPARSE_MAX_PAIR_DENSITY`` of the site pairs coupled) are sampled
            with kernels that read only the coupled pairs.
        nsweeps: Sweeps of ``L`` updates each.
        beta: Inverse temperature.
        sites: Optional site of every update, overriding the uniform random draws
            (``nsweeps`` is then ignored).

    Returns:
        The updated states, an int32 tensor of shape ``(N, L)``.
    """
    _check_method(method)
    bias = params["bias"].detach().contiguous()
    _check_parameters(bias)
    source = params["coupling_matrix"]
    prepared = _prepared_couplings(method, source.detach().to(bias.dtype)[None], source=source)
    result = states.detach().to(device="cpu", dtype=torch.int32).contiguous().clone()
    explicit = (np.empty(0, dtype=np.int32) if sites is None
                else sites.detach().to(device="cpu", dtype=torch.int32).contiguous().numpy())
    num_steps = nsweeps * result.shape[1] if sites is None else len(explicit)
    _run(method, result[None], bias[None], prepared, num_steps, explicit, beta)
    return result


def sample_replicas(
    method: str,
    states: torch.Tensor,
    biases: torch.Tensor,
    couplings: torch.Tensor,
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Run ``nsweeps * L`` random-site updates in each of a stack of models.

    Replica ``r`` updates ``states[r]`` under ``biases[r]`` and ``couplings[r]``
    with its own random sites; one call parallelizes over replicas and chains.

    Args:
        method: ``"gibbs"``, ``"metropolis"`` or ``"metropolized_gibbs"``.
        states: CPU integer tensor of shape ``(R, N, L)``; it is not modified.
        biases: Float32 or float64 tensor of shape ``(R, L, q)``.
        couplings: Stacked couplings ``(R, L, q, L, q)`` with the dtype of
            ``biases``. Their kernel layout (dense, or sparse when few site pairs
            are coupled) is prepared once and reused while the tensor is unchanged.
        nsweeps: Sweeps of ``L`` updates each.
        beta: Inverse temperature.

    Returns:
        The updated states, an int32 tensor of shape ``(R, N, L)``.
    """
    _check_method(method)
    biases = biases.detach().contiguous()
    _check_parameters(biases)
    if states.ndim != 3 or biases.shape[:2] != (states.shape[0], states.shape[2]):
        raise ValueError("Replica states must have shape (R, N, L) and biases (R, L, q).")
    num_replicas, _, length = states.shape
    q = biases.shape[2]
    if couplings.shape != (num_replicas, length, q, length, q) or couplings.dtype != biases.dtype:
        raise ValueError(f"Replica couplings must have shape {(num_replicas, length, q, length, q)} and the dtype "
                         "of the biases.")
    prepared = _prepared_couplings(method, couplings.detach().contiguous(), source=couplings)
    result = states.detach().to(device="cpu", dtype=torch.int32).contiguous().clone()
    _run(method, result, biases, prepared, nsweeps * length, np.empty(0, dtype=np.int32), beta)
    return result


def sample_profile_states(bias: torch.Tensor, num_chains: int, beta: float = 1.0) -> torch.Tensor:
    """Exact samples of the independent-site model ``p_i(a) ∝ exp(beta * h_i(a))`` as int32 states."""
    probabilities = torch.softmax(beta * bias.detach().double().cpu(), dim=-1)
    cdf = probabilities.cumsum(-1)
    cdf[:, -1] = 1.0
    set_kernel_threads()
    return torch.from_numpy(_profile_kernel(cdf.numpy(), num_chains, draw_seed()))


def categorical_sampler(method: str) -> Callable[..., torch.Tensor]:
    """A sampler of integer states with the signature ``(states, params, nsweeps, beta=1.0)``."""

    def sampler(states, params, nsweeps, beta=1.0, *, sites=None):
        return sample_categorical(method, states, params, nsweeps, beta, sites=sites)

    sampler.__name__ = f"{method}_sampling_numba"
    return sampler


def replica_sampler(method: str) -> Callable[..., torch.Tensor]:
    """A stacked-replica sampler ``(states, biases, couplings, nsweeps, beta=1.0)``, see :func:`sample_replicas`."""

    def sampler(states, biases, couplings, nsweeps, beta=1.0):
        return sample_replicas(method, states, biases, couplings, nsweeps, beta)

    sampler.__name__ = f"{method}_sampling_replicas_numba"
    return sampler


def onehot_sampler(method: str) -> Callable[..., torch.Tensor]:
    """A sampler of one-hot chains ``(N, L, q)``, the interface of :mod:`adabmDCA.sampling`."""

    def sampler(chains, params, nsweeps, beta=1.0):
        states = sample_categorical(method, chains.argmax(-1), params, nsweeps, beta)
        return torch.nn.functional.one_hot(states.long(), chains.shape[-1]).to(chains.dtype)

    sampler.__name__ = f"{method}_sampling_numba"
    return sampler
