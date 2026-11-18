import warnings
import weakref
from typing import Dict

import torch

_SYMMETRY_CACHE: dict[int, tuple] = {}
_MAX_ENUMERATED_STATES = 5**11
_DEFAULT_STATE_BATCH_SIZE = 4096


def couplings_are_symmetric(couplings: torch.Tensor) -> bool:
    """Whether ``J[i, a, j, b] == J[j, b, i, a]`` exactly, as for every trained Potts model.

    The answer is cached per tensor and recomputed after in-place changes, so
    kernels can ask on every call. Kernels use it to sum pair terms once.
    """
    if couplings.is_inference():
        return torch.equal(couplings, couplings.permute(2, 3, 0, 1))
    version = couplings._version
    entry = _SYMMETRY_CACHE.get(id(couplings))
    if entry is not None and entry[0]() is couplings and entry[1] == version:
        return entry[2]
    symmetric = bool(torch.equal(couplings, couplings.permute(2, 3, 0, 1)))
    key = id(couplings)
    reference = weakref.ref(couplings, lambda _, key=key: _SYMMETRY_CACHE.pop(key, None))
    _SYMMETRY_CACHE[key] = (reference, version, symmetric)
    return symmetric


def compute_energy(
    x: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute the DCA energy for a batch of sequences.
    
    Args:
        x (torch.Tensor): Tensor of shape (batch_size, L, q) - batch of one-hot encoded sequences.
        params (Dict[str, torch.Tensor]): Parameters of the model.
            - "bias": Tensor of shape (L, q) - local biases.
            - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
        
    
    Returns:
        torch.Tensor: Tensor of shape (batch_size,) - DCA energy for each sequence in the batch.
    """
    L, q = params["bias"].shape
    x_flat = x.reshape(x.shape[0], L * q)
    bias_flat = params["bias"].reshape(-1)
    couplings_flat = params["coupling_matrix"].reshape(L * q, L * q)
    bias_term = x_flat @ bias_flat
    coupling_term = torch.sum(x_flat * (x_flat @ couplings_flat), dim=1)
    energy = - bias_term - 0.5 * coupling_term
    
    return energy


def get_cde(
    x: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute per-site context-dependent entropy of one-hot sequences.

    For each site ``i``, hold all other residues fixed, evaluate the model
    probability of every state at ``i``, and return the Shannon entropy of
    that conditional distribution in nats. The conditional probabilities
    follow the same energy convention as :func:`compute_energy`, including
    asymmetric couplings and nonzero same-site terms.

    Args:
        x: One-hot sequence of shape ``(L, q)`` or batch of shape
            ``(N, L, q)``. It is moved to the model's device and dtype.
        params: Model parameters with ``bias`` of shape ``(L, q)`` and
            ``coupling_matrix`` of shape ``(L, q, L, q)``.

    Returns:
        Tensor of shape ``(L,)`` for one sequence or ``(N, L)`` for a batch.

    Raises:
        ValueError: If the parameter or sequence dimensions are incompatible,
            or ``x`` is not one-hot encoded.
    """
    bias = params["bias"]
    couplings = params["coupling_matrix"]
    if bias.ndim != 2:
        raise ValueError("bias must have shape (L, q).")
    length, states = bias.shape
    if couplings.shape != (length, states, length, states):
        raise ValueError("coupling_matrix must have shape (L, q, L, q).")
    if x.ndim not in (2, 3) or x.shape[-2:] != (length, states):
        raise ValueError("x must have shape (L, q) or (N, L, q) matching bias.")

    single_sequence = x.ndim == 2
    sequences = x.unsqueeze(0) if single_sequence else x
    if not (((sequences == 0) | (sequences == 1)).all() & ((sequences == 1).sum(dim=-1) == 1).all()):
        raise ValueError("x must contain finite one-hot encoded sequences.")
    sequences = sequences.to(device=bias.device, dtype=bias.dtype)

    sites = torch.arange(length, device=couplings.device)
    same_site = couplings[sites, :, sites, :]
    outgoing = torch.einsum("iajb,njb->nia", couplings, sequences)
    if (
        not (torch.is_grad_enabled() and couplings.requires_grad)
        and not torch.count_nonzero(same_site)
        and couplings_are_symmetric(couplings)
    ):
        logits = bias + outgoing
    else:
        # Remove the current state's same-site terms and add the candidate's diagonal term.
        incoming = torch.einsum("jbia,njb->nia", couplings, sequences)
        own_outgoing = torch.einsum("iab,nib->nia", same_site, sequences)
        own_incoming = torch.einsum("iba,nib->nia", same_site, sequences)
        candidate_diagonal = same_site.diagonal(dim1=1, dim2=2)
        logits = bias + 0.5 * (outgoing + incoming - own_outgoing - own_incoming + candidate_diagonal)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    cde = -(probabilities * log_probabilities).sum(dim=-1)
    return cde[0] if single_sequence else cde


def _compute_log_likelihood(
    fi: torch.Tensor,
    fij: torch.Tensor,
    params: Dict[str, torch.Tensor],
    logZ: float,
) -> float:
    
    mean_energy_data = - torch.sum(fi * params["bias"]) - 0.5 * torch.sum(fij * params["coupling_matrix"])
    L = params["bias"].shape[0]

    return (- mean_energy_data.item() - logZ) / L


def compute_log_likelihood(
    fi: torch.Tensor,
    fij: torch.Tensor,
    params: Dict[str, torch.Tensor],
    logZ: float,
) -> float:
    """Compute the log-likelihood per residue of the model.

    Args:
        fi (torch.Tensor): Single-site frequencies of the data.
        fij (torch.Tensor): Two-site frequencies of the data.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        logZ (float): Log-partition function of the model.

    Returns:
        float: Log-likelihood per residue of the model.
    """
    return _compute_log_likelihood(fi, fij, params, logZ)


def _enumeration_size(length: int, states: int) -> int:
    if length < 1 or states < 1:
        raise ValueError("L and q must be positive.")
    total = states**length
    if total > _MAX_ENUMERATED_STATES:
        raise ValueError("The number of states is too large to enumerate.")
    return total


def _state_batches(length, states, total, batch_size, device):
    divisors = torch.tensor([states**i for i in reversed(range(length))], device=device)
    for start in range(0, total, batch_size):
        indices = torch.arange(start, min(start + batch_size, total), device=device)
        yield (indices[:, None] // divisors).remainder(states)


def enumerate_states(
    L: int,
    q: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Enumerate all possible states of a system of L sites and q states.

    Args:
        L (int): Number of sites.
        q (int): Number of states.
        device (torch.device, optional): Device to store the states. Defaults to "cpu".

    Returns:
        torch.Tensor: All possible states as float32 one-hot vectors. To compute an
            exact partition function without storing this tensor, use
            ``compute_logZ_exact(params=params)``.
    """
    total = _enumeration_size(L, q)
    all_states = torch.empty((total, L, q), dtype=torch.float32, device=device)
    for start, states in enumerate(_state_batches(L, q, total, _DEFAULT_STATE_BATCH_SIZE, device)):
        begin = start * _DEFAULT_STATE_BATCH_SIZE
        all_states[begin:begin + len(states)].zero_().scatter_(2, states.unsqueeze(-1), 1)
    return all_states


def compute_logZ_exact(
    all_states: torch.Tensor | None = None,
    params: Dict[str, torch.Tensor] | None = None,
    *,
    batch_size: int = _DEFAULT_STATE_BATCH_SIZE,
) -> float:
    """Compute the log-partition function of the model.

    Args:
        all_states: All possible states as one-hot vectors. If omitted, states
            are generated from the model dimensions in batches.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        batch_size: Maximum number of states to score together.

    Returns:
        float: Log-partition function of the model.
    """
    if params is None:
        raise TypeError("params is required.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    bias = params["bias"]
    length, states = bias.shape
    if all_states is None:
        total = _enumeration_size(length, states)
        batches = (
            bias.new_zeros((len(batch), length, states)).scatter_(2, batch.unsqueeze(-1), 1)
            for batch in _state_batches(length, states, total, batch_size, bias.device)
        )
    else:
        batches = (batch.to(bias) for batch in all_states.split(batch_size))
    dtype = torch.float64 if bias.device.type != "mps" else torch.float32
    log_z = torch.full((), -torch.inf, dtype=dtype, device=bias.device)
    with torch.no_grad():
        for batch in batches:
            block_log_z = torch.logsumexp(-compute_energy(batch, params).to(dtype), dim=0)
            log_z = torch.logaddexp(log_z, block_log_z)
    return log_z.item()


def compute_entropy(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    logZ: float,
) -> float:
    """Compute the entropy of the DCA model.

    Args:
        chains (torch.Tensor): Chains that are supposed to be an equilibrium realization of the model.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        logZ (float): Log-partition function of the model.

    Returns:
        float: Entropy of the model.
    """
    mean_energy = compute_energy(chains, params).mean()
    entropy = mean_energy + logZ
    
    return entropy.item()


def exchange_log_acceptance(prev_params, curr_params, prev_chains, curr_chains):
    """Deterministic log Metropolis acceptance for Hamiltonian exchange."""
    delta = {key: curr_params[key] - prev_params[key] for key in ("bias", "coupling_matrix")}
    dtype = torch.float32 if prev_chains.device.type == "mps" else torch.float64
    prev_energy = compute_energy(prev_chains, delta).to(dtype)
    curr_energy = compute_energy(curr_chains, delta).to(dtype)
    return (curr_energy - prev_energy).clamp_max(0.0)


def _tap_residue(
    idx: int,
    mag: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    N, L, q = mag.shape
    coupling_residue = params["coupling_matrix"][idx] # (q, L, q)
    bias_residue = params["bias"][idx] # (q,)
    mag_i = mag[:, idx] # (n, q)
    
    mf_term = bias_residue + mag.view(N, L * q) @ coupling_residue.reshape(q, L * q).T
    reaction_term_temp = (
        0.5 * coupling_residue.view(1, q, L, q) + # (1, q, L, q)
        (torch.einsum("nd,djc,njc->nj", mag_i, coupling_residue, mag)).view(N, 1, L, 1) - # nd,djc,njc->nj
        0.5 * torch.einsum("njc,ajc->naj", mag, coupling_residue).view(N, q, L, 1) -      # njc,ajc->naj
        torch.einsum("nd,djb->njb", mag_i, coupling_residue).view(N, 1, L, q)             # nd,djb->njb
    )
    reaction_term = (
        (reaction_term_temp * coupling_residue.view(1, q, L, q)) * mag.view(N, 1, L, q)
    ).sum(dim=3).sum(dim=2) # najb,ajb,njb->na
    tap_residue = torch.softmax(mf_term + reaction_term, dim=1)
    
    return tap_residue


def _sweep_tap(
    residue_idxs: torch.Tensor,
    mag: torch.Tensor,
    params: Dict[str, torch.Tensor],    
) -> torch.Tensor:
    """Updates the magnetizations using the TAP equations.

    Args:
        residue_idxs (torch.Tensor): List of residue indices in random order.
        mag (torch.Tensor): Magnetizations of the residues.
        params (Dict[str, torch.Tensor]): Parameters of the model.

    Returns:
        torch.Tensor: Updated magnetizations.
    """
    for idx in residue_idxs:
        mag[:, idx] = _tap_residue(idx, mag, params)  
    
    return mag


def iterate_tap(
    mag: torch.Tensor,
    params: Dict[str, torch.Tensor],
    max_iter: int = 500,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Iterate the TAP equations up to the sweep limit.

    Args:
        mag (torch.Tensor): Initial magnetizations.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        max_iter (int, optional): Maximum number of iterations. Defaults to 500.
        epsilon (float, optional): Convergence threshold. Defaults to 1e-4.

    Returns:
        torch.Tensor: The last magnetizations. A warning is emitted if a positive
            sweep limit is reached before the convergence threshold.
    """
    # ensure that mag is a 3D tensor
    if mag.dim() != 3:
        raise ValueError("Input tensor mag must be 3-dimensional of size (_, L, q)")
    
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative.")
    mag_ = mag.clone(memory_format=torch.contiguous_format)
    if max_iter == 0 or mag_.numel() == 0:
        return mag_

    bias = params["bias"]
    couplings = params["coupling_matrix"]
    if bias.shape != mag.shape[1:] or couplings.shape != (*bias.shape, *bias.shape):
        raise ValueError("Parameter shapes must match the magnetizations.")
    rows = 0.5 * (couplings + couplings.permute(2, 3, 0, 1))
    sites = torch.arange(bias.shape[0], device=bias.device)
    diagonal = rows[sites, :, sites, :].diagonal(dim1=-2, dim2=-1)
    tap_params = {"bias": bias + 0.5 * diagonal, "coupling_matrix": rows}
    rows[sites, :, sites, :] = 0

    for _ in range(max_iter):
        mag_old = mag_.clone()
        mag_ = _sweep_tap(torch.randperm(mag_.shape[1], device=mag_.device), mag_, tap_params)
        diff = torch.abs(mag_old - mag_).max()
        if diff < epsilon:
            return mag_
    warnings.warn(f"TAP did not converge after {max_iter} sweeps.", RuntimeWarning, stacklevel=2)
    return mag_
