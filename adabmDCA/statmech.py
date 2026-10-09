import itertools
import weakref
from typing import Dict

import torch



_SYMMETRY_CACHE: dict[int, tuple] = {}


def couplings_are_symmetric(couplings: torch.Tensor) -> bool:
    """Whether ``J[i, a, j, b] == J[j, b, i, a]`` exactly, as for every trained Potts model.

    The answer is cached per tensor and recomputed after in-place changes, so
    kernels can ask on every call. Kernels use it to sum pair terms once.
    """
    entry = _SYMMETRY_CACHE.get(id(couplings))
    if entry is not None and entry[0]() is couplings and entry[1] == couplings._version:
        return entry[2]
    symmetric = bool(torch.equal(couplings, couplings.permute(2, 3, 0, 1)))
    key = id(couplings)
    reference = weakref.ref(couplings, lambda _, key=key: _SYMMETRY_CACHE.pop(key, None))
    _SYMMETRY_CACHE[key] = (reference, couplings._version, symmetric)
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
    batch_size = x.shape[0]
    x_flat = x.view(batch_size, -1)
    bias_flat = params["bias"].view(-1)
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
    sequences = sequences.to(device=bias.device, dtype=bias.dtype)
    if not torch.isfinite(sequences).all() or not (
        ((sequences == 0) | (sequences == 1)).all()
        and (sequences.sum(dim=-1) == 1).all()
    ):
        raise ValueError("x must contain finite one-hot encoded sequences.")

    # For an energy -h*x - x*J*x/2, both J[i,a,j,b] and J[j,b,i,a]
    # contribute when site i changes. Remove the current site's state before
    # adding the candidate state's same-site diagonal contribution.
    outgoing = torch.einsum("iajb,njb->nia", couplings, sequences)
    incoming = torch.einsum("jbia,njb->nia", couplings, sequences)
    sites = torch.arange(length, device=couplings.device)
    same_site = couplings[sites, :, sites, :]
    own_outgoing = torch.einsum("iab,nib->nia", same_site, sequences)
    own_incoming = torch.einsum("iba,nib->nia", same_site, sequences)
    candidate_diagonal = same_site.diagonal(dim1=1, dim2=2)
    logits = bias.unsqueeze(0) + 0.5 * (
        outgoing + incoming - own_outgoing - own_incoming + candidate_diagonal.unsqueeze(0)
    )
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
        torch.Tensor: All possible states.
    """
    if q**L > 5**11:
        raise ValueError("The number of states is too large to enumerate.")
    
    all_states = torch.tensor(list(itertools.product(range(q), repeat=L)), device=device).long()
    return torch.nn.functional.one_hot(all_states, q).float()


def compute_logZ_exact(
    all_states: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> float:
    """Compute the log-partition function of the model.

    Args:
        all_states (torch.Tensor): All possible states of the system.
        params (Dict[str, torch.Tensor]): Parameters of the model.

    Returns:
        float: Log-partition function of the model.
    """
    energies = compute_energy(all_states, params)
    logZ = torch.logsumexp(-energies, dim=0)
    
    return logZ.item()


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
    return (
        compute_energy(prev_chains, prev_params).double()
        + compute_energy(curr_chains, curr_params).double()
        - compute_energy(curr_chains, prev_params).double()
        - compute_energy(prev_chains, curr_params).double()
    ).clamp_max(0.0)


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
    """Iterates the TAP equations until convergence.

    Args:
        mag (torch.Tensor): Initial magnetizations.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        max_iter (int, optional): Maximum number of iterations. Defaults to 500.
        epsilon (float, optional): Convergence threshold. Defaults to 1e-4.

    Returns:
        torch.Tensor: Fixed point magnetizations of the TAP equations.
    """
    # ensure that mag is a 3D tensor
    if mag.dim() != 3:
        raise ValueError("Input tensor mag must be 3-dimensional of size (_, L, q)")
    
    mag_ = mag.clone()
    iterations = 0
    while True:
        mag_old = mag_.clone()
        mag_ = _sweep_tap(torch.randperm(mag_.shape[1], device=mag_.device), mag_, params)
        diff = torch.abs(mag_old - mag_).max()
        iterations += 1
        if diff < epsilon or iterations > max_iter:
            break
    
    return mag_
