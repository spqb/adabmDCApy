from typing import Dict
import itertools

import torch

from adabmDCA.functional import one_hot


@torch.jit.script
def _compute_energy_sequence(
    x: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    L, q = params["bias"].shape
    x_oh = x.ravel()
    bias_oh = params["bias"].ravel()
    couplings_oh = params["coupling_matrix"].view(L * q, L * q)
    energy = - x_oh @ bias_oh - 0.5 * x_oh @ (couplings_oh @ x_oh)
    
    return energy


def compute_energy(
    X: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute the DCA energy of the sequences in X.
    
    Args:
        X (torch.Tensor): Sequences in one-hot encoding format.
        params (Dict[str, torch.Tensor]): Parameters of the model.
    
    Returns:
        torch.Tensor: DCA Energy of the sequences.
    """
    
    if X.dim() != 3:
        raise ValueError("Input tensor X must be 3-dimensional of size (batch_size, L, q)")
    
    return torch.vmap(_compute_energy_sequence, in_dims=(0, None))(X, params)


def _update_weights_AIS(
    prev_params: Dict[str, torch.Tensor],
    curr_params: Dict[str, torch.Tensor],
    chains: torch.Tensor,
    log_weights: torch.Tensor,
) -> torch.Tensor:
    """Update the weights used during  the trajectory Annealed Importance Sampling (AIS) algorithm.

    Args:
        prev_params (Dict[str, torch.Tensor]): Params at time t-1.
        curr_params (Dict[str, torch.Tensor]): Params at time t.
        chains (torch.Tensor): Chains at time t-1.
        log_weights (torch.Tensor): Log-weights at time t-1.

    Returns:
        torch.Tensor: Log-weights at time t.
    """
    energy_prev = compute_energy(chains, prev_params)
    energy_curr = compute_energy(chains, curr_params)
    log_weights += energy_prev - energy_curr
    
    return log_weights


def _compute_ess(log_weights: torch.Tensor) -> float:
    """Computes the Effective Sample Size of the chains.

    Args:
        log_weights: log-weights of the chains.
    """
    lwc = log_weights - log_weights.min()
    numerator = torch.square(torch.mean(torch.exp(-lwc))).item()
    denominator = torch.mean(torch.exp(-2.0 * lwc)).item()

    return numerator / denominator


@torch.jit.script
def _compute_log_likelihood(
    fi: torch.Tensor,
    fij: torch.Tensor,
    params: Dict[str, torch.Tensor],
    logZ: float,
) -> float:
    
    mean_energy_data = - torch.sum(fi * params["bias"]) - 0.5 * torch.sum(fij * params["coupling_matrix"])

    return - mean_energy_data.item() - logZ


def compute_log_likelihood(
    fi: torch.Tensor,
    fij: torch.Tensor,
    params: Dict[str, torch.Tensor],
    logZ: float,
) -> float:
    """Compute the log-likelihood of the model.

    Args:
        fi (torch.Tensor): Single-site frequencies of the data.
        fij (torch.Tensor): Two-site frequencies of the data.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        logZ (float): Log-partition function of the model.

    Returns:
        float: Log-likelihood of the model.
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
    
    all_states = torch.tensor(list(itertools.product(range(q), repeat=L)), device=device)
    return one_hot(all_states, q)


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


def _get_acceptance_rate(
    prev_params: Dict[str, torch.Tensor],
    curr_params: Dict[str, torch.Tensor],
    prev_chains: torch.Tensor,
    curr_chains: torch.Tensor,
) -> float:
    """Compute the acceptance rate of swapping the configurations between two models alonge the training.

    Args:
        prev_params (Dict[str, torch.Tensor]): Parameters at time t-1.
        curr_params (Dict[str, torch.Tensor]): Parameters at time t.
        prev_chains (torch.Tensor): Chains at time t-1.
        curr_chains (torch.Tensor): Chains at time t.

    Returns:
        float: Acceptance rate of swapping the configurations between two models alonge the training.
    """
    nchains = len(prev_chains)
    delta_energy = (
        - compute_energy(curr_chains, prev_params)
        + compute_energy(prev_chains, prev_params)
        + compute_energy(curr_chains, curr_params)
        - compute_energy(prev_chains, curr_params)
    )
    swap = torch.exp(delta_energy) > torch.rand(size=(nchains,), device=delta_energy.device)
    acceptance_rate = swap.float().mean().item()
    
    return acceptance_rate


@torch.jit.script
def _tap_residue(
    idx: int,
    mag: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    N, L, q = mag.shape
    coupling_residue = params["coupling_matrix"][idx] # (q, L, q)
    bias_residue = params["bias"][idx] # (q,)
    mag_i = mag[:, idx] # (n, q)
    
    mf_term = bias_residue + mag.view(N, L * q) @ coupling_residue.view(q, L * q).T
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
):
    """Iterates the TAP equations until convergence.

    Args:
        mag (torch.Tensor): Initial magnetizations.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
        epsilon (float, optional): Convergence threshold. Defaults to 1e-6.

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
        mag_ = _sweep_tap(torch.randperm(mag_.shape[1]), mag_, params)
        diff = torch.abs(mag_old - mag_).max()
        iterations += 1
        if diff < epsilon or iterations > max_iter:
            break
    
    return mag_