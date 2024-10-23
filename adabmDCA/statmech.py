from typing import Dict, Tuple
import itertools

import torch

from adabmDCA.functional import one_hot


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
        raise ValueError("Input tensor X must be 3-dimensional of size (_, L, q)")
    
    @torch.jit.script
    def compute_energy_sequence(x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        L, q = params["bias"].shape
        x_oh = x.ravel()
        bias_oh = params["bias"].ravel()
        couplings_oh = params["coupling_matrix"].view(L * q, L * q)
        energy = - x_oh @ bias_oh - 0.5 * x_oh @ (couplings_oh @ x_oh)
        
        return energy
    
    return torch.vmap(compute_energy_sequence, in_dims=(0, None))(X, params)


def update_weights_AIS(
    prev_params: Dict[str, torch.Tensor],
    curr_params: Dict[str, torch.Tensor],
    chains: torch.Tensor,
    log_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update the weights used during  the trajectory Annealed Importance Sampling (AIS) algorithm.

    Args:
        prev_params (Dict[str, torch.Tensor]): Params at time t-1.
        curr_params (Dict[str, torch.Tensor]): Params at time t.
        chains (torch.Tensor): Chains at time t-1.
        log_weights (torch.Tensor): Log-weights at time t-1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Log-weights and chains at time t.
    """
    energy_prev = compute_energy(chains, prev_params)
    energy_curr = compute_energy(chains, curr_params)
    log_weights += energy_prev - energy_curr
    
    return log_weights


@torch.jit.script
def compute_log_likelihood_(
    fi: torch.Tensor,
    fij: torch.Tensor,
    params: Dict[str, torch.Tensor],
    logZ: float,
) -> float:
    
    mean_energy_data = - torch.sum(fi * params["bias"]) - 0.5 * torch.sum(fij * params["coupling_matrix"])
    
    return - mean_energy_data - logZ


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
    return compute_log_likelihood_(fi, fij, params, logZ)


def enumerate_states(L: int, q: int, device: torch.device=torch.device("cpu")) -> torch.Tensor:
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