from typing import Dict, Tuple

import torch

from adabmDCA.sampling import gibbs_sampling


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
    #chains = gibbs_sampling(chains, params=curr_params, nsweeps=1)
    energy_prev = compute_energy(chains, prev_params)
    energy_curr = compute_energy(chains, curr_params)
    log_weights += energy_prev - energy_curr
    
    return log_weights#, chains


@torch.jit.script
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
    mean_energy_data = - torch.sum(fi * params["bias"]) - 0.5 * torch.sum(fij * params["coupling_matrix"])
    
    return - mean_energy_data - logZ