from typing import Dict, Callable

import torch
from torch.nn.functional import one_hot
from adabmDCA.functional import multinomial_one_hot


def gibbs_mutate(
    chains: torch.Tensor,
    num_mut: int,
    params: Dict[str, torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """Attempts to perform num_mut mutations using the Gibbs sampler.

    Args:
        chains (torch.Tensor): One-hot encoded sequences of shape (batch_size, L, q).
        num_mut (int): Number of proposed mutations at random sites.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float): Inverse temperature.

    Returns:
        torch.Tensor: Updated chains.
    """
    N, L, q = chains.shape
    idx_array = torch.randint(0, L, (num_mut,), device=chains.device)
    for i in idx_array:
        # Select the couplings attached to the residue we are considering (i) and flatten along the other residues ({j})
        couplings_residue = params["coupling_matrix"][i].view(q, L * q)
        # Update the chains
        logit_residue = beta * (params["bias"][i].unsqueeze(0) + chains.reshape(N, L * q) @ couplings_residue.T) # (N, q)
        chains[:, i, :] = multinomial_one_hot(logit_residue)
        
    return chains


def gibbs_sampling(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Gibbs sampling.
    
    Args:
        chains (torch.Tensor): Initial one-hot encoded chains of size (batch_size, L, q).
        params (Dict[str, torch.Tensor]): Parameters of the model.
        nsweeps (int): Number of sweeps, where one sweep corresponds to attempting L mutations.
        beta (float, optional): Inverse temperature. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Updated chains.
    """
    L = params["bias"].shape[0]
    chains_mutate = chains.clone() # avoids to modify the chains inplace
    for _ in torch.arange(nsweeps):
        chains_mutate = gibbs_mutate(chains_mutate, L, params, beta)

    return chains_mutate


def _get_deltaE(
        idx: int | torch.Tensor,
        chain: torch.Tensor,
        residue_old: torch.Tensor,
        residue_new: torch.Tensor,
        params: Dict[str, torch.Tensor],
        L: int,
        q: int,
    ) -> torch.Tensor:
    """Computes the energy difference between the new and old residue at position i.
    
    Args:
        idx (int | torch.Tensor): Index of the residue to mutate.
        chain (torch.Tensor): Current chain.
        residue_old (torch.Tensor): One-hot encoded representation of the old residue.
        residue_new (torch.Tensor): One-hot encoded representation of the new residue.
        params (Dict[str, torch.Tensor]): Model parameters.
        L (int): Length of the sequence.
        q (int): Number of possible states.
    
    Returns:
        torch.Tensor: Energy difference between the new and old residue.
    """
    coupling_residue = chain.view(-1, L * q) @ params["coupling_matrix"][:, :, idx, :].view(L * q, q) # (N, q)
    E_old = - residue_old @ params["bias"][idx] - torch.vmap(torch.dot, in_dims=(0, 0))(coupling_residue, residue_old)
    E_new = - residue_new @ params["bias"][idx] - torch.vmap(torch.dot, in_dims=(0, 0))(coupling_residue, residue_new)
    
    return E_new - E_old


def metropolis_mutate(
    chains: torch.Tensor,
    num_mut: int,
    params: Dict[str, torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """Attempts to perform num_mut mutations using the Metropolis sampler.

    Args:
        chains (torch.Tensor): One-hot encoded sequences of shape (batch_size, L, q).
        num_mut (int): Number of proposed mutations at random sites.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float): Inverse temperature.

    Returns:
        torch.Tensor: Updated chains.
    """
    N, L, q = chains.shape
    idx_array = torch.randint(0, L, (num_mut,), device=chains.device)
    for i in idx_array:
        res_old = chains[:, i, :]
        res_new = one_hot(torch.randint(0, q, (N,), device=chains.device), num_classes=q).type(chains.dtype)
        delta_E = _get_deltaE(i, chains, res_old, res_new, params, L, q)
        accept_prob = torch.exp(- beta * delta_E).unsqueeze(-1)
        chains[:, i, :] = torch.where(accept_prob > torch.rand((N, 1), device=chains.device, dtype=chains.dtype), res_new, res_old)

    return chains
    

def metropolis(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Metropolis sampling.

    Args:
        chains (torch.Tensor): One-hot encoded sequences of shape (batch_size, L, q).
        params (Dict[str, torch.Tensor]): Parameters of the model.
        nsweeps (int): Number of sweeps to be performed, where one sweep corresponds to attempting L mutations.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        torch.Tensor: Updated chains.
    """
    L = params["bias"].shape[0]
    chains_mutate = chains.clone() # avoids to modify the chains inplace
    for _ in range(nsweeps):
        chains_mutate = metropolis_mutate(chains_mutate, L, params, beta)

    return chains_mutate


def get_sampler(sampling_method: str) -> Callable:
    """Returns the sampling function corresponding to the chosen method.

    Args:
        sampling_method (str): String indicating the sampling method. Choose between 'metropolis' and 'gibbs'.

    Raises:
        KeyError: Unknown sampling method.

    Returns:
        Callable: Sampling function.
    """
    if sampling_method == "gibbs":
        return gibbs_sampling
    elif sampling_method == "metropolis":
        return metropolis
    else:
        raise KeyError("Unknown sampling method. Choose between 'metropolis' and 'gibbs'.")
