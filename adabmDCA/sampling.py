from typing import Dict
import torch
from adabmDCA.custom_fn import one_hot
from torch.nn.functional import one_hot as one_hot_torch


@torch.jit.script
def gibbs_sweep(
    chains: torch.Tensor,
    residue_idxs: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """Performs a Gibbs sweep over the chains.

    Args:
        chains (torch.Tensor): One-hot encoded sequences.
        residue_idxs (torch.Tensor): Indices of the residues to update.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float): Inverse temperature.

    Returns:
        torch.Tensor: Updated chains.
    """
    N, L, q = chains.shape
    for i in residue_idxs:
        # Select the couplings attached to the residue we are considering (i) and flatten along the other residues ({j})
        couplings_residue = params["coupling_matrix"][i].view(q, L * q)
        # Update the chains
        logit_residue = beta * (params["bias"][i].unsqueeze(0) + chains.reshape(N, L * q) @ couplings_residue.T) # (N, q)
        chains[:, i, :] = one_hot(torch.multinomial(torch.softmax(logit_residue, -1), 1), num_classes=q).squeeze(1)
        
    return chains


def gibbs_sampling(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Gibbs sampling.
    
    Args:
        chains (torch.Tensor): Initial chains.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        nsweeps (int): Number of sweeps.
        beta (float, optional): Inverse temperature. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Updated chains.
    """
    L = params["bias"].shape[0]
    
    for t in torch.arange(nsweeps):
        # Random permutation of the residues
        residue_idxs = torch.randperm(L)
        chains = gibbs_sweep(chains, residue_idxs, params, beta)
        
    return chains


def get_deltaE(
        idx: int,
        chain: torch.Tensor,
        residue_old: torch.Tensor,
        residue_new: torch.Tensor,
        params: Dict[str, torch.Tensor],
        L: int,
        q: int,
    ) -> float:
    
        coupling_residue = chain.view(-1, L * q) @ params["coupling_matrix"][:, :, idx, :].view(L * q, q) # (N, q)
        E_old = - residue_old @ params["bias"][idx] - torch.vmap(torch.dot, in_dims=(0, 0))(coupling_residue, residue_old)
        E_new = - residue_new @ params["bias"][idx] - torch.vmap(torch.dot, in_dims=(0, 0))(coupling_residue, residue_new)
        
        return E_new - E_old
    

def metropolis_sweep(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """Performs a Metropolis sweep over the chains.

    Args:
        chains (torch.Tensor): One-hot encoded sequences.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float): Inverse temperature.

    Returns:
        torch.Tensor: Updated chains.
    """
    
    N, L, q = chains.shape
    residue_idxs = torch.randperm(L)
    for i in residue_idxs:
        res_old = chains[:, i, :]
        res_new = one_hot_torch(torch.randint(0, q, (N,), device=chains.device), num_classes=q).float()
        delta_E = get_deltaE(i, chains, res_old, res_new, params, L, q)
        accept_prob = torch.exp(- beta * delta_E).unsqueeze(-1)
        chains[:, i, :] = torch.where(accept_prob > torch.rand((N, 1), device=chains.device), res_new, res_old)

    return chains
    

def metropolis(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Metropolis sampling.

    Args:
        chains (torch.Tensor): One-hot encoded sequences.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        nsweeps (int): Number of sweeps to be performed.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        torch.Tensor: Updated chains.
    """

    for _ in range(nsweeps):
        chains = metropolis_sweep(chains, params, beta)

    return chains


def get_sampler(sampling_method: str):
    if sampling_method == "gibbs":
        return gibbs_sampling
    elif sampling_method == "metropolis":
        return metropolis
    else:
        raise KeyError("Unknown sampling method. Choose between 'metropolis' and 'gibbs'.")