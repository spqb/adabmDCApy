from typing import Dict, Tuple

import torch

from adabmDCA.utils import get_mask_save


def compute_density(mask: torch.Tensor) -> float:
    """Computes the density of active couplings in the coupling matrix.

    Args:
        mask (torch.Tensor): Mask.

    Returns:
        float: Density.
    """
    L, q, _, _ = mask.shape
    density = mask.sum() / (q**2 * L * (L-1))
    
    return density.item()


# Element-wise activation functions

def compute_Dkl_element_activation(
    fij: torch.Tensor,
    pij: torch.Tensor,
) -> torch.Tensor:
    """Computes the Kullback-Leibler divergence matrix of all the possible couplings.
    
    Args:
        fij (torch.Tensor): Two-point frequences of the dataset.
        pij (torch.Tensor): Two-point marginals of the model.
    
    Returns:
        torch.Tensor: Kullback-Leibler divergence matrix.
    """
    L = fij.shape[0]
    # Compute the Dkl of each coupling
    Dkl = fij * (torch.log(fij) - torch.log(pij)) + (1. - fij) * (torch.log(1. - fij) - torch.log(1. - pij))
    # The auto-correlations have not to be considered
    Dkl[torch.arange(L), :, torch.arange(L), :] = -float("inf")
    
    return Dkl


def select_inactive_elements(
    Dkl: torch.Tensor,
    mask: torch.Tensor,
    fraction: float,
) -> Tuple[torch.Tensor, int, int]:
    """Activates the inactive off-diagonal coupling entries with the largest Dkl.

    Entries are counted once per symmetric pair (i < j). Only inactive entries are
    eligible: active ones are already refitted at every gradient update. At least
    one entry is activated while inactive entries remain, and never more than remain.

    Args:
        Dkl (torch.Tensor): Element-wise Kullback-Leibler divergence matrix.
        mask (torch.Tensor): Symmetric boolean mask of the active couplings.
        fraction (float): Fraction of the inactive unique entries to activate.

    Returns:
        Tuple[torch.Tensor, int, int]: Updated symmetric mask, number of entries
        requested (``int(fraction * inactive)``) and number actually activated.
    """
    candidates = rank_inactive_elements(Dkl, mask)
    inactive = len(candidates)
    requested = int(inactive * fraction)
    count = min(inactive, max(1, requested))
    return activate_elements(mask, candidates[:count]), requested, count


def rank_inactive_elements(Dkl: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Flat indices of the inactive unique (i < j) coupling entries, by decreasing Dkl."""
    L, q = mask.shape[:2]
    eligible = get_mask_save(L, q, device=mask.device) & ~mask.bool()
    candidates = eligible.flatten().nonzero().squeeze(1)
    scores = torch.nan_to_num(Dkl.flatten()[candidates], nan=-float("inf"))
    return candidates[torch.argsort(scores, descending=True, stable=True)]


def activate_elements(mask: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Activates the unique entries at the flat ``indices`` together with their transposes."""
    L, q = mask.shape[:2]
    selected = torch.zeros(L * q * L * q, dtype=torch.bool, device=mask.device)
    selected[indices] = True
    selected = selected.reshape(L, q, L, q)
    return (mask.bool() | selected | selected.permute(2, 3, 0, 1)).to(mask.dtype)


def activate_graph_elements(
    mask: torch.Tensor,
    fij: torch.Tensor,
    pij: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Updates the interaction graph by activating a fraction of the inactive couplings.

    Args:
        mask (torch.Tensor): Mask.
        fij (torch.Tensor): Two-point frequencies of the dataset.
        pij (torch.Tensor): Two-point marginals of the model.
        fraction (float): Fraction of the inactive unique coupling entries to activate.

    Returns:
        torch.Tensor: Updated mask.
    """
    Dkl = compute_Dkl_element_activation(fij=fij, pij=pij)
    mask, _, _ = select_inactive_elements(Dkl=Dkl, mask=mask, fraction=fraction)

    return mask


# Edge-wise activation functions

def compute_Dkl_edge_activation(
    fij: torch.Tensor,
    pij: torch.Tensor,
) -> torch.Tensor:
    """Computes the Kullback-Leibler divergence matrix of all the possible edges.
    
    Args:
        fij (torch.Tensor): Two-point frequences of the dataset.
        pij (torch.Tensor): Two-point marginals of the model.
    
    Returns:
        torch.Tensor: Kullback-Leibler divergence matrix.
    """
    L = fij.shape[0]
    # Compute the Dkl of each edge
    Dkl = torch.sum(fij * (torch.log(fij) - torch.log(pij)), dim=(1, 3))
    # The auto-correlations have not to be considered and the lower triangular part of the Dkl matrix is set to -inf
    Dkl_idx_inf = torch.tril_indices(L, L, offset=0)
    Dkl[Dkl_idx_inf[0], Dkl_idx_inf[1]] = -float("inf")
    
    return Dkl

# Graph decimation functions

def compute_sym_Dkl(
    params: Dict[str, torch.Tensor],
    pij: torch.Tensor,
) -> torch.Tensor:
    """Computes the symmetric Kullback-Leibler divergence matrix between the initial distribution and the same 
    distribution once removing one coupling J_ij(a, b).

    Args:
        params (Dict[str, torch.Tensor]): Parameters of the model.
        pij (torch.Tensor): Two-point marginal probability distribution.

    Returns:
        torch.Tensor: Kullback-Leibler divergence matrix.
    """
    
    exp_J = torch.exp(-params["coupling_matrix"])
    denominator = pij * (exp_J - 1.) + 1.
    # Add small epsilon for numerical stability to avoid division by zero
    denominator = torch.clamp(denominator, min=1e-10)
    Dkl = pij * params["coupling_matrix"] * (1. - exp_J / denominator)
    
    return Dkl


def compute_Dkl_decimation(
    params: Dict[str, torch.Tensor],
    pij: torch.Tensor,
) -> torch.Tensor:
    """Computes the Kullback-Leibler divergence matrix between the initial distribution and the same 
    distribution once removing one coupling J_ij(a, b).

    Args:
        params (Dict[str, torch.Tensor]): Parameters of the model.
        pij (torch.Tensor): Two-point marginal probability distribution.

    Returns:
        torch.Tensor: Kullback-Leibler divergence matrix.
    """
    
    exp_J = torch.exp(-params["coupling_matrix"])
    Dkl = pij * params["coupling_matrix"] + torch.log(exp_J * pij + 1 - pij)
    
    return Dkl


def update_mask_decimation(
    mask: torch.Tensor,
    Dkl: torch.Tensor,
    drate: float,
) -> torch.Tensor:
    """Updates the mask by removing the n_remove couplings with the smallest Dkl.

    Args:
        mask (torch.Tensor): Mask.
        Dkl (torch.Tensor): Kullback-Leibler divergence matrix.
        drate (float): Percentage of active couplings to be pruned at each decimation step.

    Returns:
        torch.Tensor: Updated mask.
    """
    
    n_remove = int((mask.sum().item() // 2) * drate) * 2
    # Only consider the active couplings
    Dkl_active = torch.where(mask, Dkl, float("inf")).reshape(-1)
    _, idx_remove = torch.topk(-Dkl_active, n_remove)
    mask = mask.reshape(-1).scatter_(0, idx_remove, 0.0).reshape(mask.shape)
    
    return mask


def decimate_graph(
    pij: torch.Tensor,
    params: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    drate: float,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Performs one decimation step and updates the parameters and mask.

    Args:
        pij (torch.Tensor): Two-point marginal probability distribution.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        mask (torch.Tensor): Mask.
        drate (float): Percentage of active couplings to be pruned at each decimation step.

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: Updated parameters and mask.
    """
    
    Dkl = compute_Dkl_decimation(params=params, pij=pij)
    mask = update_mask_decimation(mask=mask, Dkl=Dkl, drate=drate)
    params["coupling_matrix"] *= mask
    
    return params, mask
