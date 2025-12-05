from typing import Dict, Tuple
import torch


@torch.jit.script
def compute_Dkl_activation(
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


def update_mask_activation(
    Dkl: torch.Tensor,
    mask: torch.Tensor,
    nactivate: int,
) -> torch.Tensor:
    """Updates the mask by removing the nactivate couplings with the smallest Dkl.
    
    Args:
        Dkl (torch.Tensor): Kullback-Leibler divergence matrix.
        mask (torch.Tensor): Mask.
        nactivate (int): Number of couplings to be activated at each graph update.
    
    Returns:
        torch.Tensor: Updated mask.
    """
    # Flatten the Dkl tensor and sort it in descending order
    Dkl_flat_sorted, _ = torch.sort(Dkl.flatten(), descending=True)
    # Get the threshold value for the top nactivate*2 elements (since Dkl is symmetric)
    Dkl_th = Dkl_flat_sorted[2 * nactivate]
    # Update the mask where Dkl is greater than the threshold
    mask = torch.where(Dkl > Dkl_th, torch.ones_like(mask), mask)
    
    return mask


@torch.jit.script
def activate_graph(
    mask: torch.Tensor,
    fij: torch.Tensor,
    pij: torch.Tensor,
    nactivate: int,
) -> torch.Tensor: 
    """Updates the interaction graph by activating a maximum of nactivate couplings.

    Args:
        mask (torch.Tensor): Mask.
        fij (torch.Tensor): Two-point frequencies of the dataset.
        pij (torch.Tensor): Two-point marginals of the model.
        nactivate (int): Number of couplings to activate.
        
    Returns:
        torch.Tensor: Updated mask.
    """
    
    # Compute the Kullback-Leibler divergence of all the couplings
    Dkl = compute_Dkl_activation(fij=fij, pij=pij)
    # Update the graph
    mask = update_mask_activation(Dkl=Dkl, mask=mask, nactivate=nactivate)
    
    return mask


@torch.jit.script
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


@torch.jit.script
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


@torch.jit.script
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