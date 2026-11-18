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
    if L < 2 or q == 0:
        return 0.0
    active = mask.count_nonzero() - mask.diagonal(dim1=0, dim2=2).count_nonzero()
    density = active / (q**2 * L * (L - 1))
    return density.item()


# Element-wise activation functions

def _kl_terms(target: torch.Tensor, model: torch.Tensor) -> torch.Tensor:
    return torch.xlogy(target, target) - torch.xlogy(target, model)


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
    Dkl = _kl_terms(fij, pij) + _kl_terms(1 - fij, 1 - pij)
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
    scores = Dkl.flatten()[candidates]
    scores = torch.where(scores.isnan(), -torch.inf, scores)
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
    Dkl = _kl_terms(fij, pij).sum(dim=(1, 3))
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
    
    coupling = params["coupling_matrix"]
    dtype = torch.promote_types(torch.promote_types(pij.dtype, coupling.dtype), torch.float32)
    p, coupling = pij.to(dtype), coupling.to(dtype)
    after_removal = torch.sigmoid(p.log() - torch.log1p(-p) - coupling)
    # expm1 keeps the difference in probabilities accurate when the coupling is small.
    change = torch.expm1(-coupling.clamp(min=-1, max=1))
    small_delta = -p * (1 - p) * change / (1 + p * change)
    delta = torch.where(coupling.abs() <= 1, small_delta, p - after_removal)
    return (coupling * delta).clamp_min(0)


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
    
    coupling = params["coupling_matrix"]
    dtype = torch.promote_types(torch.promote_types(pij.dtype, coupling.dtype), torch.float32)
    p, coupling = pij.to(dtype), coupling.to(dtype)
    log_z = torch.logaddexp(torch.log1p(-p), p.log() - coupling)
    small_log_z = torch.log1p(p * torch.expm1(-coupling.clamp(min=-1, max=1)))
    log_z = torch.where(coupling.abs() <= 1, small_log_z, log_z)
    dkl = (p * coupling + log_z).clamp_min(0)
    return torch.where((p == 0) | (p == 1), 0.0, dkl)


def update_mask_decimation(
    mask: torch.Tensor,
    Dkl: torch.Tensor,
    drate: float,
) -> torch.Tensor:
    """Remove the lowest-scoring active off-site entries from a symmetric mask.

    Count each entry once and clear both orientations. A positive rate removes
    at least one entry when candidates remain.

    Args:
        mask (torch.Tensor): Mask.
        Dkl (torch.Tensor): Kullback-Leibler divergence matrix.
        drate (float): Percentage of active couplings to be pruned at each decimation step.

    Returns:
        torch.Tensor: Updated mask.
    """
    
    if not 0 <= drate <= 1:
        raise ValueError("The decimation rate must be in [0, 1].")
    if drate == 0:
        return mask
    L, q = mask.shape[:2]
    eligible = mask.bool() & get_mask_save(L, q, device=mask.device)
    active = eligible.flatten().nonzero().squeeze(1)
    if active.numel() == 0:
        return mask
    count = min(active.numel(), max(1, int(active.numel() * drate)))
    if count < active.numel():
        scores = Dkl.flatten()[active]
        scores = torch.where(scores.isnan(), torch.inf, scores)
        cutoff = torch.topk(scores, count, largest=False, sorted=False).values.max()
        better = (scores < cutoff).nonzero().squeeze(1)
        # Flat indices are already ordered, so ties are reproducible across devices.
        tied = (scores == cutoff).nonzero().squeeze(1)[:count - len(better)]
        active = active[torch.cat((better, tied))]
    i, a, j, b = torch.unravel_index(active, mask.shape)
    mask[i, a, j, b] = 0
    mask[j, b, i, a] = 0
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
