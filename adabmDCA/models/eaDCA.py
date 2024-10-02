import time
import numpy as np
from tqdm import tqdm
from typing import Callable

import torch

from adabmDCA.io import save_chains, save_params
from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from adabmDCA.grad import train_graph
from adabmDCA.methods import compute_density, get_mask_save


@torch.jit.script
def compute_Dkl(
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


def update_mask(
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
    mask = torch.where(Dkl > Dkl_th, torch.tensor(1, device=mask.device), mask)
    
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
    Dkl = compute_Dkl(fij=fij, pij=pij)
    # Update the graph
    mask = update_mask(Dkl=Dkl, mask=mask, nactivate=nactivate)
    
    return mask


def fit(
    sampler: Callable,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    params: dict,
    mask: torch.Tensor,
    chains: torch.Tensor,
    tokens: str,
    target_pearson: float,
    nsweeps: int = 5,
    nepochs: int = 50000,
    pseudo_count: float = 0.1,
    lr: float = 0.01,
    factivate: float = 0.001,
    gsteps: int = 10,
    file_paths: dict = None,
    device: str = "cpu",
    *args, **kwargs
) -> None:
    """
    Fits an eaDCA model on the training data and saves the results in a file.

    Args:
        sampler (Callable): Sampling function to be used.
        fi_target (torch.Tensor): Single-point frequencies of the data.
        fij_target (torch.Tensor): Two-point frequencies of the data.
        params (dict): Initialization of the model's parameters.
        mask (torch.Tensor): Initialization of the coupling matrix's mask.
        chains (torch.Tensor): Initialization of the Markov chains.
        tokens (str): Tokens used for encoding the sequences.
        target_pearson (float): Pearson correlation coefficient on the two-points statistics to be reached.
        nsweeps (int): Number of Monte Carlo steps to update the state of the model. Defaults to 5.
        nepochs (int): Maximum number of epochs to be performed. Defaults to 50000.
        pseudo_count (float): Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.1.
        lr (float): Learning rate. Defaults to 0.01.
        factivate (float): Fraction of inactive couplings to activate at each step. Defaults to 0.001.
        gsteps (int): Number of gradient updates to be performed on a given graph. Defaults to 10.
        file_paths (dict): Dictionary containing the paths where to save log, params, and chains. Defaults to None.
        device (str): Device to be used. Defaults to "cpu".
    """
    
    # Check the input sizes
    if fi_target.dim() != 2:
        raise ValueError("fi_target must be a 2D tensor")
    if fij_target.dim() != 4:
        raise ValueError("fij_target must be a 4D tensor")
    if chains.dim() != 3:
        raise ValueError("chains must be a 3D tensor")
    
    graph_upd = 0
    density = compute_density(mask) * 100
    L, q = fi_target.shape
        
    # Mask for saving only the upper-diagonal matrix
    mask_save = get_mask_save(L, q, device=device)
    
    # Compute the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(data=chains, weights=None, pseudo_count=0.)
    pij = get_freq_two_points(data=chains, weights=None, pseudo_count=0.)
    pearson = max(0, float(get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)[0]))
    
    # Save the chains
    save_chains(fname=file_paths["chains"], chains=chains.argmax(-1), tokens=tokens)
    
    # Number of active couplings
    nactive = mask.sum()
    
    # Training loop
    time_start = time.time()
    pbar = tqdm(initial=max(0, float(pearson)), total=target_pearson, colour="red", dynamic_ncols=True, ascii="-#",
                bar_format="{desc}: {percentage:.2f}%[{bar}] Pearson: {n:.3f}/{total_fmt} [{elapsed}]")
    pbar.set_description(f"Training eaDCA - Graph updates: {graph_upd} - Density: {density:.3f}% - New active couplings: {0}")
    # Template for wrinting the results
    template = "{0:10} {1:10} {2:10} {3:10}\n"
    
    while pearson < target_pearson:
        
        # Old number of active couplings
        nactive_old = nactive
        
        # Compute the two-points frequencies of the simulated data with pseudo-count
        pij_Dkl = get_freq_two_points(data=chains, weights=None, pseudo_count=pseudo_count)
        
        # Update the graph
        nactivate = int(((L**2 * q**2) - mask.sum().item()) * factivate)
        mask = activate_graph(
            mask=mask,
            fij=fij_target,
            pij=pij_Dkl,
            nactivate=nactivate,
        )
        
        # New number of active couplings
        nactive = mask.sum()
        
        # Bring the model at convergence on the graph
        chains, params = train_graph(
            sampler=sampler,
            chains=chains,
            mask=mask,
            fi=fi_target,
            fij=fij_target,
            pi=pi,
            pij=pij,
            params=params,
            nsweeps=nsweeps,
            lr=lr,
            max_epochs=gsteps,
            target_pearson=target_pearson,
            tokens=tokens,
            check_slope=False,
            file_paths=None,
            progress_bar=False,
            device=device,
        )

        graph_upd += 1
        
        # Compute the single-point and two-points frequencies of the simulated data
        pi = get_freq_single_point(data=chains, weights=None, pseudo_count=0.)
        pij = get_freq_two_points(data=chains, weights=None, pseudo_count=0.)
        
        # Global Pearson
        pearson, _ = get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)
        density = compute_density(mask) * 100
        pbar.set_description(f"Training eaDCA - Graph updates: {graph_upd} - Density: {density:.3f}% - New active couplings: {int(nactive - nactive_old)}")

        # Save the model if a checkpoint is reached
        if (graph_upd % 10 == 0) or (graph_upd == nepochs):
            save_params(fname=file_paths["params"], params=params, mask=torch.logical_and(mask, mask_save), tokens=tokens)
            save_chains(fname=file_paths["chains"], chains=chains.argmax(-1), tokens=tokens)
            with open(file_paths["log"], "a") as f:
                f.write(template.format(f"{graph_upd}", f"{pearson:.3f}", f"{density:.3f}", f"{(time.time() - time_start):.1f}"))
                #f.write(f"{graph_upd}\t{pearson:.3f}\t{density:.3f}\t{(time.time() - time_start):.1f}\n")
        pbar.n = min(max(0, float(pearson)), target_pearson)

    save_params(fname=file_paths["params"], params=params, mask=torch.logical_and(mask, mask_save), tokens=tokens)
    save_chains(fname=file_paths["chains"], chains=chains.argmax(-1), tokens=tokens)
    with open(file_paths["log"], "a") as f:
        f.write(f"{graph_upd}\t{pearson:.3f}\t{density:.3f}\t{(time.time() - time_start):.1f}\n")

    pbar.close()