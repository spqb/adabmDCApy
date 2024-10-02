from tqdm import tqdm
import numpy as np
from pathlib import Path
import time
from typing import Tuple, Optional, Callable, Dict

from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from adabmDCA.io import save_chains, save_params
from adabmDCA.methods import compute_density, get_mask_save
import torch


@torch.jit.script
def compute_gradient(
    fi: torch.Tensor,
    fij: torch.Tensor,
    pi: torch.Tensor,
    pij: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Computes the gradient of the log-likelihood of the model using PyTorch.

    Args:
        fi (torch.Tensor): Single-point frequencies of the data.
        fij (torch.Tensor): Target two-points frequencies.
        pi (torch.Tensor): Single-point marginals of the model.
        pij (torch.Tensor): Two-points marginals of the model.

    Returns:
        Dict[str, torch.Tensor]: Gradient.
    """
    
    grad = {}
    grad["bias"] = fi - pi
    grad["coupling_matrix"] = fij - pij
    
    return grad


def update(
    sampler: Callable,
    chains: torch.Tensor,
    fi: torch.Tensor,
    fij: torch.Tensor,
    pi: torch.Tensor,
    pij: torch.Tensor,
    params: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    lr: float,
    nsweeps: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Updates the parameters of the model and the Markov chains.
    Args:
        sampler (Callable): Sampling function.
        chains (torch.Tensor): Markov chains simulated with the model.
        fi (torch.Tensor): Single-point frequencies of the data.
        fij (torch.Tensor): Two-points frequencies of the data.
        pi (torch.Tensor): Single-point marginals of the model.
        pij (torch.Tensor): Two-points marginals of the model.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        mask (torch.Tensor): Mask of the interaction graph.
        lr (float): Learning rate.
        nsweeps (int): Number of Monte Carlo updates.
        
    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Updated chains and parameters.
    """
    
    # Compute the gradient
    grad = compute_gradient(fi=fi, fij=fij, pi=pi, pij=pij)
    
    # Update parameters
    with torch.no_grad():
        for key in params:
            params[key] += lr * grad[key]
        params["coupling_matrix"] *= mask # Remove autocorrelations
        
    # Sample from the model
    chains = sampler(chains=chains, params=params, nsweeps=nsweeps)
    
    return chains, params


def train_graph(
    sampler: Callable,
    chains: torch.Tensor,
    mask: torch.Tensor,
    fi: torch.Tensor,
    fij: torch.Tensor,
    pi: torch.Tensor,
    pij: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    lr: float,
    max_epochs: int,
    target_pearson: float,
    tokens: str = "protein",
    check_slope: bool = False,
    file_paths: Optional[Dict[str, Path]] = None,
    progress_bar: bool = True,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Trains the model on a given graph until the target Pearson correlation is reached or the maximum number of epochs is exceeded.

    Args:
        sampler (Callable): Sampling function.
        chains (torch.Tensor): Markov chains simulated with the model.
        mask (torch.Tensor): Mask encoding the sparse graph.
        fi (torch.Tensor): Single-point frequencies of the data.
        fij (torch.Tensor): Two-point frequencies of the data.
        pi (torch.Tensor): Single-point marginals of the model.
        pij (torch.Tensor): Two-points marginals of the model.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        nsweeps (int): Number of Gibbs steps for each gradient estimation.
        lr (float): Learning rate.
        max_epochs (int): Maximum number of gradient updates to be done.
        target_pearson (float): Target Pearson coefficient.
        tokens (str): Alphabet to be used for the encoding.
        check_slope (bool): Whether to take into account the slope for the convergence criterion or not.
        file_paths (Optional[Dict[str, Path]]): Dictionary containing the paths where to save log, params, and chains.
        progress_bar (bool): Whether to display a progress bar or not.
        device (str): Device to be used.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Updated chains and parameters.
    """
    
    L, q = fi.shape
    time_start = time.time()
    density = compute_density(mask)
    
    def halt_condition(epochs, pearson, slope, check_slope):
        c1 = pearson < target_pearson
        c2 = epochs < max_epochs
        if check_slope:
            c3 = abs(slope - 1.) > 0.1
        else:
            c3 = False
        return not c2 * ((not c1) * c3 + c1)
    
    # Mask for saving only the upper-diagonal matrix
    mask_save = get_mask_save(L, q, device=device)

    pearson, slope = get_correlation_two_points(
        fij=fij,
        pij=pij,
        fi=fi,
        pi=pi,
    )
    
    epochs = 0   
    
    if progress_bar: 
        pbar = tqdm(
            initial=max(0, float(pearson)),
            total=target_pearson,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
            bar_format="{desc} {percentage:.2f}%[{bar}] Pearson: {n:.3f}/{total_fmt} [{elapsed}]"
        )
        pbar.set_description(f"Train graph - Epochs: {epochs} - Slope: {slope:.2f}")
   
    # Template for wrinting the results
    template = "{0:10} {1:10} {2:10} {3:10}\n"
    
    while not halt_condition(epochs, pearson, slope, check_slope):
        chains, params = update(
            sampler=sampler,
            chains=chains,
            fi=fi,
            fij=fij,
            pi=pi,
            pij=pij,
            params=params,
            mask=mask,
            lr=lr,
            nsweeps=nsweeps
        )
        epochs += 1
        
        # Compute the single-point and two-points frequencies of the simulated data
        pi = get_freq_single_point(data=chains, weights=None, pseudo_count=0.)
        pij = get_freq_two_points(data=chains, weights=None, pseudo_count=0.)
        
        pearson, slope = get_correlation_two_points(fij=fij, pij=pij, fi=fi, pi=pi)
        
        if progress_bar:
            pbar.n = min(max(0, float(pearson)), target_pearson)
            pbar.set_description(f"Train graph - Epochs: {epochs} - Slope: {slope:.2f}")
            
        # Save the model if a checkpoint is reached
        if (file_paths is not None) and (epochs % 50 == 0 or epochs == max_epochs):
            save_params(fname=file_paths["params"], params=params, mask=mask_save, tokens=tokens)
            save_chains(fname=file_paths["chains"], chains=chains.argmax(dim=-1), tokens=tokens)
            with open(file_paths["log"], "a") as f:
                f.write(template.format(f"{epochs}", f"{pearson:.3f}", f"{slope:.3f}", f"{(time.time() - time_start):.1f}"))
                
    if progress_bar:
        pbar.close()
    
    if file_paths is not None:
        save_params(fname=file_paths["params"], params=params, mask=mask_save, tokens=tokens)
        save_chains(fname=file_paths["chains"], chains=chains.argmax(dim=-1), tokens=tokens)
        with open(file_paths["log"], "a") as f:
            f.write(template.format(f"{epochs}", f"{pearson:.3f}", f"{slope:.3f}", f"{(time.time() - time_start):.1f}"))

    return chains, params
