from pathlib import Path
from typing import Callable, Tuple, Dict
import time

import torch

from adabmDCA.stats import get_correlation_two_points
from adabmDCA.grad import train_graph
from adabmDCA.methods import compute_density, get_mask_save
from adabmDCA.io import save_chains, save_params
from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points

MAX_EPOCHS = 10000


@torch.jit.script
def compute_sym_Dkl(
    params: Dict[str, torch.Tensor],
    pij: torch.Tensor,
) -> torch.Tensor:
    """Computes the symmetric Kullback-Leibler divergence matrix between the initial distribution and the same 
    distribution once removing one coupling J_ij(a, b).

    Args:
        params (Dict[ste, torch.Tensor]): Parameters of the model.
        pij (torch.Tensor): Two-point marginal probability distribution.

    Returns:
        torch.Tensor: Kullback-Leibler divergence matrix.
    """
    
    exp_J = torch.exp(-params["coupling_matrix"])
    Dkl = pij * params["coupling_matrix"] * (1. - exp_J / (pij * (exp_J - 1.) + 1.))
    
    return Dkl


@torch.jit.script
def compute_Dkl(
    params: Dict[str, torch.Tensor],
    pij: torch.Tensor,
) -> torch.Tensor:
    """Computes the Kullback-Leibler divergence matrix between the initial distribution and the same 
    distribution once removing one coupling J_ij(a, b).

    Args:
        params (Dict[ste, torch.Tensor]): Parameters of the model.
        pij (torch.Tensor): Two-point marginal probability distribution.

    Returns:
        torch.Tensor: Kullback-Leibler divergence matrix.
    """
    
    exp_J = torch.exp(-params["coupling_matrix"])
    Dkl = pij * params["coupling_matrix"] + torch.log(exp_J * pij + 1 - pij)
    
    return Dkl


def update_mask(
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
    Dkl_active = torch.where(mask, Dkl, float("inf")).view(-1)
    _, idx_remove = torch.topk(-Dkl_active, n_remove)
    mask = mask.view(-1).scatter_(0, idx_remove, 0).view(mask.shape)
    
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
    
    Dkl = compute_Dkl(params=params, pij=pij)
    mask = update_mask(mask=mask, Dkl=Dkl, drate=drate)
    params["coupling_matrix"] *= mask
    
    return params, mask


def fit(
    sampler: Callable,
    chains: torch.Tensor,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    params: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    lr: float,
    nsweeps: int,
    target_pearson: float,
    target_density: float,
    drate: float,
    tokens: str,
    file_paths: Dict[str, Path],
    device: str = "cpu",
    *args, **kwargs,
):
    """Fits an edDCA model on the training data and saves the results in a file.
    
    Args:
        sampler (Callable): Sampling function to be used.
        chains (torch.Tensor): Initialization of the Markov chains.
        fi_target (torch.Tensor): Single-point frequencies of the data.
        fij_target (torch.Tensor): Two-point frequencies of the data.
        params (Dict[str, torch.Tensor]): Initialization of the model's parameters.
        mask (torch.Tensor): Initialization of the coupling matrix's mask.
        lr (float): Learning rate.
        nsweeps (int): Number of Monte Carlo steps to update the state of the model.
        target_pearson (float): Pearson correlation coefficient on the two-points statistics to be reached.
        target_density (float): Target density of the coupling matrix.
        drate (float): Percentage of active couplings to be pruned at each decimation step.
        tokens (str): Tokens used for encoding the sequences.
        file_paths (Dict[str, Path]): Dictionary containing the paths where to save log, params and chains.
        device (str): Device to be used. Defaults to "cpu".
    """
    time_start = time.time()
    
    # Check the input sizes
    if fi_target.dim() != 2:
        raise ValueError("fi_target must be a 2D tensor")
    if fij_target.dim() != 4:
        raise ValueError("fij_target must be a 4D tensor")
    if chains.dim() != 3:
        raise ValueError("chains must be a 3D tensor")
    
    L, q = params["bias"].shape
    
    print("Bringing the model to the convergence threshold...")
    pi = get_freq_single_point(data=chains, weights=None, pseudo_count=0.)
    pij = get_freq_two_points(data=chains, weights=None, pseudo_count=0.)
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
        max_epochs=MAX_EPOCHS,
        target_pearson=target_pearson,
        tokens=tokens,
        check_slope=True,
        file_paths=file_paths,
        device=device,
    )
    
    # Get the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(data=chains, weights=None, pseudo_count=0.)
    pij = get_freq_two_points(data=chains, weights=None, pseudo_count=0.)
    
    # Mask for saving only the upper-diagonal matrix
    mask_save = get_mask_save(L, q, device=device)
    
    # Filenames for the decimated parameters and chains
    parent, name = file_paths["params"].parent, file_paths["params"].name
    new_name = name.replace(".dat", "_dec.dat")
    file_paths["params_dec"] = Path(parent).joinpath(new_name)
    
    name = file_paths["chains"].name
    new_name = name.replace(".fasta", "_dec.fasta")
    file_paths["chains_dec"] = Path(parent).joinpath(new_name)
    
    print(f"\nStarting the decimation (target density = {target_density}):")
    template_log = "{0:10} {1:10} {2:10} {3:10}\n"
    with open(file_paths["log"], "a") as f:
        f.write("\nDecimation\n")
        f.write(f"Target density: {target_density}\n")
        f.write(f"Decimation rate: {drate}\n\n")
        f.write(template_log.format("Epoch", "Pearson", "Density", "Time [s]"))
        
    # Template for frinting the results
    template = "{0:15} | {1:15} | {2:15} | {3:15}"
    density = compute_density(mask)
    count = 0
    
    while density > target_density:
        count += 1
        
        # Decimate the model
        params, mask = decimate_graph(
            pij=pij,
            params=params,
            mask=mask,
            drate=drate
        )
        
        # Equilibrate the model
        chains = sampler(
            chains=chains,
            params=params,
            nsweeps=nsweeps,
        )
        
        # Compute the single-point and two-points frequencies of the simulated data
        pi = get_freq_single_point(data=chains, weights=None, pseudo_count=0.)
        pij = get_freq_two_points(data=chains, weights=None, pseudo_count=0.)
        
        # bring the model at convergence on the graph
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
            max_epochs=MAX_EPOCHS,
            target_pearson=target_pearson,
            tokens=tokens,
            check_slope=True,
            progress_bar=False,
            device=device,
        )
        
        # Compute the single-point and two-points frequencies of the simulated data
        pi = get_freq_single_point(data=chains, weights=None, pseudo_count=0.)
        pij = get_freq_two_points(data=chains, weights=None, pseudo_count=0.)
        
        pearson, slope = get_correlation_two_points(fi=fi_target, pi=pi, fij=fij_target, pij=pij)
        density = compute_density(mask)
        
        print(template.format(f"Dec. step: {count}", f"Density: {density:.3f}", f"Pearson: {pearson:.3f}", f"Slope: {slope:.3f}"))
                
        if count % 10 == 0:
            save_params(fname=file_paths["params_dec"], params=params, mask=torch.logical_and(mask, mask_save), tokens=tokens)
            save_chains(fname=file_paths["chains_dec"], chains=chains.argmax(-1), tokens=tokens)
            with open(file_paths["log"], "a") as f:
                f.write(template_log.format(f"{count}", f"{pearson:.3f}", f"{density:.3f}", f"{(time.time() - time_start):.1f}"))
    
    save_params(fname=file_paths["params_dec"], params=params, mask=torch.logical_and(mask, mask_save), tokens=tokens)
    save_chains(fname=file_paths["chains_dec"], chains=chains.argmax(-1), tokens=tokens)
    with open(file_paths["log"], "a") as f:
        f.write(template_log.format(f"{count}", f"{pearson:.3f}", f"{density:.3f}", f"{(time.time() - time_start):.1f}"))
    print(f"Completed, decimated model parameters saved in {file_paths['params_dec']}")