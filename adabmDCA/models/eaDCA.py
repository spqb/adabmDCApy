import time
from tqdm import tqdm
from typing import Callable

import torch

from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from adabmDCA.training import train_graph
from adabmDCA.utils import get_mask_save
from adabmDCA.graph import activate_graph, compute_density
from adabmDCA.statmech import compute_log_likelihood, compute_entropy
from adabmDCA.checkpoint import Checkpoint


def fit(
    sampler: Callable,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    params: dict,
    mask: torch.Tensor,
    chains: torch.Tensor,
    log_weights: torch.Tensor,
    target_pearson: float,
    nsweeps: int,
    nepochs: int,
    pseudo_count: float,
    lr: float,
    factivate: float,
    gsteps: int,
    checkpoint: Checkpoint | None = None,
    *args, **kwargs,
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
        log_weights (torch.Tensor): Log-weights of the chains. Used to estimate the log-likelihood.
        target_pearson (float): Pearson correlation coefficient on the two-points statistics to be reached.
        nsweeps (int): Number of Monte Carlo steps to update the state of the model.
        nepochs (int): Maximum number of epochs to be performed. Defaults to 50000.
        pseudo_count (float): Pseudo count for the single and two points statistics. Acts as a regularization.
        lr (float): Learning rate.
        factivate (float): Fraction of inactive couplings to activate at each step.
        gsteps (int): Number of gradient updates to be performed on a given graph.
        checkpoint (Checkpoint | None): Checkpoint class to be used to save the model. Defaults to None.
    """
    
    # Check the input sizes
    if fi_target.dim() != 2:
        raise ValueError("fi_target must be a 2D tensor")
    if fij_target.dim() != 4:
        raise ValueError("fij_target must be a 4D tensor")
    if chains.dim() != 3:
        raise ValueError("chains must be a 3D tensor")
    
    device = fi_target.device
    dtype = fi_target.dtype
    checkpoint.checkpt_interval = 10 # Save the model every 10 graph updates
    checkpoint.max_epochs = nepochs
    
    graph_upd = 0
    density = compute_density(mask) * 100
    L, q = fi_target.shape
        
    # Mask for saving only the upper-diagonal matrix
    mask_save = get_mask_save(L, q, device=device)
    
    # log_weights used for the online computing of the log-likelihood
    logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
    
    # Compute the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(data=chains, weights=None, pseudo_count=0.)
    pij = get_freq_two_points(data=chains, weights=None, pseudo_count=0.)
    pearson = max(0, float(get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)[0]))
    
    # Number of active couplings
    nactive = mask.sum()
    
    # Training loop
    time_start = time.time()
    log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)
    entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
        
    pbar = tqdm(initial=max(0, float(pearson)), total=target_pearson, colour="red", dynamic_ncols=True, ascii="-#",
                bar_format="{desc}: {percentage:.2f}%[{bar}] Pearson: {n:.3f}/{total_fmt} [{elapsed}]")
    pbar.set_description(f"Graph updates: {graph_upd} - Density: {density:.3f}% - New active couplings: {0} - LL: {log_likelihood:.3f}")
    
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
        chains, params, log_weights = train_graph(
            sampler=sampler,
            chains=chains,
            mask=mask,
            fi=fi_target,
            fij=fij_target,
            params=params,
            nsweeps=nsweeps,
            lr=lr,
            max_epochs=gsteps,
            target_pearson=target_pearson,
            log_weights=log_weights,
            check_slope=False,
            checkpoint=None,
            progress_bar=False,
        )

        graph_upd += 1
        
        # Compute the single-point and two-points frequencies of the simulated data
        pi = get_freq_single_point(data=chains, weights=None, pseudo_count=0.)
        pij = get_freq_two_points(data=chains, weights=None, pseudo_count=0.)
        
        # Compute statistics of the training
        pearson, slope = get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)
        density = compute_density(mask) * 100
        logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
        log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)
        pbar.set_description(f"Graph updates: {graph_upd} - Density: {density:.3f}% - New active couplings: {int(nactive - nactive_old)} - LL: {log_likelihood:.3f}")

        # Save the model if a checkpoint is reached
        if checkpoint.check(graph_upd, params, chains):
            entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
            checkpoint.save(
                params=params,
                mask=torch.logical_and(mask, mask_save),
                chains=chains,
                log_weights=log_weights,
                epochs=graph_upd,
                pearson=pearson,
                slope=slope,
                log_likelihood=log_likelihood,
                entropy=entropy,
                density=density,
                time_start=time_start,
                )
        pbar.n = min(max(0, float(pearson)), target_pearson)

    entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
    checkpoint.save(
        params=params,
        mask=torch.logical_and(mask, mask_save),
        chains=chains,
        log_weights=log_weights,
        epochs=graph_upd,
        pearson=pearson,
        slope=slope,
        log_likelihood=log_likelihood,
        entropy=entropy,
        density=density,
        time_start=time_start,
        )
    print(f"Completed, model parameters saved in {checkpoint.file_paths['params']}")
    pbar.close()