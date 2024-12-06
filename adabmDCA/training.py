from tqdm import tqdm
import time
from typing import Tuple, Callable, Dict
import torch

from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from adabmDCA.utils import get_mask_save
from adabmDCA.statmech import _update_weights_AIS, compute_log_likelihood, compute_entropy
from adabmDCA.checkpoint import Checkpoint


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


@torch.jit.script
def compute_gradient_centred(
    fi: torch.Tensor,
    fij: torch.Tensor,
    pi: torch.Tensor,
    pij: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Computes the gradient of the log-likelihood of the model using PyTorch.
    Implements the centred gradient, which empirically improves the convergence of the model.

    Args:
        fi (torch.Tensor): Single-point frequencies of the data.
        fij (torch.Tensor): Target two-points frequencies.
        pi (torch.Tensor): Single-point marginals of the model.
        pij (torch.Tensor): Two-points marginals of the model.

    Returns:
        Dict[str, torch.Tensor]: Gradient.
    """
    grad = {}
    
    C_data = fij - torch.einsum("ij,kl->ijkl", fi, fi)
    C_model = pij - torch.einsum("ij,kl->ijkl", pi, pi)
    grad["coupling_matrix"] = C_data - C_model
    grad["bias"] = fi - pi - torch.einsum("iajb,jb->ia", grad["coupling_matrix"], fi)
    
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
    grad = compute_gradient_centred(fi=fi, fij=fij, pi=pi, pij=pij)
    
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
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    lr: float,
    max_epochs: int,
    target_pearson: float,
    checkpoint: Checkpoint | None = None,
    check_slope: bool = False,
    log_weights: torch.Tensor | None = None,
    progress_bar: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """Trains the model on a given graph until the target Pearson correlation is reached or the maximum number of epochs is exceeded.

    Args:
        sampler (Callable): Sampling function.
        chains (torch.Tensor): Markov chains simulated with the model.
        mask (torch.Tensor): Mask encoding the sparse graph.
        fi (torch.Tensor): Single-point frequencies of the data.
        fij (torch.Tensor): Two-point frequencies of the data.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        nsweeps (int): Number of Gibbs steps for each gradient estimation.
        lr (float): Learning rate.
        max_epochs (int): Maximum number of gradient updates to be done.
        target_pearson (float): Target Pearson coefficient.
        checkpoint (Checkpoint | None, optional): Checkpoint class to be used for saving the model. Defaults to None.
        check_slope (bool, optional): Whether to take into account the slope for the convergence criterion or not. Defaults to False.
        log_weights (torch.Tensor, optional): Log-weights used for the online computation of the log-likelihood. Defaults to None.
        progress_bar (bool, optional): Whether to display a progress bar or not. Defaults to True.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: Updated chains and parameters, log-weights for the log-likelihood computation.
    """
    device = fi.device
    dtype = fi.dtype
    L, q = fi.shape
    time_start = time.time()
    
    # log_weights used for the online computing of the log-likelihood
    if log_weights is None:
        log_weights = torch.zeros(len(chains), device=device, dtype=dtype)
    logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
    log_likelihood = compute_log_likelihood(fi=fi, fij=fij, params=params, logZ=logZ)
    entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
    
    # Compute the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(data=chains, weights=None, pseudo_count=0.)
    pij = get_freq_two_points(data=chains, weights=None, pseudo_count=0.)
    
    def halt_condition(epochs, pearson, slope, check_slope):
        c1 = pearson < target_pearson
        c2 = epochs < max_epochs
        if check_slope:
            c3 = abs(slope - 1.) > 0.1
        else:
            c3 = False
        return not c2 * ((not c1) * c3 + c1)
    
    # Mask for saving only the upper-diagonal coupling matrix
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
        pbar.set_description(f"Epochs: {epochs} - Slope: {slope:.2f} - LL: {log_likelihood:.2f}")
    
    while not halt_condition(epochs, pearson, slope, check_slope):
        
        # Store the previous parameters
        params_prev = {key: value.clone() for key, value in params.items()}
        
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
        
        # Compute the log-likelihood
        log_weights = _update_weights_AIS(
            prev_params=params_prev,
            curr_params=params,
            chains=chains,
            log_weights=log_weights,
        )
        
        logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
        log_likelihood = compute_log_likelihood(fi=fi, fij=fij, params=params, logZ=logZ)
        
        if progress_bar:
            pbar.n = min(max(0, float(pearson)), target_pearson)
            pbar.set_description(f"Epochs: {epochs} - Slope: {slope:.2f} - LL: {log_likelihood:.2f}")
            
        # Save the model if a checkpoint is reached
        if checkpoint is not None:
            if checkpoint.check(epochs, params, chains):
                entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
                checkpoint.save(
                    params=params,
                    mask=mask_save,
                    chains=chains,
                    log_weights=log_weights,
                    epochs=epochs,
                    pearson=pearson,
                    slope=slope,
                    log_likelihood=log_likelihood,
                    entropy=entropy,
                    density=1.0,
                    time_start=time_start,
                )
                
    if progress_bar:
        pbar.close()
    
    if checkpoint is not None:
        entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
        checkpoint.save(
            params=params,
            mask=mask_save,
            chains=chains,
            log_weights=log_weights,
            epochs=epochs,
            pearson=pearson,
            slope=slope,
            log_likelihood=log_likelihood,
            entropy=entropy,
            density=1.0,
            time_start=time_start,
        )
        
    return chains, params, log_weights
