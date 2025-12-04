import os
from typing import Callable, Dict
import time

import torch

from adabmDCA.stats import get_correlation_two_points
from adabmDCA.training import train_graph
from adabmDCA.utils import get_mask_save
from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from adabmDCA.graph import decimate_graph, compute_density
from adabmDCA.statmech import compute_log_likelihood, _update_weights_AIS, compute_entropy, _compute_ess
from adabmDCA.checkpoint import Checkpoint

MAX_EPOCHS = 10000

def fit(
    sampler: Callable,
    chains: torch.Tensor,
    log_weights: torch.Tensor,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    params: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    lr: float,
    nsweeps: int,
    target_pearson: float,
    target_density: float,
    drate: float,
    checkpoint: Checkpoint,
    fi_test: torch.Tensor | None = None,
    fij_test: torch.Tensor | None = None,
    *args, **kwargs,
):
    """Fits an edDCA model on the training data and saves the results in a file.
    
    Args:
        sampler (Callable): Sampling function to be used.
        chains (torch.Tensor): Initialization of the Markov chains.
        log_weights (torch.Tensor): Log-weights of the chains. Used to estimate the log-likelihood.
        fi_target (torch.Tensor): Single-point frequencies of the data.
        fij_target (torch.Tensor): Two-point frequencies of the data.
        params (Dict[str, torch.Tensor]): Initialization of the model's parameters.
        mask (torch.Tensor): Initialization of the coupling matrix's mask.
        lr (float): Learning rate.
        nsweeps (int): Number of Monte Carlo steps to update the state of the model.
        target_pearson (float): Pearson correlation coefficient on the two-points statistics to be reached.
        target_density (float): Target density of the coupling matrix.
        drate (float): Percentage of active couplings to be pruned at each decimation step.
        checkpoint (Checkpoint): Checkpoint class to be used to save the model.
        fi_test (torch.Tensor | None, optional): Single-point frequencies of the test data. Defaults to None.
        fij_test (torch.Tensor | None, optional): Two-point frequencies of the test data. Defaults to None.
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
    device = fi_target.device
    dtype = fi_target.dtype
    
    # Get the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(data=chains)
    pij = get_freq_two_points(data=chains)
    pearson, _ = get_correlation_two_points(fi=fi_target, pi=pi, fij=fij_target, pij=pij)
    if pearson < target_pearson:
        print("  Initial Pearson correlation: {:.4f} (target: {:.2f})".format(pearson, target_pearson))
        print("  Bringing model to convergence threshold...")
        chains, params, log_weights, _ = train_graph(
            sampler=sampler,
            chains=chains,
            log_weights=log_weights,
            mask=mask,
            fi=fi_target,
            fij=fij_target,
            fi_test=fi_test,
            fij_test=fij_test,
            params=params,
            nsweeps=nsweeps,
            lr=lr,
            max_epochs=MAX_EPOCHS,
            target_pearson=target_pearson,
            check_slope=False,
            checkpoint=checkpoint,
        )
        pi = get_freq_single_point(data=chains)
        pij = get_freq_two_points(data=chains)
        pearson, _ = get_correlation_two_points(fi=fi_target, pi=pi, fij=fij_target, pij=pij)
        # Save the equilibrated parameters
        checkpoint.save(
            params=params,
            mask=mask,
            chains=chains,
            log_weights=log_weights,
        )
        print("  ✓ Equilibrated model saved")
    print("  Current Pearson correlation: {:.4f}".format(pearson))
    
    # Mask for saving only the upper-diagonal matrix
    mask_save = get_mask_save(L, q, device=device)
    
    # Filenames for the decimated parameters and chains
    parent, name = os.path.dirname(checkpoint.file_paths["params"]), os.path.basename(checkpoint.file_paths["params"])
    new_name = name.replace(".dat", "_dec.dat")
    checkpoint.file_paths["params_dec"] = os.path.join(parent, new_name)
    
    name = os.path.basename(checkpoint.file_paths["chains"])
    new_name = name.replace(".fasta", "_dec.fasta")
    checkpoint.file_paths["chains_dec"] = os.path.join(parent, new_name)
    
    print("\n" + "-" * 80)
    print("[DECIMATION PHASE]")
    print("-" * 80)
    print(f"  Target density: {target_density:.3f}")
    print(f"  Decimation rate: {drate:.2f}")
    initial_density = compute_density(mask)
    print(f"  Initial density: {initial_density:.3f}")
    print("-" * 80)
    with open(checkpoint.file_paths["log"], "a") as f:
        f.write("\nDecimation\n")
        template = "{0:<20} {1:<50}\n"
        f.write(template.format("Target density:", target_density))
        f.write(template.format("Decimation rate:", drate))
        f.write("\n")
        header_string = " ".join([f"{key:<10}" for key in checkpoint.logs.keys()])
        f.write("{0:<10} {1}\n".format("Epoch", header_string))
        
    # Template for writing the results
    print("\n  {0:<8} {1:>12} {2:>12} {3:>12} {4:>12}".format("Step", "Density", "Log-Like", "Pearson", "Slope"))
    print("  " + "-" * 60)
    density = compute_density(mask)
    count = 0
    checkpoint.checkpt_interval = 10
    
    while density > target_density:
        count += 1
        
        # Store the previous parameters
        prev_params = {key: value.clone() for key, value in params.items()}
        
        # Decimate the model
        params, mask = decimate_graph(
            pij=pij,
            params=params,
            mask=mask,
            drate=drate
        )
        
        # Update the log-weights
        log_weights = _update_weights_AIS(
            prev_params=prev_params,
            curr_params=params,
            chains=chains,
            log_weights=log_weights,
        )
        
        # Equilibrate the model
        chains = sampler(
            chains=chains,
            params=params,
            nsweeps=nsweeps,
        )
        
        # Bring the model at convergence on the graph
        chains, params, log_weights, _ = train_graph(
            sampler=sampler,
            chains=chains,
            log_weights=log_weights,
            mask=mask,
            fi=fi_target,
            fij=fij_target,
            params=params,
            nsweeps=nsweeps,
            lr=lr,
            max_epochs=MAX_EPOCHS,
            target_pearson=target_pearson,
            check_slope=False,
            progress_bar=False,
            checkpoint=None,
        )
        
        # Compute the single-point and two-points frequencies of the simulated data
        pi = get_freq_single_point(data=chains)
        pij = get_freq_two_points(data=chains)
        
        pearson, slope = get_correlation_two_points(fi=fi_target, pi=pi, fij=fij_target, pij=pij)
        density = compute_density(mask)
        logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
        log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)
        
        print("  {0:<8} {1:>12.4f} {2:>12.3f} {3:>12.4f} {4:>12.4f}".format(count, density, log_likelihood, pearson, slope))
                
        if checkpoint.check(count):
            entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
            ess = _compute_ess(log_weights)
            checkpoint.log(
                {
                    "Epochs": count,
                    "Pearson": pearson,
                    "Slope": slope,
                    "LL_train": log_likelihood,
                    "ESS": ess,
                    "Entropy": entropy,
                    "Density": density,
                    "Time": time.time() - time_start,
                }
            )
            if fi_test is not None and fij_test is not None:
                log_likelihood_test = compute_log_likelihood(fi=fi_test, fij=fij_test, params=params, logZ=logZ)
                checkpoint.log({"LL_test": log_likelihood_test})
            else:
                checkpoint.log({"LL_test": float("nan")})
            
            checkpoint.save(
                params=params,
                mask=torch.logical_and(mask, mask_save),
                chains=chains,
                log_weights=log_weights,
            )
    
    entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
    ess = _compute_ess(log_weights)
    checkpoint.log(
        {
            "Epochs": count,
            "Pearson": pearson,
            "Slope": slope,
            "LL_train": log_likelihood,
            "ESS": ess,
            "Entropy": entropy,
            "Density": density,
            "Time": time.time() - time_start,
        }
    )
    if fi_test is not None and fij_test is not None:
        log_likelihood_test = compute_log_likelihood(fi=fi_test, fij=fij_test, params=params, logZ=logZ)
        checkpoint.log({"LL_test": log_likelihood_test})
    else:
        checkpoint.log({"LL_test": float("nan")})
                
    checkpoint.save(
        params=params,
        mask=torch.logical_and(mask, mask_save),
        chains=chains,
        log_weights=log_weights,
    )
    
    print("\n" + "-" * 80)
    print("  DECIMATION COMPLETED")
    print("-" * 80)
    print(f"  Final density: {density:.4f}")
    print(f"  Final Pearson: {pearson:.4f}")
    print(f"  Final log-likelihood: {log_likelihood:.3f}")
    print(f"  Total decimation steps: {count}")
    print(f"\n  Decimated model saved:")
    print(f"    • Parameters: {checkpoint.file_paths['params_dec']}")
    print(f"    • Chains:     {checkpoint.file_paths['chains_dec']}")
    print("-" * 80)