from tqdm.autonotebook import tqdm
import time
from typing import Tuple, Callable, Dict, List, Optional
import torch
import os

from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from adabmDCA.utils import get_mask_save
from adabmDCA.statmech import _update_weights_AIS, compute_log_likelihood, compute_entropy, _compute_ess
from adabmDCA.checkpoint import Checkpoint
from adabmDCA.graph import activate_graph, compute_density, decimate_graph


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


def update_params(
    fi: torch.Tensor,
    fij: torch.Tensor,
    pi: torch.Tensor,
    pij: torch.Tensor,
    params: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    lr: float,
) -> Dict[str, torch.Tensor]:
    """Updates the parameters of the model.
    
    Args:
        fi (torch.Tensor): Single-point frequencies of the data.
        fij (torch.Tensor): Two-points frequencies of the data.
        pi (torch.Tensor): Single-point marginals of the model.
        pij (torch.Tensor): Two-points marginals of the model.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        mask (torch.Tensor): Mask of the interaction graph.
        lr (float): Learning rate.
        
    Returns:
        Dict[str, torch.Tensor]: Updated parameters.
    """
    
    # Compute the gradient
    grad = compute_gradient(fi=fi, fij=fij, pi=pi, pij=pij)
    
    # Update parameters
    with torch.no_grad():
        for key in params:
            params[key] += lr * grad[key]
        params["coupling_matrix"] *= mask # Remove autocorrelations
    
    return params


def train_graph(
    sampler: Callable,
    chains: torch.Tensor,
    mask: torch.Tensor,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    lr: float,
    max_epochs: int,
    target_pearson: float,
    fi_test: Optional[torch.Tensor] = None,
    fij_test: Optional[torch.Tensor] = None,
    checkpoint: Optional[Checkpoint] = None,
    check_slope: bool = False,
    log_weights: Optional[torch.Tensor] = None,
    progress_bar: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]:
    """Trains the model on a given graph until the target Pearson correlation is reached or the maximum number of epochs is exceeded.

    Args:
        sampler (Callable): Sampling function.
        chains (torch.Tensor): Markov chains simulated with the model.
        mask (torch.Tensor): Mask encoding the sparse graph.
        fi_target (torch.Tensor): Single-point frequencies of the data.
        fij_target (torch.Tensor): Two-point frequencies of the data.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        nsweeps (int): Number of Gibbs steps for each gradient estimation.
        lr (float): Learning rate.
        max_epochs (int): Maximum number of gradient updates to be done.
        target_pearson (float): Target Pearson coefficient.
        fi_test (Optional[torch.Tensor], optional): Single-point frequencies of the test data. Defaults to None.
        fij_test (Optional[torch.Tensor], optional): Two-point frequencies of the test data. Defaults to None.
        checkpoint (Optional[Checkpoint], optional): Checkpoint class to be used for saving the model. Defaults to None.
        check_slope (bool, optional): Whether to take into account the slope for the convergence criterion or not. Defaults to False.
        log_weights (Optional[torch.Tensor], optional): Log-weights used for the online computation of the log-likelihood. Defaults to None.
        progress_bar (bool, optional): Whether to display a progress bar or not. Defaults to True.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]: Updated chains and parameters, log-weights for the log-likelihood computation.
    """
    device = fi_target.device
    dtype = fi_target.dtype
    L, q = fi_target.shape
    time_start = time.time()
    
    # log_weights used for the online computing of the log-likelihood
    if log_weights is None:
        log_weights = torch.zeros(len(chains), device=device, dtype=dtype)
    logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
    log_likelihood = compute_log_likelihood(fi=fi, fij=fij, params=params, logZ=logZ)
    
    # Compute the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(data=chains)
    pij = get_freq_two_points(data=chains)
    history = {
        "epochs": [],
        "pearson": [],
        "slope": [],
    }
    
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

    pearson, slope = get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)
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
        pbar.set_description(f"Epochs: {epochs} - LL: {log_likelihood:.2f}")
    
    while not halt_condition(epochs, pearson, slope, check_slope):
        
        # Store the previous parameters
        params_prev = {key: value.clone() for key, value in params.items()}
        
        # Update the parameters
        params = update_params(
            fi=fi_target,
            fij=fij_target,
            pi=pi,
            pij=pij,
            params=params,
            mask=mask,
            lr=lr,
        )
        
        # Compute the weights for the AIS
        log_weights = _update_weights_AIS(
            prev_params=params_prev,
            curr_params=params,
            chains=chains,
            log_weights=log_weights,
        )
        
        # Update the Markov chains
        chains = sampler(chains=chains, params=params, nsweeps=nsweeps)
        epochs += 1
        
        # Compute the single-point and two-points frequencies of the simulated data
        pi = get_freq_single_point(data=chains)
        pij = get_freq_two_points(data=chains)
        pearson, slope = get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)
        logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
        log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)
        
        if progress_bar:
            pbar.n = min(max(0, float(pearson)), target_pearson)
            pbar.set_description(f"Epochs: {epochs} - LL: {log_likelihood:.2f}")
        
        history["epochs"].append(epochs)
        history["pearson"].append(pearson)
        history["slope"].append(slope)
            
        if checkpoint is not None:
            entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
            ess = _compute_ess(log_weights)
            if fi_test is not None and fij_test is not None:
                log_likelihood_test = compute_log_likelihood(fi=fi_test, fij=fij_test, params=params, logZ=logZ)
            else:
                log_likelihood_test = float("nan")
            checkpoint.log(
                {
                    "Epochs": epochs,
                    "Pearson": pearson,
                    "Slope": slope,
                    "LL_train": log_likelihood,
                    "LL_test": log_likelihood_test,
                    "ESS": ess,
                    "Entropy": entropy,
                    "Density": 1.0,
                    "Time": time.time() - time_start,
                }
            )
            
            # Save the model if a checkpoint is reached
            if checkpoint.check(epochs):
                checkpoint.save(
                    params=params,
                    mask=mask_save,
                    chains=chains,
                    log_weights=log_weights,
                )
                
    if progress_bar:
        pbar.close()
    
    if checkpoint is not None:            
        checkpoint.save(
            params=params,
            mask=mask_save,
            chains=chains,
            log_weights=log_weights,
        )
        
    return chains, params, log_weights, history


def train_eaDCA(
    sampler: Callable,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    params: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    chains: torch.Tensor,
    log_weights: torch.Tensor,
    target_pearson: float,
    nsweeps: int,
    max_epochs: int,
    pseudo_count: float,
    lr: float,
    factivate: float,
    gsteps: int,
    fi_test: Optional[torch.Tensor] = None,
    fij_test: Optional[torch.Tensor] = None,
    checkpoint: Optional[Checkpoint] = None,
    *args, **kwargs,
) -> None:
    """
    Fits an eaDCA model on the training data and saves the results in a file.

    Args:
        sampler (Callable): Sampling function to be used.
        fi_target (torch.Tensor): Single-point frequencies of the data.
        fij_target (torch.Tensor): Two-point frequencies of the data.
        params (Dict[str, torch.Tensor]): Initialization of the model's parameters.
        mask (torch.Tensor): Initialization of the coupling matrix's mask.
        chains (torch.Tensor): Initialization of the Markov chains.
        log_weights (torch.Tensor): Log-weights of the chains. Used to estimate the log-likelihood.
        target_pearson (float): Pearson correlation coefficient on the two-points statistics to be reached.
        nsweeps (int): Number of Monte Carlo steps to update the state of the model.
        max_epochs (int): Maximum number of epochs to be performed.
        pseudo_count (float): Pseudo count for the single and two points statistics. Acts as a regularization.
        lr (float): Learning rate.
        factivate (float): Fraction of inactive couplings to activate at each step.
        gsteps (int): Number of gradient updates to be performed on a given graph.
        fi_test (Optional[torch.Tensor], optional): Single-point frequencies of the test data. Defaults to None.
        fij_test (Optional[torch.Tensor], optional): Two-point frequencies of the test data. Defaults to None.
        checkpoint (Optional[Checkpoint], optional): Checkpoint class to be used to save the model. Defaults to None.
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
    if checkpoint is not None:
        checkpoint.checkpt_interval = 10 # Save the model every 10 graph updates
        checkpoint.max_epochs = max_epochs
    
    graph_upd = 0
    density = compute_density(mask) * 100
    L, q = fi_target.shape
    
    print("\n" + "-" * 80)
    print("[ACTIVATION PHASE]")
    print("-" * 80)
    print(f"  Target Pearson: {target_pearson:.2f}")
    print(f"  Activation rate: {factivate:.2%}")
    print(f"  Gradient steps per graph update: {gsteps}")
    print(f"  Initial density: {density:.3f}%")
        
    # Mask for saving only the upper-diagonal matrix
    mask_save = get_mask_save(L, q, device=device)
    
    # log_weights used for the online computing of the log-likelihood
    logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
    
    # Compute the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(data=chains)
    pij = get_freq_two_points(data=chains)
    pearson = max(0, float(get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)[0]))
    
    # Number of active couplings
    nactive = mask.sum()
    
    # Training loop
    time_start = time.time()
    log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)
    entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
    
    print(f"  Initial Pearson: {pearson:.4f}")
    print(f"  Initial log-likelihood: {log_likelihood:.3f}")
    print("-" * 80 + "\n")
        
    pbar = tqdm(initial=max(0, float(pearson)), total=target_pearson, colour="red", dynamic_ncols=True, ascii="-#",
                bar_format="{desc}: {percentage:.2f}% [{bar}] Pearson: {n:.3f}/{total_fmt} [{elapsed}]")
    pbar.set_description(f"Update: {graph_upd:3d} | Density: {density:6.3f}% | New: {0:4d} | LL: {log_likelihood:8.3f}")
    
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
        chains, params, log_weights, _ = train_graph(
            sampler=sampler,
            chains=chains,
            mask=mask,
            fi_target=fi_target,
            fij_target=fij_target,
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
        pi = get_freq_single_point(data=chains)
        pij = get_freq_two_points(data=chains)
        
        # Compute statistics of the training
        pearson, slope = get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)
        density = compute_density(mask) * 100
        logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
        log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)
        pbar.set_description(f"Update: {graph_upd:3d} | Density: {density:6.3f}% | New: {int(nactive - nactive_old):4d} | LL: {log_likelihood:8.3f}")

        # Save the model if a checkpoint is reached
        if checkpoint is not None:
            entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
            ess = _compute_ess(log_weights)
            checkpoint.log(
                {
                    "Epochs": graph_upd,
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
                
            if checkpoint.check(graph_upd):
                checkpoint.save(
                    params=params,
                    mask=torch.logical_and(mask, mask_save),
                    chains=chains,
                    log_weights=log_weights,
                    )
        pbar.n = min(max(0, float(pearson)), target_pearson)

    entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
    ess = _compute_ess(log_weights)
    if checkpoint is not None:
        checkpoint.log(
            {
                "Epochs": graph_upd,
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
    pbar.close()
    
    print("\n" + "-" * 80)
    print("  ACTIVATION COMPLETED")
    print("-" * 80)
    print(f"  Final density: {density:.3f}%")
    print(f"  Final Pearson: {pearson:.4f}")
    print(f"  Final log-likelihood: {log_likelihood:.3f}")
    print(f"  Total graph updates: {graph_upd}")
    if checkpoint is not None:
        print(f"\n  Model saved:")
        print(f"    • Parameters: {checkpoint.file_paths['params']}")
        print(f"    • Chains:     {checkpoint.file_paths['chains']}")
        print(f"    • Log file:   {checkpoint.file_paths['log']}")
    print("-" * 80)
    
    
def train_edDCA(
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
    fi_test: Optional[torch.Tensor] = None,
    fij_test: Optional[torch.Tensor] = None,
    *args, **kwargs,
) -> None:
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
        fi_test (Optional[torch.Tensor], optional): Single-point frequencies of the test data. Defaults to None.
        fij_test (Optional[torch.Tensor], optional): Two-point frequencies of the test data. Defaults to None.
    """
    MAX_EPOCHS = 10000
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
            fi_target=fi_target,
            fij_target=fij_target,
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
            fi_target=fi_target,
            fij_target=fij_target,
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
