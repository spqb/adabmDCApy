import time
from collections.abc import Callable

import torch

from adabmDCA.graph import (
    activate_graph_elements,
    compute_density,
    compute_Dkl_edge_activation,
    decimate_graph,
)
from adabmDCA.statmech import (
    _compute_ess,
    _update_logZ_edge_activation,
    _update_weights_AIS,
    compute_entropy,
    compute_log_likelihood,
)
from adabmDCA.stats import get_correlation_two_points, get_freq_single_point, get_freq_two_points
from adabmDCA.training_config import (
    DEFAULT_INNER_GRADIENT_STEPS,
    EDGE_EMPIRICAL_PSEUDOCOUNT,
    EDGE_LOGZ_CHAIN_FRACTION,
    SLOPE_TOLERANCE,
)
from adabmDCA.training_control import (
    StopReason,
    TrainingController,
    TrainingHistory,
    TrainingLimits,
    TrainingMetrics,
)
from adabmDCA.utils import get_mask_save

Sampler = Callable[..., torch.Tensor]


def compute_gradient(
    fi: torch.Tensor,
    fij: torch.Tensor,
    pi: torch.Tensor,
    pij: torch.Tensor,
) -> dict[str, torch.Tensor]:
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
    params: dict[str, torch.Tensor],
    mask: torch.Tensor,
    lr: float,
    l2_reg: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Updates the parameters of the model.

    Args:
        fi (torch.Tensor): Single-point frequencies of the data.
        fij (torch.Tensor): Two-points frequencies of the data.
        pi (torch.Tensor): Single-point marginals of the model.
        pij (torch.Tensor): Two-points marginals of the model.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        mask (torch.Tensor): Mask of the interaction graph.
        lr (float): Learning rate.
        l2_reg (float, optional): L2 regularization coefficient. Defaults to 0.0.

    Returns:
        Dict[str, torch.Tensor]: Updated parameters.
    """

    # Compute the gradient
    grad = compute_gradient(fi=fi, fij=fij, pi=pi, pij=pij)

    # Update parameters
    with torch.no_grad():
        for key, value in params.items():
            if key == "coupling_matrix":
                value += lr * (grad[key] - l2_reg * value)
            else:
                value += lr * grad[key]

        params["coupling_matrix"] *= mask  # Remove autocorrelations

    return params


def update_params_edge_activation(
    fij: torch.Tensor,
    pij: torch.Tensor,
    params: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> tuple[tuple[int, int], torch.Tensor, dict[str, torch.Tensor]]:
    """Updates the mask and the coupling parameters using the edge-activation algorithm.

    Args:
        fij (torch.Tensor): Two-point frequences of the dataset.
        pij (torch.Tensor): Two-point marginals of the model.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        mask (torch.Tensor): Mask.

    Returns:
        torch.Tensor: Indices of the activated edge.
        torch.Tensor: Updated mask.
        Dict[str, torch.Tensor]: Updated parameters.
    """
    Dkl = compute_Dkl_edge_activation(fij=fij, pij=pij)
    # (i,j) indices of the edge with the largest Dkl
    idx_edge = torch.argmax(Dkl)
    i_edge = int((idx_edge // Dkl.shape[1]).item())
    j_edge = int((idx_edge % Dkl.shape[1]).item())
    # Activate the edge in the mask
    mask[i_edge, :, j_edge, :] = 1.0
    mask[j_edge, :, i_edge, :] = 1.0
    # Update the coupling parameters of the activated edge
    params["coupling_matrix"][i_edge, :, j_edge, :] += torch.log(fij[i_edge, :, j_edge, :]) - torch.log(
        pij[i_edge, :, j_edge, :]
    )
    params["coupling_matrix"][j_edge, :, i_edge, :] += torch.log(fij[j_edge, :, i_edge, :]) - torch.log(
        pij[j_edge, :, i_edge, :]
    )

    return (i_edge, j_edge), mask, params


def train_graph(
    sampler: Sampler,
    chains: torch.Tensor,
    mask: torch.Tensor,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    params: dict[str, torch.Tensor],
    nsweeps: int,
    lr: float,
    max_epochs: int,
    target_pearson: float,
    fi_val: torch.Tensor | None = None,
    fij_val: torch.Tensor | None = None,
    check_slope: bool = False,
    log_weights: torch.Tensor | None = None,
    l2_reg: float = 0.0,
    controller: TrainingController | None = None,
    slope_tolerance: float = SLOPE_TOLERANCE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, TrainingHistory]:
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
        fi_val (Optional[torch.Tensor], optional): Single-point frequencies of the validation data. Defaults to None.
        fij_val (Optional[torch.Tensor], optional): Two-point frequencies of the validation data. Defaults to None.
        check_slope (bool, optional): Whether to take into account the slope for the convergence criterion or not. Defaults to False.
        log_weights (Optional[torch.Tensor], optional): Log-weights used for the online computation of the log-likelihood. Defaults to None.
        l2_reg (float, optional): L2 regularization coefficient. Defaults to 0.0.

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
    logZ = (
        torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))
    ).item()
    log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)

    # Compute the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(data=chains)
    pij = get_freq_two_points(data=chains)
    controller = controller or TrainingController(
        limits=TrainingLimits(max_gradient_steps=max_epochs),
    )
    controller.begin_stage("optimization")
    history = controller.history

    def should_continue(epoch: int, pearson: float, slope: float) -> bool:
        target_not_reached = pearson < target_pearson
        slope_not_converged = check_slope and abs(slope - 1.0) > slope_tolerance
        return epoch < max_epochs and (target_not_reached or slope_not_converged)

    # Mask for saving only the upper-diagonal coupling matrix
    mask_save = get_mask_save(L, q, device=device)

    pearson, slope = get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)
    epochs = 0

    while should_continue(epochs, pearson, slope):
        controller.check_cancellation()

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
            l2_reg=l2_reg,
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
        controller.add_gradient_steps(1, sweeps_per_step=nsweeps)

        # Compute the single-point and two-points frequencies of the simulated data
        pi = get_freq_single_point(data=chains)
        pij = get_freq_two_points(data=chains)
        pearson, slope = get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)
        logZ = (
            torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))
        ).item()
        log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)

        entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
        ess = _compute_ess(log_weights)
        if fi_val is not None and fij_val is not None:
            log_likelihood_val = compute_log_likelihood(fi=fi_val, fij=fij_val, params=params, logZ=logZ)
            pearson_val, slope_val = get_correlation_two_points(fij=fij_val, pij=pij, fi=fi_val, pi=pi)
        else:
            log_likelihood_val = float("nan")
            pearson_val = float("nan")
            slope_val = float("nan")

        controller.record(
            TrainingMetrics(
                pearson=pearson,
                slope=slope,
                ll_train=log_likelihood,
                ll_val=log_likelihood_val,
                pearson_val=pearson_val,
                slope_val=slope_val,
                ess=ess,
                entropy=entropy,
                density=compute_density(mask),
                elapsed_time=time.time() - time_start,
            ),
            epoch=epochs,
            snapshot={
                "params": params,
                "mask": mask_save,
                "chains": chains,
                "log_weights": log_weights,
            },
        )

    if pearson >= target_pearson and (not check_slope or abs(slope - 1.0) <= slope_tolerance):
        controller.set_stop_reason(StopReason.TARGET_PEARSON)
    else:
        controller.set_stop_reason(StopReason.MAX_GRADIENT_STEPS)
    controller.finalize(
        {
            "params": params,
            "mask": mask_save,
            "chains": chains,
            "log_weights": log_weights,
        }
    )

    return chains, params, log_weights, history


def train_eaDCA(
    sampler: Sampler,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    params: dict[str, torch.Tensor],
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
    fi_val: torch.Tensor | None = None,
    fij_val: torch.Tensor | None = None,
    l2_reg: float = 0.0,
    controller: TrainingController | None = None,
    max_gradient_steps: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, TrainingHistory]:
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
        fi_val (Optional[torch.Tensor], optional): Single-point frequencies of the validation data. Defaults to None.
        fij_val (Optional[torch.Tensor], optional): Two-point frequencies of the validation data. Defaults to None.
        l2_reg (float, optional): L2 regularization coefficient. Defaults to 0.0.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]: Updated chains and parameters, log-weights for the log-likelihood computation, and training history.
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
    controller = controller or TrainingController(
        limits=TrainingLimits(
            max_gradient_steps=max_gradient_steps,
            max_structure_steps=max_epochs,
        ),
    )
    controller.begin_stage(
        "activation",
        target_pearson=target_pearson,
        activation_fraction=factivate,
        gradient_steps_per_update=gsteps,
    )

    graph_upd = 0
    density = compute_density(mask)
    L, q = fi_target.shape

    # Mask for saving only the upper-diagonal matrix
    mask_save = get_mask_save(L, q, device=device)

    # log_weights used for the online computing of the log-likelihood
    logZ = (
        torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))
    ).item()

    # Compute the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(data=chains)
    pij = get_freq_two_points(data=chains)
    pearson = max(0, float(get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)[0]))

    # Training loop
    time_start = time.time()
    log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)
    history = controller.history

    while (
        pearson < target_pearson
        and not controller.structure_limit_reached()
        and not controller.gradient_limit_reached()
    ):
        controller.check_cancellation()
        # Compute the two-points frequencies of the simulated data with pseudo-count
        pij_Dkl = get_freq_two_points(data=chains, weights=None, pseudo_count=pseudo_count)
        # Update the graph
        nactivate = int(((L**2 * q**2) - mask.sum().item()) * factivate)
        mask = activate_graph_elements(
            mask=mask,
            fij=fij_target,
            pij=pij_Dkl,
            nactivate=nactivate,
        )
        # Bring the model at convergence on the graph
        remaining_gradient_steps = controller.remaining_gradient_steps()
        inner_steps = gsteps
        if remaining_gradient_steps is not None:
            inner_steps = min(inner_steps, remaining_gradient_steps)
        chains, params, log_weights, inner_history = train_graph(
            sampler=sampler,
            chains=chains,
            mask=mask,
            fi_target=fi_target,
            fij_target=fij_target,
            params=params,
            nsweeps=nsweeps,
            lr=lr,
            max_epochs=inner_steps,
            target_pearson=target_pearson,
            log_weights=log_weights,
            check_slope=False,
            l2_reg=l2_reg,
        )

        graph_upd += 1
        controller.add_structure_step()
        controller.add_gradient_steps(len(inner_history["Epochs"]), sweeps_per_step=nsweeps)

        # Compute the single-point and two-points frequencies of the simulated data
        pi = get_freq_single_point(data=chains)
        pij = get_freq_two_points(data=chains)

        # Compute statistics of the training
        pearson, slope = get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)
        density = compute_density(mask)
        logZ = (
            torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))
        ).item()
        log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)
        entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
        ess = _compute_ess(log_weights)
        if fi_val is not None and fij_val is not None:
            log_likelihood_val = compute_log_likelihood(fi=fi_val, fij=fij_val, params=params, logZ=logZ)
            pearson_val, slope_val = get_correlation_two_points(fij=fij_val, pij=pij, fi=fi_val, pi=pi)
        else:
            log_likelihood_val = float("nan")
            pearson_val = float("nan")
            slope_val = float("nan")
        controller.record(
            TrainingMetrics(
                pearson=pearson,
                slope=slope,
                ll_train=log_likelihood,
                ll_val=log_likelihood_val,
                pearson_val=pearson_val,
                slope_val=slope_val,
                ess=ess,
                entropy=entropy,
                density=density,
                elapsed_time=time.time() - time_start,
            ),
            epoch=graph_upd,
            snapshot={
                "params": params,
                "mask": torch.logical_and(mask, mask_save),
                "chains": chains,
                "log_weights": log_weights,
            },
        )

    if pearson >= target_pearson:
        controller.set_stop_reason(StopReason.TARGET_PEARSON)
    elif controller.gradient_limit_reached():
        controller.set_stop_reason(StopReason.MAX_GRADIENT_STEPS)
    else:
        controller.set_stop_reason(StopReason.MAX_STRUCTURE_STEPS)
    controller.finalize(
        {
            "params": params,
            "mask": torch.logical_and(mask, mask_save),
            "chains": chains,
            "log_weights": log_weights,
        }
    )
    return chains, params, log_weights, history


def train_edDCA(
    sampler: Sampler,
    chains: torch.Tensor,
    log_weights: torch.Tensor,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    params: dict[str, torch.Tensor],
    mask: torch.Tensor,
    lr: float,
    nsweeps: int,
    target_pearson: float,
    target_density: float,
    drate: float,
    fi_val: torch.Tensor | None = None,
    fij_val: torch.Tensor | None = None,
    l2_reg: float = 0.0,
    max_epochs: int = 10_000,
    controller: TrainingController | None = None,
    max_gradient_steps: int | None = None,
    inner_gradient_steps: int = DEFAULT_INNER_GRADIENT_STEPS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, TrainingHistory]:
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
        fi_val (Optional[torch.Tensor], optional): Single-point frequencies of the validation data. Defaults to None.
        fij_val (Optional[torch.Tensor], optional): Two-point frequencies of the validation data. Defaults to None.
        l2_reg (float, optional): L2 regularization coefficient. Defaults to 0.0.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]: Updated chains and parameters, log-weights for the log-likelihood computation, and training history.
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
    controller = controller or TrainingController(
        limits=TrainingLimits(
            max_gradient_steps=max_gradient_steps,
            max_structure_steps=max_epochs,
        ),
    )
    controller.begin_stage("equilibration", target_pearson=target_pearson)

    # Get the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(data=chains)
    pij = get_freq_two_points(data=chains)
    pearson, _ = get_correlation_two_points(fi=fi_target, pi=pi, fij=fij_target, pij=pij)
    if pearson < target_pearson and not controller.gradient_limit_reached():
        remaining_gradient_steps = controller.remaining_gradient_steps()
        inner_steps = inner_gradient_steps
        if remaining_gradient_steps is not None:
            inner_steps = min(inner_steps, remaining_gradient_steps)
        chains, params, log_weights, inner_history = train_graph(
            sampler=sampler,
            chains=chains,
            log_weights=log_weights,
            mask=mask,
            fi_target=fi_target,
            fij_target=fij_target,
            fi_val=fi_val,
            fij_val=fij_val,
            params=params,
            nsweeps=nsweeps,
            lr=lr,
            max_epochs=inner_steps,
            target_pearson=target_pearson,
            check_slope=False,
            l2_reg=l2_reg,
        )
        controller.add_gradient_steps(len(inner_history["Epochs"]), sweeps_per_step=nsweeps)
        pi = get_freq_single_point(data=chains)
        pij = get_freq_two_points(data=chains)
        pearson, _ = get_correlation_two_points(fi=fi_target, pi=pi, fij=fij_target, pij=pij)
        # Save the equilibrated parameters
        controller.save_snapshot(
            {
                "params": params,
                "mask": mask,
                "chains": chains,
                "log_weights": log_weights,
            }
        )

    # Mask for saving only the upper-diagonal matrix
    mask_save = get_mask_save(L, q, device=device)

    initial_density = compute_density(mask)
    controller.begin_stage(
        "decimation",
        target_density=target_density,
        decimation_rate=drate,
        initial_density=initial_density,
    )
    density = compute_density(mask)
    count = 0

    history = controller.history

    # Compute the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(data=chains)
    pij = get_freq_two_points(data=chains)
    pearson, slope = get_correlation_two_points(fi=fi_target, pi=pi, fij=fij_target, pij=pij)
    density = compute_density(mask)
    logZ = (
        torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))
    ).item()
    log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)

    while (
        density > target_density
        and not controller.structure_limit_reached()
        and not controller.gradient_limit_reached()
    ):
        controller.check_cancellation()
        count += 1

        # Store the previous parameters
        prev_params = {key: value.clone() for key, value in params.items()}

        # Decimate the model
        params, mask = decimate_graph(pij=pij, params=params, mask=mask, drate=drate)

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
        controller.add_structure_step(sweeps=nsweeps)

        # Bring the model at convergence on the graph
        remaining_gradient_steps = controller.remaining_gradient_steps()
        inner_steps = inner_gradient_steps
        if remaining_gradient_steps is not None:
            inner_steps = min(inner_steps, remaining_gradient_steps)
        chains, params, log_weights, inner_history = train_graph(
            sampler=sampler,
            chains=chains,
            log_weights=log_weights,
            mask=mask,
            fi_target=fi_target,
            fij_target=fij_target,
            params=params,
            nsweeps=nsweeps,
            lr=lr,
            max_epochs=inner_steps,
            target_pearson=target_pearson,
            check_slope=False,
            l2_reg=l2_reg,
        )
        controller.add_gradient_steps(len(inner_history["Epochs"]), sweeps_per_step=nsweeps)

        # Compute the single-point and two-points frequencies of the simulated data
        pi = get_freq_single_point(data=chains)
        pij = get_freq_two_points(data=chains)

        pearson, slope = get_correlation_two_points(fi=fi_target, pi=pi, fij=fij_target, pij=pij)
        density = compute_density(mask)
        logZ = (
            torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))
        ).item()
        log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)

        entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
        ess = _compute_ess(log_weights)
        if fi_val is not None and fij_val is not None:
            log_likelihood_val = compute_log_likelihood(fi=fi_val, fij=fij_val, params=params, logZ=logZ)
            pearson_val, slope_val = get_correlation_two_points(fij=fij_val, pij=pij, fi=fi_val, pi=pi)
        else:
            log_likelihood_val = float("nan")
            pearson_val = float("nan")
            slope_val = float("nan")
        controller.record(
            TrainingMetrics(
                pearson=pearson,
                slope=slope,
                ll_train=log_likelihood,
                ll_val=log_likelihood_val,
                pearson_val=pearson_val,
                slope_val=slope_val,
                ess=ess,
                entropy=entropy,
                density=density,
                elapsed_time=time.time() - time_start,
            ),
            epoch=count,
            snapshot={
                "params": params,
                "mask": torch.logical_and(mask, mask_save),
                "chains": chains,
                "log_weights": log_weights,
            },
        )

    if density <= target_density:
        controller.set_stop_reason(StopReason.TARGET_DENSITY)
    elif controller.gradient_limit_reached():
        controller.set_stop_reason(StopReason.MAX_GRADIENT_STEPS)
    else:
        controller.set_stop_reason(StopReason.MAX_STRUCTURE_STEPS)
    controller.finalize(
        {
            "params": params,
            "mask": torch.logical_and(mask, mask_save),
            "chains": chains,
            "log_weights": log_weights,
        }
    )

    return chains, params, log_weights, history


def train_edgeDCA(
    sampler: Sampler,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    fi_pseudocounted: torch.Tensor,
    fij_pseudocounted: torch.Tensor,
    params: dict[str, torch.Tensor],
    mask: torch.Tensor,
    chains: torch.Tensor,
    target_pearson: float,
    nsweeps: int,
    max_epochs: int,
    pseudo_count: float,
    fi_val: torch.Tensor | None = None,
    fij_val: torch.Tensor | None = None,
    controller: TrainingController | None = None,
    empirical_pseudocount: float = EDGE_EMPIRICAL_PSEUDOCOUNT,
    logz_chain_fraction: float = EDGE_LOGZ_CHAIN_FRACTION,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, TrainingHistory]:
    """
    Fits an edge activation DCA model (edgeDCA) on the training data and saves the results in a file.

    Args:
        sampler (Callable): Sampling function to be used.
        fi_target (torch.Tensor): Single-point frequencies of the data.
        fij_target (torch.Tensor): Two-point frequencies of the data.
        fi_pseudocounted (torch.Tensor): Pseudocounted single-point frequencies.
        fij_pseudocounted (torch.Tensor): Pseudocounted two-point frequencies.
        params (Dict[str, torch.Tensor]): Initialization of the model's parameters.
        mask (torch.Tensor): Initialization of the coupling matrix's mask.
        chains (torch.Tensor): Initialization of the Markov chains.
        target_pearson (float): Pearson correlation coefficient on the two-points statistics to be reached.
        nsweeps (int): Number of Monte Carlo steps to update the state of the model.
        max_epochs (int): Maximum number of epochs to be performed.
        pseudo_count (float): Pseudo count for the single and two points statistics. Acts as a regularization.
        fi_val (Optional[torch.Tensor], optional): Single-point frequencies of the validation data. Defaults to None.
        fij_val (Optional[torch.Tensor], optional): Two-point frequencies of the validation data. Defaults to None.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]: Updated chains and parameters, log-weights for the log-likelihood computation, and training history.
    """
    # Fraction of chains used to estimate the logZ for the log-likelihood computation.
    num_chains_logZ_estimate = max(1, int(chains.shape[0] * logz_chain_fraction))

    # Check the input sizes
    if fi_target.dim() != 2:
        raise ValueError("fi_target must be a 2D tensor")
    if fij_target.dim() != 4:
        raise ValueError("fij_target must be a 4D tensor")
    if chains.dim() != 3:
        raise ValueError("chains must be a 3D tensor")

    device = fi_target.device
    controller = controller or TrainingController(
        limits=TrainingLimits(max_structure_steps=max_epochs),
    )
    controller.begin_stage("activation", target_pearson=target_pearson)

    graph_upd = 0
    density = compute_density(mask)
    L, q = fi_target.shape

    # Mask for saving only the upper-diagonal matrix
    mask_save = get_mask_save(L, q, device=device)

    # Initial logZ with the profile model (without couplings)
    logZ = 1.0

    # Compute the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(
        data=chains[num_chains_logZ_estimate:],
        pseudo_count=empirical_pseudocount,
    )
    pij = get_freq_two_points(
        data=chains[num_chains_logZ_estimate:],
        pseudo_count=empirical_pseudocount,
    )
    pij_pseudocounted = get_freq_two_points(data=chains[num_chains_logZ_estimate:], pseudo_count=pseudo_count)
    pearson = max(0, float(get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)[0]))

    # Training loop
    time_start = time.time()
    log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)
    history = controller.history

    while pearson < target_pearson and not controller.structure_limit_reached():
        controller.check_cancellation()
        # Update the graph
        ids_edge, mask, params = update_params_edge_activation(
            fij=fij_pseudocounted,
            pij=pij_pseudocounted,
            params=params,
            mask=mask,
        )
        chains = sampler(chains=chains, params=params, nsweeps=nsweeps)
        graph_upd += 1
        controller.add_structure_step(sweeps=nsweeps)

        # Compute the single-point and two-points frequencies of the simulated data
        pi = get_freq_single_point(
            data=chains[num_chains_logZ_estimate:],
            pseudo_count=empirical_pseudocount,
        )
        pij = get_freq_two_points(
            data=chains[num_chains_logZ_estimate:],
            pseudo_count=empirical_pseudocount,
        )
        pij_pseudocounted = get_freq_two_points(data=chains[num_chains_logZ_estimate:], pseudo_count=pseudo_count)

        # Compute statistics of the training
        pearson, slope = get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)
        density = compute_density(mask)
        logZ = _update_logZ_edge_activation(
            logZ=logZ,
            ids_edge=ids_edge,
            fij=fij_pseudocounted,
            pij=pij_pseudocounted,
            chains_estimate=chains[:num_chains_logZ_estimate],
        )
        log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)
        entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
        if fi_val is not None and fij_val is not None:
            log_likelihood_val = compute_log_likelihood(fi=fi_val, fij=fij_val, params=params, logZ=logZ)
            pearson_val, slope_val = get_correlation_two_points(fij=fij_val, pij=pij, fi=fi_val, pi=pi)
        else:
            log_likelihood_val = float("nan")
            pearson_val = float("nan")
            slope_val = float("nan")
        controller.record(
            TrainingMetrics(
                pearson=pearson,
                slope=slope,
                ll_train=log_likelihood,
                ll_val=log_likelihood_val,
                pearson_val=pearson_val,
                slope_val=slope_val,
                ess=1.0,
                entropy=entropy,
                density=density,
                elapsed_time=time.time() - time_start,
            ),
            epoch=graph_upd,
            snapshot={
                "params": params,
                "mask": torch.logical_and(mask, mask_save),
                "chains": chains,
                "log_weights": torch.ones(len(chains), device=chains.device, dtype=chains.dtype),
            },
        )

    if pearson >= target_pearson:
        controller.set_stop_reason(StopReason.TARGET_PEARSON)
    else:
        controller.set_stop_reason(StopReason.MAX_STRUCTURE_STEPS)
    final_log_weights = torch.ones(len(chains), device=chains.device, dtype=chains.dtype)
    controller.finalize(
        {
            "params": params,
            "mask": torch.logical_and(mask, mask_save),
            "chains": chains,
            "log_weights": final_log_weights,
        }
    )
    return chains, params, final_log_weights, history
