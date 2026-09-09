"""High-level thermodynamic-integration entropy estimation."""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from adabmDCA._validation import validate_integer, validate_seed
from adabmDCA.api.exceptions import (
    ConvergenceError,
    InputValidationError,
    ModelCompatibilityError,
    OperationCancelledError,
)
from adabmDCA.api.model import load_model
from adabmDCA.api.results import (
    ThermodynamicIntegrationProgress,
    ThermodynamicIntegrationResult,
)
from adabmDCA.dca import get_seqid
from adabmDCA.functional import one_hot
from adabmDCA.input_loading import AlignmentInput, AlignmentLoadConfig, load_alignment
from adabmDCA.io import load_chains
from adabmDCA.sampling import prepare_sampler
from adabmDCA.statmech import compute_energy
from adabmDCA.utils import init_chains, resample_sequences

EntropyProgress = Callable[[ThermodynamicIntegrationProgress], None]


def estimate_entropy(
    *,
    model: str | Path,
    natural_alignment: AlignmentInput,
    target_alignment: AlignmentInput,
    initial_chains_path: str | Path | None = None,
    n_chains: int = 10_000,
    n_sweeps: int = 100,
    n_steps: int = 100,
    theta_max: float = 5.0,
    theta_sweeps: int = 100,
    zero_sweeps: int = 100,
    target_fraction: float = 0.1,
    max_theta_iterations: int = 10_000,
    sampler: str = "metropolis",
    alphabet: str = "protein",
    seed: int = 0,
    device: str = "auto",
    dtype: str = "float32",
    output_dir: str | Path | None = None,
    label: str = "entropy",
    progress: EntropyProgress | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ThermodynamicIntegrationResult:
    """Estimate model entropy with bounded, observable thermodynamic integration.

    Use the first valid sequence in ``target_alignment`` in input order. If
    multiple valid sequences are provided, emit a warning and ignore the rest.
    """
    for name, value in {
        "n_chains": n_chains,
        "n_sweeps": n_sweeps,
        "theta_sweeps": theta_sweeps,
        "zero_sweeps": zero_sweeps,
        "max_theta_iterations": max_theta_iterations,
    }.items():
        validate_integer(name, value)
    validate_integer("n_steps", n_steps, minimum=2)
    validate_seed(seed)
    if not math.isfinite(theta_max) or theta_max <= 0:
        raise InputValidationError("theta_max must be positive.")
    if not 0.0 < target_fraction < 1.0:
        raise InputValidationError("target_fraction must be between 0 and 1.")

    loaded_model = load_model(model, alphabet=alphabet, device=device, dtype=dtype)
    runtime_device = loaded_model.params["bias"].device
    runtime_dtype = loaded_model.params["bias"].dtype
    length = loaded_model.metadata.length
    num_states = loaded_model.metadata.num_states
    torch.manual_seed(seed)
    if runtime_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    load_alignment(
        natural_alignment,
        config=AlignmentLoadConfig(
            alphabet=loaded_model.alphabet,
            invalid_sequences="drop",
            expected_length=length,
        ),
    )
    target = load_alignment(
        target_alignment,
        config=AlignmentLoadConfig(
            alphabet=loaded_model.alphabet,
            invalid_sequences="drop",
            expected_length=length,
        ),
    )
    if target.alignment.num_sequences > 1:
        warnings.warn(
            f"target_alignment contains {target.alignment.num_sequences} valid sequences; "
            f"using only the first ({target.alignment.names[0]!r}) for entropy estimation.",
            UserWarning,
            stacklevel=2,
        )
    target_sequence = one_hot(
        torch.as_tensor(target.encoded_sequences[0], device=runtime_device, dtype=torch.int64),
        num_classes=num_states,
    ).to(runtime_dtype)
    sample = prepare_sampler(sampler, loaded_model.params["bias"].device)
    if initial_chains_path is None:
        chains = init_chains(n_chains, length, num_states, device=runtime_device, dtype=runtime_dtype)
    else:
        chains = load_chains(
            str(initial_chains_path),
            tokens=loaded_model.tokens,
            device=runtime_device,
            dtype=runtime_dtype,
        )[0]
        if chains.ndim != 3 or tuple(chains.shape[1:]) != (length, num_states):
            raise ModelCompatibilityError(
                "Initial chains do not match the loaded model.",
                details={
                    "chain_shape": tuple(chains.shape),
                    "expected_suffix": (length, num_states),
                },
            )
        if len(chains) != n_chains:
            weights = torch.ones(len(chains), device=runtime_device, dtype=runtime_dtype) / len(chains)
            chains = resample_sequences(chains, weights=weights, nextract=n_chains)

    def check_cancelled() -> None:
        if is_cancelled is not None and is_cancelled():
            raise OperationCancelledError("Thermodynamic integration was cancelled by the caller.")

    params = loaded_model.params
    check_cancelled()
    chains_zero = sample(chains, params, zero_sweeps)
    average_energy_zero = compute_energy(chains_zero, params).mean()
    biased_params = {name: value.clone() for name, value in params.items()}
    biased_params["bias"] = params["bias"] + theta_max * target_sequence
    biased_chains = init_chains(n_chains, length, num_states, device=runtime_device, dtype=runtime_dtype)
    biased_chains = sample(biased_chains, biased_params, theta_sweeps)
    identities = get_seqid(biased_chains, target_sequence)
    observed_fraction = float((identities == length).sum().item() / n_chains)

    for iteration in range(max_theta_iterations):
        if observed_fraction > target_fraction:
            break
        check_cancelled()
        theta_max *= 1.01
        biased_params["bias"] = params["bias"] + theta_max * target_sequence
        biased_chains = sample(biased_chains, biased_params, 100)
        identities = get_seqid(biased_chains, target_sequence)
        observed_fraction = float((identities == length).sum().item() / n_chains)
        if progress is not None:
            progress(
                ThermodynamicIntegrationProgress(
                    stage="theta_search",
                    completed=iteration + 1,
                    total=max_theta_iterations,
                    theta=theta_max,
                    mean_sequence_identity=float(identities.float().mean().item()),
                )
            )
    else:
        if observed_fraction <= target_fraction:
            raise ConvergenceError(
                "Could not reach the requested target fraction before max_theta_iterations.",
                details={
                    "theta": theta_max,
                    "observed_fraction": observed_fraction,
                    "requested_fraction": target_fraction,
                    "max_theta_iterations": max_theta_iterations,
                },
            )

    identities = get_seqid(biased_chains, target_sequence)
    exact = identities == length
    free_energy = np.log(observed_fraction) + compute_energy(biased_chains[exact], biased_params).mean()
    thetas = torch.linspace(0, theta_max, n_steps, device=runtime_device, dtype=runtime_dtype)
    factor = theta_max / (2 * n_steps)
    entropy = average_energy_zero - free_energy
    history: dict[str, list[float]] = {
        "step": [],
        "theta": [],
        "free_energy": [],
        "entropy": [],
        "mean_sequence_identity": [],
        "elapsed_seconds": [],
    }
    started = time.monotonic()
    for index, theta in enumerate(thetas):
        check_cancelled()
        biased_params["bias"] = params["bias"] + theta * target_sequence
        biased_chains = sample(biased_chains, biased_params, n_sweeps)
        identities = get_seqid(biased_chains, target_sequence)
        mean_identity = identities.float().mean()
        free_energy += factor * mean_identity if index in {0, n_steps - 1} else 2 * factor * mean_identity
        entropy = average_energy_zero - free_energy
        history["step"].append(float(index))
        history["theta"].append(float(theta.item()))
        history["free_energy"].append(float(free_energy.item()))
        history["entropy"].append(float(entropy.item()))
        history["mean_sequence_identity"].append(float(mean_identity.item()))
        history["elapsed_seconds"].append(time.monotonic() - started)
        if progress is not None:
            progress(
                ThermodynamicIntegrationProgress(
                    stage="integration",
                    completed=index + 1,
                    total=n_steps,
                    theta=float(theta.item()),
                    entropy=float(entropy.item()),
                    mean_sequence_identity=float(mean_identity.item()),
                )
            )

    result = ThermodynamicIntegrationResult(
        entropy=float(entropy.item()),
        free_energy=float(free_energy.item()),
        theta_max=theta_max,
        target_fraction=observed_fraction,
        history=history,
        model=loaded_model.metadata,
    )
    if output_dir is None:
        return result
    artifacts = result.save_bundle(output_dir, label=label)
    return replace(result, artifacts=artifacts)
