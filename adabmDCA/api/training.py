"""High-level DCA training workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from adabmDCA.api.exceptions import OperationCancelledError
from adabmDCA.api.input_loading import load_training_inputs
from adabmDCA.api.model import DCAModel
from adabmDCA.api.results import TrainingProgress, TrainingResult
from adabmDCA.api.runtime import resolve_runtime
from adabmDCA.checkpoint import Checkpoint
from adabmDCA.fasta import get_tokens
from adabmDCA.input_loading import AlignmentInput, WeightInput
from adabmDCA.sampling import prepare_sampler
from adabmDCA.training import train_eaDCA, train_edDCA, train_edgeDCA, train_graph
from adabmDCA.training_config import (
    DEFAULT_ACTIVATION_FRACTION,
    DEFAULT_ACTIVATION_STEPS,
    DEFAULT_ALPHABET,
    DEFAULT_CLUSTERING_SEQID,
    DEFAULT_DECIMATION_RATE,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_L2_REGULARIZATION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MODEL_TYPE,
    DEFAULT_N_CHAINS,
    DEFAULT_N_SWEEPS,
    DEFAULT_SAMPLER,
    DEFAULT_SEED,
    DEFAULT_TARGET_DENSITY,
    DEFAULT_TARGET_PEARSON,
    TrainingConfig,
)
from adabmDCA.training_control import (
    StopReason,
    TrainingCancelled,
    TrainingController,
    TrainingCounters,
)
from adabmDCA.utils import init_chains, init_parameters

ProgressCallback = Callable[[TrainingProgress], None]
CancellationHook = Callable[[], bool]


def _progress_observer(progress: ProgressCallback | None):
    """Translate low-level records into the stable public progress event."""
    if progress is None:
        return None

    def notify(record: dict, counters: TrainingCounters) -> None:
        metrics = {
            key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in record.items()
            if isinstance(value, (int, float, torch.Tensor))
        }
        progress(
            TrainingProgress(
                epoch=int(record.get("Epochs", 0)),
                metrics=metrics,
                stage=counters.stage,
                gradient_steps=counters.gradient_steps,
                structure_steps=counters.structure_steps,
                sweeps=counters.sweeps,
            )
        )

    return notify


def train_model(
    data_path: AlignmentInput,
    *,
    config: TrainingConfig | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
    validation_path: AlignmentInput | None = None,
    weights_path: WeightInput | None = None,
    output_dir: str | Path | None = None,
    label: str | None = None,
    initial_params_path: str | Path | None = None,
    initial_chains_path: str | Path | None = None,
    alphabet: str = DEFAULT_ALPHABET,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    n_sweeps: int = DEFAULT_N_SWEEPS,
    sampler: str = DEFAULT_SAMPLER,
    n_chains: int = DEFAULT_N_CHAINS,
    target_pearson: float = DEFAULT_TARGET_PEARSON,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    max_gradient_steps: int | None = None,
    max_structure_steps: int | None = None,
    pseudocount: float | None = None,
    l2_regularization: float = DEFAULT_L2_REGULARIZATION,
    seed: int = DEFAULT_SEED,
    clustering_seqid: float = DEFAULT_CLUSTERING_SEQID,
    no_reweighting: bool = False,
    activation_steps: int = DEFAULT_ACTIVATION_STEPS,
    activation_fraction: float = DEFAULT_ACTIVATION_FRACTION,
    target_density: float = DEFAULT_TARGET_DENSITY,
    decimation_rate: float = DEFAULT_DECIMATION_RATE,
    device: str = DEFAULT_DEVICE,
    dtype: str = DEFAULT_DTYPE,
    use_wandb: bool = False,
    allow_signed_weights: bool = False,
    progress: ProgressCallback | None = None,
    is_cancelled: CancellationHook | None = None,
) -> TrainingResult:
    """Train a DCA model from a FASTA alignment.

    Persistence is optional. Set ``output_dir`` to retain the historical
    parameter, chain, weight, and log artifacts; omit it for an in-memory
    notebook workflow.
    """
    training_config = config or TrainingConfig(
        model_type=model_type,
        alphabet=alphabet,
        learning_rate=learning_rate,
        n_sweeps=n_sweeps,
        sampler=sampler,
        n_chains=n_chains,
        target_pearson=target_pearson,
        max_epochs=max_epochs,
        max_gradient_steps=max_gradient_steps,
        max_structure_steps=max_structure_steps,
        pseudocount=pseudocount,
        l2_regularization=l2_regularization,
        seed=seed,
        clustering_seqid=clustering_seqid,
        no_reweighting=no_reweighting,
        activation_steps=activation_steps,
        activation_fraction=activation_fraction,
        target_density=target_density,
        decimation_rate=decimation_rate,
        device=device,
        dtype=dtype,
        use_wandb=use_wandb,
    )
    model_type = training_config.model_type
    alphabet = training_config.alphabet
    learning_rate = training_config.learning_rate
    n_sweeps = training_config.n_sweeps
    sampler = training_config.sampler
    n_chains = training_config.n_chains
    target_pearson = training_config.target_pearson
    max_epochs = training_config.max_epochs
    max_gradient_steps = training_config.max_gradient_steps
    max_structure_steps = training_config.max_structure_steps
    pseudocount = training_config.pseudocount
    l2_regularization = training_config.l2_regularization
    seed = training_config.seed
    clustering_seqid = training_config.clustering_seqid
    no_reweighting = training_config.no_reweighting
    activation_steps = training_config.activation_steps
    activation_fraction = training_config.activation_fraction
    target_density = training_config.target_density
    decimation_rate = training_config.decimation_rate
    device = training_config.device
    dtype = training_config.dtype
    use_wandb = training_config.use_wandb
    if is_cancelled is not None and is_cancelled():
        raise OperationCancelledError("Model training was cancelled by the caller.")

    resolved_device, resolved_dtype = resolve_runtime(device, dtype)
    tokens = get_tokens(alphabet)
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    loaded_inputs = load_training_inputs(
        data_path,
        config=training_config,
        device=resolved_device,
        dtype=resolved_dtype,
        validation=validation_path,
        weights=weights_path,
        initial_params_path=initial_params_path,
        initial_chains_path=initial_chains_path,
        allow_signed_weights=allow_signed_weights,
    )
    dataset = loaded_inputs.training
    validation = loaded_inputs.validation
    if validation is not None:
        validation_pseudocount = 1.0 / validation.get_effective_size()
        fi_val, fij_val = validation.get_frequencies(
            pseudocount=training_config.empirical_pseudocount or validation_pseudocount
        )
    else:
        fi_val = fij_val = None

    effective_size = float(dataset.get_effective_size())
    effective_pseudocount = training_config.resolve_pseudocount(effective_size)

    dataset.shuffle()
    length = dataset.get_num_residues()
    num_states = dataset.get_num_states()
    if model_type == "edgeDCA":
        fi_target, fij_target = dataset.get_frequencies(pseudocount=training_config.empirical_pseudocount)
        fi_pseudocounted, fij_pseudocounted = dataset.get_frequencies(pseudocount=effective_pseudocount)
    else:
        fi_target, fij_target = dataset.get_frequencies(pseudocount=effective_pseudocount)
        fi_pseudocounted, fij_pseudocounted = fi_target, fij_target

    if loaded_inputs.initial_params is not None:
        params = loaded_inputs.initial_params
        mask = ~torch.isclose(params["coupling_matrix"], torch.zeros_like(params["coupling_matrix"]))
    else:
        params = init_parameters(fi=fi_target)
        if model_type in {"bmDCA", "edDCA"}:
            mask = torch.ones(
                (length, num_states, length, num_states),
                dtype=torch.bool,
                device=resolved_device,
            )
            mask[torch.arange(length), :, torch.arange(length), :] = 0
        else:
            mask = torch.zeros(
                (length, num_states, length, num_states),
                dtype=torch.bool,
                device=resolved_device,
            )

    if loaded_inputs.initial_chains is not None:
        chains = loaded_inputs.initial_chains
        log_weights = loaded_inputs.initial_log_weights
        n_chains = chains.shape[0]
    else:
        chains = init_chains(
            num_chains=n_chains,
            L=length,
            q=num_states,
            fi=fi_target,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        log_weights = torch.zeros(n_chains, device=resolved_device, dtype=resolved_dtype)

    artifacts: dict[str, Path] = {}
    checkpoint = None
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        stem = label or "adabmDCA"
        file_paths = {
            "log": str(folder / f"{stem}.log"),
            "params": str(folder / (f"{label}_params.dat" if label else "params.dat")),
            "chains": str(folder / (f"{label}_chains.fasta" if label else "chains.fasta")),
        }
        checkpoint_metadata = {
            **training_config.as_dict(),
            "label": label,
            "model": model_type,
            "data": str(loaded_inputs.training_alignment.alignment.source or "<memory>"),
            "val": (
                None
                if loaded_inputs.validation_alignment is None
                else str(loaded_inputs.validation_alignment.alignment.source or "<memory>")
            ),
            "alphabet": alphabet,
            "sampler": sampler,
            "nchains": n_chains,
            "nsweeps": n_sweeps,
            "lr": learning_rate,
            "dtype": dtype,
            "target": target_pearson,
            "gsteps": activation_steps,
            "factivate": activation_fraction,
            "seed": seed,
            "nepochs": max_epochs,
            "pseudocount": effective_pseudocount,
            "checkpoint_interval": training_config.resolved_checkpoint_interval,
        }
        checkpoint = Checkpoint(
            file_paths,
            tokens,
            checkpoint_metadata,
            use_wandb=use_wandb,
            config=training_config,
        )
        artifacts = {key: Path(value) for key, value in file_paths.items()}
        if weights_path is None:
            weights_output = folder / (f"{label}_weights.dat" if label else "weights.dat")
            np.savetxt(weights_output, dataset.weights.detach().cpu().numpy())
            artifacts["weights"] = weights_output

    limits = training_config.limits
    gradient_limit = limits.max_gradient_steps
    structure_limit = limits.max_structure_steps
    controller = TrainingController(
        limits=limits,
        checkpoint=checkpoint,
        observer=_progress_observer(progress),
        is_cancelled=is_cancelled,
    )
    sampling_function = prepare_sampler(sampler, params["bias"].device)

    try:
        if model_type == "bmDCA":
            chains, params, log_weights, history = train_graph(
                sampler=sampling_function,
                chains=chains,
                mask=mask,
                fi_target=fi_target,
                fij_target=fij_target,
                params=params,
                nsweeps=n_sweeps,
                lr=learning_rate,
                max_epochs=gradient_limit,
                target_pearson=target_pearson,
                fi_val=fi_val,
                fij_val=fij_val,
                controller=controller,
                log_weights=log_weights,
                l2_reg=l2_regularization,
                slope_tolerance=training_config.slope_tolerance,
            )
        elif model_type == "eaDCA":
            chains, params, log_weights, history = train_eaDCA(
                sampler=sampling_function,
                fi_target=fi_target,
                fij_target=fij_target,
                params=params,
                mask=mask,
                chains=chains,
                log_weights=log_weights,
                target_pearson=target_pearson,
                nsweeps=n_sweeps,
                max_epochs=structure_limit,
                max_gradient_steps=gradient_limit,
                pseudo_count=effective_pseudocount,
                lr=learning_rate,
                factivate=activation_fraction,
                gsteps=activation_steps,
                fi_val=fi_val,
                fij_val=fij_val,
                controller=controller,
                l2_reg=l2_regularization,
            )
        elif model_type == "edDCA":
            chains, params, log_weights, history = train_edDCA(
                sampler=sampling_function,
                chains=chains,
                log_weights=log_weights,
                fi_target=fi_target,
                fij_target=fij_target,
                params=params,
                mask=mask,
                lr=learning_rate,
                nsweeps=n_sweeps,
                target_pearson=target_pearson,
                target_density=target_density,
                drate=decimation_rate,
                max_epochs=structure_limit,
                max_gradient_steps=gradient_limit,
                controller=controller,
                inner_gradient_steps=training_config.inner_gradient_steps,
                fi_val=fi_val,
                fij_val=fij_val,
                l2_reg=l2_regularization,
            )
        else:
            chains, params, log_weights, history = train_edgeDCA(
                sampler=sampling_function,
                fi_target=fi_target,
                fij_target=fij_target,
                fi_pseudocounted=fi_pseudocounted,
                fij_pseudocounted=fij_pseudocounted,
                params=params,
                mask=mask,
                chains=chains,
                target_pearson=target_pearson,
                nsweeps=n_sweeps,
                max_epochs=structure_limit,
                pseudo_count=effective_pseudocount,
                fi_val=fi_val,
                fij_val=fij_val,
                controller=controller,
                empirical_pseudocount=training_config.edge_empirical_pseudocount,
                logz_chain_fraction=training_config.edge_logz_chain_fraction,
            )
    except TrainingCancelled as exc:
        raise OperationCancelledError(str(exc)) from exc

    trained = DCAModel(params, alphabet=alphabet, source=artifacts.get("params"))
    return TrainingResult(
        model=trained,
        history=history,
        chains=chains,
        log_weights=log_weights,
        pseudocount=float(effective_pseudocount),
        num_sequences=len(dataset),
        effective_sequences=effective_size,
        artifacts=artifacts,
        converged=controller.stop_reason
        in {
            StopReason.TARGET_PEARSON,
            StopReason.TARGET_DENSITY,
        },
        stop_reason=(controller.stop_reason.value if controller.stop_reason is not None else None),
        gradient_steps=controller.counters.gradient_steps,
        structure_steps=controller.counters.structure_steps,
        sweeps=controller.counters.sweeps,
        config=training_config,
        input_report=loaded_inputs.training_alignment.to_dict(),
    )
