"""High-level DCA training workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from adabmDCA.api.input_loading import load_training_inputs
from adabmDCA.api.model import DCAModel
from adabmDCA.api.results import (
    TrainingDatasetSummary,
    TrainingInitialization,
    TrainingProgress,
    TrainingResult,
)
from adabmDCA.api.runtime import resolve_runtime
from adabmDCA.checkpoint import Checkpoint
from adabmDCA.exceptions import InputValidationError, OperationCancelledError
from adabmDCA.fasta import get_tokens
from adabmDCA.input_loading import AlignmentInput, WeightInput
from adabmDCA.ptt import PTTConfig, PTTSampler
from adabmDCA.sampling import prepare_training_sampler
from adabmDCA.training import train_eaDCA, train_edDCA, train_edgeDCA, train_graph
from adabmDCA.training_config import (
    DEFAULT_ACTIVATION_FRACTION,
    DEFAULT_ACTIVATION_STEPS,
    DEFAULT_ALPHABET,
    DEFAULT_CHECKPOINT_INTERVAL,
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
    StageProgress,
    StopReason,
    TrainingCancelled,
    TrainingController,
    TrainingCounters,
)
from adabmDCA.utils import init_chains, init_parameters

ProgressCallback = Callable[[TrainingProgress], None]
StageProgressCallback = Callable[[StageProgress], None]
InitializationCallback = Callable[[TrainingInitialization], None]
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
        acceptance_rates = record.get("ptt_acceptance_rates")
        if isinstance(acceptance_rates, (list, tuple)):
            metrics["ptt_acceptance_rates"] = tuple(float(value) for value in acceptance_rates)
        progress(
            TrainingProgress(
                epoch=int(record.get("Epochs", 0)),
                metrics=metrics,
                stage=counters.stage,
                gradient_steps=counters.gradient_steps,
                structure_steps=counters.structure_steps,
                sweeps=counters.sweeps,
                partition_estimate=(None if "logZ_method" not in record else {
                    "log_z": record["logZ"], "method": record["logZ_method"],
                    "status": record["logZ_status"], "model_version": record["model_version"],
                    "ladder_version": record["ladder_version"],
                }),
            )
        )

    return notify


def _dataset_summary(dataset, loaded_alignment) -> TrainingDatasetSummary:
    report = loaded_alignment.to_dict()
    return TrainingDatasetSummary(
        source=report["source"],
        original_sequences=int(report["original_sequences"]),
        retained_sequences=int(report["retained_sequences"]),
        removed_invalid=len(report["dropped_indices"]),
        removed_duplicates=len(report["duplicate_indices"]),
        sequence_length=int(report["sequence_length"]),
        num_states=dataset.get_num_states(),
        effective_sequences=float(dataset.get_effective_size()),
    )


def _finish_log(checkpoint, status: str, controller, history, *, error: BaseException | None = None) -> None:
    if checkpoint is None:
        return
    finish = getattr(checkpoint, "finish", None)
    if finish is None:
        return
    final = {key: values[-1] for key, values in history.items() if values}
    summary = {
        "stop_reason": None if controller.stop_reason is None else controller.stop_reason.value,
        "converged": controller.stop_reason in {
            StopReason.TARGET_PEARSON, StopReason.TARGET_DENSITY, StopReason.VALIDATION_PLATEAU,
        },
        "gradient_steps": controller.counters.gradient_steps,
        "structure_steps": controller.counters.structure_steps,
        "sweeps": controller.counters.sweeps,
        "elapsed_seconds": final.get("Time"),
        "final_pearson": final.get("Pearson"),
        "final_validation_pearson": final.get("Pearson_val"),
        "final_density": final.get("Density"),
    }
    if error is not None:
        summary["error"] = f"{type(error).__name__}: {error}"
    finish(status, summary)


def train_model(
    data_path: AlignmentInput,
    *,
    config: TrainingConfig | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
    ptt: PTTConfig | None = None,
    ptt_resume: str | Path | None = None,
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
    checkpoint_interval: int | None = DEFAULT_CHECKPOINT_INTERVAL,
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
    stage_progress: StageProgressCallback | None = None,
    on_initialized: InitializationCallback | None = None,
    is_cancelled: CancellationHook | None = None,
) -> TrainingResult:
    """Train a DCA model from an alignment and return its in-memory result.

    ``data_path`` may be an :class:`Alignment` or a FASTA, gzip-compressed
    FASTA, or Stockholm path. Invalid sequences are removed and duplicate
    sequences are collapsed. Supplied weights may match either the original
    rows or the retained rows. The resolved filtering decisions are available
    in ``result.input_report``.

    Pass ``config=TrainingConfig(...)`` to specify the training policy as one
    object. When supplied, ``config`` is authoritative for training settings;
    the individual settings below do not override it. Input paths, output
    location, callbacks, and cancellation are still taken from this call.

    Args:
        data_path: Training alignment or path to an alignment file.
        config: Validated training settings. If omitted, the individual
            training options below are used to construct a ``TrainingConfig``.
        model_type: ``"bmDCA"``, ``"eaDCA"``, ``"edDCA"``, or ``"edgeDCA"``.
        ptt: Optional PTT settings for bmDCA, eaDCA or edgeDCA training. May also
            be supplied through ``config.ptt``.
        ptt_resume: Full HDF5 PTT archive from which to resume a compatible
            run. Requires PTT settings.
        validation_path: Optional alignment used for validation metrics.
        weights_path: Optional path, sequence, NumPy array, or tensor of
            training-sequence weights.
        output_dir: Directory for parameter, chain, weight, and log files.
            If omitted, the result stays in memory and no files are written.
        label: Stem used for output files when ``output_dir`` is set.
        initial_params_path: Optional model-parameter file for a warm start.
        initial_chains_path: Optional Markov-chain file for a warm start.
        alphabet: Standard alphabet name or explicit ordered token string.
        learning_rate: Parameter-update step size.
        n_sweeps: Sampler sweeps per training update.
        sampler: ``"metropolis"``, ``"gibbs"`` or ``"metropolized_gibbs"``.
        n_chains: Number of model chains when none are loaded.
        target_pearson: Target correlation for the two-point statistics.
        max_epochs: Legacy training-step limit; use the explicit limits below
            when distinguishing gradient and structure steps.
        max_gradient_steps: Optional limit on parameter updates.
        max_structure_steps: Optional limit on graph updates.
        checkpoint_interval: Steps between periodic parameter and chain saves.
            The final state is saved when ``output_dir`` is set.
        pseudocount: Empirical-frequency pseudocount, or ``None`` to derive it
            from the effective sequence count.
        l2_regularization: Strength of the L2 penalty on model parameters.
        seed: Random seed for initialization and sampling.
        clustering_seqid: Sequence-identity threshold used for reweighting.
        no_reweighting: Use equal weights instead of sequence-identity weights.
        activation_steps: Gradient steps per activation update in eaDCA.
        activation_fraction: Fraction of candidate couplings activated per
            structure update in eaDCA.
        target_density: Target coupling density for edDCA.
        decimation_rate: Fraction of couplings removed per edDCA update.
        device: Runtime device; ``"auto"`` selects an available accelerator.
        dtype: Training precision, ``"float32"``, ``"float64"``, or
            ``"bfloat16"``.
        use_wandb: Enable Weights & Biases logging when output is configured.
        allow_signed_weights: Permit signed input weights for experimental
            reintegration workflows.
        progress: Optional callback taking exactly one
            :class:`TrainingProgress` event and returning ``None``. It runs
            synchronously after each recorded metrics update. The event has
            ``epoch`` (the reported step), ``stage`` (for example,
            ``"optimization"`` or ``"activation"``), cumulative
            ``gradient_steps``, ``structure_steps``, and ``sweeps`` counters,
            and a ``metrics`` dictionary. Numeric metrics are Python floats;
            keys such as ``"Pearson"`` and ``"Time"`` can vary by stage.
            For PTT, ``partition_estimate`` may contain normalization
            diagnostics; otherwise it is ``None``. Callback exceptions
            propagate and end the training run.
        stage_progress: Optional callback for transient PTT stage updates.
            It receives a :class:`StageProgress` with ``stage``, ``kind``,
            completed and total work, and stage-specific details. These
            events do not add metric rows or alter the ``progress`` callback.
        on_initialized: Optional callback taking one
            :class:`TrainingInitialization` event. It runs once after input
            loading and weighting, before numerical training begins.
        is_cancelled: Optional zero-argument function returning ``True`` to
            request cancellation or ``False`` to continue.

    Returns:
        A :class:`TrainingResult` containing the trained model, metric
        history, chains, stop reason, input report, and any saved artifacts.

    Raises:
        InputValidationError: If the inputs or requested settings are invalid.
        OperationCancelledError: If ``is_cancelled`` requests cancellation.

    Example:
        A progress function receives an event object, not separate epoch and
        metric arguments. Use ``dict.get`` because metrics may differ between
        stages::

            from adabmDCA import train_model
            from adabmDCA.api import TrainingProgress

            def show_progress(event: TrainingProgress) -> None:
                pearson = event.metrics.get("Pearson")
                if pearson is not None:
                    print(
                        f"{event.stage}: epoch={event.epoch}, "
                        f"gradient_steps={event.gradient_steps}, "
                        f"Pearson={pearson:.3f}"
                    )

            result = train_model("alignment.fasta", progress=show_progress)

    Notes:
        ``dtype="bfloat16"`` uses BF16 coupling copies for CUDA/Triton
        sampling and float32 master parameters, statistics, chains, and saved
        models; it requires an NVIDIA Ampere or newer GPU. PTT requires bmDCA, eaDCA or edgeDCA
        with float32 or float64 and uses PTT bridges for normalization. For PTT resume, ``max_epochs`` is the
        total committed-step limit, including steps already saved: gradient steps for bmDCA and
        edgeDCA, graph-activation (structure) steps for eaDCA.
    """
    training_config = config or TrainingConfig(
        model_type=model_type,
        ptt=ptt,
        alphabet=alphabet,
        learning_rate=learning_rate,
        n_sweeps=n_sweeps,
        sampler=sampler,
        n_chains=n_chains,
        target_pearson=target_pearson,
        max_epochs=max_epochs,
        max_gradient_steps=max_gradient_steps,
        max_structure_steps=max_structure_steps,
        checkpoint_interval=checkpoint_interval,
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
    if ptt_resume is not None and training_config.ptt is None:
        raise InputValidationError("ptt_resume requires a PTTConfig.")
    if training_config.ptt is not None and (initial_params_path is not None or initial_chains_path is not None or allow_signed_weights):
        raise InputValidationError("PTT does not support parameter/chain warm starts or signed-weight reintegration; resume from a PTT archive.")
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

    master_dtype = "float32" if dtype == "bfloat16" else dtype
    resolved_device, resolved_dtype = resolve_runtime(device, master_dtype)
    try:
        sampling_function = (
            prepare_training_sampler(sampler, resolved_device, dtype)
            if training_config.ptt is None else None
        )
    except ValueError as exc:
        raise InputValidationError(str(exc)) from exc
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

    effective_size = dataset.get_effective_size()
    effective_pseudocount = training_config.resolve_pseudocount(effective_size)

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

    ptt_sampler = None
    if training_config.ptt is not None:
        if ptt_resume is not None:
            ptt_sampler = PTTSampler.from_archive(ptt_resume, device=resolved_device, mode="resume")
            if ptt_sampler.tokens != tokens or not ptt_sampler.training_state:
                raise InputValidationError("PTT archive token order or training restart state is incompatible.")
        else:
            ptt_sampler = PTTSampler(params, tokens=tokens, n_chains=n_chains, sampler=sampler,
                                     config=training_config.ptt, seed=seed)

    initialization = TrainingInitialization(
        training=_dataset_summary(dataset, loaded_inputs.training_alignment),
        validation=(
            None
            if validation is None or loaded_inputs.validation_alignment is None
            else _dataset_summary(validation, loaded_inputs.validation_alignment)
        ),
        device=str(resolved_device),
        dtype=str(resolved_dtype).removeprefix("torch."),
        n_chains=int(n_chains),
        effective_pseudocount=float(effective_pseudocount),
        config=training_config,
    )
    if on_initialized is not None:
        on_initialized(initialization)

    artifacts: dict[str, Path] = {}
    checkpoint = None
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        stem = label or "adabmDCA"
        file_paths = {
            "log": str(folder / f"{stem}.log"),
            "history": str(folder / (f"{label}_history.csv" if label else "history.csv")),
            "events": str(folder / (f"{label}_events.jsonl" if label else "events.jsonl")),
            "params": str(folder / (f"{label}_params.dat.gz" if label else "params.dat.gz")),
            "chains": str(folder / (f"{label}_chains.fasta" if label else "chains.fasta")),
        }
        if ptt_sampler is not None:
            file_paths["ptt_archive"] = str(folder / (f"{label}_ptt.h5" if label else "ptt.h5"))
        from adabmDCA import __version__

        limits = training_config.limits
        optimization = {
            "sampler": sampler,
            "ptt": None if training_config.ptt is None else training_config.as_dict()["ptt"],
            "chains": n_chains,
            "sweeps_per_step": n_sweeps,
            "target_pearson": target_pearson,
            "max_gradient_steps": limits.max_gradient_steps,
            "max_structure_steps": limits.max_structure_steps,
            "checkpoint_interval": training_config.resolved_checkpoint_interval,
            "effective_pseudocount": effective_pseudocount,
            "clustering_seqid": clustering_seqid,
            "no_reweighting": no_reweighting,
        }
        if model_type != "edgeDCA":
            optimization.update(learning_rate=learning_rate, l2_regularization=l2_regularization)
        if model_type == "eaDCA":
            optimization.update(activation_steps=activation_steps, activation_fraction=activation_fraction)
        elif model_type == "edDCA":
            optimization.update(target_density=target_density, decimation_rate=decimation_rate)
        elif model_type == "edgeDCA":
            optimization.update(empirical_pseudocount=training_config.edge_empirical_pseudocount)
        checkpoint_metadata = {
            "run": {
                "label": label or "adabmDCA",
                "model": model_type,
                "package_version": __version__,
                "seed": seed,
            },
            "training data": initialization.training.__dict__,
            "validation data": None if initialization.validation is None else initialization.validation.__dict__,
            "runtime": {"device": initialization.device, "dtype": initialization.dtype},
            "optimization": optimization,
        }
        checkpoint = Checkpoint(
            file_paths,
            tokens,
            checkpoint_metadata,
            use_wandb=use_wandb,
            config=training_config,
            resume=ptt_resume is not None,
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
        stage_observer=stage_progress if ptt_sampler is not None else None,
        is_cancelled=is_cancelled,
    )

    try:
        if ptt_sampler is not None:
            from adabmDCA.ptt.training import train_ptt

            chains, params, history = train_ptt(
                ptt_sampler=ptt_sampler, fi_target=fi_target, fij_target=fij_target, mask=mask,
                config=training_config, controller=controller, fi_val=fi_val, fij_val=fij_val,
                fij_raw=(dataset.get_frequencies(pseudocount=0.0)[1] if model_type == "edgeDCA" else None),
                edge_pseudocount=(effective_pseudocount if model_type == "edgeDCA" else None),
                activation_pseudocount=(effective_pseudocount if model_type == "eaDCA" else None),
                effective_size=effective_size,
            )
        elif model_type == "bmDCA":
            chains, params, history = train_graph(
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
                l2_reg=l2_regularization,
            )
        elif model_type == "eaDCA":
            chains, params, history = train_eaDCA(
                sampler=sampling_function,
                fi_target=fi_target,
                fij_target=fij_target,
                params=params,
                mask=mask,
                chains=chains,
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
            chains, params, history = train_edDCA(
                sampler=sampling_function,
                chains=chains,
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
            chains, params, history = train_edgeDCA(
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
            )
    except OperationCancelledError as exc:
        controller.set_stop_reason(StopReason.CANCELLED)
        _finish_log(checkpoint, "cancelled", controller, controller.history, error=exc)
        raise
    except TrainingCancelled as exc:
        _finish_log(checkpoint, "cancelled", controller, controller.history, error=exc)
        raise OperationCancelledError(str(exc)) from exc
    except KeyboardInterrupt as exc:
        _finish_log(checkpoint, "interrupted", controller, controller.history, error=exc)
        raise
    except Exception as exc:
        _finish_log(checkpoint, "failed", controller, controller.history, error=exc)
        raise

    _finish_log(checkpoint, "completed", controller, history)

    trained = DCAModel(params, alphabet=alphabet, source=artifacts.get("params"))
    return TrainingResult(
        model=trained,
        history=history,
        chains=chains,
        pseudocount=float(effective_pseudocount),
        num_sequences=len(dataset),
        effective_sequences=effective_size,
        artifacts=artifacts,
        converged=controller.stop_reason
        in {
            StopReason.TARGET_PEARSON,
            StopReason.TARGET_DENSITY,
            StopReason.VALIDATION_PLATEAU,
            StopReason.GRAPH_CONVERGED,
        },
        stop_reason=(controller.stop_reason.value if controller.stop_reason is not None else None),
        gradient_steps=controller.counters.gradient_steps,
        structure_steps=controller.counters.structure_steps,
        sweeps=controller.counters.sweeps,
        config=training_config,
        input_report=loaded_inputs.training_alignment.to_dict(),
        initialization=initialization,
        partition_estimate=None if ptt_sampler is None else ptt_sampler.partition_estimate(),
        ptt_sampler=ptt_sampler,
    )
