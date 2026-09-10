"""High-level sequence-generation operations."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import torch

from adabmDCA._validation import validate_integer, validate_seed
from adabmDCA.api.exceptions import InputValidationError, OperationCancelledError
from adabmDCA.api.model import DCAModel, load_model
from adabmDCA.api.results import SamplingProgress, SamplingResult
from adabmDCA.dataset import DatasetDCA
from adabmDCA.fasta import decode_sequence
from adabmDCA.input_loading import (
    AlignmentInput,
    AlignmentLoadConfig,
    WeightInput,
)
from adabmDCA.resampling import compute_mixing_time
from adabmDCA.sampling import prepare_fixed_model_sampler
from adabmDCA.statmech import compute_energy
from adabmDCA.stats import (
    extract_Cij_from_freq,
    get_correlation_two_points,
    get_freq_single_point,
    get_freq_two_points,
)
from adabmDCA.utils import init_chains, resample_sequences

ProgressCallback = Callable[[SamplingProgress], None]
CancellationHook = Callable[[], bool]


def _notify(
    callback: ProgressCallback | None,
    *,
    stage: str,
    completed: int,
    total: int,
    pearson: float | None = None,
    slope: float | None = None,
) -> None:
    if callback is not None:
        callback(SamplingProgress(stage, completed, total, pearson, slope))


def _check_cancelled(hook: CancellationHook | None) -> None:
    if hook is not None and hook():
        raise OperationCancelledError("Sequence generation was cancelled by the caller.")


def _compute_pca_scores(
    reference: torch.Tensor,
    generated: torch.Tensor,
    *,
    n_components: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit PCA on natural sequences and project both datasets into that basis."""
    reference_flat = reference.reshape(reference.shape[0], -1).float()
    generated_flat = generated.reshape(generated.shape[0], -1).float()
    mean = reference_flat.mean(dim=0, keepdim=True)
    centered_reference = reference_flat - mean
    available = min(n_components, max(reference_flat.shape[0] - 1, 0), reference_flat.shape[1])
    reference_scores = torch.zeros(
        (reference_flat.shape[0], n_components), device=reference.device, dtype=reference_flat.dtype
    )
    generated_scores = torch.zeros(
        (generated_flat.shape[0], n_components), device=generated.device, dtype=generated_flat.dtype
    )
    explained_variance_ratio = torch.zeros(n_components, device=reference.device, dtype=reference_flat.dtype)
    if available == 0:
        return reference_scores, generated_scores, explained_variance_ratio

    _, singular_values, components = torch.pca_lowrank(
        centered_reference,
        q=available,
        center=False,
        niter=4,
    )
    reference_scores[:, :available] = centered_reference @ components
    generated_scores[:, :available] = (generated_flat - mean) @ components
    total_variance = centered_reference.square().sum()
    if total_variance > 0:
        explained_variance_ratio[:available] = singular_values.square() / total_variance
    return reference_scores, generated_scores, explained_variance_ratio


def sample_sequences(
    *,
    model: DCAModel | str | Path,
    n_sequences: int,
    n_sweeps: int = 1000,
    sampler: str = "metropolis",
    beta: float = 1.0,
    seed: int = 0,
    reference_fasta: AlignmentInput | None = None,
    weights_path: WeightInput | None = None,
    n_measure: int = 10_000,
    mixing_multiplier: int = 2,
    pseudocount: float | None = None,
    clustering_seqid: float = 0.8,
    no_reweighting: bool = False,
    alphabet: str = "protein",
    device: str = "auto",
    dtype: str = "float32",
    collect_diagnostics: bool = False,
    progress: ProgressCallback | None = None,
    is_cancelled: CancellationHook | None = None,
) -> SamplingResult:
    """Generate sequences from a DCA model.

    When ``reference_fasta`` is provided, ``n_sweeps`` is the maximum number
    of sweeps used to estimate mixing and generation runs for
    ``mixing_multiplier`` times the estimated mixing time. Otherwise,
    generation runs for exactly ``n_sweeps``.
    """
    validate_integer("n_sequences", n_sequences)
    validate_integer("n_sweeps", n_sweeps, minimum=0)
    validate_integer("mixing_multiplier", mixing_multiplier)
    validate_integer("n_measure", n_measure)
    validate_seed(seed)
    if reference_fasta is not None and n_sweeps < 2:
        raise InputValidationError("n_sweeps must be at least 2 when estimating mixing time.")
    if collect_diagnostics and reference_fasta is None:
        raise InputValidationError("reference_fasta is required when collect_diagnostics=True.")
    if sampler not in {"gibbs", "metropolis"}:
        raise InputValidationError("sampler must be either 'gibbs' or 'metropolis'.")
    if dtype not in {"float32", "float64", "bfloat16"}:
        raise InputValidationError("dtype must be one of 'float32', 'float64', or 'bfloat16'.")
    if not math.isfinite(beta) or beta <= 0:
        raise InputValidationError("beta must be greater than zero.")
    if pseudocount is not None and not 0.0 <= pseudocount <= 1.0:
        raise InputValidationError("pseudocount must be between 0 and 1.")
    if not 0.0 < clustering_seqid <= 1.0:
        raise InputValidationError("clustering_seqid must be greater than 0 and at most 1.")
    _check_cancelled(is_cancelled)

    model_dtype = "float32" if dtype == "bfloat16" else dtype
    loaded = (
        model
        if isinstance(model, DCAModel)
        else load_model(model, alphabet=alphabet, device=device, dtype=model_dtype)
    )
    torch.manual_seed(seed)
    if loaded.params["bias"].device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    try:
        sampling_function, sampling_params = prepare_fixed_model_sampler(
            sampler,
            loaded.params["bias"].device,
            dtype,
            loaded.params,
        )
    except ValueError as exc:
        raise InputValidationError(str(exc)) from exc
    metadata = loaded.metadata
    samples = init_chains(
        num_chains=n_sequences,
        L=metadata.length,
        q=metadata.num_states,
        device=loaded.params["bias"].device,
        dtype=loaded.params["bias"].dtype,
    )
    mixing_history: dict[str, list[float]] = {}
    sampling_history: dict[str, list[float]] = {}
    cij_reference = cij_generated = None
    pca_reference = pca_generated = pca_explained_variance_ratio = None

    if reference_fasta is not None:
        dataset = DatasetDCA.from_alignment(
            reference_fasta,
            weights=weights_path,
            load_config=AlignmentLoadConfig(
                alphabet=loaded.tokens,
                invalid_sequences="drop",
                remove_duplicates=True,
                expected_length=metadata.length,
            ),
            clustering_th=clustering_seqid,
            no_reweighting=no_reweighting,
            device=loaded.params["bias"].device,
            dtype=loaded.params["bias"].dtype,
        )
        effective_pseudocount = pseudocount
        if effective_pseudocount is None:
            effective_pseudocount = 1.0 / dataset.weights.sum().item()
        measured = min(n_measure, len(dataset))
        reference = resample_sequences(dataset.to_one_hot(), dataset.weights, measured)
        _check_cancelled(is_cancelled)
        raw_mixing = compute_mixing_time(
            sampler=sampling_function,
            data=reference,
            params=sampling_params,
            n_max_sweeps=n_sweeps,
            beta=beta,
        )
        mixing_history = {key: list(value) for key, value in raw_mixing.items()}
        mixing_time = int(raw_mixing["t_half"][-1])
        total_sweeps = mixing_multiplier * mixing_time
        fi, fij = dataset.get_frequencies(pseudocount=effective_pseudocount)
        sampling_history = {"nsweeps": [], "pearson": [], "slope": []}
    else:
        total_sweeps = n_sweeps
        fi = fij = None

    if total_sweeps > 0:
        _notify(progress, stage="sampling", completed=0, total=total_sweeps)
    for sweep in range(total_sweeps):
        _check_cancelled(is_cancelled)
        samples = sampling_function(
            chains=samples,
            params=sampling_params,
            nsweeps=1,
            beta=beta,
        )
        pearson = slope = None
        if fi is not None and fij is not None:
            pi = get_freq_single_point(data=samples, weights=None, pseudo_count=0.0)
            pij = get_freq_two_points(data=samples, weights=None, pseudo_count=0.0)
            pearson, slope = get_correlation_two_points(fi=fi, pi=pi, fij=fij, pij=pij)
            sampling_history["nsweeps"].append(sweep + 1)
            sampling_history["pearson"].append(float(pearson))
            sampling_history["slope"].append(float(slope))
        _notify(
            progress,
            stage="sampling",
            completed=sweep + 1,
            total=total_sweeps,
            pearson=None if pearson is None else float(pearson),
            slope=None if slope is None else float(slope),
        )

    if collect_diagnostics:
        pi = get_freq_single_point(data=samples, weights=None, pseudo_count=0.0)
        pij = get_freq_two_points(data=samples, weights=None, pseudo_count=0.0)
        cij_reference_tensor, cij_generated_tensor = extract_Cij_from_freq(
            fij=fij,
            pij=pij,
            fi=fi,
            pi=pi,
        )
        cij_reference = cij_reference_tensor.detach().cpu().numpy()
        cij_generated = cij_generated_tensor.detach().cpu().numpy()
        reference_scores, generated_scores, explained_variance_ratio = _compute_pca_scores(reference, samples)
        pca_reference = reference_scores.detach().cpu().numpy()
        pca_generated = generated_scores.detach().cpu().numpy()
        pca_explained_variance_ratio = explained_variance_ratio.detach().cpu().numpy()

    energies = compute_energy(samples, params=loaded.params).detach().cpu().numpy()
    decoded = decode_sequence(samples.detach().cpu().numpy(), loaded.tokens)
    return SamplingResult(
        sequences=tuple(str(sequence) for sequence in decoded),
        energies=energies,
        num_sweeps=total_sweeps,
        sampler=sampler,
        beta=beta,
        seed=seed,
        sampling_dtype=dtype,
        model=metadata,
        mixing_history=mixing_history,
        sampling_history=sampling_history,
        cij_reference=cij_reference,
        cij_generated=cij_generated,
        pca_reference=pca_reference,
        pca_generated=pca_generated,
        pca_explained_variance_ratio=pca_explained_variance_ratio,
    )


def generate_sequences(**kwargs) -> tuple[str, ...]:
    """Notebook-friendly shortcut returning only generated sequences."""
    return sample_sequences(**kwargs).sequences
