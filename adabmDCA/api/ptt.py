"""Archive-based generation and direct entropy using the shared PTT sampler."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from adabmDCA._validation import validate_integer
from adabmDCA.api.model import DCAModel
from adabmDCA.api.results import SamplingProgress, SamplingResult
from adabmDCA.exceptions import InputValidationError
from adabmDCA.fasta import decode_sequence, get_tokens
from adabmDCA.input_loading import AlignmentInput, WeightInput
from adabmDCA.ptt import PTTSampler
from adabmDCA.ptt.precision import device_accumulation
from adabmDCA.serialization import result_document, write_json
from adabmDCA.statmech import compute_energy
from adabmDCA.steering import (
    Steering,
    SteeringInput,
    SteeringPotential,
    importance_summary,
    importance_warning,
)


def load_ptt_backend(
    source: PTTSampler | str | Path,
    *,
    device: str = "cpu",
    seed: int | None = None,
    alphabet: str | None = None,
) -> PTTSampler:
    """Return a PTT sampler ready for generation, loaded or forked from ``source``.

    An in-memory :class:`PTTSampler` is forked, so the original is not
    modified. An archive keeps its precision and alphabet.

    Args:
        source: A :class:`PTTSampler` or a path to a PTT archive.
        device: Device to load the archive on; ``"auto"`` prefers CUDA, then MPS, then CPU.
            Changing device requires a new ``seed``.
        seed: New random seed, or ``None`` to continue the archived random stream.
        alphabet: Optional alphabet; must match the archive's tokens.

    Returns:
        A :class:`PTTSampler` in generation mode.

    Raises:
        InputValidationError: If the archive is invalid or ``alphabet`` conflicts with it.
    """
    if isinstance(source, PTTSampler):
        backend = source.fork(seed=seed)
    else:
        # 'auto' respects the archived backend for a seedless restart, and
        # otherwise selects the standard runtime. CPU remains fully supported.
        if device == "auto":
            from adabmDCA.api.runtime import resolve_runtime

            device, _ = resolve_runtime("auto", "float32")
        backend = PTTSampler.from_archive(source, device=device, seed=seed)
    backend.mode = "generate"
    backend.training_state = {}
    if alphabet not in (None, "auto") and get_tokens(alphabet) != backend.tokens:
        raise InputValidationError("Explicit alphabet conflicts with the PTT archive token order.")
    return backend


# The stationary renewal stops at the first block end after the ladder is renewed;
# blocks of a quarter of the warmup waste at most a quarter of it after renewal.
STATIONARY_BLOCKS_PER_WARMUP = 4


def sample_ptt_sequences(
    *,
    source: PTTSampler | str | Path,
    n_sequences: int,
    local_sweeps: int = 10,
    reference_fasta: AlignmentInput | None = None,
    weights_path: WeightInput | None = None,
    test_fasta: AlignmentInput | None = None,
    privet_window: tuple[float, float] = (0.01, 0.5),
    clustering_seqid: float = 0.8,
    no_reweighting: bool = False,
    seed: int = 0,
    device: str = "cpu",
    alphabet: str | None = None,
    sampler: str | None = None,
    dtype: str | None = None,
    n_measure: int = 10_000,
    pseudocount: float | None = None,
    max_rounds: int = 20_000,
    mixing_method: str = "renewal",
    renewal_tolerance: float = 0.01,
    stationary: bool = False,
    local_kernel: str | None = "metropolized_gibbs",
    steering_potential: SteeringPotential | None = None,
    steering_strength: float = 1.0,
    steering_input: SteeringInput = "sequences",
    steering_proposal_steps: int | None = None,
    steering_acceptance: float = 0.3,
    collect_diagnostics: bool = False,
    progress: Callable[[SamplingProgress], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> SamplingResult:
    """Equilibrate the ladder in place, then collect endpoint samples.

    ``mixing_method='renewal'`` (default) tracks the birth round of every
    configuration and waits until the ladder has twice been repopulated by
    exact rung-0 draws (see ``PTTSampler.measure_renewal``); extra batches are
    spaced by the measured renewal time. By default the renewal runs only the
    warmup, which replaces the initial populations once; ``stationary=True``
    adds the stationary renewal, which certifies that the returned population
    descends from draws made after the ladder had forgotten its start.
    ``'autocorrelation'``
    keeps the replica-index TRWA estimate of tau_int and tau_exp.
    Each round runs one pass of adjacent exchanges and population
    permutations, then the local sweeps, as in training.
    ``local_kernel`` selects the local update ('metropolized_gibbs' by
    default, 'metropolis' or 'gibbs'); ``None`` keeps the archived training
    kernel. All of them leave every replica's distribution invariant. The
    ladder is the training ladder (see ``PTTSampler.prepare_sampling_ladder``).
    With ``steering_potential``, steered rungs are added on top of it
    (``PTTSampler.prepare_steering``) and the samples come from the top one; see
    :func:`adabmDCA.api.sampling.sample_sequences` for the steering arguments.
    """
    from adabmDCA.api.results import SamplingProgress
    from adabmDCA.api.sampling import (
        _compare_with_data,
        _compute_pca_scores,
        _distance_comparison,
        _load_test_alignment,
    )
    from adabmDCA.dataset import DatasetDCA
    from adabmDCA.input_loading import AlignmentLoadConfig
    from adabmDCA.stats import (
        extract_Cij_from_freq,
        get_correlation_two_points,
        get_freq_single_point,
        get_freq_two_points,
    )
    from adabmDCA.utils import resample_sequences

    validate_integer("n_sequences", n_sequences)
    validate_integer("local_sweeps", local_sweeps)
    validate_integer("max_rounds", max_rounds)
    validate_integer("n_measure", n_measure)
    if mixing_method not in {"renewal", "autocorrelation"}:
        raise InputValidationError("PTT mixing method must be 'renewal' or 'autocorrelation'.")
    if isinstance(renewal_tolerance, bool) or not isinstance(renewal_tolerance, (int, float)) \
            or not 0.0 < renewal_tolerance < 1.0:
        raise InputValidationError("PTT renewal tolerance must be between 0 and 1.")
    if pseudocount is not None and not 0.0 <= pseudocount <= 1.0:
        raise InputValidationError("pseudocount must be between 0 and 1.")
    if reference_fasta is None:
        raise InputValidationError("PTT sampling requires reference_fasta (--data) for ladder log-likelihoods.")
    if test_fasta is not None and not collect_diagnostics:
        raise InputValidationError("test_fasta is used only with collect_diagnostics=True (CLI: --plot).")
    if steering_potential is not None:
        from adabmDCA.api.sampling import _validate_steering_strength

        _validate_steering_strength(steering_strength)
    backend = load_ptt_backend(source, device=device, seed=seed, alphabet=alphabet)
    backend.prepare_sampling_ladder(n_sequences)
    backend.set_generation_kernel(local_kernel)
    if sampler is not None and sampler != backend.local_sampler:
        raise InputValidationError(
            "PTT generation ignores the ordinary sampler option; select its local update with local_kernel "
            "(--ptt-local-kernel)."
        )
    if dtype is not None and dtype != str(backend.models[0]["bias"].dtype).removeprefix("torch."):
        raise InputValidationError("PTT generation uses the archived precision; conflicting overrides are unsupported.")
    dataset = DatasetDCA.from_alignment(
        reference_fasta, weights=weights_path,
        load_config=AlignmentLoadConfig(alphabet=backend.tokens, invalid_sequences="drop",
                                        remove_duplicates=True, expected_length=backend.models[-1]["bias"].shape[0]),
        clustering_th=clustering_seqid, no_reweighting=no_reweighting,
        device=backend.device, dtype=backend.models[-1]["bias"].dtype,
    )
    test = None if test_fasta is None else _load_test_alignment(
        test_fasta, dataset, clustering_seqid=clustering_seqid, no_reweighting=no_reweighting
    )

    def report(stage, completed, total, **details):
        if progress is not None:
            progress(SamplingProgress("ptt_" + stage, completed, total, details=details))

    before = backend.local_sweeps
    if steering_potential is not None:
        steering = Steering(steering_potential, tokens=backend.tokens, steering_input=steering_input)
        report("steering_ladder", 0, 1, strength=0.0)
        backend.prepare_steering(steering, steering_strength, proposal_steps=steering_proposal_steps,
                                 target_acceptance=steering_acceptance, local_sweeps=local_sweeps,
                                 is_cancelled=is_cancelled, on_progress=report)
    report("ladder", len(backend.models), len(backend.models), selected_updates=backend.sampling_steps,
           chains_per_model=n_sequences)
    renewal = mixing = None
    if mixing_method == "renewal":
        renewal = backend.measure_renewal(local_sweeps=local_sweeps, max_rounds=max_rounds,
                                          tolerance=renewal_tolerance,
                                          blocks_per_warmup=STATIONARY_BLOCKS_PER_WARMUP, stationary=stationary,
                                          is_cancelled=is_cancelled, on_progress=report)
        report("renewal", renewal.rounds, renewal.rounds, status=renewal.status,
               warmup_rounds=renewal.warmup_rounds, renewal_rounds=renewal.renewal_rounds,
               trapped_fraction=renewal.trapped_fraction, acceptance=list(renewal.acceptance))
        # One renewal time separates successive endpoint batches: the
        # stationary phase length (a whole number of blocks covering one
        # renewal) when it ran, otherwise the warmup length.
        renewal_time = renewal.stationary_rounds if stationary else renewal.warmup_rounds
        spacing = max(1, renewal_time) if renewal.converged else 1
    else:
        mixing = backend.estimate_mixing_time(local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                                             on_progress=report, bounded=True, max_rounds=max_rounds,
                                             in_place=True, reference=True, method="autocorrelation")
        spacing = max(1, 2 * int(mixing.tau_int)) if mixing.tau_int is not None else 1
    # Both methods leave the population at the end of the measured trajectory.
    # That population is still a useful sampling result when the finite round
    # budget is exhausted, so retain it and mark the result as nonconverged.
    batches = []
    batch_potentials = []
    batch_endpoint_fresh = []
    collected = 0
    report("generation", 0, n_sequences)
    while collected < n_sequences:
        if is_cancelled is not None and is_cancelled():
            from adabmDCA.exceptions import OperationCancelledError
            raise OperationCancelledError("PTT sampling was cancelled.")
        count = min(n_sequences - collected, len(backend.chains[-1]))
        batches.append(backend.endpoint_samples()[:count])
        if backend.steering is not None:
            batch_potentials.append(backend.steering_values[-1][:count].clone())
        collected += count
        report("generation", collected, n_sequences)
        if collected < n_sequences:
            snapshot_round = backend.rounds
            report("spacing", 0, spacing)
            backend.advance(return_samples=False, rounds=spacing, local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                            on_round=lambda done, total: report("spacing", done, total))
            # Fraction of the next batch drawn at rung 0 after this batch.
            batch_endpoint_fresh.append(float(device_accumulation(backend.birth[-1] >= snapshot_round).mean()))
    samples = torch.cat(batches)

    # Compare the endpoint samples with the weighted natural alignment.  The
    # generated chains have uniform statistical weight, as in standard
    # sampling diagnostics.
    effective_pseudocount = pseudocount
    if effective_pseudocount is None:
        effective_pseudocount = 1.0 / dataset.weights.sum().item()
    fi, fij = dataset.get_frequencies(pseudocount=effective_pseudocount)
    pi = get_freq_single_point(data=samples, weights=None, pseudo_count=0.0)
    pij = get_freq_two_points(data=samples, weights=None, pseudo_count=0.0)
    final_pearson, final_slope = get_correlation_two_points(fi=fi, pi=pi, fij=fij, pij=pij)

    cij_reference = cij_generated = None
    pca_reference = pca_generated = pca_explained_variance_ratio = None
    data_comparison = {}
    distance_comparison = {}
    if collect_diagnostics:
        cij_reference_tensor, cij_generated_tensor = extract_Cij_from_freq(
            fij=fij, pij=pij, fi=fi, pi=pi
        )
        cij_reference = cij_reference_tensor.detach().cpu().numpy()
        cij_generated = cij_generated_tensor.detach().cpu().numpy()
        measured = min(n_measure, len(dataset))
        # Draw the reference rows before one-hot encoding: the full alignment
        # can be far larger in one-hot form than the few rows needed.
        reference = torch.nn.functional.one_hot(
            resample_sequences(dataset.data, dataset.weights, measured).long(), len(dataset.tokens)
        ).to(dtype=dataset.dtype, device=dataset.device)
        reference_scores, generated_scores, explained_variance_ratio = _compute_pca_scores(
            reference, samples
        )
        pca_reference = reference_scores.detach().cpu().numpy()
        pca_generated = generated_scores.detach().cpu().numpy()
        pca_explained_variance_ratio = explained_variance_ratio.detach().cpu().numpy()
        data_comparison = _compare_with_data(reference, samples, reference_scores, generated_scores,
                                             backend.endpoint_params(), seed=seed)
        distance_comparison = _distance_comparison(dataset, samples, test=test, n_measure=n_measure, seed=seed,
                                                   privet_window=privet_window)

    report("ladder_statistics", 0, len(backend.models))
    rows = backend.ladder_statistics(dataset.data, dataset.weights)
    flagged = {p["step"] for p in backend.ptt_checkpoints}
    for row, step in zip(rows, backend.sampling_steps):
        row["training_step"] = step
        row["flag"] = ("steered" if row["steering_strength"] != 0.0
                       else "ptt" if step in flagged else "final")
    report("ladder_statistics", len(rows), len(rows))
    # Bidirectional reweighting diagnostics on the final populations; the BAR
    # bridges above use the same pairs.
    health = backend.ladder_health(seed=seed)
    report("ladder_health", len(health["pairs"]), len(health["pairs"]), **health)
    model = DCAModel(backend.endpoint_params(), alphabet=backend.tokens)
    if renewal is not None:
        result_warnings = ["PTT population renewal diagnostics are not a certificate of equilibrium."]
        if not renewal.converged:
            result_warnings.append(
                f"PTT population renewal did not complete within the {max_rounds}-round budget; "
                f"{renewal.trapped_fraction:.1%} of endpoint configurations predate the last reference "
                "round and may be trapped. Samples were retained from the final endpoint population."
            )
        history_keys = ("warmup_ladder_old", "warmup_endpoint_fresh",
                        "stationary_ladder_old", "stationary_endpoint_fresh")
        mixing_summary = {key: value for key, value in asdict(renewal).items() if key not in history_keys}
        mixing_summary["method"] = "renewal"
        mixing_summary["stationary"] = stationary
        mixing_diagnostics = {
            "mixing": mixing_summary,
            "renewal_history": {key: list(getattr(renewal, key)) for key in history_keys},
            "batch_endpoint_fresh": batch_endpoint_fresh,
            "mixing_correlation": [],
        }
        mixing_history = {"warmup_rounds": [renewal.warmup_rounds], "renewal_rounds": [renewal.renewal_rounds],
                          "stationary_rounds": [renewal.stationary_rounds],
                          "block_rounds": [renewal.chunk_rounds],
                          "warmup_decay_rounds": [renewal.warmup_decay_rounds],
                          "stationary_decay_rounds": [renewal.stationary_decay_rounds],
                          "trapped_fraction": [renewal.trapped_fraction]}
        converged = renewal.converged
    else:
        result_warnings = ["PTT replica mixing diagnostics are not a certificate of equilibrium."]
        if not mixing.converged:
            result_warnings.append(
                f"PTT mixing did not converge within the {max_rounds} measured-round budget; "
                "samples were retained from the final endpoint population."
            )
        mixing_diagnostics = {"mixing": {**asdict(mixing), "method": "autocorrelation"},
                              "mixing_correlation": backend.mixing_correlation}
        mixing_history = {"tau_int": [mixing.tau_int], "tau_exp": [mixing.tau_exp],
                          "measurement_rounds": [mixing.rounds], "required_rounds": [mixing.required_rounds]}
        converged = mixing.converged
    energies = compute_energy(samples, model.params).cpu().numpy()
    from adabmDCA.api.scoring import fit_local_lambda as fit_slope
    from adabmDCA.api.scoring import summed_cde

    cde_sum = summed_cde(samples, model.params)
    lambda_fit = None
    if not converged:
        result_warnings.append("The sampled population may be biased because PTT mixing did not converge.")
    steering_potentials = log_weights = None
    steering_summary = {}
    if backend.steering is not None:
        steering_potentials = torch.cat(batch_potentials).cpu().numpy()
        base = backend.steering_base - backend.active_start
        log_z_ratio = rows[-1]["log_z"] - rows[base]["log_z"]
        log_weights, ess = importance_summary(steering_potentials, log_z_ratio)
        kernels = [kernel for kernel in backend._steering_kernels if kernel is not None]
        steering_summary = {
            "strength": float(steering_strength), "input": steering_input,
            "strengths": [s for s in backend.steering_strengths if s != 0.0],
            "proposal_steps": [kernel.proposal_steps for kernel in kernels],
            "acceptance": [kernel.acceptance for kernel in kernels],
            "potential_evaluations": backend.steering.evaluations,
            "log_z_ratio": log_z_ratio, "effective_sample_size": ess, "weights": "bridge_normalized",
        }
        result_warnings.append("Comparisons with reference_fasta describe the steered distribution.")
        if (warning := importance_warning(ess, len(steering_potentials))) is not None:
            result_warnings.append(warning)
    try:
        lambda_fit = fit_slope(energies, cde_sum)
    except InputValidationError as exc:
        result_warnings.append(str(exc))
    return SamplingResult(
        sequences=tuple(decode_sequence(samples.argmax(-1).cpu().numpy(), backend.tokens)),
        energies=energies,
        num_sweeps=backend.local_sweeps - before, sampler="ptt", beta=1.0,
        seed=backend.seed, model=model.metadata,
        cde_sum=cde_sum, local_lambda_fit=lambda_fit,
        sampling_dtype=str(samples.dtype).removeprefix("torch."),
        mixing_history={**mixing_history, "equilibration_rounds": [0], "spacing_rounds": [spacing],
                        "local_sweeps_per_round": [local_sweeps]},
        sampling_history={"exchange_rounds": [backend.rounds],
                          "pearson": [final_pearson], "slope": [final_slope]},
        ptt_diagnostics={**mixing_diagnostics, "mixing_method": mixing_method, "converged": converged,
                         "local_kernel": local_kernel or backend.local_sampler,
                         "mixing_round_budget": max_rounds,
                         "time_unit": "exchange_rounds", "equilibration_rounds": 0,
                         "spacing_rounds": spacing, "likelihood_basis": "weighted_reference_data",
                         "retained_models": len(rows), "historical_models": backend.total_models,
                         "selected_updates": backend.sampling_steps,
                         "reservoir_anchor": backend.reservoir is not None,
                         "final_pearson": final_pearson, "final_slope": final_slope,
                         "timings_seconds": dict(backend.timings), "models": rows,
                         "ladder_health": health},
        cij_reference=cij_reference, cij_generated=cij_generated,
        pca_reference=pca_reference, pca_generated=pca_generated,
        pca_explained_variance_ratio=pca_explained_variance_ratio,
        data_comparison=data_comparison,
        distance_comparison=distance_comparison,
        steering_potentials=steering_potentials,
        log_importance_weights=log_weights,
        steering=steering_summary,
        warnings=tuple(result_warnings),
    )


@dataclass(frozen=True)
class PTTEntropyResult:
    """Entropy of a PTT model, returned by :func:`estimate_ptt_entropy`.

    Attributes:
        entropy: Entropy of the endpoint model, in nats per sequence
            (``mean_energy + log_z``).
        free_energy: ``-log_z``.
        mean_energy: Mean endpoint energy over the endpoint chains.
        log_z: Log partition function from the PTT bridges.
        method: Estimator of ``log_z``, e.g. ``"ptt_bridge"``.
        status: Status of the ``log_z`` estimate.
        model_id: Hash identifying the endpoint parameters.
        model_version: Number of training updates of the endpoint.
        ladder_version: Version of the replica ladder used.
        sample_round: Exchange round at which the chains were measured.
        artifacts: Written files by kind, when ``output_dir`` was given.
    """

    entropy: float
    free_energy: float
    mean_energy: float
    log_z: float
    method: str
    status: str
    model_id: str
    model_version: int
    ladder_version: int
    sample_round: int
    artifacts: dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible document with the result type and schema version."""
        return result_document("ptt_entropy", asdict(self))

    def to_json(self, path: str | Path) -> Path:
        """Write :meth:`to_dict` as JSON and return the written path."""
        return write_json(path, self.to_dict())


def estimate_ptt_entropy(
    *,
    model: PTTSampler | str | Path,
    n_sweeps: int = 1,
    seed: int = 0,
    device: str = "cpu",
    alphabet: str | None = None,
    output_dir: str | Path | None = None,
    label: str = "entropy",
    is_cancelled: Callable[[], bool] | None = None,
) -> PTTEntropyResult:
    """Estimate the entropy of a PTT-trained model as ``<E> + log Z``.

    A PTT archive already carries an estimate of log Z (bridges along its
    replica ladder), so no thermodynamic integration is needed: the ladder is
    advanced for the archive's ``equilibration_rounds`` exchange rounds, then
    ``log Z`` is re-estimated with Bennett's acceptance ratio and ``<E>`` is
    averaged over the endpoint chains.

    Args:
        model: A PTT archive path, or a :class:`PTTSampler` (it is forked, not modified).
        n_sweeps: Local sweeps per exchange round of the warmup.
        seed: Random seed of the warmup.
        device: ``"cpu"``, ``"cuda"``, ``"mps"`` (float32) or ``"auto"``.
        alphabet: Optional alphabet; must match the archive.
        output_dir: If given, the result is written to ``<output_dir>/<label>.json``.
        label: File-name stem of the written JSON.
        is_cancelled: Optional callable; returning ``True`` stops the warmup.

    Returns:
        A :class:`PTTEntropyResult`; ``entropy`` is in nats per sequence.

    Example:
        >>> estimate_ptt_entropy(model="output/ptt.h5", device="cuda").entropy
        131.2
    """
    backend = load_ptt_backend(model, device=device, seed=seed, alphabet=alphabet)
    backend.equilibrate(local_sweeps=n_sweeps, is_cancelled=is_cancelled)
    result = PTTEntropyResult(**backend.entropy())
    if output_dir is not None:
        path = Path(output_dir) / f"{label}.json"
        result.artifacts["summary"] = path
        result.to_json(path)
    return result
