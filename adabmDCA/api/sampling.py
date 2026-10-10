"""High-level sequence-generation operations."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adabmDCA._validation import validate_integer, validate_seed
from adabmDCA.api.model import DCAModel, load_model
from adabmDCA.api.results import SamplingProgress, SamplingResult
from adabmDCA.dataset import DatasetDCA
from adabmDCA.exceptions import InputValidationError, OperationCancelledError
from adabmDCA.fasta import decode_sequence
from adabmDCA.input_loading import (
    AlignmentInput,
    AlignmentLoadConfig,
    WeightInput,
)
from adabmDCA.privet import DEFAULT_WINDOW as DEFAULT_PRIVET_WINDOW
from adabmDCA.privet import privet
from adabmDCA.resampling import compute_mixing_time
from adabmDCA.sampling import prepare_fixed_model_sampler
from adabmDCA.statmech import compute_energy
from adabmDCA.stats import (
    extract_Cij_from_freq,
    get_correlation_two_points,
    get_freq_single_point,
    get_freq_two_points,
)
from adabmDCA.steering import (
    SteeredKernel,
    Steering,
    SteeringInput,
    SteeringPotential,
    importance_summary,
    importance_warning,
)
from adabmDCA.utils import init_chains, resample_sequences

ProgressCallback = Callable[[SamplingProgress], None]
CancellationHook = Callable[[], bool]


class _SamplerDefault(str):
    """Preserve the public string default while recognizing archive inheritance."""


_DEFAULT_SAMPLER = _SamplerDefault("metropolized_gibbs")


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


def _validate_steering_strength(strength: float) -> None:
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) \
            or strength == 0:
        raise InputValidationError("steering_strength must be a finite, non-zero number.")


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
    # PCA is a final plotting diagnostic. MPS lacks the SVD used by
    # pca_lowrank, so run the diagnostic on CPU explicitly rather than relying
    # on the process-wide unsupported-operator fallback.
    device = torch.device("cpu") if reference.device.type == "mps" else reference.device
    reference_flat = reference.reshape(reference.shape[0], -1).to(device=device, dtype=torch.float32)
    generated_flat = generated.reshape(generated.shape[0], -1).to(device=device, dtype=torch.float32)
    mean = reference_flat.mean(dim=0, keepdim=True)
    centered_reference = reference_flat - mean
    available = min(n_components, max(reference_flat.shape[0] - 1, 0), reference_flat.shape[1])
    reference_scores = torch.zeros(
        (reference_flat.shape[0], n_components), device=device, dtype=reference_flat.dtype
    )
    generated_scores = torch.zeros(
        (generated_flat.shape[0], n_components), device=device, dtype=generated_flat.dtype
    )
    explained_variance_ratio = torch.zeros(n_components, device=device, dtype=reference_flat.dtype)
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


def _kmeans(points: np.ndarray, n_clusters: int, *, seed: int = 0, iterations: int = 100) -> np.ndarray:
    """Cluster centres of ``points`` by k-means with k-means++ initialization."""
    rng = np.random.default_rng(seed)
    centres = [points[rng.integers(len(points))]]
    for _ in range(1, n_clusters):
        distance = np.min(((points[:, None, :] - np.asarray(centres)[None]) ** 2).sum(-1), axis=1)
        total = distance.sum()
        centres.append(points[rng.choice(len(points), p=distance / total)] if total > 0 else points[0])
    centres = np.asarray(centres, dtype=np.float64)
    for _ in range(iterations):
        labels = ((points[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
        updated = np.asarray([points[labels == k].mean(0) if (labels == k).any() else centres[k]
                              for k in range(n_clusters)])
        if np.allclose(updated, centres):
            break
        centres = updated
    return centres


def _compare_with_data(reference, samples, reference_scores, generated_scores, params, *,
                       n_clusters: int = 8, seed: int = 0) -> dict[str, Any]:
    """Where the samples sit relative to the (weight-resampled) data.

    Clusters the data in the principal-component space of the PCA plots
    (k-means) and assigns every sample to its nearest cluster centre; also
    compares the model energies of data and samples. A cluster whose share
    differs between data and samples is a region the model, or its sampling,
    over- or under-weights. Regions the samples never reach cannot show up.
    """
    points = reference_scores.detach().cpu().double().numpy()
    generated = generated_scores.detach().cpu().double().numpy()
    n_clusters = int(min(n_clusters, len(points)))
    centres = _kmeans(points, n_clusters, seed=seed)
    data_labels = ((points[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
    sample_labels = ((generated[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
    data_fraction = np.bincount(data_labels, minlength=n_clusters) / len(points)
    sample_fraction = np.bincount(sample_labels, minlength=n_clusters) / max(len(generated), 1)
    order = np.argsort(-data_fraction)
    data_energy = compute_energy(reference, params=params).cpu().double().numpy()
    sample_energy = compute_energy(samples, params=params).cpu().double().numpy()
    both = np.concatenate([data_energy, sample_energy])
    edges = np.linspace(np.quantile(both, 0.001), np.quantile(both, 0.999), 41)
    sorted_data, sorted_samples = np.sort(data_energy), np.sort(sample_energy)
    grid = np.concatenate([sorted_data, sorted_samples])
    ks = float(np.max(np.abs(np.searchsorted(sorted_data, grid, side="right") / len(sorted_data)
                             - np.searchsorted(sorted_samples, grid, side="right") / len(sorted_samples))))
    return {
        "n_reference": len(points), "n_samples": len(generated),
        "clusters": {
            "data_fraction": data_fraction[order].tolist(),
            "sample_fraction": sample_fraction[order].tolist(),
            "centres": centres[order].tolist(),
        },
        "energy": {
            "bin_edges": edges.tolist(),
            "data_density": np.histogram(data_energy, bins=edges, density=True)[0].tolist(),
            "sample_density": np.histogram(sample_energy, bins=edges, density=True)[0].tolist(),
            "data_mean": float(data_energy.mean()), "sample_mean": float(sample_energy.mean()),
            "ks_distance": ks,
        },
    }


_DISTANCE_BINS = 50


def _distance_bin_width(length: int) -> int:
    """Mismatches per histogram bin: whole counts, so that no bin holds more distance values than another."""
    return -(-length // _DISTANCE_BINS)


def _one_hot_rows(states: torch.Tensor, num_states: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(states.long(), num_states).reshape(len(states), -1).float()


def _distance_bins(mismatches: torch.Tensor, length: int) -> torch.Tensor:
    """Histogram bin of Hamming distances given as integer mismatch counts."""
    return mismatches // _distance_bin_width(length)


def _distance_statistics(a, b, weights_a, weights_b, num_states, *, same: bool, chunk: int = 1024):
    """Weighted all-pair distance histogram between two sets of states, and each row of ``a``'s nearest distance.

    The nearest distances are integer mismatch counts. ``same`` excludes each
    sequence's distance to itself.
    """
    length = a.shape[1]
    n_bins = length // _distance_bin_width(length) + 1
    b_one_hot = _one_hot_rows(b, num_states)
    histogram = torch.zeros(n_bins, dtype=torch.float64, device=a.device)
    nearest = torch.empty(len(a), dtype=torch.long, device=a.device)
    for start in range(0, len(a), chunk):
        rows = slice(start, start + chunk)
        mismatches = length - (_one_hot_rows(a[rows], num_states) @ b_one_hot.T).round().long()
        pair_weight = weights_a[rows, None].double() * weights_b[None, :].double()
        if same:
            index = torch.arange(mismatches.shape[0], device=a.device)
            pair_weight[index, start + index] = 0.0
            mismatches[index, start + index] = length + 1
            nearest[rows] = mismatches.min(1).values
            mismatches[index, start + index] = 0
        else:
            nearest[rows] = mismatches.min(1).values
        bins = _distance_bins(mismatches, length)
        histogram += torch.bincount(bins.reshape(-1), weights=pair_weight.reshape(-1), minlength=n_bins)
    return histogram, nearest


def _load_test_alignment(test_fasta, dataset, *, clustering_seqid: float, no_reweighting: bool) -> DatasetDCA:
    """Load a held-out alignment with the reference's alphabet, length, reweighting, device and precision."""
    return DatasetDCA.from_alignment(
        test_fasta,
        load_config=AlignmentLoadConfig(
            alphabet=dataset.tokens,
            invalid_sequences="drop",
            remove_duplicates=True,
            expected_length=dataset.data.shape[1],
        ),
        clustering_th=clustering_seqid,
        no_reweighting=no_reweighting,
        device=dataset.device,
        dtype=dataset.dtype,
    )


_PRIVET_MIN_SEQUENCES = 50


def _distance_comparison(dataset, samples, *, test=None, n_measure: int = 10_000, seed: int = 0,
                         privet_window: tuple[float, float] = DEFAULT_PRIVET_WINDOW) -> dict[str, Any]:
    """Hamming distances within and between natural and generated sequences (overfitting diagnostics).

    All-pair distance distributions show whether the samples reproduce the
    family's diversity; nearest-neighbour distances show near-copies of training
    sequences (generated -> natural) and collapse (generated -> generated). With
    a held-out ``test`` alignment, held-out -> natural is the reference for
    generated -> natural. Natural sequences carry their reweighting weights; at
    most ``n_measure`` sequences of each set are used.

    ``privet`` holds the PRIVET scores of the generated sequences (see
    :mod:`adabmDCA.privet`), with the null law fitted to the reweighted
    training distances and ``sample_index`` giving each scored sequence's
    position among the samples.
    """
    generator = torch.Generator().manual_seed(seed)

    def subset(states, weights, index=False):
        weights = weights.reshape(-1).to(states.device)
        keep = torch.arange(len(states), device=states.device)
        if len(states) > n_measure:
            keep = torch.randperm(len(states), generator=generator)[:n_measure].to(states.device)
            states, weights = states[keep], weights[keep]
        # Histogram weights/reductions use float64, which MPS cannot allocate.
        # Transfer only the selected categorical rows before encoding chunks.
        if states.device.type == "mps":
            states = states.cpu()
            weights, keep = weights.cpu(), keep.cpu()
        weights = weights.reshape(-1).to(states.device).double()
        return (states, weights / weights.sum(), keep) if index else (states, weights / weights.sum())

    num_states = samples.shape[-1]
    natural, natural_weights = subset(dataset.data.long(), dataset.weights)
    generated_states = samples.argmax(-1).to(natural.device)
    generated, generated_weights, generated_index = subset(generated_states, torch.ones(len(generated_states)),
                                                           index=True)
    length = samples.shape[1]
    bin_width = _distance_bin_width(length)
    n_bins = length // bin_width + 1
    edges = np.arange(n_bins + 1) * bin_width / length
    width = bin_width / length

    def density(histogram):
        histogram = histogram.cpu().numpy()
        total = histogram.sum()
        return (histogram / (total * width) if total > 0 else histogram).tolist()

    def nearest_density(mismatches, weights):
        bins = _distance_bins(mismatches, length)
        return density(torch.bincount(bins, weights=weights.to(bins.device), minlength=n_bins))

    def summary(mismatches, weights):
        values = mismatches.double() / length
        order = torch.argsort(values)
        cumulative = torch.cumsum(weights[order], 0).cpu().numpy()
        ordered = values[order].cpu().numpy()

        def quantile(level):
            return float(ordered[min(int(np.searchsorted(cumulative, level)), len(ordered) - 1)])

        return {"q01": quantile(0.01), "q10": quantile(0.1), "median": quantile(0.5),
                "mean": float((values * weights).sum()), "identical": float(weights[mismatches == 0].sum())}

    natural_pairs, natural_nearest = _distance_statistics(natural, natural, natural_weights, natural_weights,
                                                          num_states, same=True)
    generated_pairs, generated_nearest = _distance_statistics(generated, generated, generated_weights,
                                                              generated_weights, num_states, same=True)
    cross_pairs, generated_to_natural = _distance_statistics(generated, natural, generated_weights, natural_weights,
                                                             num_states, same=False)
    result = {
        "bin_edges": edges.tolist(),
        "n_natural": len(natural), "n_generated": len(generated), "n_test": 0,
        "all_pairs": {"natural": density(natural_pairs), "generated": density(generated_pairs),
                      "natural_generated": density(cross_pairs)},
        "nearest": {"generated_to_natural": nearest_density(generated_to_natural, generated_weights),
                    "generated_to_generated": nearest_density(generated_nearest, generated_weights),
                    "natural_to_natural": nearest_density(natural_nearest, natural_weights)},
        "summary": {"generated_to_natural": summary(generated_to_natural, generated_weights),
                    "generated_to_generated": summary(generated_nearest, generated_weights),
                    "natural_to_natural": summary(natural_nearest, natural_weights)},
    }
    if test is not None:
        held_out, held_out_weights = subset(test.data.long().to(natural.device), test.weights)
        _, test_to_natural = _distance_statistics(held_out, natural, held_out_weights, natural_weights, num_states,
                                                  same=False)
        _, generated_to_test = _distance_statistics(generated, held_out, generated_weights, held_out_weights,
                                                    num_states, same=False)
        result["n_test"] = len(held_out)
        result["nearest"]["test_to_natural"] = nearest_density(test_to_natural, held_out_weights)
        result["nearest"]["generated_to_test"] = nearest_density(generated_to_test, generated_weights)
        result["summary"]["test_to_natural"] = summary(test_to_natural, held_out_weights)
        result["summary"]["generated_to_test"] = summary(generated_to_test, generated_weights)
    if len(natural) >= _PRIVET_MIN_SEQUENCES:
        def fraction(mismatches):
            return mismatches.cpu().numpy() / length

        with_test = test is not None and result["n_test"] >= _PRIVET_MIN_SEQUENCES
        result["privet"] = privet(
            fraction(natural_nearest), fraction(generated_to_natural),
            fraction(generated_to_test) if with_test else None,
            fraction(test_to_natural) if test is not None else None,
            n_train=len(natural), n_test=result["n_test"] if with_test else None,
            resolution=1.0 / length, window=privet_window,
            train_weights=natural_weights.cpu().numpy(),
        )
        result["privet"]["sample_index"] = generated_index.cpu().tolist()
    return result


def sample_sequences(
    *,
    model: DCAModel | str | Path,
    n_sequences: int,
    ptt: bool = False,
    ptt_local_sweeps: int = 10,
    ptt_max_rounds: int = 20_000,
    ptt_mixing_method: str = "renewal",
    ptt_renewal_tolerance: float = 0.01,
    ptt_stationary: bool = False,
    ptt_local_kernel: str | None = "metropolized_gibbs",
    n_sweeps: int = 1000,
    sampler: str | None = _DEFAULT_SAMPLER,
    beta: float = 1.0,
    seed: int = 0,
    reference_fasta: AlignmentInput | None = None,
    weights_path: WeightInput | None = None,
    test_fasta: AlignmentInput | None = None,
    privet_window: tuple[float, float] = DEFAULT_PRIVET_WINDOW,
    n_measure: int = 10_000,
    mixing_multiplier: int = 2,
    pseudocount: float | None = None,
    clustering_seqid: float = 0.8,
    no_reweighting: bool = False,
    alphabet: str | None = None,
    device: str = "auto",
    dtype: str | None = None,
    steering_potential: SteeringPotential | None = None,
    steering_strength: float = 1.0,
    steering_input: SteeringInput = "sequences",
    steering_proposal_steps: int | None = None,
    ptt_steering_acceptance: float = 0.3,
    collect_diagnostics: bool = False,
    progress: ProgressCallback | None = None,
    is_cancelled: CancellationHook | None = None,
) -> SamplingResult:
    """Generate sequences from a DCA model, with an energy for each.

    Ordinary sampling runs ``n_sequences`` independent Markov chains. With
    ``reference_fasta``, their mixing time is first measured (up to ``n_sweeps``
    sweeps) and generation runs for ``mixing_multiplier`` mixing times;
    otherwise it runs for exactly ``n_sweeps`` sweeps.

    PTT sampling (``ptt=True``, a PTT archive, or a :class:`PTTSampler` as
    ``model``) instead uses the replica ladder saved during training: the exact
    profile, the models flagged during training and the endpoint. It runs at
    ``beta=1``, ignores ``n_sweeps``, and equilibrates until its mixing check
    passes. A :class:`PTTSampler` passed as ``model`` is forked, not modified.
    Every result includes the summed context-dependent entropy (CDE) of each
    sequence and, when identifiable, a least-squares fit of energy against it.

    **Steered (importance) sampling.** With ``steering_potential``, sequences are
    drawn from ``p_s(x) ∝ exp(-beta * H(x) - V(x, s))`` instead, where
    ``V(x, s) = steering_potential(x, s)`` and ``s = steering_strength``: low ``V``
    is favoured, so return ``-s * score`` to favour high scores. ``V`` is called
    on batches: a list of aligned sequence strings (``steering_input="sequences"``)
    or a one-hot tensor ``(n, L, q)`` on the model's device and precision
    (``"onehot"``), and must return one number per sequence, and ``0`` when
    ``s == 0``. Each Monte Carlo move proposes ``steering_proposal_steps`` Gibbs
    updates under the DCA model, at random sites visited in palindromic order
    (which makes the proposal reversible), and accepts them with probability
    ``min(1, exp(-ΔV))``. This samples ``p_s`` exactly while calling ``V`` once
    per chain and move; ``sampler`` and ``ptt_local_kernel`` do not apply to
    steered moves.

    With PTT, steered rungs of increasing strength are added above the trained
    endpoint until ``steering_strength`` is reached, keeping the swap acceptance
    between them at least ``ptt_steering_acceptance``; the ladder's bottom is
    unchanged, so the mixing check still applies, and PTT also estimates
    ``log Z_s - log Z_0``. The result's ``log_importance_weights`` turn averages
    over the steered sequences into averages under the original model.

    Args:
        model: A :class:`DCAModel`, a path to a parameter file or PTT archive, or
            a :class:`PTTSampler`.
        n_sequences: Number of sequences to generate.
        ptt: Sample with PTT from the archive given as ``model``.
        ptt_local_sweeps: PTT: local sweeps per exchange round.
        ptt_max_rounds: PTT: largest number of exchange rounds of the mixing check.
        ptt_mixing_method: PTT: ``"renewal"`` waits until the ladder has twice been
            repopulated by exact profile draws, leaving at most
            ``ptt_renewal_tolerance`` of older configurations at the endpoint;
            ``"autocorrelation"`` measures the replica-index autocorrelation times
            and spaces output batches by twice the integrated time.
        ptt_renewal_tolerance: PTT renewal: largest remaining fraction of old configurations.
        ptt_stationary: PTT renewal: add a stationary renewal after the warmup.
        ptt_local_kernel: PTT: ``"metropolized_gibbs"``, ``"metropolis"`` or
            ``"gibbs"``; ``None`` keeps the kernel used in training.
        n_sweeps: Sweeps of ordinary sampling, or the mixing-time budget when
            ``reference_fasta`` is given.
        sampler: ``"metropolized_gibbs"``, ``"gibbs"`` or ``"metropolis"``. For PTT,
            select the local update with ``ptt_local_kernel`` instead.
        beta: Inverse temperature (ordinary sampling only).
        seed: Random seed.
        reference_fasta: Natural alignment used to measure mixing and to compare
            the generated statistics; required for PTT and for ``collect_diagnostics``.
        weights_path: Optional weights of the reference sequences.
        test_fasta: Optional held-out alignment (e.g. the validation split). With
            ``collect_diagnostics``, nearest-neighbour distances from it to the
            reference give the yardstick for the generated sequences' distances to
            the training set, and PRIVET compares each generated sequence's
            distances to both sets (overfitting check).
        privet_window: Quantiles ``(q1, q2)`` of the training nearest-neighbour
            distances fitted by PRIVET's extreme-value law.
        n_measure: Largest number of reference sequences used for the comparison.
        mixing_multiplier: Ordinary sampling: mixing times to run after measuring one.
        pseudocount: Pseudocount of the reference statistics; ``1/Meff`` if ``None``.
        clustering_seqid: Identity threshold for reweighting the reference.
        no_reweighting: Give every reference sequence the same weight.
        alphabet: Alphabet of a text parameter file; ``None`` reads it from a PTT
            archive and assumes ``"protein"`` for text files.
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.
        dtype: ``"float32"``, ``"float64"`` or ``"bfloat16"`` (CUDA/MPS, ordinary
            unsteered sampling only); ``None`` means ``"float32"``, or the archive's precision.
        steering_potential: Optional ``potential(batch, strength)`` added to the
            DCA energy, returning one value per sequence (list, NumPy array or
            tensor) and ``0`` at strength 0. ``None`` samples the model itself.
        steering_strength: Non-zero strength ``s`` passed to the potential.
        steering_input: What the potential receives: ``"sequences"`` (a list of
            aligned strings, convenient for external tools) or ``"onehot"`` (a
            tensor ``(n, L, q)``, fastest for potentials written in PyTorch).
        steering_proposal_steps: Site updates per steered move; ``None`` adapts it
            during warmup (doubling above 50% acceptance, halving below 20%, at most
            16 sweeps' worth) and then keeps it fixed. Small values suit strong or
            rugged potentials; larger ones need fewer potential evaluations per sweep.
        ptt_steering_acceptance: PTT steering: smallest swap acceptance between
            adjacent steered rungs; higher values place more rungs.
        collect_diagnostics: Keep the correlations and PCA projections needed by
            :meth:`SamplingResult.save_diagnostic_plots`.
        progress: Optional callback receiving :class:`SamplingProgress` events.
        is_cancelled: Optional callable; returning ``True`` stops sampling.

    Returns:
        A :class:`SamplingResult` with the sequences, energies and diagnostics.

    Raises:
        InputValidationError: If a setting is invalid or incompatible with PTT.
        OperationCancelledError: If ``is_cancelled`` returns ``True``.

    Example:
        >>> result = sample_sequences(model="params.dat.gz", n_sequences=1000,
        ...                           reference_fasta="family.fasta")
        >>> result.to_fasta("samples.fasta")

        Steering towards sequences with a high GC content (RNA), as strings:

        >>> def gc_potential(sequences, strength):
        ...     return [-strength * sum(c in "GC" for c in s) / len(s) for s in sequences]
        >>> steered = sample_sequences(model="params.dat.gz", alphabet="rna", n_sequences=500,
        ...                            steering_potential=gc_potential, steering_strength=20.0)
        >>> steered.steering["effective_sample_size"], steered.log_importance_weights[:3]
    """
    from adabmDCA.ptt import PTTSampler

    if ptt or isinstance(model, PTTSampler):
        if (
            beta != 1.0 or dtype == "bfloat16"
        ):
            raise InputValidationError(
                "PTT generation supports beta=1 and full precision only."
            )
        from adabmDCA.api.ptt import sample_ptt_sequences

        _check_cancelled(is_cancelled)
        return sample_ptt_sequences(
            source=model, n_sequences=n_sequences, seed=seed,
            device=device, alphabet=alphabet,
            sampler=None if sampler is _DEFAULT_SAMPLER else sampler, dtype=dtype,
            local_sweeps=ptt_local_sweeps, reference_fasta=reference_fasta, weights_path=weights_path,
            test_fasta=test_fasta, privet_window=privet_window, clustering_seqid=clustering_seqid, no_reweighting=no_reweighting,
            n_measure=n_measure, pseudocount=pseudocount,
            max_rounds=ptt_max_rounds,
            mixing_method=ptt_mixing_method,
            renewal_tolerance=ptt_renewal_tolerance,
            stationary=ptt_stationary,
            local_kernel=ptt_local_kernel,
            steering_potential=steering_potential, steering_strength=steering_strength,
            steering_input=steering_input, steering_proposal_steps=steering_proposal_steps,
            steering_acceptance=ptt_steering_acceptance,
            collect_diagnostics=collect_diagnostics,
            progress=progress, is_cancelled=is_cancelled,
        )
    sampler = "metropolized_gibbs" if sampler is None or sampler is _DEFAULT_SAMPLER else sampler
    dtype = "float32" if dtype is None else dtype
    validate_integer("n_sequences", n_sequences)
    validate_integer("n_sweeps", n_sweeps, minimum=0)
    validate_integer("mixing_multiplier", mixing_multiplier)
    validate_integer("n_measure", n_measure)
    validate_seed(seed)
    if reference_fasta is not None and n_sweeps < 2:
        raise InputValidationError("n_sweeps must be at least 2 when estimating mixing time.")
    if collect_diagnostics and reference_fasta is None:
        raise InputValidationError("reference_fasta is required when collect_diagnostics=True.")
    if test_fasta is not None and not collect_diagnostics:
        raise InputValidationError("test_fasta is used only with collect_diagnostics=True (CLI: --plot).")
    if sampler not in {"gibbs", "metropolis", "metropolized_gibbs"}:
        raise InputValidationError("sampler must be 'gibbs', 'metropolis' or 'metropolized_gibbs'.")
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

    metadata = loaded.metadata
    samples = init_chains(
        num_chains=n_sequences,
        L=metadata.length,
        q=metadata.num_states,
        device=loaded.params["bias"].device,
        dtype=loaded.params["bias"].dtype,
    )
    steered_kernel = None
    if steering_potential is not None:
        _validate_steering_strength(steering_strength)
        if dtype == "bfloat16":
            raise InputValidationError("Steered sampling supports float32 and float64 only.")
        steering = Steering(steering_potential, tokens=loaded.tokens, steering_input=steering_input)
        steering.check_zero_strength(samples[:16], dtype=samples.dtype)
        steered_kernel = SteeredKernel(
            steering, steering_strength, length=metadata.length, device=samples.device,
            proposal_steps=steering_proposal_steps,
        )
        sampling_function, sampling_params = steered_kernel, loaded.params
    else:
        try:
            sampling_function, sampling_params = prepare_fixed_model_sampler(
                sampler,
                loaded.params["bias"].device,
                dtype,
                loaded.params,
            )
        except ValueError as exc:
            raise InputValidationError(str(exc)) from exc
    mixing_history: dict[str, list[float]] = {}
    sampling_history: dict[str, list[float]] = {}
    cij_reference = cij_generated = None
    pca_reference = pca_generated = pca_explained_variance_ratio = None
    data_comparison: dict[str, Any] = {}
    distance_comparison: dict[str, Any] = {}

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
        test = None if test_fasta is None else _load_test_alignment(
            test_fasta, dataset, clustering_seqid=clustering_seqid, no_reweighting=no_reweighting
        )
        effective_pseudocount = pseudocount
        if effective_pseudocount is None:
            effective_pseudocount = 1.0 / dataset.weights.sum().item()
        measured = min(n_measure, len(dataset))
        # Draw the reference rows before one-hot encoding: the full alignment
        # can be far larger in one-hot form than the few rows needed.
        reference = torch.nn.functional.one_hot(
            resample_sequences(dataset.data, dataset.weights, measured).long(), len(dataset.tokens)
        ).to(dtype=dataset.dtype, device=dataset.device)
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
        if steered_kernel is not None and sweep == total_sweeps // 2:
            # Adapt the proposal blocks during the first half only.
            steered_kernel.freeze()
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
        data_comparison = _compare_with_data(reference, samples, reference_scores, generated_scores, loaded.params,
                                             seed=seed)
        distance_comparison = _distance_comparison(dataset, samples, test=test, n_measure=n_measure, seed=seed,
                                                   privet_window=privet_window)

    energies = compute_energy(samples, params=loaded.params).detach().cpu().numpy()
    from adabmDCA.api.scoring import fit_local_lambda as fit_slope
    from adabmDCA.api.scoring import summed_cde

    cde_sum = summed_cde(samples, loaded.params)
    lambda_fit = fit_warning = None
    try:
        lambda_fit = fit_slope(energies, cde_sum)
    except InputValidationError as exc:
        fit_warning = str(exc)
    decoded = decode_sequence(samples.detach().cpu().numpy(), loaded.tokens)
    result_warnings = [fit_warning] if fit_warning else []
    steering_potentials = log_weights = None
    steering_summary: dict[str, Any] = {}
    if steered_kernel is not None:
        steering_potentials = steered_kernel.values(samples).cpu().numpy()
        log_weights, ess = importance_summary(steering_potentials)
        steering_summary = {
            "strength": float(steering_strength), "input": steering_input,
            "proposal_steps": steered_kernel.proposal_steps, "acceptance": steered_kernel.acceptance,
            "potential_evaluations": steered_kernel.steering.evaluations,
            "effective_sample_size": ess, "weights": "self_normalized",
        }
        if reference_fasta is not None:
            result_warnings.append("Comparisons with reference_fasta describe the steered distribution.")
        if (warning := importance_warning(ess, len(steering_potentials))) is not None:
            result_warnings.append(warning)
        if steered_kernel.acceptance < 0.05:
            result_warnings.append(
                f"Steered moves were accepted {steered_kernel.acceptance:.1%} of the time; "
                "lower steering_proposal_steps or steering_strength."
            )
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
        cde_sum=cde_sum,
        local_lambda_fit=lambda_fit,
        warnings=tuple(result_warnings),
        steering_potentials=steering_potentials,
        log_importance_weights=log_weights,
        steering=steering_summary,
        cij_reference=cij_reference,
        cij_generated=cij_generated,
        pca_reference=pca_reference,
        pca_generated=pca_generated,
        pca_explained_variance_ratio=pca_explained_variance_ratio,
        data_comparison=data_comparison,
        distance_comparison=distance_comparison,
    )


def generate_sequences(**kwargs: Any) -> tuple[str, ...]:
    """Return only the sequences of :func:`sample_sequences`.

    Args:
        **kwargs: Keyword arguments of :func:`sample_sequences`; ``model`` and
            ``n_sequences`` are required.

    Returns:
        The generated sequences.

    Example:
        >>> generate_sequences(model="params.dat.gz", n_sequences=10, n_sweeps=500)
    """
    return sample_sequences(**kwargs).sequences
