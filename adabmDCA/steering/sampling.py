"""Sequence generation with compiled fields, hard support and weighted callbacks."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from adabmDCA._validation import validate_integer, validate_seed
from adabmDCA.api.model import DCAModel, load_model
from adabmDCA.api.results import SamplingProgress, SamplingResult
from adabmDCA.exceptions import InputValidationError, OperationCancelledError
from adabmDCA.steering.diagnostics import partition, partition_point, summarize_traces
from adabmDCA.steering.kernels import PottsKernel
from adabmDCA.steering.support import ExactProfile, HammingBias, HammingWindow, Support


@dataclass(frozen=True)
class WeightedPotential:
    """Deterministic batch callback returning one finite unweighted cost per sequence.

    Input is a list of strings by default or an integer NumPy array with
    shape (batch, sites) when input is 'categorical'.
    """

    evaluate: Callable
    weight: float = 1.0
    input: str = "sequences"


@dataclass(frozen=True)
class SteeringConfig:
    """Define a structured steering target and its sampling diagnostics.

    Args:
        allowed_tokens: Subset of model tokens allowed at every site.
        windows: Inclusive Hamming constraints, applied jointly on every rung.
        distance_biases: Linear Hamming costs folded into the Potts fields.
        potentials: Weighted deterministic batch costs applied to Gibbs proposals.
        chains: Number of parallel chains per rung.
        groups: Number of independently seeded groups for convergence checks.
        block_steps: Site updates per proposal. The default visits each site once.
        check_every: Warmup rounds between convergence checks.
        min_rounds: Minimum warmup rounds.
        min_ess: Minimum bulk and tail ESS for each monitored group-mean trace.
        rhat: Upper bound on rank-normalized split R-hat.
        production_spacing: Rounds between retained batches. Zero takes one final snapshot.
        warmup_seconds: Optional wall-clock budget for setup, pilot, and warmup.
        ladder_steps: Initial number of increments from the trained to steered target.
        bridge_steps: Initial number of increments between consecutive base models.
        pilot_rounds: Rounds per PTT pilot block. Zero skips the pilot.
        max_rungs: Upper bound on the rung count for pilot insertions.
        min_swap: Pilot swap acceptance below which an intermediate rung is considered.
        cache_size: Maximum number of cached sequences per callback.
        entropy: Estimate target entropy using PTT and the exact bottom normalizer.
    """

    allowed_tokens: str | None = None
    windows: tuple[HammingWindow, ...] = ()
    distance_biases: tuple[HammingBias, ...] = ()
    potentials: tuple[WeightedPotential, ...] = ()
    chains: int = 128
    groups: int = 4
    block_steps: int | None = None
    check_every: int = 8
    min_rounds: int = 32
    min_ess: float = 40
    rhat: float = 1.01
    production_spacing: int = 1
    ladder_steps: int = 4
    bridge_steps: int = 2
    pilot_rounds: int = 8
    max_rungs: int = 24
    min_swap: float = 0.25
    cache_size: int = 8192
    entropy: bool = False
    warmup_seconds: float | None = None


@dataclass(eq=False)
class _Rung:
    kernel: PottsKernel
    beta: float
    fields: np.ndarray
    weights: np.ndarray
    label: str
    states: np.ndarray | None = None
    values: np.ndarray | None = None
    energy: np.ndarray | None = None
    births: np.ndarray | None = None
    accepted: int = 0
    proposed: int = 0

    def reduced(self, states=None, values=None, energy=None):
        if states is None:
            states, values, energy = self.states, self.values, self.energy
        if energy is None:
            energy = self.kernel.energy(states)
        linear = self.fields[np.arange(states.shape[1]), states].sum(axis=1)
        return self.beta * energy + linear + self.bias_energy(values)

    def bias_energy(self, values):
        active = self.weights != 0
        return values[:, active] @ self.weights[active]


class _PotentialCache:
    def __init__(self, potentials, tokens, capacity):
        self.potentials = potentials
        self.tokens = np.array(list(tokens))
        self.cache = [OrderedDict() for _ in potentials]
        self.capacity = capacity
        self.calls = [0] * len(potentials)
        self.seconds = 0.0

    def evaluate(self, batches):
        # Each batch contributes destinations and the subset of columns needed by its rung.
        for index, potential in enumerate(self.potentials):
            cache = self.cache[index]
            missing = {}
            for states, values, rows, needed in batches:
                if not needed[index]:
                    continue
                for row in rows[np.isnan(values[rows, index])]:
                    key = states[row].tobytes()
                    if key in cache:
                        values[row, index] = cache[key]
                        cache.move_to_end(key)
                    else:
                        missing.setdefault(key, []).append((values, row))
            if not missing:
                continue
            states = np.frombuffer(b"".join(missing), dtype=np.int32).reshape(len(missing), -1)
            argument = states if potential.input == "categorical" else ["".join(row) for row in self.tokens[states]]
            start = perf_counter()
            result = potential.evaluate(argument)
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            result = np.asarray(result, dtype=np.float64)
            self.seconds += perf_counter() - start
            self.calls[index] += len(states)
            if result.shape != (len(states),) or not np.isfinite(result).all():
                raise InputValidationError("A weighted potential must return one finite number per sequence.")
            for (key, destinations), value in zip(missing.items(), result):
                for values, row in destinations:
                    values[row, index] = value
                if self.capacity:
                    cache[key] = value
                    if len(cache) > self.capacity:
                        cache.popitem(last=False)


def _validate(config, beta, n_sequences, seed, rounds, renewal_tolerance):
    for name in ("chains", "groups", "check_every", "min_rounds", "ladder_steps", "bridge_steps"):
        validate_integer(name, getattr(config, name))
    validate_integer("cache_size", config.cache_size, minimum=0)
    validate_integer("pilot_rounds", config.pilot_rounds, minimum=0)
    validate_integer("max_rungs", config.max_rungs, minimum=2)
    if not 0 < config.min_swap < 1:
        raise InputValidationError("Steering min_swap must lie between zero and one.")
    validate_integer("n_sequences", n_sequences)
    validate_integer("production_spacing", config.production_spacing, minimum=0)
    if config.production_spacing == 0 and n_sequences > config.chains:
        raise InputValidationError("A final snapshot requires n_sequences <= chains.")
    if config.warmup_seconds is not None and (
        not math.isfinite(config.warmup_seconds) or config.warmup_seconds <= 0
    ):
        raise InputValidationError("warmup_seconds must be finite and positive.")
    validate_integer("round limit", rounds)
    validate_seed(seed)
    if config.block_steps is not None:
        validate_integer("block_steps", config.block_steps)
    if config.groups < 2 or config.chains < config.groups or config.min_rounds < 4:
        raise InputValidationError("Steering needs at least two groups, a chain per group and four warmup rounds.")
    if not math.isfinite(beta) or beta <= 0:
        raise InputValidationError("beta must be greater than zero.")
    if not math.isfinite(config.rhat) or config.rhat <= 1 or not math.isfinite(config.min_ess) or config.min_ess <= 0:
        raise InputValidationError("Steering rhat must exceed one and min_ess must be positive.")
    if not 0 < renewal_tolerance < 1:
        raise InputValidationError("The PTT renewal tolerance must lie between zero and one.")
    for potential in config.potentials:
        if (not callable(potential.evaluate) or not math.isfinite(potential.weight)
                or potential.input not in ("sequences", "categorical")):
            raise InputValidationError("Invalid weighted potential callback, weight, or input.")


def _fields(config, tokens, length):
    references = Support.encode_references(tokens, length, [bias.reference for bias in config.distance_biases])
    weights = np.array([bias.weight for bias in config.distance_biases])
    if not np.isfinite(weights).all():
        raise InputValidationError("Hamming bias weights must be finite.")
    return ((np.arange(len(tokens))[None, None, :] != references[:, :, None]) * weights[:, None, None]).sum(axis=0)


def _masked_anchor(anchor, support):
    if len(support.indices) == support.q:
        return anchor
    from adabmDCA.anchor.anchor import TreeLowRankPotts

    indices = torch.as_tensor(support.indices, device=anchor.fields.device)
    parameters = (anchor.fields[:, indices], anchor.weights[:, indices])
    if isinstance(anchor, TreeLowRankPotts):
        parameters += (anchor.edges, anchor.pairs[:, indices][:, :, indices])
    return type(anchor)(*parameters, tolerance=anchor.tolerance)


def _make_ladder(config, backend, params, support, beta, fields, weights, ptt):
    target = PottsKernel(params, support)
    anchor = None if backend is None else backend.anchor
    reference_kernel = target
    if ptt and anchor is not None and len(support.references):
        reference_kernel = PottsKernel(anchor.potts_params(params["bias"].dtype, target.device), support)
    profile_fields = reference_kernel.bias.detach().cpu().double().numpy().copy()
    # Approximate the first coupled rung with site conditionals at the
    # reference. A narrow shell visits far fewer combinations than the
    # full alphabet, where the raw fields need not be a useful bridge.
    if len(support.references):
        ref = support.references[0]
        rows = reference_kernel.rows.detach().cpu().numpy()
        profile_fields += rows[:, :, np.arange(support.length), ref].sum(axis=-1)
    if not ptt:
        profile = ExactProfile(beta * profile_fields - fields, support)
        return [_Rung(target, beta, fields, weights, "target")], profile, None, None
    zero = np.zeros_like(weights)
    empty = np.zeros_like(fields)
    points = [] if backend is None else sorted(backend.ptt_checkpoints, key=lambda point: point["step"])
    ladder = []
    exact_anchor = _masked_anchor(anchor, support) if anchor is not None and not len(support.maximum) else None
    if exact_anchor is not None:
        ladder.append(_Rung(PottsKernel(anchor.potts_params(params["bias"].dtype, params["bias"].device),
                                       support), 1, empty, zero, "archive anchor"))
        log_z0 = exact_anchor.log_z
        profile = None
    else:
        if backend is not None and anchor is None and points:
            profile_fields = points[0]["params"]["bias"].detach().cpu().double().numpy()
        bottom_params = {"bias": torch.as_tensor(profile_fields, device=target.device, dtype=params["bias"].dtype),
                         "coupling_matrix": torch.zeros_like(params["coupling_matrix"])}
        # Use identical fields in the exact draw and in exchange arithmetic.
        profile = ExactProfile(bottom_params["bias"].cpu().double().numpy(), support)
        ladder.append(_Rung(PottsKernel(bottom_params, support), 1, empty, zero, "exact profile"))
        log_z0 = profile.log_z
    if backend is not None:
        usable = [point for point in points if point["step"] != backend.model_version
                  and (point["step"] != 0 or (anchor is not None and not exact_anchor))]
        for point in usable:
            label = "restricted archive anchor" if point["step"] == 0 else f"archive step {point['step']}"
            ladder.append(_Rung(PottsKernel(point["params"], support), 1, empty, zero, label))
    endpoint_is_anchor = anchor is not None and backend.model_version == 0
    if exact_anchor is None or not endpoint_is_anchor:
        label = "restricted archive anchor" if endpoint_is_anchor else "trained endpoint"
        ladder.append(_Rung(target, 1, empty, zero, label))
    elif beta == 1 and not np.any(weights) and not np.any(fields):
        # Reuse the exact bottom as the endpoint when no trajectory exists.
        ladder[0].kernel = target

    if config.bridge_steps > 1:
        bridged = [ladder[0]]
        for left, right in zip(ladder, ladder[1:]):
            for fraction in np.linspace(0, 1, config.bridge_steps + 1)[1:-1]:
                bridged.append(_interpolate(left, right, float(fraction), support))
            bridged.append(right)
        ladder = bridged
    if beta != 1 or np.any(weights) or np.any(fields):
        for fraction in np.linspace(0, 1, config.ladder_steps + 1)[1:]:
            ladder.append(_Rung(target, 1 + (beta - 1) * fraction, fields * fraction, weights * fraction,
                                f"steering {fraction:.2g}"))
    return ladder, profile, log_z0, exact_anchor


def _interpolate(left, right, fraction, support):
    beta = (1 - fraction) * left.beta + fraction * right.beta
    kernel = left.kernel
    if left.kernel is not right.kernel:
        mix = fraction * right.beta / beta
        kernel = PottsKernel({
            "bias": torch.lerp(left.kernel.bias, right.kernel.bias, mix),
            "coupling_matrix": torch.lerp(left.kernel.rows, right.kernel.rows, mix),
        }, support)
    return _Rung(kernel, beta, (1 - fraction) * left.fields + fraction * right.fields,
                 (1 - fraction) * left.weights + fraction * right.weights,
                 f"{left.label} → {right.label} ({fraction:.2g})")


def _cross_energies(left, right):
    if left.kernel is right.kernel:
        return left.energy, right.energy
    return right.kernel.energy(left.states), left.kernel.energy(right.states)


def _pair_work(left, right, crossed=None):
    a, b = _cross_energies(left, right) if crossed is None else crossed
    return (right.reduced(left.states, left.values, a) - left.reduced(),
            right.reduced() - left.reduced(right.states, right.values, b))


def _refresh_energy(rung):
    rung.energy = rung.kernel.energy(rung.states)


def _swap(left, right, accepted, crossed):
    for name in ("states", "values", "births"):
        a, b = getattr(left, name), getattr(right, name)
        a[accepted], b[accepted] = b[accepted], a[accepted]
    a, b = crossed
    left.energy[accepted], right.energy[accepted] = b[accepted], a[accepted]


@torch.no_grad()
def _sample_steered_sequences(
    *, model, n_sequences: int, steering: SteeringConfig, beta: float = 1.0,
    ptt: bool = False, max_rounds: int = 1000, renewal_tolerance: float = 0.01,
    alphabet=None, device="auto", dtype=None, seed=0, progress=None, is_cancelled=None,
    initial_sequences=None,
) -> SamplingResult:
    """Sample the constrained Potts law with explicit reduced-energy biases.

    PTT can use a training archive or construct a ladder from a standalone
    model. Running out of warmup returns a labelled checkpoint. Entropy is
    available with PTT, from its exactly sampled profile or archived anchor.
    The result's steering history records warmup, retained production batches,
    diagnostic checks, and PTT pilot blocks.

    Args:
        model: DCA model, parameter path, PTT archive path, or PTT sampler.
        n_sequences: Number of retained draws.
        steering: Hard support, biases, and sampling settings.
        beta: Inverse Potts temperature. Explicit bias weights use reduced-energy units.
        ptt: Use a replica ladder. Passing a PTT sampler also selects this path.
        max_rounds: Combined pilot and warmup round budget.
        renewal_tolerance: Maximum number of old configurations per chain per rung,
            counting old configurations across the whole ladder.
        alphabet: Model alphabet when loading a standalone parameter file.
        device: Device when loading the model or archive.
        dtype: Precision when loading a standalone parameter file.
        seed: Random seed for the run.
        progress: Optional callback receiving sampling progress events.
        is_cancelled: Optional cancellation callback checked during sampling.
        initial_sequences: Starting population for rungs above the exact PTT bottom.
    """
    from adabmDCA.api.ptt import load_ptt_backend
    from adabmDCA.ptt import PTTSampler

    started = perf_counter()

    def time_expired():
        return steering.warmup_seconds is not None and perf_counter() - started >= steering.warmup_seconds

    _validate(steering, beta, n_sequences, seed, max_rounds, renewal_tolerance)
    if is_cancelled is not None and is_cancelled():
        raise OperationCancelledError("Sampling was cancelled.")
    ptt = ptt or isinstance(model, PTTSampler)
    if initial_sequences is not None and not ptt:
        raise InputValidationError("Use PTT to provide initial_sequences with structured steering.")
    if steering.entropy and not ptt:
        raise InputValidationError("Absolute steered entropy requires a PTT ladder.")
    backend = None
    path = isinstance(model, (str, Path)) and str(model).endswith((".h5", ".hdf5"))
    if isinstance(model, PTTSampler) or path:
        backend = load_ptt_backend(model, device=device, seed=seed, alphabet=alphabet)
        loaded = DCAModel(backend.endpoint_params(), alphabet=backend.tokens, source=model if path else None)
    else:
        loaded = model if isinstance(model, DCAModel) else load_model(
            model, alphabet=alphabet, device=device, dtype=dtype or "float32"
        )
    params = loaded.params
    length, q = params["bias"].shape
    support = Support(loaded.tokens, length, steering.allowed_tokens, steering.windows)
    fields = _fields(steering, loaded.tokens, length)
    potential_indices = [i for i, potential in enumerate(steering.potentials) if potential.weight]
    potentials = tuple(steering.potentials[i] for i in potential_indices)
    weights = np.array([potential.weight for potential in potentials])
    ladder, profile, log_z0, exact_anchor = _make_ladder(steering, backend, params, support, beta, fields, weights, ptt)
    top = ladder[-1]
    cache = _PotentialCache(potentials, loaded.tokens, steering.cache_size)
    def needed_values():
        required = [rung.weights != 0 for rung in ladder]
        for i, (left, right) in enumerate(zip(ladder, ladder[1:])):
            required[i] |= right.weights != 0
            required[i + 1] |= left.weights != 0
        return required

    needed = needed_values()

    size = steering.chains
    groups = np.array_split(np.arange(size), steering.groups)
    initial_states = None
    if initial_sequences is not None:
        from adabmDCA.api.sampling import _initial_states

        if len(ladder) == 1:
            raise InputValidationError("Initial sequences require a PTT rung above the exact bottom.")
        initial_states = _initial_states(initial_sequences, loaded, size, seed).cpu().numpy().astype(np.int32)
        if not support.contains(initial_states).all():
            raise InputValidationError("Initial sequences must satisfy the steering alphabet and hard windows.")
        initial_rng = np.random.default_rng(np.random.SeedSequence([seed % 2**64, 1]))
    rngs = [np.random.default_rng(child) for child in np.random.SeedSequence(seed % 2**64).spawn(len(groups))]

    def random():
        return np.concatenate([rng.random(len(group)) for rng, group in zip(rngs, groups)])

    def draw(source):
        if source is not None:
            return np.concatenate([source.draw(len(group), rng) for group, rng in zip(groups, rngs)])
        device = exact_anchor.fields.device
        cuda = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda):
            states = []
            for group, rng in zip(groups, rngs):
                group_seed = int(rng.integers(0, 2**62, dtype=np.int64))
                torch.random.default_generator.manual_seed(group_seed)
                if cuda:
                    torch.cuda.default_generators[cuda[0]].manual_seed(group_seed)
                states.append(support.indices[exact_anchor.sample(len(group), device=device).cpu().numpy()])
            return np.concatenate(states).astype(np.int32)

    def evaluate(changed, indices):
        cache.evaluate([(ladder[i].states, ladder[i].values, changed[i], needed[i]) for i in indices])

    indices = np.arange(size)
    for i, rung in enumerate(ladder):
        if i and initial_states is not None:
            order = np.concatenate([initial_rng.permutation(group) for group in groups])
            rung.states = initial_states[order]
        else:
            rung.states = draw(profile)
        rung.values = np.full((size, len(weights)), np.nan)
        rung.births = np.zeros(size, dtype=np.int64)
        _refresh_energy(rung)
    if not ptt and support.restricted:
        dispersed = ExactProfile(np.zeros_like(fields), support)
        for group, rng in list(zip(groups, rngs))[1::2]:
            top.states[group] = dispersed.draw(len(group), rng)
        _refresh_energy(top)
    evaluate([indices] * len(ladder), range(len(ladder)))

    moves = steering.block_steps or length
    rounds = 0
    swap_total = np.zeros(len(ladder) - 1)
    swap_count = np.zeros_like(swap_total)

    def local_update():
        proposed = {}
        batches = []
        acceptance = [None] * len(ladder)
        for i, rung in enumerate(ladder):
            if ptt and i == 0:
                rung.states = draw(profile)
                rung.values.fill(np.nan)
                rung.births.fill(rounds)
                _refresh_energy(rung)
                batches.append((rung.states, rung.values, indices, needed[i]))
                continue
            schedules, uniforms = [], []
            for group, rng in zip(groups, rngs):
                # A random ordering mixed uniformly with its reverse is a
                # reversible block, while visiting each site once per sweep.
                sites = np.concatenate([rng.permutation(length) for _ in range(math.ceil(moves / length))])[:moves]
                if rng.integers(0, 2):
                    sites = sites[::-1].copy()
                schedules.append(sites)
                uniforms.append(rng.random((len(group), len(sites))))
            states = rung.kernel.propose_groups(rung.states, rung.beta, rung.fields, groups, schedules, uniforms)
            changed = np.flatnonzero((states != rung.states).any(axis=1))
            values = rung.values.copy()
            values[changed] = np.nan
            proposed[i] = states, values, changed
            batches.append((states, values, changed, needed[i]))
        cache.evaluate(batches)
        for i, (states, values, changed) in proposed.items():
            rung = ladder[i]
            delta = rung.bias_energy(values - rung.values)
            accepted = np.log(random()) < -delta
            rung.proposed += size
            accepted_count = int(accepted.sum())
            rung.accepted += accepted_count
            acceptance[i] = accepted_count / size
            kept = changed[accepted[changed]]
            rung.states[kept] = states[kept]
            rung.values[kept] = values[kept]
            if len(kept):
                rung.energy[kept] = rung.kernel.energy(rung.states[kept])
        return acceptance

    def supplemental_move(global_move):
        proposals = []
        for i, rung in enumerate(ladder):
            if (ptt and i == 0) or not support.restricted:
                continue
            states = rung.states.copy()
            if global_move:
                states = draw(profile)
            else:
                for group, rng in zip(groups, rngs):
                    rows = np.repeat(group, 2)
                    sites = rng.integers(length, size=len(rows))
                    states[rows, sites] = rng.choice(support.indices, size=len(rows))
            changed = np.flatnonzero(support.contains(states) & (states != rung.states).any(axis=1))
            values = rung.values.copy()
            values[changed] = np.nan
            proposals.append((i, states, values, changed))
        cache.evaluate([(states, values, changed, needed[i]) for i, states, values, changed in proposals])
        for i, states, values, changed in proposals:
            if not len(changed):
                continue
            rung = ladder[i]
            candidate = states[changed]
            energy = rung.kernel.energy(candidate)
            delta = rung.reduced(candidate, values[changed], energy) - rung.reduced()[changed]
            if global_move:
                position = np.arange(length)
                delta += (profile.fields[position, candidate] - profile.fields[position, rung.states[changed]]).sum(1)
            accepted = np.log(random()[changed]) < -delta
            keep = changed[accepted]
            rung.states[keep], rung.values[keep], rung.energy[keep] = candidate[accepted], values[keep], energy[accepted]

    def step():
        nonlocal rounds
        if is_cancelled is not None and is_cancelled():
            raise OperationCancelledError("Sampling was cancelled.")
        rounds += 1
        local_acceptance = local_update()
        exchange_acceptance = [0.0] * (len(ladder) - 1)
        if len(support.maximum) and rounds % 4 == 0:
            supplemental_move(global_move=rounds % 32 == 0)
        for i in list(range(rounds % 2, len(ladder) - 1, 2)) + list(range((rounds + 1) % 2, len(ladder) - 1, 2)):
            left, right = ladder[i:i + 2]
            crossed = _cross_energies(left, right)
            a, b = _pair_work(left, right, crossed)
            accepted = np.log(random()) < b - a
            exchange_acceptance[i] = float(accepted.mean())
            swap_total[i] += exchange_acceptance[i]
            swap_count[i] += 1
            _swap(left, right, accepted, crossed)
            changed = np.flatnonzero(accepted)
            cache.evaluate([(rung.states, rung.values, changed, needed[j])
                            for j, rung in ((i, left), (i + 1, right))])
        return local_acceptance, exchange_acceptance

    # Insert interpolants only in pilot; warmup and production use a fixed
    # ladder. Preserve the archived checkpoints and prioritize weak edges.
    additions = 0
    pilot_history = []
    while ptt and steering.pilot_rounds and rounds + steering.pilot_rounds <= max_rounds // 3:
        first_round = rounds + 1
        for _ in range(steering.pilot_rounds):
            if time_expired():
                break
            step()
        if time_expired():
            break
        rates = swap_total / swap_count
        block = {
            "first_round": first_round, "round": rounds, "elapsed_seconds": perf_counter() - started,
            "ladder": [rung.label for rung in ladder], "swap_acceptance": rates.tolist(), "inserted_rungs": [],
        }
        pilot_history.append(block)
        weak = np.flatnonzero(rates < steering.min_swap)
        available = steering.max_rungs - len(ladder)
        if not len(weak) or available <= 0:
            break
        weakest = weak[np.argsort(rates[weak])[:available]]
        for i in sorted(weakest, reverse=True):
            left, right = ladder[i:i + 2]
            middle = _interpolate(left, right, 0.5, support)
            take_left = random() < 0.5
            middle.states = np.where(take_left[:, None], left.states, right.states)
            middle.values = np.where(take_left[:, None], left.values, right.values)
            middle.births = np.zeros(size, dtype=np.int64)
            _refresh_energy(middle)
            ladder.insert(i + 1, middle)
            block["inserted_rungs"].append(middle.label)
        additions += len(weakest)
        needed = needed_values()
        evaluate([indices] * len(ladder), range(len(ladder)))
        swap_total = np.zeros(len(ladder) - 1)
        swap_count = np.zeros_like(swap_total)
    pilot = rounds
    if pilot:
        for rung in ladder:
            rung.births.fill(0)
            rung.accepted = rung.proposed = 0
        swap_total = np.zeros(len(ladder) - 1)
        swap_count = np.zeros_like(swap_total)
        if progress is not None:
            progress(SamplingProgress(stage="steering_pilot", completed=pilot, total=max_rounds,
                                      details={"added_rungs": additions, "rungs": len(ladder)}))

    base_names = ["potts_energy", "reduced_energy"]
    base_names += [f"potential_{i + 1}" for i in range(len(weights))]
    distance_names = [f"distance_{i + 1}" for i in range(len(support.maximum))]
    # Distance boundary occupancy and a projection encoding the spatial
    # distribution of substitutions catch drift hidden by mean distance.
    names = base_names + distance_names + [name + "_at_max" for name in distance_names] + ["sequence_projection"]
    projection = np.random.default_rng(84279).normal(size=(length, q))

    def observe():
        distances = support.distances(top.states)
        sequence_score = projection[np.arange(length), top.states].sum(axis=1)
        observable = np.column_stack((top.energy, top.reduced(), top.values, distances,
                                      distances == support.maximum, sequence_score))
        return np.stack([observable[group].mean(axis=0) for group in groups])

    def snapshot(observed, local_acceptance, exchange_acceptance, retained=0):
        means = observed.mean(axis=0)
        old = [int(np.count_nonzero(rung.births == 0)) for rung in ladder] if ptt else []
        return {
            "round": rounds, "elapsed_seconds": perf_counter() - started,
            "observables": dict(zip(names, means.tolist())),
            "group_means": dict(zip(names, observed.T.tolist())),
            "rung_potts_energy": [float(rung.energy.mean()) for rung in ladder],
            "rung_potential_means": [
                [float(rung.values[:, j].mean()) if recorded else None for j, recorded in enumerate(required)]
                for rung, required in zip(ladder, needed)
            ],
            "block_acceptance": local_acceptance, "swap_acceptance": exchange_acceptance,
            "old_configurations_by_rung": old, "old_configurations_per_chain": sum(old) / size,
            "potential_evaluations": cache.calls.copy(), "retained": retained,
        }

    warmup_observations = []
    warmup_history = []
    checks = []
    diagnostics = {}
    converged = False
    old_fraction = float(len(ladder)) if ptt else 0.0
    for _ in range(max_rounds - pilot):
        if time_expired():
            break
        rates = step()
        observed = observe()
        warmup_observations.append(observed)
        record = snapshot(observed, *rates)
        warmup_history.append(record)
        old_fraction = record["old_configurations_per_chain"]
        warmup_round = len(warmup_observations)
        expired = time_expired()
        if warmup_round >= max(6, steering.min_rounds) and (
            warmup_round % steering.check_every == 0 or rounds == max_rounds or expired
        ):
            diagnostics = summarize_traces(warmup_observations[warmup_round // 3:], names)
            pass_observables = all(item["rhat"] < steering.rhat and item["bulk_ess"] >= steering.min_ess
                                   and item["tail_ess"] >= steering.min_ess for item in diagnostics.values())
            converged = pass_observables and old_fraction <= renewal_tolerance
            checks.append({
                "round": rounds, "warmup_round": warmup_round, "observables": diagnostics,
                "old_configurations_per_chain": old_fraction, "converged": converged,
            })
        if progress is not None and (rounds % steering.check_every == 0 or converged):
            progress(SamplingProgress(stage="mixing", completed=rounds, total=max_rounds,
                                      details={"old_fraction": old_fraction, "observables": diagnostics}))
        if converged or expired:
            break
    deadline_reached = time_expired() and not converged
    warmup = len(warmup_observations)
    before_production = rounds
    local_acceptance = [rung.accepted / rung.proposed if rung.proposed else None for rung in ladder]
    batches, energies, values, raw_values, endpoints = [], [], [], [], []
    production_history = []
    production_batches = []
    work = [[] for _ in range(len(ladder) - 1)]
    spacing = steering.production_spacing
    for start in range(0, n_sequences, size):
        previous_acceptance = [rung.accepted for rung in ladder]
        previous_swaps = swap_total.copy()
        for _ in range(spacing):
            step()
        count = min(size, n_sequences - start)
        batches.append(top.states[:count].copy())
        energies.append(top.energy[:count].copy())
        values.append((top.reduced() - beta * top.energy)[:count])
        raw_values.append(top.values[:count].copy())
        observed = observe()
        block_rates = [(rung.accepted - before) / (spacing * size) if spacing and rung.proposed else None
                       for rung, before in zip(ladder, previous_acceptance)]
        production_batches.append(snapshot(
            observed, block_rates, ((swap_total - previous_swaps) / spacing).tolist() if spacing
            else [None] * len(swap_total), retained=count,
        ))
        if count == size:
            production_history.append(observed)
        if steering.entropy:
            for i, pair in enumerate(zip(ladder, ladder[1:])):
                work[i].append(_pair_work(*pair))
            endpoints.append(top.reduced())
        if progress is not None:
            progress(SamplingProgress(stage="sampling", completed=start + count, total=n_sequences))
    entropy = None
    entropy_history = []
    if steering.entropy:
        bridge_work = [(np.stack(lower), np.stack(upper)) for lower, upper in (zip(*edge) for edge in work)]
        reduced_energy = np.stack(endpoints)
        entropy = partition(log_z0, bridge_work, reduced_energy, groups)
        entropy["provisional"] = not (converged and entropy["overlap_ok"])
        entropy["anchor_log_z"] = log_z0
        # Doubling the number of batches keeps the total work for history
        # comparable to one final partition estimate.
        count = 1
        while count < len(endpoints):
            estimate = partition_point(log_z0, [(a[:count], b[:count]) for a, b in bridge_work],
                                       reduced_energy[:count])
            estimate["provisional"] = not (converged and estimate["overlap_ok"])
            entropy_history.append({"round": production_batches[count - 1]["round"], "batches": count,
                                    **estimate})
            count *= 2
        entropy_history.append({"round": rounds, "batches": len(endpoints), **entropy})
    states = np.concatenate(batches)
    tokens = np.array(list(loaded.tokens))
    status = "converged" if converged else ("warmup_time_exhausted" if deadline_reached else "warmup_budget_exhausted")
    summary = {
        "status": status, "warmup_rounds": warmup, "production_rounds": rounds - before_production,
        "pilot_rounds": pilot, "added_rungs": additions,
        "initialization": "initial_sequences" if initial_states is not None else "default",
        "kernel": "triton" if top.kernel.cuda else "numba" if top.kernel.cpu else "torch",
        "production_observables": summarize_traces(production_history, names) if len(production_history) >= 4 else {},
        "old_configurations_per_chain": old_fraction, "observables": diagnostics,
        "trace": {name: [record["observables"][name] for record in warmup_history] for name in names},
        "history": {
            "pilot": pilot_history, "warmup": warmup_history, "production": production_batches,
            "checks": checks, "entropy": entropy_history,
            "group_sizes": [len(group) for group in groups],
            "potential_indices": potential_indices,
            "windows": [{"minimum": int(window.minimum), "maximum": int(window.maximum)}
                        for window in steering.windows],
            "thresholds": {"rhat": steering.rhat, "ess": steering.min_ess,
                           "renewal": renewal_tolerance if ptt else None},
            "rungs": [{"label": rung.label, "beta": rung.beta, "potential_weights": rung.weights.tolist()}
                      for rung in ladder],
        },
        "ladder": [rung.label for rung in ladder], "exact_archive_anchor": exact_anchor is not None,
        "swap_acceptance": (swap_total / np.maximum(swap_count, 1)).tolist(),
        "block_acceptance": local_acceptance,
        "potential_evaluations": cache.calls, "potential_seconds": cache.seconds,
        "potential_values": np.concatenate(raw_values).tolist(),
        "constraint_distances": support.distances(states).tolist(),
        "entropy": entropy,
    }
    return SamplingResult(
        sequences=tuple("".join(row) for row in tokens[states]), energies=np.concatenate(energies),
        num_sweeps=rounds * moves * (len(ladder) - int(ptt)) // length,
        sampler="ptt" if ptt else "gibbs", beta=beta, seed=seed, model=loaded.metadata,
        steering=summary, steering_potentials=np.concatenate(values),
        warnings=() if converged else ("Steered warmup reached its round budget. Draws and entropy are provisional.",),
        ptt_diagnostics={"mixing": {"status": status}, "converged": converged} if ptt else {},
    )
