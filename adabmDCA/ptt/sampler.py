"""Parallel Trajectory Tempering with snapshot retention and mixing recovery.

All replicas target beta=1. The active ladder can be shortened using a
reservoir; optional full sampler mode retains the historical models.
The initial profile normalizer is analytic and subsequent ones use bridges.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from adabmDCA._validation import validate_integer, validate_seed
from adabmDCA.exceptions import InputValidationError, OperationCancelledError
from adabmDCA.ptt.archive import load_archive, model_id, save_archive
from adabmDCA.ptt.config import DEFAULT_PTT_N_CHAINS, DEFAULT_PTT_SAMPLER, PTTConfig
from adabmDCA.ptt.health import IMMOBILE_THRESHOLD, bar, pair_health
from adabmDCA.ptt.kernels import (
    _TIMING_KEYS,
    LOCAL_KERNELS,
    _one_hot,
    _PhaseTimer,
    _prepare_categorical_sampler,
    _prepare_exchange_kernel,
    _prepare_replica_sampler,
)
from adabmDCA.ptt.mixing import MixingEstimate, RenewalEstimate, process_replica_experiment, renewal_forecast
from adabmDCA.ptt.precision import accumulation_dtype, device_accumulation, diagnostic_double
from adabmDCA.sampling import sampling_profile_categorical
from adabmDCA.statmech import compute_energy
from adabmDCA.steering import SteeredKernel, Steering


def _profile_states(params, n_chains):
    """Exact integer draws of the uncoupled profile model of ``params``."""
    if params["bias"].device.type == "cpu":
        from adabmDCA.numba_kernels import is_numba_available, sample_profile_states

        if is_numba_available():
            return sample_profile_states(params["bias"], n_chains)
    return sampling_profile_categorical(params, n_chains, 1.0).to(torch.int32)


def _replica_positions_by_lineage(lineage, active_start=0):
    """Return each active lineage's replica position, independent of slot permutations."""
    active = lineage[active_start:]
    n_replicas, n_chains = active.shape
    active_ids = active.long().reshape(-1) - active_start * n_chains
    slot_replicas = torch.arange(n_replicas, device=lineage.device).view(-1, 1).expand(-1, n_chains)
    positions = torch.empty(n_replicas * n_chains, device=lineage.device, dtype=torch.long)
    positions.scatter_(0, active_ids, slot_replicas.reshape(-1))
    return positions.reshape(n_replicas, n_chains)


def _decay_rounds(ladder_old, threshold):
    """Decay time of a renewal phase's old fraction, fitted over its recent half, or ``None``."""
    forecast = renewal_forecast(ladder_old, threshold)
    return None if forecast is None else forecast.decay_rounds



@dataclass(frozen=True)
class PartitionEstimate:
    """Estimate of the endpoint's log partition function, with its provenance.

    Attributes:
        log_z: Natural log of the partition function.
        model_version: Training update of the endpoint.
        ladder_version: Version of the ladder the bridges ran along.
        model_id: Hash of the endpoint parameters.
        sample_round: Exchange round of the chains used.
        method: ``"ptt_bridge"`` (forward bridges) or ``"ptt_bar"`` (BAR).
        status: ``"estimated"``.
    """

    log_z: float
    model_version: int
    ladder_version: int
    model_id: str
    sample_round: int
    method: str = "ptt_bridge"
    status: str = "estimated"



def _clone(params):
    return {key: value.detach().clone() for key, value in params.items()}



def bridge_increment(lower, upper, samples):
    """Return log(Z_upper/Z_lower) from samples of the lower Hamiltonian."""
    samples = _one_hot(samples, lower)
    if samples.device.type == "mps":
        delta = {key: lower[key] - upper[key] for key in lower}
        weights = diagnostic_double(compute_energy(samples, delta))
    else:
        weights = compute_energy(samples, lower).double() - compute_energy(samples, upper).double()
    if not torch.isfinite(weights).all():
        raise ValueError("Non-finite PTT bridge energies.")
    offset = weights.max()
    centered = weights - offset
    norm = torch.logsumexp(centered, 0)
    return float(offset + norm - math.log(len(weights)))


class PTTSampler:
    """A Parallel Trajectory Tempering ladder: models, their chains and its own RNG.

    During training (see :func:`train_model` with ``ptt=``) the ladder runs
    from an exactly normalized profile model to the model being trained, and
    snapshots of the training trajectory are kept as intermediate replicas.
    Each exchange round swaps configurations between adjacent replicas and
    then runs local Monte Carlo sweeps. The same ladder is saved in the PTT
    archive and reused to sample the trained model and to estimate its log Z.

    Most users only load an archive and sample or measure it:

    - :meth:`from_archive`, :meth:`save_archive`, :meth:`fork`;
    - :meth:`draw`, :meth:`endpoint_samples`, :meth:`endpoint_params`;
    - :meth:`partition_estimate`, :meth:`entropy`, :meth:`ladder_statistics`,
      :meth:`ladder_health`, :meth:`measure_renewal`.

    The remaining public methods (``transition_target``,
    ``update_replica_chain``, ``refresh_reservoir``, ...) are the training
    machinery used by the PTT trainer. Mutating the ladder directly is
    unsupported. Archives contain arrays and JSON only, never pickled objects.

    Example:
        >>> sampler = PTTSampler.from_archive("model/ptt.h5", device="cuda")
        >>> sampler.prepare_sampling_ladder(2000)
        >>> sampler.measure_renewal(local_sweeps=10).status
        'converged'
        >>> sequences = sampler.draw(1000, local_sweeps=10)   # one-hot, (1000, L, q)
        >>> sampler.entropy()["entropy"]
    """

    def __init__(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        tokens: str,
        n_chains: int = DEFAULT_PTT_N_CHAINS,
        sampler: str = DEFAULT_PTT_SAMPLER,
        config: PTTConfig | None = None,
        seed: int = 0,
    ) -> None:
        """Start a two-replica ladder at an uncoupled (profile) model.

        Both replicas start at ``params``; training then moves the endpoint.
        To continue from a saved state use :meth:`from_archive`.

        Args:
            params: ``bias`` of shape ``(L, q)`` and an all-zero ``coupling_matrix``
                of shape ``(L, q, L, q)``, float32 or float64, on CPU, CUDA or MPS (float32).
            tokens: Ordered alphabet of length ``q``.
            n_chains: Chains per replica.
            sampler: Local kernel: ``"metropolized_gibbs"``, ``"gibbs"`` or ``"metropolis"``.
            config: PTT settings; defaults to ``PTTConfig()``.
            seed: Random seed of the sampler's own generator.

        Raises:
            InputValidationError: If the parameters are coupled, invalid or
                underflow, or a setting is invalid.
        """
        self.config = config or PTTConfig()
        if not isinstance(self.config, PTTConfig):
            raise InputValidationError("config must be a PTTConfig instance.")
        validate_integer("n_chains", n_chains)
        validate_seed(seed)
        if sampler not in LOCAL_KERNELS:
            raise InputValidationError(f"PTT local sampler must be one of {', '.join(LOCAL_KERNELS)}.")
        self.tokens = tokens
        self.local_sampler = sampler
        self._validate_params(params)
        if torch.count_nonzero(params["coupling_matrix"]):
            raise InputValidationError("PTT must start at an exact uncoupled profile; use an archive to resume.")
        if not (params["bias"].softmax(-1) > 0).all():
            raise InputValidationError("PTT anchor probabilities underflow at the selected precision.")
        self.models = [_clone(params), _clone(params)]
        self.profile_params = _clone(params)
        self.ptt_checkpoints = [{"step": 0, "params": _clone(params)}]
        self.ptt_checkpoints_complete = True
        self.optimizer_state = None
        self.total_models = 2
        self.device = params["bias"].device
        self._local_kernel = _prepare_categorical_sampler(self.local_sampler, self.device)
        self._replica_kernel = _prepare_replica_sampler(self.local_sampler, self.device)
        self._replica_params_cache = None
        self._exchange_kernel = _prepare_exchange_kernel(self.device)
        self.seed = seed
        self._rng = torch.Generator(device=self.device).manual_seed(seed).get_state()
        self.model_version = 0
        self.ladder_version = 0
        self.rounds = 0
        self.local_sweeps = 0
        self.acceptance = [1.0]
        self.training_state = {}
        self.active_start = 0
        self.anchor_log_z = float(torch.logsumexp(diagnostic_double(params["bias"]), -1).sum())
        self.reservoir = None
        self.temporary = None
        self.held = None
        self.last_mixing = None
        self.mixing_correlation = []
        self.last_failure = None
        self.events = []
        self.recovery_points = []
        self.timings = {key: 0.0 for key in _TIMING_KEYS}
        self.last_advance_timing = dict(self.timings)
        self.mode = "train"
        with self._random_context():
            self.chains = [_profile_states(params, n_chains) for _ in self.models]
        self.lineage = torch.arange(2 * n_chains, device=self.device).reshape(2, n_chains)
        # Generation-only state: birth rounds of configurations (see
        # ``measure_renewal``), and whether each configuration has reached the
        # top replica since its birth (the flow profile of ``ladder_health``).
        self.birth = None
        self.reached_top = None
        # Training state of the 'adaptive' optimizer: per configuration, the
        # energy changes it received from endpoint updates while at the
        # endpoint, decayed over ``config.lag_horizon`` updates. Carried with
        # configurations like lineages; see ``_record_endpoint_update``.
        self.lag_memory = None
        self.generation_kernel = None
        # Generation-only steering (see ``prepare_steering``): rungs from
        # ``steering_base`` up carry the endpoint parameters plus the potential
        # at ``steering_strengths[k]``; ``steering_values[k]`` caches its value
        # on the chains of rung k.
        self.steering = None
        self.steering_base = None
        self.steering_strengths = None
        self.steering_values = None
        self._steering_kernels = None
        if self.config.reservoir_size is not None and self.config.reservoir_size < n_chains:
            raise InputValidationError("PTT reservoir_size must be at least n_chains.")

    def _validate_params(self, params):
        if set(params) != {"bias", "coupling_matrix"} or params["bias"].ndim != 2:
            raise InputValidationError("PTT requires bias (L,q) and coupling_matrix (L,q,L,q).")
        h, j = params["bias"], params["coupling_matrix"]
        L, q = h.shape
        if L < 1 or q < 2 or len(self.tokens) != q or len(set(self.tokens)) != q:
            raise InputValidationError("Invalid PTT dimensions or token order.")
        if h.dtype not in (torch.float32, torch.float64) or j.dtype != h.dtype or h.device != j.device:
            raise InputValidationError("PTT requires consistent float32/float64 parameters.")
        if h.device.type not in ("cpu", "cuda", "mps"):
            raise InputValidationError("PTT supports CPU, CUDA and MPS.")
        if h.device.type == "mps" and h.dtype != torch.float32:
            raise InputValidationError("PTT on MPS requires float32 parameters.")
        if j.shape != (L, q, L, q) or not torch.isfinite(h).all() or not torch.isfinite(j).all():
            raise InputValidationError(
                "PTT parameters must have compatible shapes and finite values; use positive pseudocounts."
            )
        if not torch.allclose(j, j.permute(2, 3, 0, 1)) or torch.count_nonzero(
            j[torch.arange(L), :, torch.arange(L), :]
        ):
            raise InputValidationError("PTT couplings must be symmetric with zero within-site blocks.")
        if hasattr(self, "models") and (
            h.shape != self.models[0]["bias"].shape
            or h.dtype != self.models[0]["bias"].dtype
            or h.device != self.device
        ):
            raise InputValidationError("A PTT transition cannot change dimensions, precision or device.")

    @contextmanager
    def _random_context(self):
        if self.device.type == "mps":
            global_state = torch.mps.get_rng_state()
            torch.mps.set_rng_state(self._rng)
            try:
                yield
            finally:
                self._rng = torch.mps.get_rng_state()
                torch.mps.set_rng_state(global_state)
            return
        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            if devices:
                torch.cuda.set_rng_state(self._rng, self.device)
            else:
                torch.set_rng_state(self._rng)
            try:
                yield
            finally:
                self._rng = torch.cuda.get_rng_state(self.device) if devices else torch.get_rng_state()

    def endpoint_params(self) -> dict[str, torch.Tensor]:
        """Return a copy of the endpoint (last replica) parameters: the trained model."""
        return _clone(self.models[-1])

    def endpoint_samples(self, *, copy: bool = True) -> torch.Tensor:
        """Return the endpoint chains, one-hot, shape ``(n_chains, L, q)``.

        Args:
            copy: Return a copy; ``False`` may return internal storage, which must
                not be modified.
        """
        chains, params = self.chains[-1], self.models[-1]
        signature = (params["bias"].shape[1], params["bias"].dtype)
        cache = getattr(self, "_endpoint_samples_cache", None)
        if not torch.is_inference(chains) and cache is not None and (
            cache[0] is chains and cache[1] == chains._version and cache[2] == signature
            and cache[4] == cache[3]._version
        ):
            samples = cache[3]
        else:
            samples = _one_hot(chains, params)
            # Inference tensors have no version counters, so their mutations
            # cannot be tracked safely. Rebuild those encodings every time.
            self._endpoint_samples_cache = (
                (chains, chains._version, signature, samples, samples._version)
                if not torch.is_inference(chains) and not torch.is_inference(samples) else None
            )
        return samples.clone() if copy else samples

    def prepare_sampling_ladder(self, n_chains: int) -> None:
        """Rebuild the full ladder for generation, with fresh chains drawn from the profile.

        The generation ladder is the training ladder: the exact profile, every
        update flagged ``ptt`` during training, and the final endpoint. This
        switches the sampler to generation mode and starts birth tracking, as
        :meth:`measure_renewal` requires.

        Args:
            n_chains: Chains per replica.

        Raises:
            InputValidationError: If the archive lacks the flagged models.
        """
        validate_integer("n_chains", n_chains)
        if not self.ptt_checkpoints_complete:
            raise InputValidationError(
                "This archive lacks the saved models for earlier 'ptt' checkpoints. "
                "Retrain with checkpoint-preserving training to sample the complete flagged ladder."
            )
        points = list(self.ptt_checkpoints)
        if not points or points[-1]["step"] != self.model_version:
            points.append({"step": self.model_version, "params": self.endpoint_params()})
        # The initial profile alone is exact; duplicate it internally so the
        # existing two-replica exchange machinery also handles zero updates.
        if len(points) == 1:
            points.append(points[0])
        self.sampling_steps = [point["step"] for point in points]
        self.models = [_clone(point["params"]) for point in points]
        self.steering = self.steering_base = self.steering_strengths = None
        self.steering_values = self._steering_kernels = None
        self._replica_params_cache = None
        self.active_start = 0
        self.reservoir = self.temporary = self.held = None
        self.anchor_log_z = float(torch.logsumexp(diagnostic_double(self.profile_params["bias"]), -1).sum())
        # Start every replica from exact draws of the bottom profile, i.e. the
        # distribution that enters the ladder at rung 0. Uniform random
        # sequences can relax into states that every model disfavors
        # relative to its neighbors, where exchanges cannot remove them.
        with self._random_context():
            self.chains = [_profile_states(self.models[0], n_chains) for _ in self.models]
        self._reset_lineage()
        # Initial configurations predate every round of this generation run; stamping them with the
        # round before it makes ages count from the start of generation, not of training.
        self.birth = torch.full((len(self.models), n_chains), self.rounds - 1, device=self.device, dtype=torch.int64)
        self.reached_top = torch.zeros_like(self.birth, dtype=torch.bool)
        self.lag_memory = None
        self.generation_start_round = self.rounds
        self.acceptance = [1.0] * (len(self.models) - 1)
        self.training_state = {}
        self.mode = "generate"

    def fork(self, *, seed: int | None = None) -> PTTSampler:
        """Return an independent copy of the sampler, without its recovery points.

        Args:
            seed: New random seed, or ``None`` to continue the same random stream.

        Returns:
            A new :class:`PTTSampler`; changes to it do not affect this one.
        """
        # Recovery checkpoints must not recursively contain earlier checkpoints.
        result = object.__new__(type(self))
        result.__dict__ = copy.deepcopy({
            k: v for k, v in self.__dict__.items()
            if k not in (
                "recovery_points", "ptt_checkpoints", "_local_kernel", "_replica_kernel",
                "_replica_params_cache", "_endpoint_samples_cache", "_exchange_kernel", "steering", "_steering_kernels",
            )
        })
        # The steering potential is user code (possibly a large model): share it.
        result.steering = getattr(self, "steering", None)
        kernels = getattr(self, "_steering_kernels", None)
        result._steering_kernels = None if kernels is None else [copy.copy(kernel) for kernel in kernels]
        kernel = getattr(result, "generation_kernel", None) or result.local_sampler
        result._local_kernel = _prepare_categorical_sampler(kernel, result.device)
        result._replica_kernel = _prepare_replica_sampler(kernel, result.device)
        result._replica_params_cache = None
        result._endpoint_samples_cache = None
        result._exchange_kernel = _prepare_exchange_kernel(result.device)
        result.ptt_checkpoints = list(self.ptt_checkpoints)
        result.recovery_points = []
        if seed is not None:
            validate_seed(seed)
            result.seed = seed
            result._rng = torch.Generator(device=self.device).manual_seed(seed).get_state()
        return result

    def set_generation_kernel(self, name: str | None = None) -> None:
        """Select the local kernel used during generation; ``None`` restores the archived one.

        Every kernel leaves each replica's distribution invariant, so a model
        trained with one kernel can be sampled with another. Archives keep the
        training kernel as ``local_sampler``.

        Args:
            name: ``"metropolized_gibbs"``, ``"gibbs"``, ``"metropolis"`` or ``None``.
        """
        if name is not None and name not in LOCAL_KERNELS:
            raise InputValidationError(f"PTT local kernel must be one of {', '.join(LOCAL_KERNELS)}.")
        self.generation_kernel = name
        kernel = name or self.local_sampler
        self._local_kernel = _prepare_categorical_sampler(kernel, self.device)
        self._replica_kernel = _prepare_replica_sampler(kernel, self.device)
        self._replica_params_cache = None
        self._endpoint_samples_cache = None

    @property
    def n_active(self) -> int:
        """Number of replicas in the active ladder (above the reservoir, if any)."""
        return len(self.models) - self.active_start

    def _reset_lineage(self):
        self.lineage = torch.arange(len(self.models) * len(self.chains[0]), device=self.device).reshape(len(self.models), -1)

    def capture_state(self) -> dict[str, Any]:
        """Return a deep copy of the numerical state, for a recovery checkpoint.

        Training history and recovery points are excluded; see :meth:`restore_state`.
        """
        keys = (
            "models", "chains", "lineage", "_rng", "model_version", "ladder_version", "rounds", "local_sweeps",
            "acceptance", "active_start", "anchor_log_z", "reservoir", "temporary", "held", "last_mixing",
            "mixing_correlation", "events", "profile_params", "total_models",
            "optimizer_state", "timings", "last_advance_timing", "lag_memory",
        )
        return copy.deepcopy({key: getattr(self, key) for key in keys})

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Restore a state returned by :meth:`capture_state`, clearing the training state."""
        self.__dict__.update(copy.deepcopy(state))
        self._replica_params_cache = None
        self._endpoint_samples_cache = None
        self.ptt_checkpoints = [p for p in self.ptt_checkpoints if p["step"] <= self.model_version]
        self.training_state = {}
        self.last_failure = None

    @torch.no_grad()
    def advance(
        self,
        *,
        rounds: int = 1,
        local_sweeps: int = 1,
        is_cancelled: Callable[[], bool] | None = None,
        on_round: Callable[[int, int], None] | None = None,
        until: Callable[[], bool] | None = None,
        return_samples: bool = True,
    ) -> torch.Tensor | None:
        """Run exchange rounds at fixed parameters.

        Each round swaps configurations between adjacent replicas, permutes
        chains within replicas, redraws the bottom replica, and runs local sweeps.

        Args:
            rounds: Largest number of rounds.
            local_sweeps: Local sweeps per round.
            is_cancelled: Optional callable; returning ``True`` stops the run.
            on_round: Optional callback ``on_round(completed, rounds)`` after each round.
            until: Optional callable; returning ``True`` stops after the current round.
            return_samples: Return one-hot endpoint chains; ``False`` skips
                their construction when only advancing the sampler state.

        Returns:
            A copy of the endpoint chains, one-hot, shape ``(n_chains, L, q)``,
            or ``None`` when ``return_samples=False``.

        Raises:
            OperationCancelledError: If ``is_cancelled`` returns ``True``.
        """
        validate_integer("rounds", rounds)
        validate_integer("local_sweeps", local_sweeps)
        if rounds:
            self._endpoint_samples_cache = None
        completed = rounds
        kernel = self._local_kernel
        exchange_kernel = self._exchange_kernel
        timer = _PhaseTimer(self.device)
        acceptance = torch.zeros(self.n_active - 1, dtype=torch.float64)
        birth = getattr(self, "birth", None)
        reached = getattr(self, "reached_top", None)
        if birth is None or reached is None or reached.shape != birth.shape:
            reached = None
        memory = getattr(self, "lag_memory", None)
        if memory is not None and memory.shape != (len(self.chains), len(self.chains[0])):
            # Experiments on forks replace populations; their memory is meaningless.
            memory = self.lag_memory = None
        fused_counts = None
        fused_move = None
        if self.device.type == "mps":
            from adabmDCA.mps_kernels.swaps import acceptance_rates, swap_and_permute, swaps_available

            if swaps_available(self, birth, reached, memory):
                fused_counts = torch.zeros((rounds, self.n_active - 1), dtype=torch.int32, device=self.device)
                fused_move = swap_and_permute
        elif self.device.type == "cpu" and self.mode == "generate" and len(self.models) == 7:
            from adabmDCA.numba_kernels import is_numba_available

            if is_numba_available():
                from adabmDCA.numba_kernels.swaps import swap_and_permute, swaps_available

                if swaps_available(self, birth, reached, memory):
                    fused_move = swap_and_permute
        # A training checkpoint becomes stale when populations evolve outside
        # its commit. The trainer installs fresh metadata after each commit.
        self.training_state = {}
        with self._random_context():
            for step in range(rounds):
                if is_cancelled is not None and is_cancelled():
                    raise OperationCancelledError("PTT sampling was cancelled.")
                # Reference order: one pass of adjacent exchanges and population
                # permutations, then local moves and reservoir refresh.
                for k in range(self.active_start, len(self.models) - 1):
                    if fused_move is not None:
                        with timer.measure("exchange_seconds"):
                            log_a = exchange_kernel(self.models[k], self.models[k + 1], self.chains[k], self.chains[k + 1])
                            log_u = torch.rand(len(log_a), device=self.device).log()
                            orders = tuple(torch.randperm(len(self.chains[j]), device=self.device) for j in (k, k + 1))
                            moved = fused_move(
                                (self.chains[k], self.chains[k + 1]), (self.lineage[k], self.lineage[k + 1]),
                                log_a, log_u, orders, fused_counts, step * (self.n_active - 1) + k - self.active_start,
                                birth=None if birth is None else (birth[k], birth[k + 1]),
                                reached=None if reached is None else (reached[k], reached[k + 1]),
                                memory=None if memory is None else (memory[k], memory[k + 1]),
                            )
                            if fused_counts is None:
                                acceptance[k - self.active_start] += moved["accepted"] / len(log_a)
                            self.chains[k], self.chains[k + 1] = moved["chains"].unbind(0)
                            self.lineage[k:k + 2] = moved["lineage"]
                            for key, values in (("birth", birth), ("reached", reached), ("memory", memory)):
                                if values is not None:
                                    values[k:k + 2] = moved[key]
                        continue
                    with timer.measure("exchange_seconds"):
                        steered_pair = self._steered_pair(k)
                        if steered_pair:
                            log_a, cross = self._steered_exchange(k)
                        else:
                            log_a = exchange_kernel(
                                self.models[k], self.models[k + 1], self.chains[k], self.chains[k + 1]
                            )
                        swap = torch.rand(len(log_a), device=self.device).log() < log_a
                        if steered_pair:
                            self._swap_steering_values(k, swap, cross)
                        acceptance[k - self.active_start] += device_accumulation(swap).mean().cpu()
                        saved = self.chains[k][swap].clone()
                        self.chains[k][swap] = self.chains[k + 1][swap]
                        self.chains[k + 1][swap] = saved
                        ids = self.lineage[k, swap].clone()
                        self.lineage[k, swap] = self.lineage[k + 1, swap]
                        self.lineage[k + 1, swap] = ids
                        if birth is not None:
                            born = birth[k, swap].clone()
                            birth[k, swap] = birth[k + 1, swap]
                            birth[k + 1, swap] = born
                        if reached is not None:
                            kept = reached[k, swap].clone()
                            reached[k, swap] = reached[k + 1, swap]
                            reached[k + 1, swap] = kept
                        if memory is not None:
                            kept = memory[k, swap].clone()
                            memory[k, swap] = memory[k + 1, swap]
                            memory[k + 1, swap] = kept
                    with timer.measure("permutation_seconds"):
                        for j in (k, k + 1):
                            order = torch.randperm(len(self.chains[j]), device=self.device)
                            self.chains[j] = self.chains[j][order]
                            self.lineage[j] = self.lineage[j][order]
                            if birth is not None:
                                birth[j] = birth[j][order]
                            if reached is not None:
                                reached[j] = reached[j][order]
                            if memory is not None:
                                memory[j] = memory[j][order]
                            if self.steering_values is not None and self.steering_values[j] is not None:
                                self.steering_values[j] = self.steering_values[j][order]
                if reached is not None:
                    # After the exchanges: whatever sits at the top has reached it.
                    reached[-1] = True
                with timer.measure("local_sampling_seconds"):
                    first_local = self.active_start
                    if first_local == 0 and self.reservoir is None:
                        self.chains[0] = _profile_states(self.models[0], len(self.chains[0]))
                        if birth is not None:
                            # Exact redraws are new configurations.
                            birth[0] = self.rounds
                        if reached is not None:
                            reached[0] = False
                        if memory is not None:
                            memory[0] = 0.0
                        first_local = 1
                    local_indices = [k for k in range(first_local, len(self.models)) if not self._is_steered(k)]
                    for k in range(first_local, len(self.models)):
                        if self._is_steered(k):
                            self._steered_local_update(k, local_sweeps)
                    use_replica_kernel = (
                        self._replica_kernel is not None
                        and len(local_indices) >= 2
                        and self.models[-1]["bias"].shape[0] >= self._replica_kernel.min_length
                        and (self._replica_kernel.always or (
                            self.mode == "generate" and (self._replica_params_cache is not None or rounds >= 32)))
                    )
                    # Metal amortizes launch overhead across sweeps. Keep one-sweep
                    # calls when cancellation must be checked between sweeps.
                    batch_sweeps = local_sweeps if self.device.type == "mps" and is_cancelled is None else 1
                    if use_replica_kernel:
                        for _ in range(local_sweeps // batch_sweeps):
                            if is_cancelled is not None and is_cancelled():
                                raise OperationCancelledError("PTT sampling was cancelled.")
                            biases, couplings = self._stacked_replica_params(local_indices)
                            stacked = self._replica_kernel(
                                torch.stack([self.chains[k] for k in local_indices]),
                                biases,
                                couplings,
                                nsweeps=batch_sweeps,
                            )
                            for k, chains in zip(local_indices, stacked.unbind(0)):
                                self.chains[k] = chains
                            self.local_sweeps += len(local_indices) * batch_sweeps
                    else:
                        for k in local_indices:
                            for _ in range(local_sweeps // batch_sweeps):
                                if is_cancelled is not None and is_cancelled():
                                    raise OperationCancelledError("PTT sampling was cancelled.")
                                self.chains[k] = kernel(self.chains[k], self.models[k], nsweeps=batch_sweeps)
                                self.local_sweeps += batch_sweeps
                if self.reservoir is not None:
                    # Exchange with randomly selected reservoir entries, as
                    # in ptt_paper.pre_sampler.reservoir.Reservoir.
                    idx = torch.randperm(len(self.reservoir), device=self.device)[:len(self.chains[0])]
                    fresh = self.reservoir[idx].clone()
                    self.reservoir[idx] = self.chains[self.active_start]
                    self.chains[self.active_start] = fresh
                    if birth is not None:
                        # The reservoir is the source of this ladder: a
                        # configuration enters the active ladder when emitted.
                        birth[self.active_start] = self.rounds
                    if reached is not None:
                        reached[self.active_start] = False
                    if memory is not None:
                        memory[self.active_start] = 0.0
                self.rounds += 1
                completed = step + 1
                if on_round is not None:
                    on_round(completed, rounds)
                if until is not None and until():
                    break
        self.last_advance_timing = timer.finish()
        for key, value in self.last_advance_timing.items():
            self.timings[key] += value
        self.acceptance = ((acceptance / completed).tolist() if fused_counts is None
                           else acceptance_rates(fused_counts, completed, len(self.chains[-1])))
        return self.endpoint_samples() if return_samples else None

    def _stacked_replica_params(self, indices):
        """Stacked biases and kernel-layout couplings for ``indices``, cached per replica set.

        At most two sets are kept (all local replicas and one boosted subset);
        keys include tensor addresses and versions, so replaced or modified
        models are restacked. Inference tensors are restacked on every call. Each
        entry keeps its source tensors alive, so no other tensor can reuse
        their addresses while it is cached.
        """
        sources = [(self.models[k]["bias"], self.models[k]["coupling_matrix"]) for k in indices]
        transform = self._replica_kernel.transform_couplings
        if any(torch.is_inference(value) for pair in sources for value in pair):
            return (torch.stack([bias for bias, _ in sources]),
                    torch.stack([transform(couplings) for _, couplings in sources]))
        key = tuple((k, bias.data_ptr(), bias._version, couplings.data_ptr(), couplings._version)
                    for k, (bias, couplings) in zip(indices, sources))
        cache = self._replica_params_cache or {}
        if key not in cache:
            if len(cache) >= 2:
                cache = {}  # release old stacks before materializing a new one
            cache[key] = (
                torch.stack([bias for bias, _ in sources]),
                torch.stack([transform(couplings) for _, couplings in sources]),
                sources,
            )
        self._replica_params_cache = cache
        return cache[key][:2]

    def equilibrate(
        self,
        *,
        rounds: int | None = None,
        local_sweeps: int = 1,
        is_cancelled: Callable[[], bool] | None = None,
        on_round: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        """Run a fixed warmup of :meth:`advance`; this does not certify equilibrium.

        Args:
            rounds: Rounds to run; defaults to ``config.equilibration_rounds``.
            local_sweeps: Local sweeps per round.
            is_cancelled: Optional callable; returning ``True`` stops the run.
            on_round: Optional callback ``on_round(completed, rounds)``.

        Returns:
            A copy of the endpoint chains, one-hot.
        """
        return self.advance(
            rounds=rounds or self.config.equilibration_rounds, local_sweeps=local_sweeps,
            is_cancelled=is_cancelled, on_round=on_round,
        )

    def partition_estimate(self, estimator: str = "forward") -> PartitionEstimate:
        """Estimate log Z of the endpoint from the exact anchor and one bridge per replica pair.

        ``estimator='forward'`` reweights each lower population to the model
        above (the historical bridge used during training and stored in
        archives). ``'bar'`` combines both populations of every pair with
        Bennett's acceptance ratio, which stays reliable when the forward
        weights are heavy-tailed.

        Args:
            estimator: ``"forward"`` or ``"bar"``.

        Returns:
            A :class:`PartitionEstimate` with ``log_z`` and its provenance.
        """
        if estimator not in ("forward", "bar"):
            raise InputValidationError("PTT partition estimator must be 'forward' or 'bar'.")
        log_z = self.anchor_log_z
        for k in range(self.active_start, len(self.models) - 1):
            if estimator == "forward":
                log_z += self._forward_increment(k)
            else:
                log_z -= float(bar(*self._pair_works(k)))
        return PartitionEstimate(
            log_z,
            self.model_version,
            self.ladder_version,
            model_id(self.models[-1]),
            self.rounds,
            method="ptt_bridge" if estimator == "forward" else "ptt_bar",
        )

    def _energies(self, chains, params_list, chunk=4096):
        """Energies of categorical ``chains`` under each model in ``params_list``, in chunks."""
        results = [[] for _ in params_list]
        for start in range(0, len(chains), chunk):
            x = _one_hot(chains[start:start + chunk], params_list[0])
            for out, params in zip(results, params_list):
                out.append(device_accumulation(compute_energy(x, params)))
        return [torch.cat(out) for out in results]

    @torch.no_grad()
    def _pair_works(self, k):
        """``W = E_{k+1} - E_k`` for the populations of replicas ``k`` and ``k + 1``."""
        if self._steered_pair(k):
            # Same DCA parameters on both rungs: only the potential differs.
            lower_under_upper, upper_under_lower = self._steering_cross(k)
            return (lower_under_upper - self._rung_values(k),
                    self._rung_values(k + 1) - upper_under_lower)
        lower, upper = self.models[k], self.models[k + 1]
        if self.device.type == "mps":
            delta = {key: upper[key] - lower[key] for key in lower}
            return (self._energies(self.chains[k], [delta])[0],
                    self._energies(self.chains[k + 1], [delta])[0])
        lower_under_lower, lower_under_upper = self._energies(self.chains[k], [lower, upper])
        upper_under_lower, upper_under_upper = self._energies(self.chains[k + 1], [lower, upper])
        return lower_under_upper - lower_under_lower, upper_under_upper - upper_under_lower

    @torch.no_grad()
    def ladder_health(self, *, bootstrap: int = 200, seed: int = 0) -> dict[str, Any]:
        """Bidirectional reweighting diagnostics for every adjacent active pair.

        See ``adabmDCA.ptt.health``. Per pair: forward, reverse and BAR free
        energies with bootstrap errors, hysteresis, effective sample sizes and
        top-1% weight shares in both directions, the Crooks slope (-1 at
        equilibrium), mean swap acceptance and the fraction of configurations
        whose average swap probability is below ``IMMOBILE_THRESHOLD``, and
        quantiles of the per-configuration acceptance in both directions. With
        birth tracking, the mean age of immobile and mobile upper configurations
        is included, and ``replicas`` gives, per replica, the median and 99th
        percentile of the configuration ages (rounds since birth) and the flow:
        the fraction of configurations that reached the top replica since birth.
        Also returns forward and BAR endpoint log Z.

        Args:
            bootstrap: Bootstrap resamples for the error bars.
            seed: Seed of the bootstrap.

        Returns:
            ``pairs`` (one dictionary per adjacent pair), ``replicas`` (one per
            replica, empty without birth tracking), ``log_z_forward``,
            ``log_z_bar``, ``log_z_bar_error`` and ``immobile_threshold``.
        """
        generator = torch.Generator().manual_seed(seed)
        pairs = []
        log_z_forward = log_z_bar = self.anchor_log_z
        bar_variance = 0.0
        birth = getattr(self, "birth", None)
        for k in range(self.active_start, len(self.models) - 1):
            w_lower, w_upper = self._pair_works(k)
            ages = None if birth is None else self.rounds - birth[k + 1]
            row = pair_health(w_lower, w_upper, bootstrap=bootstrap, generator=generator, ages=ages)
            row.update({"lower": k, "upper": k + 1})
            pairs.append(row)
            log_z_forward -= row["dF_forward"]
            log_z_bar -= row["dF_bar"]
            bar_variance += row["dF_bar_error"] ** 2
        replicas = []
        reached = getattr(self, "reached_top", None)
        if birth is not None:
            for k in range(self.active_start, len(self.models)):
                ages = device_accumulation(self.rounds - birth[k]).cpu()
                replicas.append({
                    "replica": k, "age_median": float(ages.median()), "age_q99": float(torch.quantile(ages, 0.99)),
                    # Fraction of the configurations that reached the top since birth (moving down).
                    "flow_down": None if reached is None else float(device_accumulation(reached[k]).mean()),
                })
        return {"pairs": pairs, "replicas": replicas, "log_z_forward": log_z_forward, "log_z_bar": log_z_bar,
                "log_z_bar_error": math.sqrt(bar_variance), "immobile_threshold": IMMOBILE_THRESHOLD}

    def estimate_mixing_time(
        self,
        *,
        local_sweeps: int = 1,
        is_cancelled: Callable[[], bool] | None = None,
        full_ladder: bool = False,
        on_progress: Callable[..., None] | None = None,
        bounded: bool = True,
        max_rounds: int | None = None,
        in_place: bool = False,
        reference: bool = False,
        method: str | None = None,
    ) -> MixingEstimate:
        """Run reference TRWA on a separate population and return its diagnostics.

        Extend the experiment until it contains at least 20 times both
        estimated correlation times (configurable), or reaches its budget.
        Training populations and RNG are preserved; diagnostic work counts.
        ``full_ladder=True`` also checks retained history in full sampler mode.
        ``bounded=False`` continues until convergence or cancellation, ignoring
        the training mixing-round budget.
        ``in_place=True`` measures and equilibrates the current populations,
        as in reference TRWA. ``reference=True`` uses its FFT and fitting rules.
        The replica index of each uniquely tagged configuration (its lineage)
        is the observable.
        ``method`` defaults to ``config.mixing_method``. ``'renewal'`` replaces
        the replica-index experiment with a population-renewal check on the
        same kind of separate population (see ``_renewal_mixing_experiment``).

        Args:
            local_sweeps: Local sweeps per exchange round.
            is_cancelled: Optional callable; returning ``True`` stops the run.
            full_ladder: Also check the retained history (full sampler mode).
            on_progress: Optional callback ``on_progress(stage, done, total, **details)``.
            bounded: Stop at ``config.mixing_max_rounds``; ``False`` runs until converged.
            max_rounds: Round budget overriding the configured one.
            in_place: Measure and equilibrate the current populations.
            reference: Use the reference TRWA fitting rules.
            method: ``"autocorrelation"`` or ``"renewal"``; defaults to
                ``config.mixing_method``.

        Returns:
            A :class:`MixingEstimate`; ``converged`` tells whether the ladder mixes.
        """
        if max_rounds is not None:
            validate_integer("max_rounds", max_rounds)
        if full_ladder and self.reservoir is not None and not self.config.full_sampler:
            raise InputValidationError("The full PTT history was not retained; use full_sampler=True when training.")
        method = self.config.mixing_method if method is None else method
        if method not in ("autocorrelation", "renewal"):
            raise InputValidationError("PTT mixing method must be 'autocorrelation' or 'renewal'.")
        self.training_state = {}
        if method == "renewal":
            if in_place:
                raise InputValidationError("In-place population renewal is measured with measure_renewal.")
            return self._renewal_mixing_experiment(
                local_sweeps=local_sweeps, is_cancelled=is_cancelled, full_ladder=full_ladder,
                on_progress=on_progress, bounded=bounded, max_rounds=max_rounds,
            )
        trial = self if in_place else self.fork()
        trial.training_state = {}
        if full_ladder:
            trial.active_start = 0
            trial.reservoir = None
            trial.anchor_log_z = float(torch.logsumexp(diagnostic_double(trial.profile_params["bias"]), -1).sum())
            trial.acceptance = [1.0] * (len(trial.models) - 1)
        n = min(self.config.mixing_chains, len(trial.reservoir)) if trial.reservoir is not None else self.config.mixing_chains
        if in_place:
            n = len(trial.chains[0])
        else:
            with trial._random_context():
                trial.chains = [_profile_states(p, n) for p in trial.models]
        trial._reset_lineage()
        before = trial.local_sweeps
        timing_before = dict(trial.timings)
        if self.config.mixing_thermalization_rounds:
            warmup_rounds = self.config.mixing_thermalization_rounds
            if on_progress is not None:
                on_progress("mixing_warmup", 0, warmup_rounds, chains=n, replicas=trial.n_active)
            trial.advance(
                return_samples=False,
                rounds=warmup_rounds, local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                on_round=(None if on_progress is None else
                          lambda done, total: on_progress("mixing_warmup", done, total, chains=n, replicas=trial.n_active)),
            )
        trial._reset_lineage()
        indices = []
        acceptance = torch.zeros(trial.n_active - 1, dtype=torch.float64)
        max_rounds = max_rounds if max_rounds is not None else (
            self.config.mixing_max_rounds if bounded else None
        )
        target = self.config.mixing_initial_rounds
        if max_rounds is not None:
            target = min(target, max_rounds)
        tau_int = tau_exp = None
        required = target
        correlation = torch.empty(0)
        if on_progress is not None:
            on_progress("mixing_measure", 0, target, chains=n, replicas=trial.n_active,
                        max_rounds=max_rounds, observable="lineage")
        while True:
            for _ in range(len(indices), target):
                trial.advance(return_samples=False, local_sweeps=local_sweeps, is_cancelled=is_cancelled)
                indices.append(_replica_positions_by_lineage(trial.lineage, trial.active_start).cpu())
                acceptance += torch.tensor(trial.acceptance, dtype=torch.float64)
                if on_progress is not None:
                    on_progress("mixing_measure", len(indices), target, chains=n, replicas=trial.n_active,
                                max_rounds=max_rounds)
            try:
                tau_int, tau_exp, correlation = process_replica_experiment(torch.stack(indices), **({"reference": True} if reference else {}))
                if tau_exp > len(indices):
                    # An exponential fit cannot resolve a time longer than the
                    # trajectory it was fitted on; extend instead of trusting it.
                    raise ValueError("Exponential correlation time exceeds the measured trajectory.")
                required = math.ceil(self.config.mixing_window_factor * max(tau_int, tau_exp))
            except ValueError:
                # An unresolved fit needs a longer trajectory, not an
                # arbitrary small/negative relaxation time.
                tau_int = tau_exp = None
                required = 2 * len(indices)
            if tau_int is not None and len(indices) >= required:
                status = "converged"
                break
            if max_rounds is not None and (len(indices) >= max_rounds or required > max_rounds):
                status = "budget_exceeded"
                break
            target = max(required, len(indices) + 1)
            if max_rounds is not None:
                target = min(max_rounds, target)
            if on_progress is not None:
                on_progress("mixing_measure", len(indices), target, chains=n, replicas=trial.n_active,
                            max_rounds=max_rounds, tau_int=tau_int, tau_exp=tau_exp,
                            required_rounds=required, observable="lineage",
                            acceptance=(acceptance / len(indices)).tolist())
        work = trial.local_sweeps - before
        result = MixingEstimate(tau_int, tau_exp, len(indices), required, status,
                                tuple((acceptance / len(indices)).tolist()), work,
                                model_version=self.model_version, ladder_version=self.ladder_version,
                                replicas=trial.n_active, full_ladder=trial.reservoir is None)
        if not in_place:
            self.local_sweeps += work
            for key in _TIMING_KEYS:
                self.timings[key] += trial.timings[key] - timing_before[key]
        self.last_mixing = asdict(result)
        self.last_mixing["acceptance"] = list(result.acceptance)
        self.last_mixing["observable"] = "lineage"
        self.mixing_correlation = correlation.tolist()
        self.events.append({"kind": "mixing", "model_version": self.model_version, **self.last_mixing})
        return result

    def start_birth_tracking(self) -> None:
        """Mark every current configuration as born before the next round.

        From then on ``advance`` stamps configurations entering the active
        ladder with the current round: exact profile draws at rung 0 in a full
        ladder, or configurations emitted by the reservoir otherwise.
        Exchanges and permutations carry birth rounds with configurations.
        """
        self.birth = torch.full(
            (len(self.models), len(self.chains[0])), self.rounds - 1, device=self.device, dtype=torch.int64
        )
        self.reached_top = torch.zeros_like(self.birth, dtype=torch.bool)

    def _renewal_phase(self, *, reference, rounds, limit, stop_when_renewed, rung=None,
                       local_sweeps=1, is_cancelled=None, on_round=None):
        """Advance up to ``rounds`` rounds and record renewal relative to ``reference``.

        Configurations born before ``reference`` are old. The phase counts old
        configurations in the whole active ladder, or only in replica ``rung``,
        and is renewed once that count is at most ``limit``. Returns the
        per-round active-ladder old fractions and endpoint fresh fractions,
        the first renewed round (or None), and the final old count.
        """
        ladder_old, endpoint_fresh = [], []
        renewed_at = None
        count = 0

        def after_round():
            nonlocal renewed_at, count
            old = self.birth[self.active_start:] < reference
            count = int(old.sum()) if rung is None else int(old[rung - self.active_start].sum())
            ladder_old.append(float(device_accumulation(old).mean()))
            endpoint_fresh.append(1.0 - float(device_accumulation(old[-1]).mean()))
            if renewed_at is None and count <= limit:
                renewed_at = len(ladder_old)
            if on_round is not None:
                on_round(len(ladder_old), ladder_old[-1], endpoint_fresh[-1])
            return stop_when_renewed and renewed_at is not None

        self.advance(return_samples=False, rounds=rounds, local_sweeps=local_sweeps, is_cancelled=is_cancelled, until=after_round)
        return ladder_old, endpoint_fresh, renewed_at, count

    def measure_renewal(
        self,
        *,
        local_sweeps: int = 1,
        max_rounds: int = 20_000,
        tolerance: float = 0.01,
        blocks_per_warmup: int = 1,
        stationary: bool = True,
        is_cancelled: Callable[[], bool] | None = None,
        on_progress: Callable[..., None] | None = None,
        stages: Sequence[str] = ("renewal_warmup", "renewal_stationary"),
    ) -> RenewalEstimate:
        """Equilibrate the active ladder in place until its population is renewed twice (or once).

        Requires birth tracking (``prepare_sampling_ladder`` or
        ``start_birth_tracking``). A configuration is born when it enters the
        active ladder: an exact profile draw at rung 0 of a full ladder, or a
        reservoir emission otherwise. The warmup ends at the first round where
        at most ``tolerance * n_chains`` configurations born before the start
        remain in the active ladder. The stationary phase then restarts the
        reference and advances in fixed blocks of the warmup length divided by
        ``blocks_per_warmup`` until the same holds at a block end. Stopping only
        at block ends keeps the stopping round independent of which
        configurations are old; shorter blocks waste fewer rounds after the
        renewal. With ``stationary=False`` only the warmup runs, and the
        measurement converges at the warmup renewal. No autocorrelation fit
        decides anything; see
        ``RenewalEstimate`` for the reported quantities. ``max_rounds`` bounds
        both phases together; exhausting it reports the old endpoint fraction
        as ``trapped_fraction``.

        Progress events carry the old fraction ``ladder_old`` of the whole
        ladder, the ``threshold`` it must reach, and every 10 rounds a
        ``renewal_forecast`` of the phase: ``decay_rounds``,
        ``predicted_round`` and the first decay time of the phase,
        ``first_decay_rounds``, whose growth signals configurations that leave
        more and more slowly.

        Args:
            local_sweeps: Local sweeps per exchange round.
            max_rounds: Round budget of both phases together.
            tolerance: Largest remaining fraction of old configurations, per replica.
            blocks_per_warmup: Stationary phase: blocks per warmup length.
            stationary: Run the stationary phase after the warmup.
            is_cancelled: Optional callable; returning ``True`` stops the run.
            on_progress: Optional callback ``on_progress(stage, done, total, **details)``.
            stages: Stage names reported for the two phases.

        Returns:
            A :class:`RenewalEstimate`; ``status`` is ``"converged"`` when renewed in time.
        """
        validate_integer("local_sweeps", local_sweeps)
        validate_integer("max_rounds", max_rounds)
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0.0 < tolerance < 1.0:
            raise InputValidationError("PTT renewal tolerance must be between 0 and 1.")
        if getattr(self, "birth", None) is None:
            raise InputValidationError(
                "PTT renewal needs birth tracking; call prepare_sampling_ladder or start_birth_tracking first."
            )
        validate_integer("blocks_per_warmup", blocks_per_warmup)
        n_chains = self.birth.shape[1]
        limit = tolerance * n_chains
        # The same bound as a fraction of all configurations of the active ladder.
        threshold = tolerance / self.n_active
        before = self.local_sweeps
        acceptance = torch.zeros(self.n_active - 1, dtype=torch.float64)
        used = 0

        forecasts = {}

        def run(stage, reference, rounds, stop_when_renewed, history, block=None):
            """Advance one phase (or one block of it); return its histories and the round of renewal, if any."""
            nonlocal used, acceptance

            def on_round(done, old_fraction, fresh_fraction):
                history.append(old_fraction)
                if on_progress is None:
                    return
                if len(history) % 10 == 0:
                    forecast = renewal_forecast(history, threshold)
                    if forecast is not None:
                        forecasts.setdefault(stage, {"first_decay_rounds": forecast.decay_rounds})
                        forecasts[stage].update(decay_rounds=forecast.decay_rounds,
                                                predicted_round=forecast.predicted_round)
                on_progress(stage, used + done, max_rounds, ladder_old=old_fraction, endpoint_fresh=fresh_fraction,
                            threshold=threshold, phase_round=len(history), block_rounds=block,
                            **forecasts.get(stage, {}))

            ladder_old, endpoint_fresh, renewed, _ = self._renewal_phase(
                reference=reference, rounds=rounds, limit=limit, stop_when_renewed=stop_when_renewed,
                local_sweeps=local_sweeps, is_cancelled=is_cancelled, on_round=on_round,
            )
            acceptance += torch.tensor(self.acceptance, dtype=torch.float64) * len(ladder_old)
            used += len(ladder_old)
            return ladder_old, endpoint_fresh, renewed

        reference = self.rounds
        warm_old, warm_fresh, warmup_rounds = run(stages[0], reference, max_rounds, True, [])
        stationary_old, stationary_fresh, stationary_history = [], [], []
        renewal_rounds = None
        chunk = math.ceil(warmup_rounds / blocks_per_warmup) if warmup_rounds else 0
        status = "budget_exceeded"
        if warmup_rounds is not None and not stationary:
            status = "converged"
        elif warmup_rounds is not None:
            reference = self.rounds
            while used < max_rounds:
                # Block ends do not depend on the birth labels, so the retained
                # population is not selected by the renewal criterion itself.
                old, fresh, renewed = run(stages[1], reference, min(chunk, max_rounds - used), False,
                                          stationary_history, chunk)
                if renewed is not None and renewal_rounds is None:
                    renewal_rounds = len(stationary_old) + renewed
                stationary_old += old
                stationary_fresh += fresh
                if renewal_rounds is not None:
                    status = "converged"
                    break
        old = self.birth[self.active_start:] < reference
        immobile = () if status == "converged" else tuple(self._immobile_old_configurations(reference, tolerance))
        result = RenewalEstimate(
            status=status, tolerance=float(tolerance), warmup_rounds=warmup_rounds,
            renewal_rounds=renewal_rounds, stationary_rounds=len(stationary_old), chunk_rounds=chunk,
            trapped_fraction=float(device_accumulation(old[-1]).mean()),
            warmup_ladder_old=tuple(warm_old), warmup_endpoint_fresh=tuple(warm_fresh),
            stationary_ladder_old=tuple(stationary_old), stationary_endpoint_fresh=tuple(stationary_fresh),
            old_fraction_by_model=tuple(device_accumulation(old).mean(1).tolist()),
            acceptance=tuple((acceptance / max(used, 1)).tolist()),
            local_sweeps=self.local_sweeps - before,
            replicas=self.n_active,
            warmup_decay_rounds=_decay_rounds(warm_old, threshold),
            stationary_decay_rounds=_decay_rounds(stationary_old, threshold),
            immobile=immobile,
        )
        self.events.append({
            "kind": "renewal", "model_version": self.model_version, "status": status,
            "warmup_rounds": warmup_rounds, "renewal_rounds": renewal_rounds,
            "trapped_fraction": result.trapped_fraction,
        })
        return result

    @torch.no_grad()
    def _immobile_old_configurations(self, reference, tolerance, *, partners=8, threshold=IMMOBILE_THRESHOLD):
        """Diagnose a failed renewal: per replica, old configurations that cannot swap toward the bottom.

        Configurations are only destroyed at the bottom of the ladder, so an old
        configuration must travel down to be renewed. One whose swap acceptance
        with the replica below is under ``threshold`` (mean over ``partners``
        random partners) can only move after local moves have carried it to
        where that model also gives it weight. When such configurations alone
        exceed the renewal tolerance, more local sweeps per round are what renewal needs.
        """
        found = []
        for k in range(self.active_start + 1, len(self.models)):
            old = self.birth[k] < reference
            if self._is_steered(k) or self._is_steered(k - 1) or not old.any():
                continue
            upper, lower = self.chains[k][old], self.chains[k - 1]
            acceptance = torch.zeros(len(upper), dtype=accumulation_dtype(self.device), device=self.device)
            with self._random_context():
                for _ in range(partners):
                    partner = torch.randint(len(lower), (len(upper),), device=self.device)
                    acceptance += self._exchange_kernel(self.models[k - 1], self.models[k], lower[partner],
                                                        upper).exp()
            immobile = int((acceptance / partners < threshold).sum())
            chains = len(self.chains[k])
            found.append({"replica": k - self.active_start, "chains": chains, "old": int(old.sum()),
                          "immobile": immobile, "blocking": immobile > tolerance * chains})
        return found

    def _renewal_mixing_experiment(self, *, local_sweeps, is_cancelled, full_ladder, on_progress, bounded, max_rounds):
        """Training mixing check by population renewal on a separate small population.

        Mirrors the autocorrelation experiment's setup (fork, optional full
        ladder, ``mixing_chains`` profile draws per replica) but needs no
        thermalization: the warmup phase is itself the measurement from the
        profile start. Passes when the active ladder is renewed twice within
        the round budget.
        """
        trial = self.fork()
        trial.training_state = {}
        if full_ladder:
            trial.active_start = 0
            trial.reservoir = None
            trial.anchor_log_z = float(torch.logsumexp(diagnostic_double(trial.profile_params["bias"]), -1).sum())
            trial.acceptance = [1.0] * (len(trial.models) - 1)
        n = self.config.mixing_chains
        if trial.reservoir is not None:
            n = min(n, len(trial.reservoir))
        with trial._random_context():
            trial.chains = [_profile_states(p, n) for p in trial.models]
        trial._reset_lineage()
        trial.start_birth_tracking()
        timing_before = dict(trial.timings)
        if max_rounds is None:
            max_rounds = self.config.mixing_max_rounds if bounded else 10**9
        stages = ("mixing_warmup", "mixing_measure")

        def progress(stage, current, total, **details):
            if on_progress is not None:
                on_progress(stage, current, total, chains=n, replicas=trial.n_active, **details)

        renewal = trial.measure_renewal(
            local_sweeps=local_sweeps, max_rounds=max_rounds, tolerance=self.config.renewal_tolerance,
            is_cancelled=is_cancelled, on_progress=progress, stages=stages,
        )
        result = MixingEstimate(
            None, None, renewal.rounds, renewal.rounds, renewal.status, renewal.acceptance, renewal.local_sweeps,
            model_version=self.model_version, ladder_version=self.ladder_version, replicas=trial.n_active,
            full_ladder=trial.reservoir is None, method="renewal", warmup_rounds=renewal.warmup_rounds,
            renewal_rounds=renewal.renewal_rounds, trapped_fraction=renewal.trapped_fraction,
            immobile=renewal.immobile,
        )
        self.local_sweeps += renewal.local_sweeps
        for key in _TIMING_KEYS:
            self.timings[key] += trial.timings[key] - timing_before[key]
        self.last_mixing = asdict(result)
        self.last_mixing["acceptance"] = list(result.acceptance)
        self.last_mixing["observable"] = "birth_round"
        self.mixing_correlation = []
        self.events.append({"kind": "mixing", "model_version": self.model_version, **self.last_mixing})
        return result

    def _collect_reservoir_by_renewal(self, rung, size, *, local_sweeps=1, is_cancelled=None, on_progress=None,
                                      before_collect=None):
        """Collect ``size`` configurations of replica ``rung`` spaced by its full renewal.

        Runs in place on the production populations. Warmup lasts until the
        replica is renewed relative to the current populations. The spacing is
        then measured as its stationary renewal time, a whole number of chunks
        of the warmup length. Between batches the populations advance in
        chunks of that spacing until, at a chunk end, at most
        ``renewal_tolerance * n_chains`` of the replica's configurations
        predate the previous batch. Returns ``(batches, details)``, or
        ``(None, details)`` when a phase exceeds ``mixing_max_rounds``.
        ``before_collect()`` runs once, right before the first batch.
        """
        n_chains = len(self.chains[rung])
        limit = self.config.renewal_tolerance * n_chains
        budget = self.config.mixing_max_rounds
        self.start_birth_tracking()
        details = {"warmup_rounds": None, "renewal_rounds": None, "spacing_rounds": None, "batch_fresh": []}

        def phase(reference, rounds, stop_when_renewed, report):
            def on_round(done, *_):
                if on_progress is not None:
                    report(done)

            return self._renewal_phase(
                reference=reference, rounds=rounds, limit=limit, stop_when_renewed=stop_when_renewed, rung=rung,
                local_sweeps=local_sweeps, is_cancelled=is_cancelled, on_round=on_round,
            )

        def report(stage, offset):
            # Each phase has its own budget of mixing_max_rounds rounds.
            return lambda done: on_progress(stage, offset + done, budget, replicas=self.config.target_replicas)

        if on_progress is not None:
            on_progress("reservoir_warmup", 0, budget, replicas=self.config.target_replicas)
        reference = self.rounds
        _, _, warmup, _ = phase(reference, budget, True, report("reservoir_warmup", 0))
        details["warmup_rounds"] = warmup
        if warmup is None:
            details["immobile"] = self._immobile_old_configurations(reference, self.config.renewal_tolerance)
            return None, details
        reference, spacing, count = self.rounds, 0, limit + 1
        while count > limit:
            if spacing >= budget:
                details["immobile"] = self._immobile_old_configurations(reference, self.config.renewal_tolerance)
                return None, details
            chunk = min(warmup, budget - spacing)
            old, _, renewed, count = phase(reference, chunk, False, report("reservoir_renewal", spacing))
            if renewed is not None and details["renewal_rounds"] is None:
                details["renewal_rounds"] = spacing + renewed
            spacing += len(old)
        details["spacing_rounds"] = spacing
        if before_collect is not None:
            before_collect()
        batches, remaining = [], size
        if on_progress is not None:
            on_progress("reservoir_collect", 0, size, replicas=self.config.target_replicas)
        while remaining:
            count = min(remaining, n_chains)
            batches.append(self.chains[rung][:count].clone())
            remaining -= count
            if on_progress is not None:
                on_progress("reservoir_collect", size - remaining, size, replicas=self.config.target_replicas)
            if remaining:
                reference, advanced, old_count = self.rounds, 0, limit + 1
                collected = size - remaining
                while old_count > limit:
                    if advanced >= budget:
                        details["immobile"] = self._immobile_old_configurations(
                            reference, self.config.renewal_tolerance)
                        return None, details
                    _, _, _, old_count = phase(
                        reference, spacing, False,
                        lambda done, advanced=advanced, collected=collected: on_progress(
                            "reservoir_collect", collected, size, replicas=self.config.target_replicas,
                            spacing_current=advanced + done, spacing_total=spacing),
                    )
                    advanced += spacing
                details["batch_fresh"].append(1.0 - old_count / n_chains)
        return batches, details

    def _snapshot_endpoint(self):
        return {"params": self.endpoint_params(), "chains": self.chains[-1].clone(), "step": self.model_version}

    @torch.no_grad()
    def _record_endpoint_update(self, params):
        """Add the energy change of an endpoint update to its configurations' lag memory.

        Right after the update the endpoint population still follows the old
        model; to linear order its mean of any observable ``m`` exceeds the new
        equilibrium value by ``Cov(m, s)``, with ``s = E_new - E_old``. Local
        moves and exchanges relax that excess, which then survives as the
        covariance between ``m`` now and ``s`` evaluated back then. Summing the
        centered ``s`` of successive updates per configuration (``lag_memory``,
        carried through exchanges and permutations) therefore gives the current
        lag of any observable as ``Cov(m, lag_memory)`` over the endpoint.
        Configurations that leave the endpoint take their share with them.
        Contributions older than ``config.lag_horizon`` updates are forgotten
        exponentially, which bounds the noise of the estimate.
        """
        shape = (len(self.chains), len(self.chains[0]))
        if self.lag_memory is None or self.lag_memory.shape != shape:
            self.lag_memory = torch.zeros(shape, dtype=accumulation_dtype(self.device), device=self.device)
        change = {key: params[key] - self.models[-1][key] for key in ("bias", "coupling_matrix")}
        (energy_change,) = self._energies(self.chains[-1], [change])
        self.lag_memory.mul_(1.0 - 1.0 / self.config.lag_horizon)
        self.lag_memory[-1] += energy_change - energy_change.mean()

    def update_replica_chain(
        self,
        *,
        local_sweeps: int = 1,
        is_cancelled: Callable[[], bool] | None = None,
        on_progress: Callable[..., None] | None = None,
    ) -> bool:
        """Training: run the snapshot procedure after an endpoint update.

        When the endpoint's swap acceptance falls below ``2 * target_acceptance`` a
        snapshot of it is stored; below ``target_acceptance`` the stored snapshot
        is inserted into the ladder (or a held one replaces the replica below).

        Returns:
            ``False`` if a required reservoir refresh failed its mixing check.
        """
        if self.mode != "train":
            raise InputValidationError("Generation freezes the PTT ladder; snapshot updates are disabled.")
        self.training_state = {}
        acc = self.acceptance[-1]
        threshold = self.config.target_acceptance
        if acc > 2 * threshold:
            return True
        if self.held is None:
            if self.temporary is None and acc < 2 * threshold:
                self.temporary = self._snapshot_endpoint()
                self.events.append({"kind": "snapshot_stored", "model_version": self.model_version, "acceptance": acc})
            elif self.temporary is not None and acc < threshold:
                self._insert_temporary(acc, {}, local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                                       on_progress=on_progress)
        elif acc < threshold:
            return self._replace_with_held(acc, {}, local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                                           on_progress=on_progress)
        return True

    def _insert_temporary(self, acc, details, *, local_sweeps, is_cancelled, on_progress):
        """Flag the endpoint as a checkpoint and insert the stored snapshot below it."""
        self.ptt_checkpoints.append({"step": self.model_version, "params": self.endpoint_params()})
        self.events.append({"kind": "checkpoint_flagged", "model_version": self.model_version,
                            "acceptance": acc, "flag": "ptt", **details})
        self.models.insert(len(self.models) - 1, self.temporary["params"])
        self.chains.insert(len(self.chains) - 1, self.temporary["chains"])
        if self.lag_memory is not None:
            self.lag_memory = torch.cat([
                self.lag_memory[:-1], torch.zeros_like(self.lag_memory[-1:]), self.lag_memory[-1:],
            ])
        self.total_models += 1
        self.temporary = None
        self._reset_lineage()
        self.ladder_version += 1
        if on_progress is not None:
            on_progress("replica_equilibration", 0, self.config.equilibration_rounds, action="insert")
        self.advance(
            return_samples=False,
            rounds=self.config.equilibration_rounds, local_sweeps=local_sweeps, is_cancelled=is_cancelled,
            on_round=(None if on_progress is None else
                      lambda done, total: on_progress("replica_equilibration", done, total, action="insert")),
        )
        self.held = self._snapshot_endpoint()
        self.events.append({"kind": "snapshot_inserted", "model_version": self.model_version, "acceptance": acc,
                            **details})

    def _replace_with_held(self, acc, details, *, local_sweeps, is_cancelled, on_progress):
        """Replace the replica below the endpoint with the held snapshot; refresh an overlong ladder."""
        self.models[-2] = self.held["params"]
        self.chains[-2] = self.held["chains"]
        if self.lag_memory is not None:
            self.lag_memory[-2] = 0.0
        self.held = None
        self._reset_lineage()
        self.ladder_version += 1
        if on_progress is not None:
            on_progress("replica_equilibration", 0, self.config.equilibration_rounds, action="replace")
        self.advance(
            return_samples=False,
            rounds=self.config.equilibration_rounds, local_sweeps=local_sweeps, is_cancelled=is_cancelled,
            on_round=(None if on_progress is None else
                      lambda done, total: on_progress("replica_equilibration", done, total, action="replace")),
        )
        self.events.append({"kind": "snapshot_replaced", "model_version": self.model_version, "acceptance": acc,
                            **details})
        if self.n_active > self.config.max_replicas:
            return self.refresh_reservoir(local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                                          on_progress=on_progress)
        return True

    def force_ladder_update(
        self,
        *,
        local_sweeps: int = 1,
        is_cancelled: Callable[[], bool] | None = None,
        on_progress: Callable[..., None] | None = None,
    ) -> bool:
        """Advance the snapshot procedure now, as if acceptance had just dropped below its target.

        With a held snapshot, it replaces the replica below the endpoint (and
        the reservoir is refreshed when the active ladder grows too long);
        otherwise the stored snapshot, or a snapshot of the current endpoint
        when none is stored, is inserted below the endpoint and the endpoint
        is flagged as a checkpoint. Nothing is inserted twice at the same
        model version. Returns False when a reservoir refresh fails.
        """
        if self.mode != "train":
            raise InputValidationError("Generation freezes the PTT ladder; snapshot updates are disabled.")
        self.training_state = {}
        acc = self.acceptance[-1]
        details = {"reason": "lag"}
        if self.held is not None:
            return self._replace_with_held(acc, details, local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                                           on_progress=on_progress)
        if self.ptt_checkpoints and self.ptt_checkpoints[-1]["step"] == self.model_version:
            return True
        if self.temporary is None:
            self.temporary = self._snapshot_endpoint()
            self.events.append({"kind": "snapshot_stored", "model_version": self.model_version, "acceptance": acc,
                                **details})
        self._insert_temporary(acc, details, local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                               on_progress=on_progress)
        return True

    def equilibrate_for_lag(
        self,
        relaxed: Callable[[PTTSampler], bool],
        *,
        max_rounds: int,
        chunk: int = 10,
        local_sweeps: int = 1,
        is_cancelled: Callable[[], bool] | None = None,
        on_progress: Callable[..., None] | None = None,
    ) -> tuple[bool, int, int]:
        """Advance the ladder, then evolve at fixed parameters until ``relaxed(sampler)`` holds.

        The response of the ``lag`` optimizer when the endpoint chains trail
        the model: parameters stay fixed while ``force_ladder_update`` gives
        the endpoint a recent neighbour and the ladder evolves in chunks of
        ``chunk`` rounds, up to ``max_rounds``. Runs on a fork and commits only
        a healthy result. Returns ``(healthy, rounds, work)``.
        """
        trial = self.fork()
        healthy = trial.force_ladder_update(local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                                            on_progress=on_progress)
        rounds = 0
        while healthy and rounds < max_rounds and not relaxed(trial):
            step = min(chunk, max_rounds - rounds)
            if on_progress is not None:
                on_progress("lag_equilibration", rounds, max_rounds)
            trial.advance(return_samples=False, rounds=step, local_sweeps=local_sweeps, is_cancelled=is_cancelled)
            rounds += step
        work = trial.local_sweeps - self.local_sweeps
        if not healthy:
            self.last_failure = {"reason": "mixing_budget_exceeded", "model_version": trial.model_version,
                                 "mixing": trial.last_mixing, "events": trial.events[len(self.events):]}
            return False, rounds, work
        recovery_points = self.recovery_points
        self.__dict__.update(trial.__dict__)
        self.recovery_points = recovery_points
        self.last_failure = None
        return True, rounds, work

    def refresh_reservoir(
        self,
        *,
        local_sweeps: int = 1,
        is_cancelled: Callable[[], bool] | None = None,
        on_progress: Callable[..., None] | None = None,
    ) -> bool:
        """Training: collect a new reservoir from the ladder and shorten the active ladder.

        Returns:
            ``False`` if the ladder failed the mixing check needed to collect it.
        """
        if self.mode != "train":
            raise InputValidationError("Generation freezes the PTT ladder; reservoir rebuilding is disabled.")
        self.training_state = {}
        source = self.fork()
        if self.config.full_sampler:
            source.active_start = 0
            source.reservoir = None
            source.anchor_log_z = float(torch.logsumexp(diagnostic_double(source.models[0]["bias"]), -1).sum())
            source.acceptance = [1.0] * (len(source.models) - 1)
        before = source.local_sweeps
        timing_before = dict(source.timings)
        mixing = source.estimate_mixing_time(local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                                             on_progress=on_progress)
        self.last_mixing = source.last_mixing
        self.mixing_correlation = source.mixing_correlation
        self.events.append({"kind": "mixing", "model_version": self.model_version, **self.last_mixing})
        if not mixing.converged:
            self.local_sweeps += source.local_sweeps - before
            for key in _TIMING_KEYS:
                self.timings[key] += source.timings[key] - timing_before[key]
            return False
        new_start = len(self.models) - self.config.target_replicas
        size = self.config.reservoir_size or 10 * len(self.chains[0])
        renewal_details = None
        if self.config.mixing_method == "renewal":
            # Collection is spaced by full renewal of the collection replica,
            # measured on the production populations; the bridge normalizer
            # uses those populations right after the stationary renewal.
            log_z_holder = {}

            def bridge_normalizer():
                log_z_holder["log_z"] = source.anchor_log_z + sum(
                    bridge_increment(source.models[k], source.models[k + 1], source.chains[k])
                    for k in range(source.active_start, new_start)
                )

            batches, renewal_details = source._collect_reservoir_by_renewal(
                new_start, size, local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                on_progress=on_progress, before_collect=bridge_normalizer,
            )
            source.birth = None
            source.reached_top = None
            if batches is None:
                self.local_sweeps += source.local_sweeps - before
                for key in _TIMING_KEYS:
                    self.timings[key] += source.timings[key] - timing_before[key]
                failure = {**self.last_mixing, "status": "budget_exceeded", "reservoir_renewal": renewal_details}
                self.last_mixing = failure
                self.events.append({"kind": "mixing", "model_version": self.model_version, **failure})
                return False
            log_z = log_z_holder["log_z"]
            self._finish_reservoir_refresh(source, new_start, batches, log_z, size, before, timing_before,
                                           renewal_details)
            return True
        warmup_rounds = max(1, 20 * int(mixing.tau_exp))
        if on_progress is not None:
            on_progress("reservoir_warmup", 0, warmup_rounds, replicas=source.n_active)
        source.advance(
            return_samples=False,
            rounds=warmup_rounds, local_sweeps=local_sweeps, is_cancelled=is_cancelled,
            on_round=(None if on_progress is None else
                      lambda done, total: on_progress("reservoir_warmup", done, total, replicas=source.n_active)),
        )
        log_z = source.anchor_log_z
        for k in range(source.active_start, new_start):
            log_z += bridge_increment(source.models[k], source.models[k + 1], source.chains[k])
        batches = []
        remaining = size
        if on_progress is not None:
            on_progress("reservoir_collect", 0, size, replicas=self.config.target_replicas)
        while remaining:
            count = min(remaining, len(source.chains[new_start]))
            batches.append(source.chains[new_start][:count].clone())
            remaining -= count
            if on_progress is not None:
                on_progress("reservoir_collect", size - remaining, size, replicas=self.config.target_replicas)
            if remaining:
                spacing = max(1, 2 * int(mixing.tau_int))
                collected = size - remaining
                source.advance(
                    return_samples=False,
                    rounds=spacing, local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                    on_round=(None if on_progress is None else
                              lambda done, total, collected=collected: on_progress(
                                  "reservoir_collect", collected, size, replicas=self.config.target_replicas,
                                  spacing_current=done, spacing_total=total)),
                )
        self._finish_reservoir_refresh(source, new_start, batches, log_z, size, before, timing_before, None)
        return True

    def _finish_reservoir_refresh(self, source, new_start, batches, log_z, size, before, timing_before,
                                  renewal_details):
        """Install a collected reservoir and shorten the active ladder."""
        self.reservoir = torch.cat(batches)
        self.anchor_log_z = log_z
        if self.config.full_sampler:
            self.active_start = new_start
            self.chains = source.chains
            self.lineage = source.lineage
            self.lag_memory = source.lag_memory
        else:
            # As in the reference's default reservoir mode, release inactive
            # models and populations. Full-history mode retains them for the
            # next reservoir-generation experiment.
            self.models = self.models[new_start:]
            self.chains = source.chains[new_start:]
            self.lag_memory = None if source.lag_memory is None else source.lag_memory[new_start:].clone()
            self.active_start = 0
            self._reset_lineage()
        self._rng = source._rng
        self.local_sweeps += source.local_sweeps - before
        for key in _TIMING_KEYS:
            self.timings[key] += source.timings[key] - timing_before[key]
        self.rounds = source.rounds
        self.acceptance = source.acceptance[-(self.n_active - 1):]
        self.ladder_version += 1
        event = {"kind": "reservoir_refreshed", "model_version": self.model_version,
                 "active_replicas": self.n_active, "reservoir_size": size}
        if renewal_details is not None:
            event.update({
                "renewal_warmup_rounds": renewal_details["warmup_rounds"],
                "renewal_spacing_rounds": renewal_details["spacing_rounds"],
                "min_batch_fresh": min(renewal_details["batch_fresh"], default=1.0),
            })
        self.events.append(event)

    def transition_target(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        local_sweeps: int = 1,
        is_cancelled: Callable[[], bool] | None = None,
        on_progress: Callable[..., None] | None = None,
    ) -> tuple[bool, int]:
        """Training: move the endpoint to ``params``, then run the snapshot procedure.

        The transition runs on a fork and is committed only if the ladder stays
        healthy; otherwise ``last_failure`` describes the mixing failure.

        Args:
            params: New endpoint parameters, same shape, precision and device.
            local_sweeps: Local sweeps per exchange round.
            is_cancelled: Optional callable; returning ``True`` stops the run.
            on_progress: Optional progress callback.

        Returns:
            ``(accepted, work)``: whether the transition was committed and the
            local sweeps it used.
        """
        if self.mode != "train":
            raise InputValidationError("Generation freezes the archived Hamiltonians; target transitions are disabled.")
        self._validate_params(params)
        trial = self.fork()
        if self.config.optimizer == "adaptive":
            trial._record_endpoint_update(params)
        trial.models[-1] = _clone(params)
        trial.model_version += 1
        trial.advance(
            return_samples=False,
            rounds=max(1, int(self.config.swaps * math.sqrt(trial.n_active))),
            local_sweeps=local_sweeps,
            is_cancelled=is_cancelled,
        )
        healthy = trial.update_replica_chain(local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                                             on_progress=on_progress)
        if healthy and min(trial.acceptance) < self.config.min_acceptance:
            healthy = trial.estimate_mixing_time(local_sweeps=local_sweeps, is_cancelled=is_cancelled,
                                                 on_progress=on_progress).converged
        work = trial.local_sweeps - self.local_sweeps
        if not healthy:
            self.last_failure = {"reason": "mixing_budget_exceeded", "model_version": trial.model_version,
                                 "mixing": trial.last_mixing, "events": trial.events[len(self.events):]}
            return False, work
        recovery_points = self.recovery_points
        self.__dict__.update(trial.__dict__)
        self.recovery_points = recovery_points
        self.last_failure = None
        return True, work

    def draw(
        self,
        n_samples: int,
        *,
        spacing: int = 1,
        local_sweeps: int = 1,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> torch.Tensor:
        """Collect endpoint samples, taking a batch of chains every ``spacing`` rounds.

        Samples are only as good as the ladder's equilibration: run
        :meth:`measure_renewal` (or use :func:`sample_sequences`) first.

        Args:
            n_samples: Number of samples.
            spacing: Exchange rounds between successive batches of ``n_chains``.
            local_sweeps: Local sweeps per round.
            is_cancelled: Optional callable; returning ``True`` stops the run.

        Returns:
            One-hot samples of shape ``(n_samples, L, q)``.
        """
        validate_integer("n_samples", n_samples)
        validate_integer("spacing", spacing)
        samples = []
        remaining = n_samples
        while remaining:
            batch = self.advance(rounds=spacing, local_sweeps=local_sweeps, is_cancelled=is_cancelled)
            count = min(remaining, len(batch))
            samples.append(batch[:count])
            remaining -= count
        return torch.cat(samples)

    def entropy(self, estimator: str = "bar") -> dict[str, Any]:
        """Estimate the endpoint entropy as ``<E> + log Z`` from the current chains.

        Args:
            estimator: log Z estimator, ``"bar"`` or ``"forward"`` (see :meth:`partition_estimate`).

        Returns:
            ``entropy``, ``mean_energy`` and ``free_energy`` (``-log Z``), in nats,
            with the fields of the :class:`PartitionEstimate`.
        """
        estimate = self.partition_estimate(estimator)
        mean_energy = float((device_accumulation(compute_energy(self.endpoint_samples(copy=False), self.models[-1]))
                             + self._rung_values(len(self.models) - 1)).mean())
        return {
            "entropy": mean_energy + estimate.log_z,
            "mean_energy": mean_energy,
            "free_energy": -estimate.log_z,
            **asdict(estimate),
        }

    @torch.no_grad()
    def ladder_statistics(
        self, reference: torch.Tensor, weights: torch.Tensor, estimator: str = "bar",
    ) -> list[dict[str, Any]]:
        """Estimate every active model's entropy and weighted data likelihood.

        Energies, entropy and logZ are in nats per sequence; likelihood also
        includes a per-residue column, matching the training convention.
        logZ uses BAR bridges by default (see ``partition_estimate``);
        ``log_z_forward`` always reports the forward bridges for comparison.

        Args:
            reference: Data sequences, categorical ``(M, L)`` or one-hot ``(M, L, q)``.
            weights: Weight of each reference sequence.
            estimator: ``"bar"`` or ``"forward"``.

        Returns:
            One dictionary per active replica: ``log_z``, ``mean_energy``,
            ``entropy``, ``log_likelihood``, ``log_likelihood_per_residue``, ...
        """
        if estimator not in ("forward", "bar"):
            raise InputValidationError("PTT partition estimator must be 'forward' or 'bar'.")
        rows = []
        log_z = log_z_forward = self.anchor_log_z
        weights = device_accumulation(weights) / device_accumulation(weights).sum()
        for k in range(self.active_start, len(self.models)):
            if k > self.active_start:
                forward = self._forward_increment(k - 1)
                log_z_forward += forward
                log_z += forward if estimator == "forward" else -float(bar(*self._pair_works(k - 1)))
            samples = _one_hot(self.chains[k], self.models[k])
            mean_energy = float((device_accumulation(compute_energy(samples, self.models[k])) + self._rung_values(k)).mean())
            if self._is_steered(k):
                # The data likelihood of a steered rung would need the potential on
                # the whole reference alignment; it is not part of the diagnostics.
                likelihood = None
            else:
                # Categorical references are encoded in chunks: large alignments do
                # not fit in memory as one one-hot tensor.
                (reference_energy,) = (self._energies(reference, [self.models[k]]) if reference.ndim == 2
                                       else (device_accumulation(compute_energy(reference, self.models[k])),))
                likelihood = -float((reference_energy * weights).sum()) - log_z
            rows.append({
                "replica": k, "model_id": model_id(self.models[k]),
                "log_z": log_z, "log_z_forward": log_z_forward, "log_z_estimator": estimator,
                "mean_energy": mean_energy, "entropy": mean_energy + log_z,
                "log_likelihood": likelihood,
                "log_likelihood_per_residue": None if likelihood is None else likelihood / reference.shape[1],
                "steering_strength": self._strength(k),
                "sample_round": self.rounds, "n_chains": len(self.chains[k]),
            })
        return rows

    # Steering (generation only)

    def _strength(self, k: int) -> float:
        strengths = getattr(self, "steering_strengths", None)
        return 0.0 if strengths is None else strengths[k]

    def _is_steered(self, k: int) -> bool:
        return self._strength(k) != 0.0

    def _steered_pair(self, k: int) -> bool:
        """Whether rungs ``k`` and ``k + 1`` belong to the steering ladder (same DCA parameters)."""
        return getattr(self, "steering", None) is not None and k >= self.steering_base

    def _rung_values(self, k: int) -> torch.Tensor | float:
        """Cached potential of the chains of rung ``k`` (0 for unsteered rungs)."""
        values = None if getattr(self, "steering_values", None) is None else self.steering_values[k]
        return 0.0 if values is None else values

    def _steering_cross(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``V(x_k, s_{k+1})`` and ``V(x_{k+1}, s_k)``: each population under its neighbour's strength."""
        dtype = self.models[k]["bias"].dtype
        return (self.steering(self.chains[k], self._strength(k + 1), dtype=dtype),
                self.steering(self.chains[k + 1], self._strength(k), dtype=dtype))

    def _steered_exchange(self, k: int) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        lower_under_upper, upper_under_lower = self._steering_cross(k)
        log_a = -(upper_under_lower + lower_under_upper - self._rung_values(k) - self._rung_values(k + 1))
        return log_a.clamp_max(0.0), (lower_under_upper, upper_under_lower)

    def _swap_steering_values(self, k: int, swap: torch.Tensor, cross) -> None:
        """Move cached potentials with swapped configurations, re-evaluated at their new strength."""
        lower_under_upper, upper_under_lower = cross
        values = self.steering_values
        if values[k] is not None:
            values[k] = torch.where(swap, upper_under_lower, values[k])
        values[k + 1] = torch.where(swap, lower_under_upper, values[k + 1])

    def _steered_local_update(self, k: int, local_sweeps: int) -> None:
        params = self.models[k]
        chains, values = self._steering_kernels[k].run(
            _one_hot(self.chains[k], params), params, local_sweeps, values=self.steering_values[k],
        )
        self.chains[k] = chains.argmax(-1).to(torch.int32)
        self.steering_values[k] = values
        self.local_sweeps += local_sweeps

    def _forward_increment(self, k: int) -> float:
        """Forward bridge estimate of ``log Z_{k+1} - log Z_k`` from the chains of rung ``k``."""
        if not self._steered_pair(k):
            return bridge_increment(self.models[k], self.models[k + 1], self.chains[k])
        work = diagnostic_double(self._pair_works(k)[0])
        return float(torch.logsumexp(-work, 0) - math.log(len(work)))

    def _append_steered_rung(self, strength: float, proposal_steps: int | None) -> None:
        base = self.steering_base
        previous = self._steering_kernels[-1]
        kernel = SteeredKernel(self.steering, strength, length=self.models[base]["bias"].shape[0],
                               device=self.device, proposal_steps=proposal_steps)
        if previous is not None and proposal_steps is None:
            kernel.proposal_steps = previous.proposal_steps
        chains = self.chains[-1].clone()
        self.models.append(_clone(self.models[base]))
        self.chains.append(chains)
        self.lineage = torch.cat([self.lineage, torch.arange(
            self.lineage.numel(), self.lineage.numel() + len(chains), device=self.device).unsqueeze(0)])
        if self.birth is not None:
            self.birth = torch.cat([self.birth, self.birth[-1:].clone()])
        if getattr(self, "reached_top", None) is not None:
            self.reached_top = torch.cat([self.reached_top, torch.zeros_like(self.reached_top[-1:])])
        self.steering_strengths.append(float(strength))
        self.steering_values.append(self.steering(chains, strength, dtype=self.models[base]["bias"].dtype))
        self._steering_kernels.append(kernel)
        self.sampling_steps.append(self.sampling_steps[base])
        self.acceptance.append(1.0)
        self._replica_params_cache = None

    def _pop_steered_rung(self) -> None:
        for name in ("models", "chains", "steering_strengths", "steering_values", "_steering_kernels",
                     "sampling_steps", "acceptance"):
            getattr(self, name).pop()
        self.lineage = self.lineage[:-1]
        if self.birth is not None:
            self.birth = self.birth[:-1]
        if getattr(self, "reached_top", None) is not None:
            self.reached_top = self.reached_top[:-1]
        self._replica_params_cache = None

    def prepare_steering(
        self,
        steering: Steering,
        strength: float,
        *,
        proposal_steps: int | None = None,
        target_acceptance: float = 0.3,
        local_sweeps: int = 1,
        trial_rounds: int = 20,
        max_rungs: int = 64,
        is_cancelled: Callable[[], bool] | None = None,
        on_progress: Callable[..., None] | None = None,
    ) -> list[float]:
        """Extend the generation ladder with steered rungs up to ``strength``.

        Every new rung has the endpoint's DCA parameters plus the steering
        potential at a strength between 0 and ``strength``; the last one is at
        ``strength`` and becomes the sampled endpoint. Rungs are placed one at a
        time: a candidate strength is kept when, after ``trial_rounds`` rounds of
        warmup and ``trial_rounds`` of measurement, the swap acceptance with the
        rung below is at least ``target_acceptance``; otherwise the step is
        halved. The bottom of the ladder, the exact profile, is unchanged, so
        population renewal still certifies mixing. Proposal blocks of the steered
        rungs adapt during placement and are then frozen.

        Call after :meth:`prepare_sampling_ladder`. A steered ladder cannot be
        saved with :meth:`save_archive`.

        Args:
            steering: The validated potential.
            strength: Final steering strength.
            proposal_steps: Site updates per steered proposal, or ``None`` to adapt.
            target_acceptance: Smallest swap acceptance between adjacent steered rungs.
            local_sweeps: Local sweeps per exchange round.
            trial_rounds: Rounds of warmup, then of measurement, per candidate rung.
            max_rungs: Largest number of steered rungs.
            is_cancelled: Optional callable; returning ``True`` stops the run.
            on_progress: Optional callback ``on_progress(stage, done, total, **details)``.

        Returns:
            The strengths of the steered rungs, increasing to ``strength``.

        Raises:
            InputValidationError: If the ladder is not a generation ladder, the
                potential is invalid, or ``strength`` is 0.
            ConvergenceError: If ``max_rungs`` rungs do not reach ``strength``.
        """
        from adabmDCA.exceptions import ConvergenceError

        if self.mode != "generate" or not hasattr(self, "sampling_steps"):
            raise InputValidationError("Call prepare_sampling_ladder before prepare_steering.")
        if getattr(self, "steering", None) is not None:
            raise InputValidationError("This ladder is already steered.")
        if not math.isfinite(strength) or strength == 0.0:
            raise InputValidationError("steering_strength must be finite and non-zero.")
        validate_integer("trial_rounds", trial_rounds)
        validate_integer("max_rungs", max_rungs)
        if not 0.0 < target_acceptance < 1.0:
            raise InputValidationError("The steering target acceptance must be between 0 and 1.")
        steering.check_zero_strength(self.chains[-1][:16], dtype=self.models[-1]["bias"].dtype)
        self.steering = steering
        self.steering_base = len(self.models) - 1
        self.steering_strengths = [0.0] * len(self.models)
        self.steering_values = [None] * len(self.models)
        self._steering_kernels = [None] * len(self.models)
        reached, step, rungs = 0.0, 1.0, 0
        while reached < 1.0:
            if rungs >= max_rungs:
                raise ConvergenceError(
                    f"Steering needs more than {max_rungs} rungs to reach strength {strength}; "
                    f"reached {reached * strength:.4g}.",
                    details={"reached_strength": reached * strength, "max_rungs": max_rungs},
                )
            candidate = min(1.0, reached + step)
            self._append_steered_rung(candidate * strength, proposal_steps)
            self.advance(return_samples=False, rounds=trial_rounds, local_sweeps=local_sweeps, is_cancelled=is_cancelled)
            self.advance(return_samples=False, rounds=trial_rounds, local_sweeps=local_sweeps, is_cancelled=is_cancelled)
            acceptance = self.acceptance[-1]
            if acceptance >= target_acceptance or step <= 2.0 ** -12:
                reached, rungs = candidate, rungs + 1
                if acceptance > 2 * target_acceptance:
                    step *= 2.0
                if on_progress is not None:
                    on_progress("steering_ladder", rungs, max_rungs, strength=candidate * strength,
                                acceptance=acceptance)
            else:
                self._pop_steered_rung()
                step /= 2.0
        for kernel in self._steering_kernels:
            if kernel is not None:
                kernel.freeze()
        return [s for s in self.steering_strengths if s != 0.0]

    def steering_log_z_ratio(self, estimator: str = "bar") -> float:
        """Estimate ``log Z_s - log Z_0``: steered top rung against the unsteered endpoint.

        ``-log_z_ratio`` is the free-energy cost of the steering, and
        ``exp(log_z_ratio)`` equals ``<exp(-V(x, s))>`` under the unsteered model.

        Args:
            estimator: ``"bar"`` or ``"forward"`` bridges.

        Returns:
            The log ratio of partition functions.
        """
        if getattr(self, "steering", None) is None:
            raise InputValidationError("This ladder is not steered.")
        total = 0.0
        for k in range(self.steering_base, len(self.models) - 1):
            total += self._forward_increment(k) if estimator == "forward" else -float(bar(*self._pair_works(k)))
        return total

    def save_archive(self, path: str | Path) -> Path:
        """Write the complete sampler to an HDF5 archive, replacing ``path`` atomically.

        The archive is validated before it replaces an existing file, so an
        interrupted save never leaves a broken archive.

        Args:
            path: Archive path, usually ``ptt.h5``.

        Returns:
            The written path.
        """
        if getattr(self, "steering", None) is not None:
            raise InputValidationError(
                "A steered generation ladder cannot be saved: its extra rungs are not part of the trained model."
            )
        return save_archive(self, path)

    @classmethod
    def from_archive(
        cls, path: str | Path, *, device: str = "cpu", mode: str = "generate", seed: int | None = None,
    ) -> PTTSampler:
        """Load a sampler from an HDF5 archive written by training or :meth:`save_archive`.

        Every array is validated on load. Generation never writes the source
        archive. Loading on a different device than the one that saved the
        archive requires a new ``seed`` (and then a warmup): the random stream
        cannot be continued exactly across devices.

        Args:
            path: Archive path.
            device: ``"cpu"``, a CUDA device such as ``"cuda"``, or
                ``"mps"`` (float32 archives).
            mode: ``"generate"`` (sampling; training state dropped), ``"resume"``
                (continue training exactly) or ``"inspect"`` (read only).
            seed: New random seed, or ``None`` to continue the archived stream.

        Returns:
            The loaded :class:`PTTSampler`.

        Raises:
            InputValidationError: If the archive is invalid, or the device changes
                without a new seed.
        """
        return load_archive(cls, path, device=device, mode=mode, seed=seed)

    def _validate_algorithm_state(self):
        validate_integer("active_start", self.active_start, minimum=0)
        for timing in (self.timings, self.last_advance_timing):
            if set(timing) != set(_TIMING_KEYS) or any(
                not math.isfinite(value) or value < 0 for value in timing.values()
            ):
                raise ValueError("Invalid PTT timing diagnostics.")
        if not 2 <= self.n_active <= self.config.max_replicas + 1 or not math.isfinite(self.anchor_log_z):
            raise ValueError("Invalid PTT active ladder or anchor normalizer.")
        if self.reservoir is None and self.active_start != 0:
            raise ValueError("PTT compressed ladders require a reservoir.")
        validate_integer("total_models", self.total_models, minimum=len(self.models))
        self._validate_params(self.profile_params)
        if torch.count_nonzero(self.profile_params["coupling_matrix"]):
            raise ValueError("PTT initial profile must be uncoupled.")
        if self.reservoir is None and model_id(self.models[0]) != model_id(self.profile_params):
            raise ValueError("PTT requires a profile anchor or a reservoir.")

        def validate_population(chains, *, size=None):
            if (not isinstance(chains, torch.Tensor) or chains.ndim != 2
                    or chains.shape[1] != self.models[0]["bias"].shape[0]
                    or (size is not None and len(chains) != size)
                    or chains.dtype != torch.int32
                    or chains.numel() == 0
                    or chains.min() < 0 or chains.max() >= self.models[0]["bias"].shape[1]):
                raise ValueError("Invalid PTT stored population.")

        if self.reservoir is not None:
            validate_population(self.reservoir)
            if len(self.reservoir) < len(self.chains[0]):
                raise ValueError("PTT reservoir is smaller than the chain population.")
        memory = getattr(self, "lag_memory", None)
        if memory is not None and (
            not isinstance(memory, torch.Tensor) or memory.dtype not in (torch.float32, torch.float64)
            or memory.shape != (len(self.chains), len(self.chains[0])) or not torch.isfinite(memory).all()
        ):
            raise ValueError("Invalid PTT lag memory.")
        for snapshot in (self.temporary, self.held):
            if snapshot is not None:
                self._validate_params(snapshot["params"])
                validate_population(snapshot["chains"], size=len(self.chains[0]))
                validate_integer("snapshot step", snapshot["step"], minimum=0)
                if snapshot["step"] > self.model_version:
                    raise ValueError("PTT snapshot is newer than its endpoint.")
        for point in self.recovery_points:
            state = point["state"]
            if point["step"] != state["model_version"] or not 0 <= point["step"] <= self.model_version:
                raise ValueError("Invalid PTT recovery checkpoint version.")
            if not math.isfinite(point["learning_rate"]) or point["learning_rate"] <= 0:
                raise ValueError("Invalid PTT recovery learning rate.")
            checkpoint = self.fork()
            checkpoint.restore_state(state)
            checkpoint.recovery_points = []
            for params, chains in zip(checkpoint.models, checkpoint.chains):
                checkpoint._validate_params(params)
                validate_population(chains, size=len(self.chains[0]))
            checkpoint._validate_algorithm_state()
