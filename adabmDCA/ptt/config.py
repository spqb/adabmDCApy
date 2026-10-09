"""Configuration for PTT snapshot retention, reservoirs and mixing analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

from adabmDCA._validation import validate_integer
from adabmDCA.exceptions import InputValidationError

# One local sampler call is one Potts sweep. PTT and standard PCD share the
# same sampling defaults unless the caller provides an explicit override.
DEFAULT_PTT_N_SWEEPS = 10
DEFAULT_PTT_N_CHAINS = 2_000
DEFAULT_PTT_SAMPLER = "metropolized_gibbs"
DEFAULT_PTT_MAX_EPOCHS = 50_000
PTT_OPTIMIZERS = ("sgd", "adaptive")
PTT_ACTIVATIONS = ("fixed", "adaptive")


@dataclass(frozen=True)
class PTTConfig:
    """Settings of Parallel Trajectory Tempering, for training and sampling.

    Pass an instance as ``ptt=`` to :func:`train_model` (or through
    ``TrainingConfig(ptt=...)``) to train with PTT. The defaults suit most
    families; see ``docs/ptt.md`` for the method. Round counts are exchange
    rounds: one pass of replica exchanges followed by the local sweeps.

    Attributes:
        swaps: Exchange-round factor: each update runs
            ``int(swaps * sqrt(active replicas))`` rounds.
        target_acceptance: Swap acceptance below which the current endpoint is
            kept as a new snapshot in the ladder.
        min_acceptance: Swap acceptance below which a mixing check is run.
        max_replicas: Active ladder size above which the reservoir is refreshed
            and the ladder compressed.
        target_replicas: Active replicas kept after compression.
        reservoir_size: Size of the reservoir population; ``None`` means
            ``10 * n_chains``. Must be at least ``n_chains``.
        full_sampler: Use the full historical ladder when collecting a new reservoir.
        equilibration_rounds: Extra rounds after a snapshot is inserted or replaced.
        initialization_rounds: Warmup rounds before the first update.
        mixing_chains: Chains per replica in the autocorrelation mixing experiment.
        mixing_initial_rounds: Initial trajectory length of that experiment.
        mixing_thermalization_rounds: Warmup before a mixing measurement and after a recovery.
        mixing_max_rounds: Round budget of one mixing check; exceeding it is a
            mixing failure, which triggers recovery.
        mixing_window_factor: Required trajectory length relative to the longest
            correlation time (autocorrelation method).
        mixing_method: Training mixing check: ``"renewal"`` (the ladder is
            repopulated by fresh configurations) or ``"autocorrelation"``.
        renewal_tolerance: Fraction of old configurations, per replica, below which
            the ladder counts as renewed.
        max_recoveries: Largest number of learning-rate halvings after mixing failures.
        min_learning_rate: Learning-rate floor of recovery (``1 - pseudocount`` for edgeDCA).
        optimizer: ``"adaptive"`` (steps bounded by a KL trust region, training
            paused while the endpoint chains lag behind the model) or ``"sgd"``
            (fixed learning rate).
        trust_radius: Adaptive optimizer: largest predicted KL divergence of one update.
        lag_tolerance: Adaptive optimizer: largest lag of the endpoint chains, in
            population standard deviations, before training pauses.
        lag_horizon: Adaptive optimizer: memory of past updates, in updates.
        lag_pause_rounds: Adaptive optimizer: longest pause, in rounds.
        validation_stop: Stop when the validation log-likelihood plateaus instead
            of at the target Pearson; requires a validation alignment.
        validation_window: Updates per window of the plateau test, which compares
            the medians of the last two windows.
        validation_min_gain: Stop when the median gain between the two windows
            is below this value (log-likelihood per site).
        activation: eaDCA: ``"fixed"`` activates ``activation_fraction`` of the
            inactive couplings per block; ``"adaptive"`` activates at most that
            fraction, only significant candidates within a KL budget.
        activation_kl_share: Adaptive activation: share of ``trust_radius`` for the
            first update of the new couplings; halved by each recovery.
        activation_significance: Adaptive activation: standard errors by which a
            candidate's pair frequency must differ from the data; 0 disables the test.

    Example:
        >>> from adabmDCA import PTTConfig, train_model
        >>> train_model("family.fasta", alphabet="rna", ptt=PTTConfig(validation_stop=True),
        ...             validation_path="validation.fasta", output_dir="model")
    """

    swaps: int = 1
    target_acceptance: float = 0.25
    min_acceptance: float = 0.1
    max_replicas: int = 2
    target_replicas: int = 2
    reservoir_size: int | None = None
    full_sampler: bool = False
    equilibration_rounds: int = 10
    initialization_rounds: int = 100
    mixing_chains: int = 100
    mixing_initial_rounds: int = 100
    mixing_thermalization_rounds: int = 1000
    mixing_max_rounds: int = 20_000
    mixing_window_factor: float = 20.0
    mixing_method: str = "renewal"
    renewal_tolerance: float = 0.01
    max_recoveries: int = 3
    min_learning_rate: float = 1e-8
    optimizer: str = "adaptive"
    trust_radius: float = 0.01
    lag_tolerance: float = 0.25
    lag_horizon: int = 200
    lag_pause_rounds: int = 100
    validation_stop: bool = False
    validation_window: int = 100
    validation_min_gain: float = 0.0
    activation: str = "fixed"
    activation_kl_share: float = 0.5
    activation_significance: float = 3.0

    def __post_init__(self):
        for name in ("swaps", "equilibration_rounds", "initialization_rounds", "mixing_chains", "max_recoveries",
                     "lag_horizon", "lag_pause_rounds", "validation_window"):
            validate_integer(name, getattr(self, name))
        validate_integer("max_replicas", self.max_replicas, minimum=2)
        validate_integer("target_replicas", self.target_replicas, minimum=2)
        validate_integer("mixing_initial_rounds", self.mixing_initial_rounds, minimum=8)
        validate_integer("mixing_max_rounds", self.mixing_max_rounds, minimum=self.mixing_initial_rounds)
        validate_integer("mixing_thermalization_rounds", self.mixing_thermalization_rounds, minimum=0)
        if self.target_replicas > self.max_replicas:
            raise InputValidationError("PTT target_replicas cannot exceed max_replicas.")
        if self.reservoir_size is not None:
            validate_integer("reservoir_size", self.reservoir_size)
        if not isinstance(self.full_sampler, bool):
            raise InputValidationError("PTT full_sampler must be boolean.")
        if not isinstance(self.validation_stop, bool):
            raise InputValidationError("PTT validation_stop must be boolean.")
        if self.mixing_method not in ("autocorrelation", "renewal"):
            raise InputValidationError("PTT mixing_method must be 'autocorrelation' or 'renewal'.")
        if (isinstance(self.renewal_tolerance, bool) or not isinstance(self.renewal_tolerance, (int, float))
                or not 0.0 < self.renewal_tolerance < 1.0):
            raise InputValidationError("PTT renewal_tolerance must be between 0 and 1.")
        if self.optimizer not in PTT_OPTIMIZERS:
            raise InputValidationError("PTT optimizer must be 'sgd' or 'adaptive'.")
        if self.activation not in PTT_ACTIVATIONS:
            raise InputValidationError("PTT activation must be 'fixed' or 'adaptive'.")
        for name in (
            "min_acceptance", "target_acceptance", "min_learning_rate",
            "mixing_window_factor", "trust_radius", "lag_tolerance", "validation_min_gain",
            "activation_kl_share", "activation_significance",
        ):
            value = getattr(self, name)
            try:
                valid = not isinstance(value, bool) and math.isfinite(value)
            except TypeError:
                valid = False
            if not valid:
                raise InputValidationError(f"PTT {name} must be a finite number.")
        if not 0 < self.min_acceptance < self.target_acceptance < 1:
            raise InputValidationError("PTT requires 0 < min_acceptance < target_acceptance < 1.")
        if not math.isfinite(self.min_learning_rate) or self.min_learning_rate <= 0:
            raise InputValidationError("min_learning_rate must be finite and positive.")
        if self.mixing_window_factor < 2:
            raise InputValidationError("PTT mixing_window_factor must be finite and at least 2.")
        if self.trust_radius <= 0:
            raise InputValidationError("PTT trust_radius must be positive.")
        if self.lag_tolerance <= 0:
            raise InputValidationError("PTT lag_tolerance must be positive.")
        if not 0 < self.activation_kl_share <= 1:
            raise InputValidationError("PTT activation_kl_share must be larger than 0 and at most 1.")
        if self.activation_significance < 0:
            raise InputValidationError("PTT activation_significance cannot be negative.")
