"""Canonical configuration values and validation for DCA training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from adabmDCA._validation import validate_integer, validate_seed
from adabmDCA.api.exceptions import InputValidationError
from adabmDCA.training_control import TrainingLimits

MODEL_TYPES = ("bmDCA", "eaDCA", "edDCA", "edgeDCA")
SAMPLERS = ("metropolis", "gibbs")

DEFAULT_MODEL_TYPE = "bmDCA"
DEFAULT_ALPHABET = "protein"
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_N_SWEEPS = 10
DEFAULT_SAMPLER = "metropolis"
DEFAULT_N_CHAINS = 10_000
DEFAULT_TARGET_PEARSON = 0.95
DEFAULT_MAX_EPOCHS = 50_000
DEFAULT_L2_REGULARIZATION = 0.0
DEFAULT_SEED = 0
DEFAULT_CLUSTERING_SEQID = 0.8
DEFAULT_ACTIVATION_STEPS = 10
DEFAULT_ACTIVATION_FRACTION = 0.001
DEFAULT_TARGET_DENSITY = 0.02
DEFAULT_DECIMATION_RATE = 0.01
DEFAULT_DEVICE = "auto"
DEFAULT_DTYPE = "float32"

DEFAULT_CHECKPOINT_INTERVAL = 100
# Compatibility alias: all training models now share the same default.
SPARSE_CHECKPOINT_INTERVAL = DEFAULT_CHECKPOINT_INTERVAL
DEFAULT_INNER_GRADIENT_STEPS = 10_000
EDGE_DEFAULT_PSEUDOCOUNT = 0.1
EDGE_EMPIRICAL_PSEUDOCOUNT = 1e-6
EDGE_LOGZ_CHAIN_FRACTION = 0.2
SLOPE_TOLERANCE = 0.1


class ConfigurationError(InputValidationError):
    """Raised when a training configuration is internally inconsistent."""


@dataclass(frozen=True)
class TrainingConfig:
    """Validated, transport-neutral configuration for one training run."""

    model_type: str = DEFAULT_MODEL_TYPE
    alphabet: str = DEFAULT_ALPHABET
    learning_rate: float = DEFAULT_LEARNING_RATE
    n_sweeps: int = DEFAULT_N_SWEEPS
    sampler: str = DEFAULT_SAMPLER
    n_chains: int = DEFAULT_N_CHAINS
    target_pearson: float = DEFAULT_TARGET_PEARSON
    max_epochs: int = DEFAULT_MAX_EPOCHS
    max_gradient_steps: int | None = None
    max_structure_steps: int | None = None
    pseudocount: float | None = None
    l2_regularization: float = DEFAULT_L2_REGULARIZATION
    seed: int = DEFAULT_SEED
    clustering_seqid: float = DEFAULT_CLUSTERING_SEQID
    no_reweighting: bool = False
    activation_steps: int = DEFAULT_ACTIVATION_STEPS
    activation_fraction: float = DEFAULT_ACTIVATION_FRACTION
    target_density: float = DEFAULT_TARGET_DENSITY
    decimation_rate: float = DEFAULT_DECIMATION_RATE
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    use_wandb: bool = False
    checkpoint_interval: int | None = DEFAULT_CHECKPOINT_INTERVAL
    inner_gradient_steps: int = DEFAULT_INNER_GRADIENT_STEPS
    slope_tolerance: float = SLOPE_TOLERANCE
    edge_empirical_pseudocount: float = EDGE_EMPIRICAL_PSEUDOCOUNT
    edge_logz_chain_fraction: float = EDGE_LOGZ_CHAIN_FRACTION

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "float64", "bfloat16"):
            raise ConfigurationError("dtype must be float32, float64 or bfloat16.")
        if self.model_type not in MODEL_TYPES:
            raise ConfigurationError(f"Unsupported model_type '{self.model_type}'.")
        if self.sampler not in SAMPLERS:
            raise ConfigurationError(f"sampler must be one of {SAMPLERS}.")

        for name, value in (
            ("learning_rate", self.learning_rate),
            ("target_pearson", self.target_pearson),
            ("pseudocount", self.pseudocount),
            ("l2_regularization", self.l2_regularization),
            ("clustering_seqid", self.clustering_seqid),
            ("activation_fraction", self.activation_fraction),
            ("target_density", self.target_density),
            ("decimation_rate", self.decimation_rate),
            ("slope_tolerance", self.slope_tolerance),
            ("edge_empirical_pseudocount", self.edge_empirical_pseudocount),
            ("edge_logz_chain_fraction", self.edge_logz_chain_fraction),
        ):
            if value is not None:
                try:
                    finite = math.isfinite(value)
                except TypeError as exc:
                    raise ConfigurationError(
                        f"{name} must be numeric.",
                        details={"name": name, "value": value},
                    ) from exc
                if not finite:
                    raise ConfigurationError(
                        f"{name} must be finite.",
                        details={"name": name, "value": value},
                    )

        self._positive("n_chains", self.n_chains)
        self._positive("n_sweeps", self.n_sweeps)
        self._positive("max_epochs", self.max_epochs)
        if self.max_gradient_steps is not None:
            self._positive("max_gradient_steps", self.max_gradient_steps)
        if self.max_structure_steps is not None:
            self._positive("max_structure_steps", self.max_structure_steps)
        if self.checkpoint_interval is not None:
            self._positive("checkpoint_interval", self.checkpoint_interval)
        self._positive("inner_gradient_steps", self.inner_gradient_steps)
        self._positive("activation_steps", self.activation_steps)
        validate_seed(self.seed, error_type=ConfigurationError)

        if self.model_type != "edgeDCA" and self.learning_rate <= 0.0:
            raise ConfigurationError("learning_rate must be positive.")
        if self.model_type != "edgeDCA" and self.l2_regularization < 0.0:
            raise ConfigurationError("l2_regularization cannot be negative.")
        if not 0.0 <= self.target_pearson < 1.0:
            raise ConfigurationError("target_pearson must be at least 0 and smaller than 1.")
        if self.pseudocount is not None and not 0.0 <= self.pseudocount <= 1.0:
            raise ConfigurationError("pseudocount must be between 0 and 1.")
        if not 0.0 < self.clustering_seqid <= 1.0:
            raise ConfigurationError("clustering_seqid must be larger than 0 and at most 1.")
        if self.slope_tolerance < 0.0:
            raise ConfigurationError("slope_tolerance cannot be negative.")
        if not 0.0 < self.edge_empirical_pseudocount <= 1.0:
            raise ConfigurationError("edge_empirical_pseudocount must be larger than 0 and at most 1.")
        if not 0.0 < self.edge_logz_chain_fraction < 1.0:
            raise ConfigurationError("edge_logz_chain_fraction must be larger than 0 and smaller than 1.")
        if self.model_type == "eaDCA":
            if not 0.0 < self.activation_fraction <= 1.0:
                raise ConfigurationError("activation_fraction must be larger than 0 and at most 1.")
        if self.model_type == "edDCA":
            if not 0.0 <= self.target_density <= 1.0:
                raise ConfigurationError("target_density must be between 0 and 1.")
            if not 0.0 < self.decimation_rate <= 1.0:
                raise ConfigurationError("decimation_rate must be larger than 0 and at most 1.")

    @staticmethod
    def _positive(name: str, value: int) -> None:
        validate_integer(name, value, error_type=ConfigurationError)

    @property
    def limits(self) -> TrainingLimits:
        """Resolve the legacy epoch limit into explicit model-aware limits."""
        if self.model_type == "bmDCA":
            return TrainingLimits(
                max_gradient_steps=self.max_gradient_steps or self.max_epochs,
                max_structure_steps=self.max_structure_steps,
            )
        return TrainingLimits(
            max_gradient_steps=self.max_gradient_steps,
            max_structure_steps=self.max_structure_steps or self.max_epochs,
        )

    @property
    def resolved_checkpoint_interval(self) -> int:
        if self.checkpoint_interval is not None:
            return self.checkpoint_interval
        return DEFAULT_CHECKPOINT_INTERVAL

    @property
    def empirical_pseudocount(self) -> float | None:
        """Pseudocount used for target statistics before regularization."""
        if self.model_type == "edgeDCA":
            return self.edge_empirical_pseudocount
        return None

    def resolve_pseudocount(self, effective_size: float) -> float:
        if self.pseudocount is not None:
            return self.pseudocount
        if self.model_type == "edgeDCA":
            return EDGE_DEFAULT_PSEUDOCOUNT
        if effective_size <= 0.0:
            raise ConfigurationError("effective_size must be positive.")
        return 1.0 / effective_size

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot suitable for run metadata."""
        return asdict(self)
