"""Canonical configuration values and validation for DCA training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from adabmDCA._validation import validate_integer, validate_seed
from adabmDCA.exceptions import InputValidationError
from adabmDCA.ptt.config import (
    DEFAULT_PTT_MAX_EPOCHS,
    DEFAULT_PTT_N_CHAINS,
    DEFAULT_PTT_N_SWEEPS,
    DEFAULT_PTT_SAMPLER,
    PTTConfig,
)
from adabmDCA.training_control import TrainingLimits


class _DefaultInt(int):
    """Recognize omitted numeric defaults without overriding explicit values."""


class _DefaultStr(str):
    """Recognize the omitted local kernel while preserving the public default."""


MODEL_TYPES = ("bmDCA", "eaDCA", "edDCA", "edgeDCA")
SAMPLERS = ("metropolis", "gibbs", "metropolized_gibbs")

DEFAULT_MODEL_TYPE = "bmDCA"
DEFAULT_ALPHABET = "protein"
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_N_SWEEPS = _DefaultInt(10)
DEFAULT_SAMPLER = _DefaultStr("metropolized_gibbs")
DEFAULT_N_CHAINS = _DefaultInt(2_000)
DEFAULT_TARGET_PEARSON = 0.95
DEFAULT_MAX_EPOCHS = _DefaultInt(50_000)
DEFAULT_L2_REGULARIZATION = 0.0
DEFAULT_SEED = 0
DEFAULT_CLUSTERING_SEQID = 0.8
DEFAULT_ACTIVATION_STEPS = 10
DEFAULT_ACTIVATION_FRACTION = 0.001
DEFAULT_TARGET_DENSITY = 0.02
DEFAULT_DECIMATION_RATE = 0.01
DEFAULT_DEVICE = "auto"
DEFAULT_DTYPE = "float32"

DEFAULT_CHECKPOINT_INTERVAL = 500
# Compatibility alias: all training models now share the same default.
SPARSE_CHECKPOINT_INTERVAL = DEFAULT_CHECKPOINT_INTERVAL
DEFAULT_INNER_GRADIENT_STEPS = 10_000
EDGE_DEFAULT_PSEUDOCOUNT = 0.1
EDGE_EMPIRICAL_PSEUDOCOUNT = 1e-6


class ConfigurationError(InputValidationError):
    """Raised when a training configuration is internally inconsistent."""


@dataclass(frozen=True)
class TrainingConfig:
    """All settings of one training run, validated on construction.

    Pass it as ``config=`` to :func:`train_model`; it then takes precedence
    over the function's individual keyword arguments. Settings that do not
    apply to the chosen ``model_type`` are ignored.

    Attributes:
        model_type: ``"bmDCA"`` (fully connected), ``"eaDCA"`` (coupling
            activation), ``"edDCA"`` (decimation) or ``"edgeDCA"`` (edge activation).
        alphabet: ``"protein"``, ``"rna"``, ``"dna"`` or an ordered custom token string.
        learning_rate: Step size of the parameter updates (ignored by edgeDCA).
        n_sweeps: Monte Carlo sweeps per update.
        sampler: ``"metropolized_gibbs"``, ``"gibbs"`` or ``"metropolis"``.
        n_chains: Number of Markov chains.
        target_pearson: Training stops when the Pearson correlation between data and
            model connected correlations reaches this value.
        max_epochs: Step budget: gradient steps for bmDCA and PTT edgeDCA, graph
            steps for the other sparse models. Superseded by the two limits below.
        max_gradient_steps: Optional limit on parameter updates.
        max_structure_steps: Optional limit on graph activations or decimations.
        pseudocount: Pseudocount of the training statistics; ``None`` means ``1/Meff``
            (0.1 for edgeDCA).
        l2_regularization: Strength of the L2 penalty on the couplings.
        seed: Random seed.
        clustering_seqid: Sequence identity above which sequences share weight.
        no_reweighting: Give every training sequence the same weight.
        activation_steps: eaDCA: gradient updates between graph activations.
        activation_fraction: eaDCA: fraction of the inactive couplings activated
            per graph update (the upper limit for PTT adaptive activation).
        target_density: edDCA: coupling density at which decimation stops.
        decimation_rate: edDCA: fraction of active couplings removed per decimation.
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.
        dtype: ``"float32"``, ``"float64"`` or ``"bfloat16"`` (CUDA, PCD only).
        use_wandb: Log to Weights & Biases.
        checkpoint_interval: Updates between saved checkpoints; ``None`` uses the default.
        inner_gradient_steps: edDCA: largest number of gradient updates to
            re-converge the model after each decimation.
        edge_empirical_pseudocount: edgeDCA: small pseudocount of the target
            statistics, which avoids zero frequencies.
        ptt: :class:`PTTConfig` to train with PTT, or ``None`` for PCD.

    Raises:
        ConfigurationError: If a setting is invalid or unsupported for the model.

    Example:
        >>> config = TrainingConfig(model_type="eaDCA", alphabet="rna",
        ...                         ptt=PTTConfig(activation="adaptive"))
        >>> train_model("family.fasta", config=config, output_dir="model")
    """

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
    edge_empirical_pseudocount: float = EDGE_EMPIRICAL_PSEUDOCOUNT
    ptt: PTTConfig | None = None

    def __post_init__(self) -> None:
        if self.ptt is not None:
            if not isinstance(self.ptt, PTTConfig):
                raise ConfigurationError("ptt must be a PTTConfig instance.")
            for name, default in (
                ("n_sweeps", DEFAULT_PTT_N_SWEEPS),
                ("n_chains", DEFAULT_PTT_N_CHAINS),
                ("sampler", DEFAULT_PTT_SAMPLER),
                ("max_epochs", DEFAULT_PTT_MAX_EPOCHS),
            ):
                if isinstance(getattr(self, name), (_DefaultInt, _DefaultStr)):
                    object.__setattr__(self, name, default)
            if self.model_type not in ("bmDCA", "eaDCA", "edgeDCA") or self.dtype == "bfloat16":
                raise ConfigurationError("PTT currently supports bmDCA, eaDCA and edgeDCA with float32/float64 only.")
            if self.ptt.activation != "fixed" and self.model_type != "eaDCA":
                raise ConfigurationError("PTT adaptive activation applies to eaDCA only.")
            if self.model_type == "edgeDCA" and self.pseudocount is not None and not 0.0 < self.pseudocount < 1.0:
                raise ConfigurationError("PTT edgeDCA pseudocount must be strictly between 0 and 1.")
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
            ("edge_empirical_pseudocount", self.edge_empirical_pseudocount),
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

        if self.ptt is not None:
            if self.model_type == "edgeDCA":
                if 1.0 - self.resolve_pseudocount(1.0) < self.ptt.min_learning_rate:
                    raise ConfigurationError("PTT edgeDCA effective step cannot start below min_learning_rate.")
            elif self.learning_rate < self.ptt.min_learning_rate:
                raise ConfigurationError("PTT learning rates cannot start below min_learning_rate.")
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
        if not 0.0 < self.edge_empirical_pseudocount <= 1.0:
            raise ConfigurationError("edge_empirical_pseudocount must be larger than 0 and at most 1.")
        if self.model_type == "eaDCA" and not 0.0 < self.activation_fraction <= 1.0:
            raise ConfigurationError("activation_fraction must be larger than 0 and at most 1.")
        if self.model_type == "edDCA" and not 0.0 <= self.target_density <= 1.0:
            raise ConfigurationError("target_density must be between 0 and 1.")
        if self.model_type == "edDCA" and not 0.0 < self.decimation_rate <= 1.0:
            raise ConfigurationError("decimation_rate must be larger than 0 and at most 1.")

    @staticmethod
    def _positive(name: str, value: int) -> None:
        validate_integer(name, value, error_type=ConfigurationError)

    @property
    def limits(self) -> TrainingLimits:
        """Resolve the legacy epoch limit into explicit model-aware limits."""
        if self.model_type == "bmDCA" or (self.model_type == "edgeDCA" and self.ptt is not None):
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
        """Checkpoint interval in updates, with the default filled in."""
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
        """Return the pseudocount to use for a training set of ``effective_size`` (``Meff``).

        Args:
            effective_size: Sum of the training sequence weights.

        Returns:
            ``pseudocount`` if set; otherwise 0.1 for edgeDCA and ``1/Meff`` for
            the other models.

        Raises:
            ConfigurationError: If ``effective_size`` is not positive.
        """
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
