"""Aggregate input loading for high-level workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from adabmDCA.alignment import Alignment
from adabmDCA.dataset import DatasetDCA
from adabmDCA.exceptions import (
    AdabmDCAError,
    ChainLoadError,
    ModelCompatibilityError,
    ModelLoadError,
)
from adabmDCA.input_loading import (
    AlignmentInput,
    AlignmentLoadConfig,
    WeightInput,
    load_alignment,
)
from adabmDCA.io import load_chains, load_params
from adabmDCA.training_config import TrainingConfig


@dataclass(frozen=True)
class TrainingInputs:
    """Loaded and cross-checked inputs of one training run.

    Attributes:
        training: Weighted training dataset.
        training_alignment: Training alignment after filtering.
        validation: Weighted validation dataset, if any.
        validation_alignment: Validation alignment after filtering, if any.
        initial_params: Parameters to start from, if given.
        initial_chains: One-hot chains to start from, if given.
    """

    training: DatasetDCA
    training_alignment: Alignment
    validation: DatasetDCA | None = None
    validation_alignment: Alignment | None = None
    initial_params: dict[str, torch.Tensor] | None = None
    initial_chains: torch.Tensor | None = None


def _check_parameter_compatibility(
    params: dict[str, torch.Tensor],
    *,
    length: int,
    num_states: int,
) -> None:
    bias = params.get("bias")
    couplings = params.get("coupling_matrix")
    if bias is None or couplings is None:
        raise ModelCompatibilityError("Initial parameters must contain 'bias' and 'coupling_matrix'.")
    if tuple(bias.shape) != (length, num_states):
        raise ModelCompatibilityError(
            "Initial parameter dimensions do not match the training alignment.",
            details={
                "parameter_shape": tuple(bias.shape),
                "alignment_shape": (length, num_states),
            },
        )
    expected_couplings = (length, num_states, length, num_states)
    if tuple(couplings.shape) != expected_couplings:
        raise ModelCompatibilityError(
            "Initial coupling dimensions do not match the training alignment.",
            details={
                "coupling_shape": tuple(couplings.shape),
                "expected_shape": expected_couplings,
            },
        )


def _check_chain_compatibility(
    chains: torch.Tensor,
    *,
    length: int,
    num_states: int,
) -> None:
    if chains.ndim != 3 or tuple(chains.shape[1:]) != (length, num_states):
        raise ModelCompatibilityError(
            "Initial chains do not match the training alignment.",
            details={
                "chain_shape": tuple(chains.shape),
                "expected_suffix": (length, num_states),
            },
        )


def load_training_inputs(
    training: AlignmentInput,
    *,
    config: TrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
    validation: AlignmentInput | None = None,
    weights: WeightInput | None = None,
    initial_params_path: str | Path | None = None,
    initial_chains_path: str | Path | None = None,
    allow_signed_weights: bool = False,
) -> TrainingInputs:
    """Load the training inputs of :func:`train_model` and check they fit together.

    Sequences with unknown tokens are dropped and duplicates removed. The
    validation alignment, initial parameters and initial chains must have the
    training alignment's length and alphabet.

    Args:
        training: Training alignment (path, :class:`Alignment` or sequences).
        config: Training settings; provides alphabet and reweighting options.
        device: Device of the loaded tensors.
        dtype: Precision of the loaded tensors.
        validation: Optional validation alignment.
        weights: Optional training-sequence weights (path, array or tensor).
        initial_params_path: Optional parameter file to start from.
        initial_chains_path: Optional FASTA of starting chains.
        allow_signed_weights: Accept negative weights (experimental reintegration).

    Returns:
        The loaded :class:`TrainingInputs`.

    Raises:
        ModelCompatibilityError: If validation data, parameters or chains do not
            match the training alignment.
        InputLoadError: If a file cannot be read.
    """
    load_policy = AlignmentLoadConfig(
        alphabet=config.alphabet,
        invalid_sequences="drop",
        remove_duplicates=True,
    )
    training_alignment = load_alignment(training, config=load_policy)
    training_dataset = DatasetDCA.from_alignment(
        training_alignment,
        weights=weights,
        clustering_th=config.clustering_seqid,
        no_reweighting=config.no_reweighting,
        device=device,
        dtype=dtype,
        allow_signed_weights=allow_signed_weights,
    )

    validation_alignment = None
    validation_dataset = None
    if validation is not None:
        validation_alignment = load_alignment(
            validation,
            config=AlignmentLoadConfig(
                alphabet=config.alphabet,
                invalid_sequences="drop",
                remove_duplicates=True,
                expected_length=training_alignment.sequence_length,
            ),
        )
        validation_dataset = DatasetDCA.from_alignment(
            validation_alignment,
            clustering_th=config.clustering_seqid,
            no_reweighting=config.no_reweighting,
            device=device,
            dtype=dtype,
        )

    length = training_dataset.get_num_residues()
    num_states = training_dataset.get_num_states()
    params = None
    if initial_params_path is not None:
        path = Path(initial_params_path)
        if not path.is_file():
            raise ModelLoadError(
                f"Initial parameter file '{path}' was not found.",
                details={"path": str(path)},
            )
        try:
            params = load_params(
                str(path),
                tokens=training_dataset.tokens,
                device=device,
                dtype=dtype,
            )
        except AdabmDCAError:
            raise
        except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ModelLoadError(
                f"Could not load initial parameters from '{path}': {exc}",
                details={"path": str(path)},
            ) from exc
        _check_parameter_compatibility(
            params,
            length=length,
            num_states=num_states,
        )

    chains = None
    if initial_chains_path is not None:
        path = Path(initial_chains_path)
        if not path.is_file():
            raise ChainLoadError(
                f"Initial chain file '{path}' was not found.",
                details={"path": str(path)},
            )
        try:
            chains, = load_chains(
                str(path),
                tokens=training_dataset.tokens,
                device=device,
                dtype=dtype,
            )
        except AdabmDCAError:
            raise
        except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ChainLoadError(
                f"Could not load initial chains from '{path}': {exc}",
                details={"path": str(path)},
            ) from exc
        _check_chain_compatibility(
            chains,
            length=length,
            num_states=num_states,
        )

    return TrainingInputs(
        training=training_dataset,
        training_alignment=training_alignment,
        validation=validation_dataset,
        validation_alignment=validation_alignment,
        initial_params=params,
        initial_chains=chains,
    )
