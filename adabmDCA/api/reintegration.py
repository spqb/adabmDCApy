"""High-level orchestration for experimental sequence reintegration."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from adabmDCA.alignment import Alignment
from adabmDCA.api.results import ReintegrationResult, TrainingProgress
from adabmDCA.api.training import train_model
from adabmDCA.dataset import DatasetDCA
from adabmDCA.exceptions import InputValidationError
from adabmDCA.input_loading import (
    AlignmentInput,
    AlignmentLoadConfig,
    WeightInput,
    load_alignment,
    load_sequence_weights,
)
from adabmDCA.training_config import TrainingConfig
from adabmDCA.utils import get_device, get_dtype


def reintegrate_model(
    natural_alignment: AlignmentInput,
    experimental_alignment: AlignmentInput,
    adjustments: WeightInput,
    *,
    config: TrainingConfig | None = None,
    natural_weights: WeightInput | None = None,
    lambda_value: float | None = None,
    output_dir: str | Path | None = None,
    label: str | None = None,
    initial_params_path: str | Path | None = None,
    initial_chains_path: str | Path | None = None,
    progress: Callable[[TrainingProgress], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ReintegrationResult:
    """Train a model on natural sequences plus experimentally scored ones.

    Experimental sequences enter the training statistics with signed weights
    proportional to their ``adjustments``: positive values pull the model
    towards a sequence, negative values push it away. The experimental weights
    are scaled by ``lambda_value * Meff / n_experimental``, so ``lambda_value``
    sets the weight of the experimental data relative to the natural alignment.
    Training uses PCD (PTT does not support signed weights).

    Args:
        natural_alignment: Natural sequences (path, :class:`Alignment` or sequences).
        experimental_alignment: Tested sequences, aligned like the natural ones.
        adjustments: One signed score per experimental sequence (path, array or
            sequence of numbers), e.g. +1 for functional and -1 for non-functional.
        config: Training settings; defaults to ``TrainingConfig()``.
            A pseudocount of 1e-6 (0.1 for edgeDCA) is used unless set.
        natural_weights: Optional weights of the natural sequences; computed
            by sequence-identity reweighting otherwise.
        lambda_value: Relative weight of the experimental data; defaults to
            ``1 / max(|adjustments|)``.
        output_dir: If given, training files and the combined alignment are written there.
        label: File-name prefix; the written label is ``<label>-lambda_<value>``.
        initial_params_path: Optional parameter file to start from.
        initial_chains_path: Optional FASTA of starting chains.
        progress: Optional callback receiving a :class:`TrainingProgress` per update.
        is_cancelled: Optional callable; returning ``True`` stops training.

    Returns:
        A :class:`ReintegrationResult` with the training result, the combined
        alignment and its weights.

    Raises:
        InputValidationError: If all adjustments are zero, ``lambda_value`` is
            not positive, or the alignments are incompatible.
    """
    selected = config or TrainingConfig()
    device = get_device(selected.device, message=False)
    dtype = get_dtype(selected.dtype)
    policy = AlignmentLoadConfig(
        alphabet=selected.alphabet,
        invalid_sequences="drop",
        remove_duplicates=True,
    )
    natural = load_alignment(natural_alignment, config=policy)
    experimental = load_alignment(
        experimental_alignment,
        config=replace(policy, expected_length=natural.sequence_length),
    )
    natural_dataset = DatasetDCA.from_alignment(
        natural,
        weights=natural_weights,
        clustering_th=selected.clustering_seqid,
        no_reweighting=selected.no_reweighting,
        device=device,
        dtype=dtype,
    )
    adjustment_values = load_sequence_weights(
        adjustments,
        loaded_alignment=experimental,
        no_reweighting=False,
        clustering_seqid=selected.clustering_seqid,
        device=device,
        dtype=dtype,
        allow_negative=True,
        require_positive_sum=False,
    )
    span = float(adjustment_values.abs().max().item())
    if span == 0.0:
        raise InputValidationError("The adjustment vector cannot contain only zeros.")
    resolved_lambda = 1.0 / span if lambda_value is None else float(lambda_value)
    if not math.isfinite(resolved_lambda) or resolved_lambda <= 0:
        raise InputValidationError("lambda_value must be positive.")
    scaling_factor = resolved_lambda * natural_dataset.get_effective_size() / len(experimental)

    combined = Alignment(
        names=natural.names + experimental.names,
        sequences=natural.sequences + experimental.sequences,
    )
    combined_weights = torch.cat((natural_dataset.weights, scaling_factor * adjustment_values)).detach().cpu().numpy()
    base_label = label or "reintegrated"
    resolved_label = f"{base_label}-lambda_{resolved_lambda:.2f}"
    training_config = replace(
        selected,
        no_reweighting=False,
        pseudocount=(
            selected.pseudocount
            if selected.pseudocount is not None
            else (0.1 if selected.model_type == "edgeDCA" else 1e-6)
        ),
    )
    training = train_model(
        combined,
        config=training_config,
        weights_path=combined_weights,
        output_dir=output_dir,
        label=resolved_label,
        initial_params_path=initial_params_path,
        initial_chains_path=initial_chains_path,
        allow_signed_weights=True,
        progress=progress,
        is_cancelled=is_cancelled,
    )
    result = ReintegrationResult(
        training=training,
        alignment=combined,
        weights=np.asarray(combined_weights),
        lambda_value=resolved_lambda,
        scaling_factor=scaling_factor,
        label=resolved_label,
    )
    if output_dir is None:
        return result
    artifacts = result.save_bundle(output_dir)
    return replace(result, artifacts=artifacts)
