"""High-level contact-prediction operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adabmDCA.api.exceptions import InputValidationError
from adabmDCA.api.model import DCAModel, load_model
from adabmDCA.api.results import ContactMapResult
from adabmDCA.api.runtime import resolve_runtime
from adabmDCA.dataset import DatasetDCA
from adabmDCA.dca import get_contact_map, get_mf_contact_map
from adabmDCA.fasta import get_tokens
from adabmDCA.input_loading import AlignmentInput, AlignmentLoadConfig


def predict_contacts(
    *,
    model: DCAModel | str | Path | None = None,
    fasta_path: AlignmentInput | None = None,
    alphabet: str = "protein",
    pseudocount: float = 0.5,
    device: str = "auto",
    dtype: str = "float32",
) -> ContactMapResult:
    """Compute APC-corrected contact scores from a model or an alignment."""
    if (model is None) == (fasta_path is None):
        raise InputValidationError("Provide exactly one of 'model' or 'fasta_path'.")

    if model is not None:
        loaded = (
            model if isinstance(model, DCAModel) else load_model(model, alphabet=alphabet, device=device, dtype=dtype)
        )
        if "-" not in loaded.tokens:
            raise InputValidationError(
                "Contact prediction requires an alphabet containing the '-' gap token.",
                details={"tokens": loaded.tokens},
            )
        scores = get_contact_map(loaded.params, loaded.tokens)
        return ContactMapResult(
            scores=scores,
            method="model",
            alphabet=loaded.alphabet,
            tokens=loaded.tokens,
            model=loaded.metadata,
        )

    if not 0.0 <= pseudocount <= 1.0:
        raise InputValidationError("pseudocount must be between 0 and 1.")
    resolved_device, resolved_dtype = resolve_runtime(device, dtype)
    tokens = get_tokens(alphabet)
    if "-" not in tokens:
        raise InputValidationError(
            "Mean-field contact prediction requires an alphabet containing the '-' gap token.",
            details={"tokens": tokens},
        )
    dataset = DatasetDCA.from_alignment(
        fasta_path,
        load_config=AlignmentLoadConfig(
            alphabet=alphabet,
            invalid_sequences="drop",
            remove_duplicates=True,
        ),
        device=resolved_device,
        dtype=resolved_dtype,
    )
    scores = get_mf_contact_map(
        dataset.to_one_hot(),
        tokens=tokens,
        weights=dataset.weights,
        pseudo_count=pseudocount,
    )
    return ContactMapResult(
        scores=scores,
        method="mean_field",
        alphabet=alphabet,
        tokens=tokens,
    )


def compute_contact_map(
    *,
    model: DCAModel | str | Path | None = None,
    fasta_path: AlignmentInput | None = None,
    alphabet: str = "protein",
    pseudocount: float = 0.5,
    device: str = "auto",
    dtype: str = "float32",
) -> np.ndarray:
    """Notebook-friendly shortcut returning only the contact-score matrix."""
    return predict_contacts(
        model=model,
        fasta_path=fasta_path,
        alphabet=alphabet,
        pseudocount=pseudocount,
        device=device,
        dtype=dtype,
    ).scores
