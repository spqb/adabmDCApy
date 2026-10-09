"""High-level contact-prediction operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adabmDCA.api.model import DCAModel, load_model
from adabmDCA.api.results import ContactMapResult
from adabmDCA.api.runtime import resolve_runtime
from adabmDCA.dataset import DatasetDCA
from adabmDCA.dca import get_contact_map, get_mf_contact_map
from adabmDCA.exceptions import InputValidationError
from adabmDCA.fasta import get_tokens
from adabmDCA.input_loading import AlignmentInput, AlignmentLoadConfig


def predict_contacts(
    *,
    model: DCAModel | str | Path | None = None,
    fasta_path: AlignmentInput | None = None,
    alphabet: str | None = None,
    pseudocount: float = 0.5,
    device: str = "auto",
    dtype: str = "float32",
) -> ContactMapResult:
    """Predict residue contacts from a DCA model or directly from an alignment.

    With ``model``, scores are the average-product-corrected (APC) Frobenius
    norms of the zero-sum-gauge couplings, excluding the gap state. With
    ``fasta_path``, a mean-field DCA model is inferred from the reweighted
    alignment and scored the same way. Larger scores mean likelier contacts.

    Args:
        model: A :class:`DCAModel`, or a path to a parameter file or PTT archive.
        fasta_path: Aligned sequences (path, :class:`Alignment` or sequence
            strings) for mean-field prediction. Provide exactly one of ``model``
            and ``fasta_path``.
        alphabet: ``"protein"``, ``"rna"``, ``"dna"`` or an ordered custom token
            string, for the alignment or a text parameter file. ``None`` reads it
            from a PTT archive and otherwise means ``"protein"``. Ignored when
            ``model`` is already a :class:`DCAModel`. Must contain ``"-"``.
        pseudocount: Mean-field only: pseudocount in ``[0, 1]`` that regularizes the
            covariance matrix before inversion.
        device: ``"auto"`` (CUDA when available, else CPU), ``"cpu"``,
            ``"cuda"`` or ``"mps"``. Ignored for an in-memory model.
        dtype: ``"float32"`` or ``"float64"`` for loaded parameters. Ignored for an
            in-memory model.

    Returns:
        A :class:`ContactMapResult` whose ``scores`` is a symmetric ``(L, L)``
        array with a zero diagonal.

    Raises:
        InputValidationError: If both or neither of ``model`` and ``fasta_path``
            are given, the alphabet lacks the ``"-"`` gap token, or the
            pseudocount is outside ``[0, 1]``.

    Example:
        >>> result = predict_contacts(model="params.dat.gz", alphabet="rna")
        >>> top = result.to_long_dataframe().nlargest(20, "score")  # 0-based positions
    """
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
    alphabet = "protein" if alphabet is None else alphabet
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
    alphabet: str | None = None,
    pseudocount: float = 0.5,
    device: str = "auto",
    dtype: str = "float32",
) -> np.ndarray:
    """Return only the ``(L, L)`` contact-score matrix of :func:`predict_contacts`.

    Takes the same arguments as :func:`predict_contacts`; see it for details.

    Returns:
        A symmetric NumPy array of APC-corrected scores with a zero diagonal.
    """
    return predict_contacts(
        model=model,
        fasta_path=fasta_path,
        alphabet=alphabet,
        pseudocount=pseudocount,
        device=device,
        dtype=dtype,
    ).scores
