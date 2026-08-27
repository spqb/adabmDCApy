"""High-level sequence-energy operations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import one_hot

from adabmDCA.api.model import DCAModel, load_model
from adabmDCA.api.results import EnergyResult
from adabmDCA.api.runtime import load_fasta_sequences, normalize_sequences
from adabmDCA.fasta import encode_sequence
from adabmDCA.input_loading import AlignmentInput
from adabmDCA.statmech import compute_energy


def _coerce_model(
    model: DCAModel | str | Path,
    *,
    alphabet: str,
    device: str,
    dtype: str,
) -> DCAModel:
    return model if isinstance(model, DCAModel) else load_model(model, alphabet=alphabet, device=device, dtype=dtype)


def score_sequences(
    sequences: str | Iterable[str] | None = None,
    *,
    model: DCAModel | str | Path,
    fasta_path: AlignmentInput | None = None,
    alphabet: str = "protein",
    device: str = "auto",
    dtype: str = "float32",
    remove_duplicates: bool = False,
) -> EnergyResult:
    """Compute DCA energies and return a structured result.

    Provide exactly one of ``sequences`` or ``fasta_path``. When ``model`` is a
    :class:`DCAModel`, its alphabet and runtime configuration are reused.
    """
    if (sequences is None) == (fasta_path is None):
        from adabmDCA.api.exceptions import InputValidationError

        raise InputValidationError("Provide exactly one of 'sequences' or 'fasta_path'.")

    loaded = _coerce_model(model, alphabet=alphabet, device=device, dtype=dtype)
    if fasta_path is not None:
        names, normalized = load_fasta_sequences(
            fasta_path,
            alphabet=loaded.alphabet,
            expected_length=loaded.metadata.length,
            remove_duplicates=remove_duplicates,
        )
    else:
        names = ()
        normalized = normalize_sequences(sequences, tokens=loaded.tokens, expected_length=loaded.metadata.length)

    encoded = torch.as_tensor(
        encode_sequence(list(normalized), loaded.tokens),
        dtype=torch.int64,
        device=loaded.params["bias"].device,
    )
    data = one_hot(encoded, num_classes=len(loaded.tokens)).to(dtype=loaded.params["bias"].dtype)
    energies = compute_energy(data, loaded.params).detach().cpu().numpy()
    return EnergyResult(
        sequences=normalized,
        energies=energies,
        model=loaded.metadata,
        names=names,
    )


def compute_energies(
    sequences: str | Iterable[str] | None = None,
    *,
    model: DCAModel | str | Path,
    fasta_path: AlignmentInput | None = None,
    alphabet: str = "protein",
    device: str = "auto",
    dtype: str = "float32",
    remove_duplicates: bool = False,
) -> np.ndarray:
    """Notebook-friendly shortcut returning only the energy vector."""
    return score_sequences(
        sequences,
        model=model,
        fasta_path=fasta_path,
        alphabet=alphabet,
        device=device,
        dtype=dtype,
        remove_duplicates=remove_duplicates,
    ).energies
