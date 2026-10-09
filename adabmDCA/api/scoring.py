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
from adabmDCA.statmech import compute_energy, get_cde


def summed_cde(data: torch.Tensor, params: dict[str, torch.Tensor], *, batch_size: int = 64) -> np.ndarray:
    """Return summed conditional entropy for each one-hot sequence, in nats."""
    values = []
    for start in range(0, len(data), batch_size):
        values.append(get_cde(data[start:start + batch_size], params).sum(dim=-1).detach().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def fit_local_lambda(energies: np.ndarray, cde_sum: np.ndarray) -> dict[str, float | int]:
    """Fit E = intercept + lambda * summed CDE by ordinary least squares."""
    from adabmDCA.exceptions import InputValidationError

    energy = np.asarray(energies, dtype=np.float64)
    entropy = np.asarray(cde_sum, dtype=np.float64)
    if energy.ndim != 1 or entropy.shape != energy.shape or len(energy) < 2:
        raise InputValidationError("Lambda fitting requires at least two paired energies and CDE sums.")
    if not np.isfinite(energy).all() or not np.isfinite(entropy).all():
        raise InputValidationError("Lambda fitting requires finite energies and CDE sums.")
    centered_entropy = entropy - entropy.mean()
    variance = np.dot(centered_entropy, centered_entropy)
    if variance <= 0:
        raise InputValidationError("Lambda cannot be fitted because all sampled CDE sums are equal.")
    slope = float(np.dot(centered_entropy, energy - energy.mean()) / variance)
    intercept = float(energy.mean() - slope * entropy.mean())
    residual = energy - (intercept + slope * entropy)
    total = float(np.dot(energy - energy.mean(), energy - energy.mean()))
    r_squared = float(1 - np.dot(residual, residual) / total) if total > 0 else 1.0
    return {"lambda": slope, "intercept": intercept, "r_squared": r_squared, "n_samples": len(energy)}


def _coerce_model(
    model: DCAModel | str | Path,
    *,
    alphabet: str | None,
    device: str,
    dtype: str,
) -> DCAModel:
    return model if isinstance(model, DCAModel) else load_model(model, alphabet=alphabet, device=device, dtype=dtype)


def score_sequences(
    sequences: str | Iterable[str] | None = None,
    *,
    model: DCAModel | str | Path,
    fasta_path: AlignmentInput | None = None,
    alphabet: str | None = None,
    device: str = "auto",
    dtype: str = "float32",
    remove_duplicates: bool = False,
    local_lambda: float | None = 1.0,
) -> EnergyResult:
    """Compute DCA energies of aligned sequences, with their local free energies.

    Energies follow ``E(s) = -sum_i h_i(s_i) - sum_{i<j} J_ij(s_i, s_j)``: lower is
    more probable under the model. Unless ``local_lambda`` is ``None``, each
    sequence also gets its summed context-dependent entropy (CDE) and the local
    free energy ``E - local_lambda * CDE``.

    Args:
        sequences: One aligned sequence or an iterable of them, each of length
            ``L`` over the model's tokens. Provide exactly one of ``sequences``
            and ``fasta_path``.
        model: A :class:`DCAModel`, or a path to a parameter file or PTT archive.
        fasta_path: Alignment to score (path, :class:`Alignment` or sequences);
            sequence names are kept in the result.
        alphabet: Alphabet of a text parameter file: ``"protein"``, ``"rna"``,
            ``"dna"`` or an ordered custom token string. ``None`` reads it from a
            PTT archive and assumes ``"protein"`` for text files. Ignored when
            ``model`` is already a :class:`DCAModel`.
        device: ``"auto"`` (CUDA when available, else CPU), ``"cpu"``,
            ``"cuda"`` or ``"mps"``. Ignored for an in-memory model.
        dtype: ``"float32"`` or ``"float64"`` for loaded parameters. Ignored for an
            in-memory model.
        remove_duplicates: With ``fasta_path``, score each distinct sequence once.
        local_lambda: Weight of the summed CDE in the local free energy, or
            ``None`` to skip the CDE computation.

    Returns:
        An :class:`EnergyResult` with one energy per sequence, in input order.

    Raises:
        InputValidationError: If both or neither of ``sequences`` and
            ``fasta_path`` are given, a sequence is incompatible with the model,
            or ``local_lambda`` is not finite.

    Example:
        >>> result = score_sequences(fasta_path="designs.fasta", model="params.dat.gz")
        >>> result.to_dataframe().sort_values("energy").head()
    """
    if (sequences is None) == (fasta_path is None):
        from adabmDCA.exceptions import InputValidationError

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
    if local_lambda is not None and not np.isfinite(local_lambda):
        from adabmDCA.exceptions import InputValidationError

        raise InputValidationError("local_lambda must be a finite number.")
    energies = compute_energy(data, loaded.params).detach().cpu().numpy()
    cde_sum = summed_cde(data, loaded.params) if local_lambda is not None else None
    return EnergyResult(
        sequences=normalized,
        energies=energies,
        model=loaded.metadata,
        names=names,
        cde_sum=cde_sum,
        local_free_energies=None if cde_sum is None else energies - local_lambda * cde_sum,
        local_lambda=local_lambda,
    )


def compute_energies(
    sequences: str | Iterable[str] | None = None,
    *,
    model: DCAModel | str | Path,
    fasta_path: AlignmentInput | None = None,
    alphabet: str | None = None,
    device: str = "auto",
    dtype: str = "float32",
    remove_duplicates: bool = False,
) -> np.ndarray:
    """Return only the DCA energies of :func:`score_sequences`, without CDE.

    Takes the same arguments as :func:`score_sequences` except ``local_lambda``.

    Returns:
        A one-dimensional NumPy array with one energy per sequence, also for a
        single sequence string.

    Example:
        >>> compute_energies(["ACGU...", "ACGA..."], model=model)
        array([-212.4, -208.9])
    """
    return score_sequences(
        sequences,
        model=model,
        fasta_path=fasta_path,
        alphabet=alphabet,
        device=device,
        dtype=dtype,
        remove_duplicates=remove_duplicates,
        local_lambda=None,
    ).energies
