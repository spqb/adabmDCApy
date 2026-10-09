"""High-level mutation-scanning operations."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn.functional import one_hot

from adabmDCA.api.model import DCAModel, load_model
from adabmDCA.api.results import MutationRecord, MutationScanResult
from adabmDCA.api.runtime import normalize_sequences
from adabmDCA.exceptions import InputValidationError
from adabmDCA.fasta import decode_sequence, encode_sequence
from adabmDCA.statmech import compute_energy


def scan_mutations(
    wild_type: str,
    *,
    model: DCAModel | str | Path,
    name: str = "wild_type",
    alphabet: str | None = None,
    device: str = "auto",
    dtype: str = "float32",
    include_gap: bool = True,
) -> MutationScanResult:
    """Score every single-site substitution of an aligned wild-type sequence.

    For each position and each alternative token, the mutant's energy is
    compared with the wild type's. With the convention ``E = -sum h - sum J``,
    a negative ``delta_energy`` means the model favours the mutant.

    Args:
        wild_type: Aligned sequence of length ``L`` using the model's tokens.
        model: A :class:`DCAModel`, or a path to a parameter file or PTT archive.
        name: Label stored in the result and used in exported files.
        alphabet: Alphabet of a text parameter file: ``"protein"``, ``"rna"``,
            ``"dna"`` or an ordered custom token string. ``None`` reads it from a
            PTT archive and assumes ``"protein"`` for text files. Ignored when
            ``model`` is already a :class:`DCAModel`.
        device: ``"auto"`` (CUDA when available, else CPU), ``"cpu"``,
            ``"cuda"`` or ``"mps"``. Ignored for an in-memory model.
        dtype: ``"float32"`` or ``"float64"`` for loaded parameters. Ignored for an
            in-memory model.
        include_gap: Whether substitutions to the gap token ``"-"`` are scored.

    Returns:
        A :class:`MutationScanResult` with the wild-type energy and one
        :class:`MutationRecord` per mutant, ``L * (q - 1)`` in total (fewer
        when gaps are excluded).

    Raises:
        InputValidationError: If ``wild_type`` has the wrong length or unknown
            tokens, or fewer than two tokens are eligible.

    Example:
        >>> scan = scan_mutations("MKT...", model="params.dat.gz")
        >>> scan.to_dataframe().sort_values("delta_energy").head()
    """
    loaded = model if isinstance(model, DCAModel) else load_model(model, alphabet=alphabet, device=device, dtype=dtype)
    normalized = normalize_sequences(
        wild_type,
        tokens=loaded.tokens,
        expected_length=loaded.metadata.length,
    )[0]
    target_states = [i for i, token in enumerate(loaded.tokens) if include_gap or token != "-"]
    if len(target_states) < 2:
        raise InputValidationError(
            "Mutation scanning requires at least two eligible alphabet states.",
            details={"tokens": loaded.tokens, "include_gap": include_gap},
        )
    encoded_wt = torch.as_tensor(
        encode_sequence(normalized, loaded.tokens),
        dtype=torch.int64,
        device=loaded.params["bias"].device,
    )
    categorical = []
    descriptors: list[tuple[int, str, str]] = []
    for position in range(len(normalized)):
        for state in target_states:
            if encoded_wt[position].item() == state:
                continue
            mutant = encoded_wt.clone()
            mutant[position] = state
            categorical.append(mutant)
            descriptors.append((position, normalized[position], loaded.tokens[state]))

    mutants = torch.stack(categorical)
    mutants_oh = one_hot(mutants, num_classes=len(loaded.tokens)).to(dtype=loaded.params["bias"].dtype)
    wt_oh = one_hot(encoded_wt.unsqueeze(0), num_classes=len(loaded.tokens)).to(dtype=loaded.params["bias"].dtype)
    energies = compute_energy(mutants_oh, loaded.params)
    wt_energy = compute_energy(wt_oh, loaded.params)[0]
    delta = (energies - wt_energy).detach().cpu().numpy()
    decoded = decode_sequence(mutants.detach().cpu().numpy(), loaded.tokens)

    records = tuple(
        MutationRecord(
            position=position,
            wild_type=old,
            mutant=new,
            sequence=str(sequence),
            delta_energy=float(delta_energy),
        )
        for (position, old, new), sequence, delta_energy in zip(descriptors, decoded, delta)
    )
    return MutationScanResult(
        wild_type=normalized,
        wild_type_energy=float(wt_energy.detach().cpu()),
        mutations=records,
        model=loaded.metadata,
        name=name,
    )
