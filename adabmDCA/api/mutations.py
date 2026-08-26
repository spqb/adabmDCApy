"""High-level mutation-scanning operations."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn.functional import one_hot

from adabmDCA.api.model import DCAModel, load_model
from adabmDCA.api.results import MutationRecord, MutationScanResult
from adabmDCA.api.runtime import normalize_sequences
from adabmDCA.fasta import decode_sequence, encode_sequence
from adabmDCA.statmech import compute_energy


def scan_mutations(
    wild_type: str,
    *,
    model: DCAModel | str | Path,
    name: str = "wild_type",
    alphabet: str = "protein",
    device: str = "auto",
    dtype: str = "float32",
    include_gap: bool = True,
) -> MutationScanResult:
    """Score all single-residue substitutions of ``wild_type``."""
    loaded = model if isinstance(model, DCAModel) else load_model(
        model, alphabet=alphabet, device=device, dtype=dtype
    )
    normalized = normalize_sequences(
        wild_type,
        tokens=loaded.tokens,
        expected_length=loaded.metadata.length,
    )[0]
    target_states = [i for i, token in enumerate(loaded.tokens) if include_gap or token != "-"]
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
    mutants_oh = one_hot(mutants, num_classes=len(loaded.tokens)).to(
        dtype=loaded.params["bias"].dtype
    )
    wt_oh = one_hot(encoded_wt.unsqueeze(0), num_classes=len(loaded.tokens)).to(
        dtype=loaded.params["bias"].dtype
    )
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
