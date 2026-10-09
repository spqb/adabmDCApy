"""Sequence clustering and cluster-preserving alignment partitioning."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from adabmDCA.exceptions import ComputationError, InputValidationError


def _mmseqs_clusters(sequences: tuple[str, ...], identity: float) -> list[list[int]]:
    """Run MMseqs2 easy-cluster using stable numeric IDs, independent of FASTA headers."""
    with TemporaryDirectory(prefix="adabmdca-cluster-") as directory:
        root = Path(directory)
        source = root / "sequences.fasta"
        with source.open("w") as handle:
            for index, sequence in enumerate(sequences):
                handle.write(f">s{index}\n{sequence}\n")
        output = root / "clusters"
        command = ["mmseqs", "easy-cluster", str(source), str(output), str(root / "tmp"),
                   "--min-seq-id", str(identity), "-v", "1"]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise ComputationError("MMseqs2 easy-cluster failed.",
                                   details={"output": (completed.stdout + completed.stderr).strip()[-2000:]})
        groups: dict[str, list[int]] = {}
        assigned: set[int] = set()
        for line in (root / "clusters_cluster.tsv").read_text().splitlines():
            representative, member = line.split("\t")
            index = int(member[1:])
            if index in assigned:
                raise ComputationError("MMseqs2 assigned a sequence to multiple clusters.")
            assigned.add(index)
            groups.setdefault(representative, []).append(index)
        if assigned != set(range(len(sequences))):
            raise ComputationError("MMseqs2 did not assign every input sequence to a cluster.")
        return list(groups.values())


def _torch_clusters(encoded: np.ndarray, identity: float, seed: int, device: str) -> list[list[int]]:
    """Greedy representative clustering of aligned sequences, in device sized batches."""
    data = torch.as_tensor(encoded, device=device)
    remaining = np.ones(len(encoded), dtype=bool)
    order = np.random.default_rng(seed).permutation(len(encoded))
    groups: list[list[int]] = []
    # Limit the comparison tensor to about eight million elements per batch.
    batch_size = max(1, min(len(encoded), 8_000_000 // max(1, encoded.shape[1])))
    for representative in order:
        if not remaining[representative]:
            continue
        candidates = np.flatnonzero(remaining)
        members: list[int] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            indices = torch.as_tensor(batch, device=data.device)
            matches = (data.index_select(0, indices) == data[representative]).sum(dim=1)
            selected = matches >= identity * data.shape[1] - 1e-7
            members.extend(batch[selected.cpu().numpy()].tolist())
        remaining[members] = False
        groups.append(members)
    return groups


def cluster_sequences(sequences: tuple[str, ...], encoded: np.ndarray, identity: float,
                      seed: int, device: str) -> list[list[int]]:
    """Use MMseqs2 for larger inputs, with an aligned identity fallback on CPU/GPU."""
    if len(sequences) >= 100 and shutil.which("mmseqs"):
        try:
            return _mmseqs_clusters(sequences, identity)
        except (ComputationError, OSError, ValueError):
            # MMseqs2 can reject short or low-complexity alignments.
            pass
    return _torch_clusters(encoded, identity, seed, device)


def partition_clusters(groups: list[list[int]], count: int, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Choose whole clusters to get close to the requested sequence fraction."""
    if len(groups) < 2:
        raise InputValidationError(
            "The identity threshold produced fewer than two clusters; a train/test split is impossible."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    order = sorted(order, key=lambda index: len(groups[index]), reverse=True)
    target = train_fraction * count
    train_groups: set[int] = set()
    train_size = 0
    for index in order:
        size = len(groups[index])
        if abs(train_size + size - target) < abs(train_size - target):
            train_groups.add(index)
            train_size += size
    if not train_groups:
        chosen = min(range(len(groups)), key=lambda index: abs(len(groups[index]) - target))
        train_groups.add(chosen)
    if len(train_groups) == len(groups):
        chosen = min(train_groups, key=lambda index: abs(count - len(groups[index]) - target))
        train_groups.remove(chosen)
    train = sorted(member for index in train_groups for member in groups[index])
    test = sorted(member for index, group in enumerate(groups) if index not in train_groups for member in group)
    return train, test
