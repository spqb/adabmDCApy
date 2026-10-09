"""High-level API for homology-aware alignment splitting."""

from __future__ import annotations

import numpy as np
import torch

from adabmDCA._validation import validate_integer, validate_seed
from adabmDCA.alignment import Alignment
from adabmDCA.api.results import ProfileSplitResult
from adabmDCA.cobalt import run_cobalt
from adabmDCA.exceptions import ConvergenceError, InputValidationError
from adabmDCA.fasta import decode_sequence
from adabmDCA.input_loading import AlignmentInput, AlignmentLoadConfig, load_alignment
from adabmDCA.utils import get_device


def split_alignment(
    alignment: AlignmentInput,
    *,
    method: str = "clustering",
    identity: float = 0.8,
    train_fraction: float = 0.8,
    t1: float = 0.5,
    t2: float = 0.5,
    t3: float = 1.0,
    max_train: int | None = None,
    max_test: int | None = None,
    attempts: int = 1,
    alphabet: str = "protein",
    seed: int = 0,
    device: str = "auto",
) -> ProfileSplitResult:
    """Split an alignment into training and test sets that are not too similar.

    ``method="clustering"`` clusters sequences at ``identity`` (MMseqs2 when
    installed and there are at least 100 sequences, otherwise a built-in greedy
    clustering) and assigns whole clusters to the training set until about
    ``train_fraction`` of the sequences are in it. ``method="cobalt"`` runs the
    Cobalt algorithm (Petti and Eddy, 2022), whose thresholds ``t1``-``t3`` bound
    the identities between and within the two sets.

    Args:
        alignment: Alignment to split (path, :class:`Alignment` or sequences).
        method: ``"clustering"`` or ``"cobalt"``.
        identity: Clustering: sequence identity, in ``(0, 1]``, that joins two
            sequences into one cluster.
        train_fraction: Clustering: target fraction of sequences in the training set.
        t1: Cobalt: no test sequence is more than ``t1`` identical to a training sequence.
        t2: Cobalt: no two test sequences are more than ``t2`` identical.
        t3: Cobalt: no two training sequences are more than ``t3`` identical.
        max_train: Cobalt: largest training set, or ``None`` for no limit.
        max_test: Cobalt: largest test set, or ``None`` for no limit.
        attempts: Cobalt: random attempts; the one maximizing
            ``len(training) * len(test)`` is kept.
        alphabet: ``"protein"``, ``"rna"``, ``"dna"`` or a custom token string.
        seed: Random seed.
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"`` for identity computations.

    Returns:
        A :class:`ProfileSplitResult` with the two alignments.

    Raises:
        InputValidationError: If a setting is out of range or the alignment has
            invalid sequences.
        ConvergenceError: If Cobalt finds no split with both sets non-empty.

    Example:
        >>> split = split_alignment("family.fasta", alphabet="rna", identity=0.8)
        >>> split.save_bundle("splits/family")
    """
    if method not in {"clustering", "cobalt"}:
        raise InputValidationError("method must be clustering or cobalt.")
    if not 0.0 < identity <= 1.0:
        raise InputValidationError("identity must be greater than 0 and at most 1.")
    if not 0.0 < train_fraction < 1.0:
        raise InputValidationError("train_fraction must be between 0 and 1 (exclusive).")
    validate_integer("attempts", attempts)
    validate_seed(seed)
    for name, value in {"max_train": max_train, "max_test": max_test}.items():
        if value is not None:
            validate_integer(name, value)
    for name, threshold in {"t1": t1, "t2": t2, "t3": t3}.items():
        if not 0.0 <= threshold <= 1.0:
            raise InputValidationError(f"{name} must be between 0 and 1.")

    loaded = load_alignment(
        alignment,
        config=AlignmentLoadConfig(alphabet=alphabet, invalid_sequences="error"),
    )
    selected_device = get_device(device, message=False)
    if method == "clustering":
        from adabmDCA.clustering import cluster_sequences, partition_clusters

        groups = cluster_sequences(
            tuple(loaded.sequences), loaded.encoded_sequences, identity, seed, selected_device
        )
        train_indices, test_indices = partition_clusters(groups, len(loaded.names), train_fraction, seed)
        return ProfileSplitResult(
            training=Alignment(
                tuple(loaded.names[index] for index in train_indices),
                tuple(loaded.sequences[index] for index in train_indices),
            ),
            test=Alignment(
                tuple(loaded.names[index] for index in test_indices),
                tuple(loaded.sequences[index] for index in test_indices),
            ),
            score=len(train_indices) * len(test_indices),
            attempts=1,
            tokens=loaded.tokens,
            seed=seed,
            method=method,
            identity=identity,
            train_fraction=train_fraction,
        )
    headers = np.asarray(loaded.names)
    encoded = torch.as_tensor(loaded.encoded_sequences, device=selected_device)
    generator = torch.Generator(device=selected_device).manual_seed(seed)

    best = None
    best_score = 0
    for _ in range(attempts):
        candidate = run_cobalt(
            headers,
            encoded,
            t1,
            t2,
            t3,
            max_train,
            max_test,
            generator,
        )
        score = len(candidate[1]) * len(candidate[3])
        if score > best_score:
            best = candidate
            best_score = score
    if best is None:
        raise ConvergenceError(
            "Cobalt could not find a non-empty training/test split.",
            details={"t1": t1, "t2": t2, "t3": t3, "attempts": attempts},
        )

    train_names, train, test_names, test = best
    return ProfileSplitResult(
        training=Alignment(
            tuple(map(str, train_names)),
            tuple(decode_sequence(train, loaded.tokens)),
        ),
        test=Alignment(
            tuple(map(str, test_names)),
            tuple(decode_sequence(test, loaded.tokens)),
        ),
        score=best_score,
        attempts=attempts,
        tokens=loaded.tokens,
        seed=seed,
        method="cobalt",
    )
