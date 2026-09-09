"""High-level API for profile-model alignment splitting."""

from __future__ import annotations

import numpy as np
import torch

from adabmDCA._validation import validate_integer, validate_seed
from adabmDCA.alignment import Alignment
from adabmDCA.api.exceptions import ConvergenceError, InputValidationError
from adabmDCA.api.results import ProfileSplitResult
from adabmDCA.cobalt import run_cobalt
from adabmDCA.fasta import decode_sequence
from adabmDCA.input_loading import AlignmentInput, AlignmentLoadConfig, load_alignment
from adabmDCA.utils import get_device


def split_alignment(
    alignment: AlignmentInput,
    *,
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
    """Split an alignment and retain the best Cobalt partition."""
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
    headers = np.asarray(loaded.alignment.names)
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
    )
