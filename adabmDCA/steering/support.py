"""Compiled Hamming costs, hard support, and an exactly sampled profile."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

from adabmDCA._validation import validate_integer
from adabmDCA.exceptions import InputValidationError


@dataclass(frozen=True)
class HammingWindow:
    """Inclusive substitution-distance bounds on an aligned reference.

    Args:
        reference: Model-length sequence in the model alphabet.
        minimum: Smallest allowed number of substitutions.
        maximum: Largest allowed number of substitutions.
    """

    reference: str
    minimum: int
    maximum: int


@dataclass(frozen=True)
class HammingBias:
    """Reduced energy equal to weight times the distance from reference.

    Args:
        reference: Model-length sequence in the model alphabet.
        weight: Cost per substitution. Positive values favor proximity.
    """

    reference: str
    weight: float


class Support:
    def __init__(self, tokens, length, allowed_tokens, windows):
        self.length = length
        self.q = len(tokens)
        if allowed_tokens is not None and (not allowed_tokens or set(allowed_tokens) - set(tokens)):
            raise InputValidationError("allowed_tokens must be a nonempty subset of the model alphabet.")
        self.allowed = np.array([allowed_tokens is None or token in allowed_tokens for token in tokens])
        self.indices = np.flatnonzero(self.allowed)
        self.references = self.encode_references(tokens, length, [window.reference for window in windows])
        for window in windows:
            validate_integer("Hamming minimum", window.minimum, minimum=0, maximum=length)
            validate_integer("Hamming maximum", window.maximum, minimum=window.minimum, maximum=length)
        self.minimum = np.array([window.minimum for window in windows], dtype=np.int32)
        self.maximum = np.array([window.maximum for window in windows], dtype=np.int32)
        self.increments = np.ascontiguousarray(
            np.arange(self.q)[None, :, None] != self.references.T[:, None, :], dtype=np.int32,
        )
        self.restricted = bool(windows) or not self.allowed.all()

    @staticmethod
    def encode_references(tokens, length, references):
        lookup = {token: index for index, token in enumerate(tokens)}
        if any(len(ref) != length or set(ref) - lookup.keys() for ref in references):
            raise InputValidationError("Hamming references must match the model length and alphabet.")
        return np.array([[lookup[token] for token in ref] for ref in references], dtype=np.int32).reshape(-1, length)

    def distances(self, states):
        return np.count_nonzero(states[:, None, :] != self.references[None], axis=2).astype(np.int32)

    def contains(self, states):
        distances = self.distances(states)
        return self.allowed[states].all(axis=1) & (
            (distances >= self.minimum) & (distances <= self.maximum)
        ).all(axis=1)


class ExactProfile:
    """Suffix log partitions and independent samples from a constrained profile."""

    def __init__(self, fields, support):
        self.fields = fields.astype(np.float64, copy=False)
        self.support = support
        shape = tuple(int(bound) + 1 for bound in support.maximum)
        if (support.length + 1) * prod(shape) > 16_000_000:
            raise InputValidationError("The joint Hamming count table is too large. Use fewer or narrower constraints.")
        self.table = np.full((support.length + 1, *shape), -np.inf)
        terminal = tuple(slice(lo, hi + 1) for lo, hi in zip(support.minimum, support.maximum))
        self.table[(-1, *terminal)] = 0
        for site in range(support.length - 1, -1, -1):
            for letter in support.indices:
                inc = support.increments[site, letter]
                before = tuple(slice(None, size - int(step)) for size, step in zip(shape, inc))
                after = tuple(slice(int(step), None) for step in inc)
                target = (site, *before)
                self.table[target] = np.logaddexp(
                    self.table[target], self.fields[site, letter] + self.table[(site + 1, *after)]
                )
        self.log_z = float(self.table[(0, *(0 for _ in shape))])
        if not np.isfinite(self.log_z):
            raise InputValidationError("The Hamming windows and allowed alphabet have empty support.")

    def draw(self, size, rng):
        support = self.support
        if not len(support.maximum):
            logits = self.fields[:, support.indices]
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            cdf = np.cumsum(weights, axis=1)
            indices = np.sum(rng.random((size, support.length, 1)) * cdf[:, -1:] >= cdf, axis=-1)
            return support.indices[indices].astype(np.int32)
        totals = np.zeros((size, len(support.maximum)), dtype=np.int32)
        states = np.empty((size, support.length), dtype=np.int32)
        for site in range(support.length):
            proposed = totals[:, None, :] + support.increments[site, support.indices]
            valid = (proposed <= support.maximum).all(axis=-1)
            safe = np.minimum(proposed, support.maximum)
            suffix = self.table[(site + 1, *np.moveaxis(safe, -1, 0))]
            logits = np.where(valid, suffix + self.fields[site, support.indices], -np.inf)
            cdf = np.cumsum(np.exp(logits - logits.max(axis=1, keepdims=True)), axis=1)
            letter = support.indices[np.sum(rng.random((size, 1)) * cdf[:, -1:] >= cdf, axis=1)]
            states[:, site] = letter
            totals += support.increments[site, letter]
        return states
