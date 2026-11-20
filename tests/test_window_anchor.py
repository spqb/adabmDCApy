from itertools import product

import numpy as np
import pytest
import torch

from adabmDCA.anchor.anchor import TreeLowRankPotts
from adabmDCA.anchor.lowrank import LowRankPotts
from adabmDCA.anchor.window import WindowedAnchor

CHAIN = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)]


def anchor(rank, edges, seed):
    generator = torch.Generator().manual_seed(seed)
    fields = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    weights = 0.6 * torch.randn(6, 3, rank, generator=generator, dtype=torch.float64)
    pairs = torch.randn(len(edges), 3, 3, generator=generator, dtype=torch.float64)
    return TreeLowRankPotts(fields, weights, torch.tensor(edges, dtype=torch.long).reshape(-1, 2), pairs)


@pytest.mark.parametrize("rank, edges, minimum, maximum, kind", [
    (0, CHAIN, 2, 3, "tree"),
    (2, CHAIN, 1, 3, "tree"),
    (2, CHAIN, 4, 6, "far"),            # counts matches instead of mismatches
    (1, [(0, 1), (2, 3)], 2, 2, "forest"),
    (2, [], 0, 1, "low-rank"),
    (2, CHAIN, 3, 4, "forced"),         # the reference letter is excluded at one site
])
def test_window_matches_enumeration(rank, edges, minimum, maximum, kind):
    model = anchor(rank, edges, seed=rank + minimum + 7 * maximum)
    reference = np.array([0, 2, 1, 1, 0, 2])
    if kind == "forced":
        reference[2] = 3  # a letter outside the anchor's alphabet: a substitution at every state
    increments = (np.arange(3)[None] != reference[:, None]).astype(int)
    restricted = WindowedAnchor(LowRankPotts(model.fields, model.weights) if kind == "low-rank" else model,
                                [reference], [minimum], [maximum])
    states = torch.tensor(list(product(range(3), repeat=6)))
    distances = increments[np.arange(6), states.numpy()].sum(1)
    inside = (distances >= minimum) & (distances <= maximum)
    scores = model._scores(states).numpy()[inside]
    log_z = np.logaddexp.reduce(scores)
    assert restricted.log_z == pytest.approx(log_z, abs=1e-7)
    torch.manual_seed(3)
    drawn = restricted.sample(200_000).numpy()
    codes = drawn @ 3 ** np.arange(5, -1, -1)
    frequencies = np.bincount(codes, minlength=3 ** 6) / len(drawn)
    assert frequencies[~inside].sum() == 0
    np.testing.assert_allclose(frequencies[inside], np.exp(scores - log_z), atol=4e-3)


def test_empty_window_is_rejected():
    with pytest.raises(ValueError):
        WindowedAnchor(anchor(1, CHAIN, 0), [[3] * 6], [0], [5])
    # Pairwise feasible but jointly empty: 0 substitutions from both of two different references.
    with pytest.raises(ValueError):
        WindowedAnchor(anchor(1, CHAIN, 0), [[0] * 6, [1] + [0] * 5], [0, 0], [0, 0])


@pytest.mark.parametrize("rank, edges, references, minimum, maximum, tokens", [
    (2, CHAIN, [[0, 2, 1, 1, 0, 2], [1, 2, 0, 1, 0, 0]], [1, 2], [3, 4], None),
    (1, [(0, 1), (2, 3)], [[0, 2, 1, 1, 0, 2], [1, 2, 0, 1, 0, 0]], [2, 2], [2, 3], None),     # forest, exact shell
    (2, [], [[0, 2, 1, 1, 0, 2], [1, 2, 0, 1, 0, 0]], [0, 3], [3, 6], None),                  # low rank, far window
    (0, CHAIN, [[0, 2, 1, 1, 0, 2], [1, 2, 0, 1, 0, 0], [1, 1, 0, 1, 2, 2]], [0, 1, 2], [3, 4, 4], None),
    (2, CHAIN, [[0, 2, 1, 1, 0, 2], [1, 2, 0, 1, 0, 0], [1, 2, 0, 1, 0, 0]], [1, 0, 2], [4, 3, 3], None),
    (1, CHAIN, [[0, 2, 1, 1, 0, 2], [1, 3, 0, 1, 3, 0]], [1, 2], [3, 4], [0, 1, 2]),          # masked letters
    (2, CHAIN, [[0, 4, 1, 1, 0, 2], [1, 2, 0, 3, 0, 0]], [2, 1], [4, 3], [0, 2, 4]),          # token map
    (2, CHAIN, [[0, 2, 1, 1, 0, 2], [0, 2, 1, 1, 0, 2], [1, 2, 0, 1, 0, 0]], [0, 1, 0], [3, 4, 6], None),  # merged, implied
    (1, CHAIN, [[0, 3, 1, 1, 0, 2], [0, 4, 1, 1, 0, 2]], [1, 2], [3, 3], None),               # same letters inside
])
def test_several_windows_match_enumeration(rank, edges, references, minimum, maximum, tokens):
    model = anchor(rank, edges, seed=sum(minimum) + 11 * sum(maximum))
    if not edges:
        model = LowRankPotts(model.fields, model.weights)
    restricted = WindowedAnchor(model, references, minimum, maximum, tokens=tokens)
    tree = restricted.anchor
    states = torch.tensor(list(product(range(3), repeat=6)))
    letters = np.arange(3) if tokens is None else np.array(tokens)
    distances = (letters[states.numpy()][None] != np.array(references)[:, None]).sum(-1)
    inside = ((distances >= np.array(minimum)[:, None]) & (distances <= np.array(maximum)[:, None])).all(0)
    scores = tree._scores(states).numpy()[inside]
    log_z = np.logaddexp.reduce(scores)
    assert restricted.log_z == pytest.approx(log_z, abs=1e-7)
    torch.manual_seed(5)
    drawn = restricted.sample(200_000).numpy()
    frequencies = np.bincount(drawn @ 3 ** np.arange(5, -1, -1), minlength=3 ** 6) / len(drawn)
    assert frequencies[~inside].sum() == 0
    np.testing.assert_allclose(frequencies[inside], np.exp(scores - log_z), atol=4e-3)
