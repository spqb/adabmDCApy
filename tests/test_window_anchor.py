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
    increments = (np.arange(3)[None] != reference[:, None]).astype(int)
    if kind == "forced":
        increments[2] = 1
    restricted = WindowedAnchor(LowRankPotts(model.fields, model.weights) if kind == "low-rank" else model,
                                increments, minimum, maximum)
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
    increments = np.ones((6, 3), dtype=int)
    with pytest.raises(ValueError):
        WindowedAnchor(anchor(1, CHAIN, 0), increments, 0, 5)
