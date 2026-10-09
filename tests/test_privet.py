from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adabmDCA.api.results import SamplingResult
from adabmDCA.api.sampling import _distance_comparison
from adabmDCA.privet import LEAK_THRESHOLD, fit_nearest_distances, privet


def _family(rng, n, length=60, q=4, ancestors=20, rate=0.25):
    root = np.random.default_rng(0).integers(q, size=(ancestors, length))
    states = root[rng.integers(ancestors, size=n)].copy()
    mutated = rng.random((n, length)) < rate
    states[mutated] = rng.integers(q, size=mutated.sum())
    return states


def _nearest(a, b, same=False):
    distance = (a[:, None, :] != b[None]).sum(-1)
    if same:
        np.fill_diagonal(distance, a.shape[1] + 1)
    return distance.min(1) / a.shape[1]


def test_weibull_fit_recovers_discretized_law():
    rng = np.random.default_rng(3)
    resolution = 1 / 200
    # P(d <= u) = 1 - exp(-scale * u**shape), drawn continuously and rounded to the grid.
    scale, shape = 400.0, 3.0
    continuous = (rng.exponential(size=20_000) / scale) ** (1 / shape)
    distances = np.round(continuous / resolution) * resolution

    fit = fit_nearest_distances(distances, resolution=resolution, window=(0.01, 0.5))

    assert fit.family == "weibull"
    assert fit.shape == pytest.approx(shape, rel=0.05)
    assert fit.scale == pytest.approx(scale, rel=0.25)


def test_copies_are_flagged_and_fresh_samples_are_not():
    rng = np.random.default_rng(1)
    train, test, fresh = _family(rng, 1000), _family(rng, 300), _family(rng, 500)
    copies = fresh.copy()
    copies[:50] = train[:50]

    def score(generated):
        return privet(_nearest(train, train, same=True), _nearest(generated, train), _nearest(generated, test),
                      _nearest(test, train), n_train=len(train), n_test=len(test), resolution=1 / 60)

    clean, leaky = score(fresh), score(copies)

    assert clean["n_flagged"] == 0
    assert clean["excess_train"]["log10_p"] > LEAK_THRESHOLD
    assert clean["control"]["n_flagged"] == 0
    assert leaky["excess_train"]["log10_p"] < -10
    assert leaky["excess_train"]["distance"] == 0.0
    assert leaky["n_pleaks"] > 0
    assert np.all(np.asarray(leaky["delta_p"])[:50] < LEAK_THRESHOLD)
    assert np.sum(np.asarray(leaky["delta_p"])[50:] < LEAK_THRESHOLD) == 0


def test_distance_comparison_scores_samples_and_exports_columns():
    rng = np.random.default_rng(2)
    train, test = _family(rng, 200, length=30), _family(rng, 80, length=30)
    generated = np.vstack([train[:5], _family(rng, 75, length=30)])
    dataset = SimpleNamespace(data=torch.as_tensor(train), weights=torch.ones(200))
    held_out = SimpleNamespace(data=torch.as_tensor(test), weights=torch.ones(80))
    samples = torch.nn.functional.one_hot(torch.as_tensor(generated), 4).float()

    result = _distance_comparison(dataset, samples, test=held_out, n_measure=60)

    scores = result["privet"]
    assert len(scores["sample_index"]) == len(scores["delta_p"]) == 60
    sampling = SamplingResult(sequences=("A" * 30,) * 80, energies=np.zeros(80), num_sweeps=0, sampler="gibbs",
                              beta=1.0, seed=0, model=None, distance_comparison=result)
    frame = sampling.to_dataframe()
    assert frame["privet_delta_p"].notna().sum() == 60
    assert {"privet_log10_p_train", "privet_log10_p_test"} <= set(frame.columns)
