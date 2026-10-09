"""Numba CPU kernels: exact distributions, determinism, explicit sites, replica stacks and dispatch."""

import os
from itertools import product

import pytest
import torch

pytest.importorskip("numba")

from adabmDCA.numba_kernels import (
    METHODS,
    categorical_sampler,
    is_numba_available,
    sample_categorical,
    sample_replicas,
)
from adabmDCA.statmech import compute_energy

pytestmark = pytest.mark.skipif(not is_numba_available(), reason="Numba kernels disabled")


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def random_potts(length, num_states, dtype, seed=0):
    generator = torch.Generator().manual_seed(seed)
    couplings = torch.randn(length, num_states, length, num_states, generator=generator, dtype=torch.float64) * 0.6
    couplings = 0.5 * (couplings + couplings.permute(2, 3, 0, 1))
    couplings[torch.arange(length), :, torch.arange(length), :] = 0.0
    bias = torch.randn(length, num_states, generator=generator, dtype=torch.float64) * 0.5
    return {"bias": bias.to(dtype), "coupling_matrix": couplings.to(dtype)}


def distance_to_exact(final, params):
    """Total variation between the empirical distribution of ``final`` and the exact one."""
    length, num_states = params["bias"].shape
    states = torch.tensor(list(product(range(num_states), repeat=length)))
    energies = compute_energy(torch.nn.functional.one_hot(states, num_states).double(),
                              {key: value.double() for key, value in params.items()})
    exact = torch.softmax(-energies, 0)
    index = (final.long() * num_states ** torch.arange(length - 1, -1, -1)).sum(1)
    histogram = torch.bincount(index, minlength=len(exact)).double() / len(final)
    return 0.5 * float((histogram - exact).abs().sum())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", METHODS)
def test_kernels_sample_the_exact_distribution(method, dtype):
    params = random_potts(4, 3, dtype)
    torch.manual_seed(0)
    final = sample_categorical(method, torch.randint(0, 3, (100_000, 4)), params, 30)
    assert distance_to_exact(final, params) < 0.015


@pytest.mark.parametrize("method", METHODS)
def test_each_replica_samples_its_own_model(method):
    models = [random_potts(4, 3, torch.float32, seed=seed) for seed in (0, 1, 2)]
    biases = torch.stack([model["bias"] for model in models])
    couplings = torch.stack([model["coupling_matrix"] for model in models])
    torch.manual_seed(0)
    final = sample_replicas(method, torch.randint(0, 3, (3, 60_000, 4)), biases, couplings, 30)
    for states, model in zip(final, models):
        assert distance_to_exact(states, model) < 0.02


@pytest.mark.parametrize("method", METHODS)
def test_a_single_replica_matches_the_single_model_sampler(method):
    params = random_potts(6, 4, torch.float64, seed=3)
    start = torch.randint(0, 4, (200, 6))
    torch.manual_seed(1)
    single = sample_categorical(method, start, params, 2)
    torch.manual_seed(1)
    stacked = sample_replicas(method, start[None], params["bias"][None], params["coupling_matrix"][None], 2)
    assert torch.equal(single, stacked[0])


def test_replica_couplings_must_match_the_biases():
    params = random_potts(3, 2, torch.float32)
    with pytest.raises(ValueError, match="Replica couplings must have shape"):
        sample_replicas("gibbs", torch.zeros(1, 4, 3, dtype=torch.int32), params["bias"][None],
                        params["coupling_matrix"][None].double(), 1)


@pytest.mark.parametrize("method", METHODS)
def test_results_depend_on_the_seed_not_on_the_thread_count(method):
    params = random_potts(7, 4, torch.float32, seed=1)
    start = torch.randint(0, 4, (300, 7))
    outputs = []
    for threads in (1, 3, 1):
        torch.set_num_threads(threads)
        torch.manual_seed(5)
        outputs.append(sample_categorical(method, start, params, 3))
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[0], outputs[2])
    torch.manual_seed(6)
    assert not torch.equal(outputs[0], sample_categorical(method, start, params, 3))
    assert torch.equal(start, start.clone())  # input untouched


def test_explicit_sites_update_only_those_sites():
    params = random_potts(6, 3, torch.float64, seed=2)
    start = torch.randint(0, 3, (500, 6), dtype=torch.int32)
    final = categorical_sampler("gibbs")(start, params, 0, 1.0, sites=torch.tensor([1, 4, 1]))
    untouched = [0, 2, 3, 5]
    assert torch.equal(final[:, untouched], start[:, untouched])
    assert not torch.equal(final[:, [1, 4]], start[:, [1, 4]])


def test_cpu_dispatch_uses_numba_unless_disabled(monkeypatch):
    from adabmDCA.ptt import PTTSampler
    from adabmDCA.sampling import prepare_sampler

    assert prepare_sampler("gibbs", torch.device("cpu")).__name__ == "gibbs_sampling_numba"
    params = random_potts(3, 2, torch.float64)
    profile = {"bias": params["bias"], "coupling_matrix": torch.zeros_like(params["coupling_matrix"])}
    sampler = PTTSampler(profile, tokens="AB", n_chains=8, sampler="metropolized_gibbs")
    assert sampler._local_kernel.__name__ == "metropolized_gibbs_sampling_numba"
    assert sampler._replica_kernel.function.__name__ == "metropolized_gibbs_sampling_replicas_numba"
    assert sampler._replica_kernel.always
    chains = torch.nn.functional.one_hot(torch.randint(0, 2, (10, 3)), 2).double()
    assert prepare_sampler("metropolis", torch.device("cpu"))(chains, params, 2).shape == chains.shape
    monkeypatch.setenv("ADABMDCA_NUMBA", "0")
    assert not is_numba_available()
    assert "numba" not in prepare_sampler("gibbs", torch.device("cpu")).name


def test_invalid_inputs_are_rejected():
    params = random_potts(3, 2, torch.float64)
    with pytest.raises(ValueError, match="Unknown sampling method"):
        sample_categorical("heat_bath", torch.zeros(2, 3, dtype=torch.int32), params, 1)
    with pytest.raises(ValueError, match="float32 or float64"):
        sample_categorical("gibbs", torch.zeros(2, 3, dtype=torch.int32),
                           {key: value.to(torch.bfloat16) for key, value in params.items()}, 1)


def test_profile_states_follow_the_independent_site_model():
    from adabmDCA.numba_kernels import sample_profile_states

    bias = torch.tensor([[0.5, -1.0, 0.2], [2.0, 0.0, -3.0]], dtype=torch.float32)
    torch.manual_seed(0)
    states = sample_profile_states(bias, 200_000)
    assert states.dtype == torch.int32 and states.shape == (200_000, 2)
    for site in range(2):
        frequencies = torch.bincount(states[:, site].long(), minlength=3).double() / len(states)
        torch.testing.assert_close(frequencies, torch.softmax(bias[site].double(), 0), atol=0.004, rtol=0)
    torch.manual_seed(0)
    assert torch.equal(states, sample_profile_states(bias, 200_000))


def test_counter_based_uniforms_are_uniform_and_independent():
    import numpy as np
    from numba import njit

    from adabmDCA.numba_kernels.random import uniform

    @njit
    def draws(seed, n):
        out = np.empty((n, 2))
        for i in range(n):
            out[i, 0] = uniform(seed, i, 7, 0)
            out[i, 1] = uniform(seed, i, 7, 1)
        return out

    values = draws(np.uint64(12345), 200_000)
    assert values.min() >= 0.0 and values.max() < 1.0
    assert abs(values.mean() - 0.5) < 0.003 and abs(values.var() - 1 / 12) < 0.002
    assert abs(np.corrcoef(values[:, 0], values[:, 1])[0, 1]) < 0.01  # slots are independent streams
    assert abs(np.corrcoef(values[:-1, 0], values[1:, 0])[0, 1]) < 0.01  # successive steps too
    histogram, _ = np.histogram(values[:, 0], bins=20, range=(0, 1))
    assert histogram.min() > 0.95 * len(values) / 20


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux only")
def test_worker_placement_lists_physical_cores_first():
    from adabmDCA.numba_kernels.threads import _core_of, _cpus_by_core

    allowed = frozenset(os.sched_getaffinity(0))
    cpus, cores = _cpus_by_core(allowed)
    assert sorted(cpus) == sorted(allowed)
    assert len({_core_of(cpu) for cpu in cpus[:cores]}) == cores == len({_core_of(cpu) for cpu in allowed})


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exchange_acceptance_matches_the_one_hot_energies(dtype, symmetric):
    from adabmDCA.numba_kernels import exchange_log_acceptance
    from adabmDCA.statmech import couplings_are_symmetric

    lower, upper = random_potts(5, 4, dtype, seed=4), random_potts(5, 4, dtype, seed=5)
    noise = 0.1 * torch.randn_like(upper["coupling_matrix"])
    if symmetric:
        # Symmetric, with non-zero diagonal blocks: pairs are summed once and the diagonal with weight 1/2.
        noise = 0.5 * (noise + noise.permute(2, 3, 0, 1))
    upper["coupling_matrix"] = upper["coupling_matrix"] + noise
    assert couplings_are_symmetric(upper["coupling_matrix"]) == symmetric
    x, y = torch.randint(0, 4, (2, 300, 5), dtype=torch.int32)

    def energy(states, params):
        return compute_energy(torch.nn.functional.one_hot(states.long(), 4).double(),
                              {key: value.double() for key, value in params.items()})

    expected = (energy(y, upper) - energy(y, lower) - energy(x, upper) + energy(x, lower)).clamp_max(0.0)
    result = exchange_log_acceptance(lower, upper, x, y)
    assert result.dtype == torch.float64 and (result < 0).any() and (result == 0).any()
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-4 if dtype == torch.float32 else 1e-10)


def test_symmetry_check_is_cached_and_follows_in_place_changes():
    from adabmDCA.statmech import couplings_are_symmetric

    couplings = random_potts(4, 3, torch.float64)["coupling_matrix"]
    assert couplings_are_symmetric(couplings) and couplings_are_symmetric(couplings)
    couplings[0, 1, 2, 0] += 1.0
    assert not couplings_are_symmetric(couplings)


def sparse_potts(length, num_states, dtype, coupled_pairs, seed=0):
    """A Potts model whose couplings are non-zero only on ``coupled_pairs`` (whole blocks)."""
    params = random_potts(length, num_states, dtype, seed=seed)
    pair = torch.zeros(length, length, dtype=torch.bool)
    for i, j in coupled_pairs:
        pair[i, j] = pair[j, i] = True
    params["coupling_matrix"] = params["coupling_matrix"] * pair[:, None, :, None]
    return params


@pytest.mark.parametrize("method", METHODS)
def test_sparse_kernels_sample_the_exact_distribution(method):
    from adabmDCA.numba_kernels.sampling import _prepared_couplings

    params = sparse_potts(5, 3, torch.float64, [(0, 1), (2, 4)])
    assert _prepared_couplings(method, params["coupling_matrix"][None])[0] == "sparse"
    torch.manual_seed(0)
    final = sample_categorical(method, torch.randint(0, 3, (100_000, 5)), params, 30)
    assert distance_to_exact(final, params) < 0.02


@pytest.mark.parametrize("method", METHODS)
def test_sparse_and_dense_kernels_give_identical_chains(method, monkeypatch):
    import adabmDCA.numba_kernels.sampling as kernels

    models = [sparse_potts(8, 4, torch.float32, [(0, 3), (1, 6)], seed=1),
              sparse_potts(8, 4, torch.float32, [(2, 5)], seed=2)]
    biases = torch.stack([model["bias"] for model in models])
    start = torch.randint(0, 4, (2, 300, 8))
    results = []
    for threshold in (kernels.SPARSE_MAX_PAIR_DENSITY, 0.0):  # sparse, then forced dense
        monkeypatch.setattr(kernels, "SPARSE_MAX_PAIR_DENSITY", threshold)
        couplings = torch.stack([model["coupling_matrix"] for model in models])  # a new tensor: no cached layout
        assert kernels._prepared_couplings(method, couplings)[0] == ("sparse" if threshold else "dense")
        torch.manual_seed(3)
        results.append(sample_replicas(method, start, biases, couplings, 3))
    assert torch.equal(results[0], results[1])


def test_dense_graphs_keep_the_dense_kernels():
    from adabmDCA.numba_kernels.sampling import _prepared_couplings

    assert _prepared_couplings("gibbs", random_potts(6, 3, torch.float32)["coupling_matrix"][None])[0] == "dense"


@pytest.mark.parametrize("symmetric", [False, True])
def test_sparse_exchange_matches_the_one_hot_energies(symmetric):
    from adabmDCA.numba_kernels import exchange_log_acceptance
    from adabmDCA.numba_kernels.exchange import _sparse_layout_of

    lower = sparse_potts(7, 3, torch.float64, [(0, 4), (2, 3)], seed=4)
    upper = sparse_potts(7, 3, torch.float64, [(0, 4), (1, 6), (2, 5)], seed=5)
    if not symmetric:
        upper["coupling_matrix"] = upper["coupling_matrix"] * (1 + 0.3 * torch.rand_like(upper["coupling_matrix"]))
    assert _sparse_layout_of(lower["coupling_matrix"]) is not None
    assert _sparse_layout_of(upper["coupling_matrix"]) is not None
    x, y = torch.randint(0, 3, (2, 400, 7), dtype=torch.int32)

    def energy(states, params):
        return compute_energy(torch.nn.functional.one_hot(states.long(), 3).double(), params)

    expected = (energy(y, upper) - energy(y, lower) - energy(x, upper) + energy(x, lower)).clamp_max(0.0)
    torch.testing.assert_close(exchange_log_acceptance(lower, upper, x, y), expected, rtol=0, atol=1e-10)
