"""Exactness of the Metropolized Gibbs kernels used for PTT generation."""

from itertools import product

import numpy as np
import pytest
import torch

from adabmDCA import PTTSampler
from adabmDCA.exceptions import InputValidationError
from adabmDCA.sampling import metropolized_gibbs_sampling_categorical
from adabmDCA.statmech import compute_energy
from tests import test_ptt as helpers

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def random_potts(length, num_states, *, scale=1.0, seed=0, dtype=torch.float64, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    bias = torch.randn(length, num_states, generator=generator, dtype=dtype) * scale
    couplings = torch.randn(length, num_states, length, num_states, generator=generator, dtype=dtype) * scale
    couplings = 0.5 * (couplings + couplings.permute(2, 3, 0, 1))
    couplings[torch.arange(length), :, torch.arange(length), :] = 0.0
    return {"bias": bias.to(device), "coupling_matrix": couplings.to(device)}


def exact_distribution(params):
    length, num_states = params["bias"].shape
    states = torch.tensor(list(product(range(num_states), repeat=length)))
    one_hot = torch.nn.functional.one_hot(states, num_states).to(params["bias"].dtype)
    energies = compute_energy(one_hot.to(params["bias"].device), params).cpu()
    return states, (-energies).softmax(0).numpy()


def empirical_distribution(samples, num_states):
    length = samples.shape[1]
    codes = (samples.long().cpu() * num_states ** torch.arange(length - 1, -1, -1)).sum(1)
    return np.bincount(codes.numpy(), minlength=num_states ** length) / len(samples)


def test_reference_kernel_samples_exact_distribution():
    torch.manual_seed(3)
    params = random_potts(3, 3, scale=1.2, seed=5)
    _, exact = exact_distribution(params)
    start = torch.zeros(40_000, 3, dtype=torch.int32)
    samples = metropolized_gibbs_sampling_categorical(start, params, nsweeps=12)
    np.testing.assert_allclose(empirical_distribution(samples, 3), exact, atol=0.01)


def test_reference_kernel_handles_nearly_deterministic_conditionals():
    torch.manual_seed(0)
    params = random_potts(2, 3, scale=0.0)
    params["bias"][:, 0] = 60.0  # other states underflow relative to state 0 in float32
    params = {key: value.float() for key, value in params.items()}
    samples = metropolized_gibbs_sampling_categorical(torch.zeros(500, 2, dtype=torch.int32), params, 5)
    assert torch.equal(samples, torch.zeros_like(samples))
    samples = metropolized_gibbs_sampling_categorical(torch.full((500, 2), 2, dtype=torch.int32), params, 5)
    assert (samples == 0).double().mean() > 0.99


def controlled_reference(states, biases, couplings, sites, uniforms, beta=1.0):
    """Step-by-step Metropolized Gibbs with explicit random inputs, original coupling layout."""
    states = states.clone().long()
    num_replicas, num_chains, length = states.shape
    for step in range(sites.shape[0]):
        for replica in range(num_replicas):
            site = int(sites[step, replica])
            for chain in range(num_chains):
                x = states[replica, chain]
                field = biases[replica, site] + couplings[replica, site, :, torch.arange(length), x].sum(1)
                weights = torch.exp(beta * field - (beta * field).max())
                old = int(x[site])
                others = weights.clone()
                others[old] = 0.0
                threshold = uniforms[step, replica, chain, 0] * others.sum()
                proposed = min(int((others.cumsum(0) <= threshold).sum()), len(weights) - 1)
                excluded = weights.clone()
                excluded[proposed] = 0.0
                if others.sum() > 0 and uniforms[step, replica, chain, 1] * excluded.sum() < others.sum():
                    states[replica, chain, site] = proposed
    return states.int()


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_triton_kernel_matches_controlled_reference(dtype):
    from adabmDCA.sampling_triton import (
        _replica_metropolized_gibbs_steps_triton,
        transpose_couplings_for_metropolized_gibbs,
    )

    models = [random_potts(7, 5, scale=0.8, seed=seed, dtype=dtype, device="cuda") for seed in (1, 2)]
    biases = torch.stack([m["bias"] for m in models])
    couplings = torch.stack([m["coupling_matrix"] for m in models])
    transposed = torch.stack([transpose_couplings_for_metropolized_gibbs(m["coupling_matrix"]) for m in models])
    generator = torch.Generator(device="cuda").manual_seed(4)
    states = torch.randint(0, 5, (2, 9, 7), device="cuda", dtype=torch.int32, generator=generator)
    sites = torch.randint(0, 7, (40, 2), device="cuda", dtype=torch.int32, generator=generator)
    uniforms = torch.rand((40, 2, 9, 2), device="cuda", dtype=dtype, generator=generator)
    expected = controlled_reference(states, biases, couplings, sites, uniforms)
    for steps_per_launch in (1, 16):
        result = states.clone()
        _replica_metropolized_gibbs_steps_triton(
            result, biases, transposed, sites, uniforms, 1.0,
            steps_per_launch=steps_per_launch, block_l=8, block_q=8, num_warps=4,
        )
        assert torch.equal(result, expected)


@cuda
def test_triton_replica_kernel_samples_exact_distributions():
    from adabmDCA.sampling_triton import (
        metropolized_gibbs_sampling_replicas_categorical_triton,
        transpose_couplings_for_metropolized_gibbs,
    )

    torch.manual_seed(11)
    models = [random_potts(3, 3, scale=1.0, seed=seed, device="cuda") for seed in (7, 8)]
    biases = torch.stack([m["bias"] for m in models])
    couplings = torch.stack([transpose_couplings_for_metropolized_gibbs(m["coupling_matrix"]) for m in models])
    states = torch.zeros(2, 40_000, 3, device="cuda", dtype=torch.int32)
    samples = metropolized_gibbs_sampling_replicas_categorical_triton(states, biases, couplings, nsweeps=12)
    for replica, params in enumerate(models):
        _, exact = exact_distribution(params)
        np.testing.assert_allclose(empirical_distribution(samples[replica], 3), exact, atol=0.01)


@pytest.mark.parametrize("scripted", [False, True])
def test_one_hot_reference_sampler_samples_exact_distribution(scripted):
    from adabmDCA.sampling import get_sampler

    torch.manual_seed(21)
    params = random_potts(3, 3, scale=1.2, seed=6)
    _, exact = exact_distribution(params)
    sampler = get_sampler("metropolized_gibbs")
    sampler = torch.jit.script(sampler) if scripted else sampler
    chains = torch.nn.functional.one_hot(torch.zeros(40_000, 3, dtype=torch.long), 3).double()
    samples = sampler(chains, params, 12, 1.0).argmax(-1)
    np.testing.assert_allclose(empirical_distribution(samples, 3), exact, atol=0.01)
    assert torch.equal(chains.argmax(-1), torch.zeros(40_000, 3, dtype=torch.long))


@cuda
@pytest.mark.parametrize("coupling_dtype", [None, torch.bfloat16])
def test_single_model_triton_sampler_samples_exact_distribution(coupling_dtype):
    from adabmDCA.sampling_triton import metropolized_gibbs_sampling_triton

    torch.manual_seed(12)
    params = random_potts(3, 3, scale=1.0, seed=9, dtype=torch.float32, device="cuda")
    if coupling_dtype is torch.bfloat16:
        # BF16 storage samples the rounded model; compare with that model.
        params["coupling_matrix"] = params["coupling_matrix"].to(torch.bfloat16).float()
    _, exact = exact_distribution(params)
    chains = torch.nn.functional.one_hot(torch.zeros(40_000, 3, dtype=torch.long, device="cuda"), 3).float()
    samples = metropolized_gibbs_sampling_triton(chains, params, 12, coupling_dtype=coupling_dtype)
    assert samples.dtype == chains.dtype and torch.equal(samples.sum(-1), torch.ones(40_000, 3, device="cuda"))
    np.testing.assert_allclose(empirical_distribution(samples.argmax(-1), 3), exact, atol=0.01)


def test_sampler_registry_and_public_sampling_accept_metropolized_gibbs(tmp_path, monkeypatch):
    from adabmDCA import DCAModel
    from adabmDCA.sampling import metropolized_gibbs_sampling, prepare_sampler
    from adabmDCA.training_config import SAMPLERS

    assert "metropolized_gibbs" in SAMPLERS
    monkeypatch.setenv("ADABMDCA_NUMBA", "0")  # the TorchScript fallback on CPU
    assert prepare_sampler("metropolized_gibbs", torch.device("cpu")).name == metropolized_gibbs_sampling.__name__
    monkeypatch.delenv("ADABMDCA_NUMBA")
    if torch.cuda.is_available():
        from adabmDCA.sampling_triton import is_triton_available, metropolized_gibbs_sampling_triton

        if is_triton_available():
            assert prepare_sampler("metropolized_gibbs", torch.device("cuda")) is metropolized_gibbs_sampling_triton
    params = {key: value.float() for key, value in random_potts(4, 3, scale=0.5, seed=2).items()}
    model = DCAModel(params, alphabet="ABC")
    result = model.sample_sequences(16, n_sweeps=3, sampler="metropolized_gibbs", device="cpu", seed=1)
    assert len(result.sequences) == 16 and all(len(s) == 4 for s in result.sequences)


def test_wide_index_threshold():
    from adabmDCA.sampling_triton import _needs_wide_indices

    assert not _needs_wide_indices(14_000, 307, 21)
    assert _needs_wide_indices(10, 2207, 21)  # (L*q)**2 >= 2**31
    assert _needs_wide_indices(2**31 // 100 + 1, 100, 2)  # N*L >= 2**31


@cuda
def test_wide_index_kernels_match_int32_kernels(monkeypatch):
    """Forcing the int64 path must not change any trajectory."""
    from adabmDCA import sampling_triton as st

    generator = torch.Generator(device="cuda").manual_seed(9)
    models = [random_potts(11, 6, scale=0.7, seed=seed, dtype=torch.float32, device="cuda") for seed in (3, 4)]
    biases = torch.stack([m["bias"] for m in models])
    couplings = torch.stack([m["coupling_matrix"] for m in models])
    transposed = torch.stack([st.transpose_couplings_for_metropolized_gibbs(m["coupling_matrix"]) for m in models])
    states = torch.randint(0, 6, (2, 37, 11), device="cuda", dtype=torch.int32, generator=generator)
    sites = torch.randint(0, 11, (30, 2), device="cuda", dtype=torch.int32, generator=generator)
    proposals = torch.randint(0, 6, (30, 2, 37), device="cuda", dtype=torch.int32, generator=generator)
    uniforms = torch.rand((30, 2, 37, 2), device="cuda", generator=generator)
    independent_sites = torch.randint(0, 11, (37,), device="cuda", dtype=torch.int32, generator=generator)

    def run_all():
        single = states[0].clone()
        st._metropolis_steps_triton(single, models[0], sites[:, 0].contiguous(), proposals[:, 0].contiguous(),
                                    uniforms[:, 0, :, 0].contiguous(), 0.9)
        independent = states[0].clone()
        st._metropolis_step_independent_triton(independent, models[0], independent_sites, proposals[0, 0],
                                               uniforms[0, 0, :, 0].contiguous(), 0.9)
        replica = states.clone()
        st._replica_metropolis_steps_triton(replica, biases, couplings, sites, proposals,
                                            uniforms[..., 0].contiguous(), 0.9, block_n=4, num_warps=1)
        gibbs = states.clone()
        st._replica_metropolized_gibbs_steps_triton(gibbs, biases, transposed, sites, uniforms, 0.9,
                                                    steps_per_launch=8, block_l=16, block_q=8, num_warps=1)
        return single, independent, replica, gibbs

    narrow = run_all()
    monkeypatch.setattr(st, "_needs_wide_indices", lambda *args: True)
    wide = run_all()
    for a, b in zip(narrow, wide):
        assert torch.equal(a, b)


def test_ptt_generation_kernel_selection_and_exact_endpoint(tmp_path):
    from adabmDCA.api import sample_sequences

    data = tmp_path / "reference.fasta"
    data.write_text(">a\nAA\n>b\nBB\n")
    backend = PTTSampler(helpers.profile(), tokens="AB", n_chains=8)
    backend.models[-1] = helpers.coupled()
    backend.model_version = 1
    with pytest.raises(InputValidationError, match="local kernel"):
        backend.set_generation_kernel("heat_bath")
    result = sample_sequences(model=backend, reference_fasta=data, n_sequences=20_000, ptt_local_sweeps=1,
                              ptt_local_kernel="metropolized_gibbs")
    assert result.ptt_diagnostics["local_kernel"] == "metropolized_gibbs"
    index = {"AA": 0, "AB": 1, "BA": 2, "BB": 3}
    counts = np.bincount([index[s] for s in result.sequences], minlength=4) / len(result.sequences)
    exact = (-compute_energy(helpers.states(), helpers.coupled())).softmax(0).numpy()
    np.testing.assert_allclose(counts, exact, atol=0.02)
    # The source sampler keeps its archived kernel.
    assert backend.generation_kernel is None


def sparse_potts(length, num_states, pairs, *, seed=0, dtype=torch.float64, device="cuda"):
    """A Potts model coupled only on ``pairs`` (whole blocks)."""
    params = random_potts(length, num_states, scale=0.8, seed=seed, dtype=dtype, device=device)
    pair = torch.zeros(length, length, dtype=torch.bool, device=device)
    for i, j in pairs:
        pair[i, j] = pair[j, i] = True
    params["coupling_matrix"] = params["coupling_matrix"] * pair[:, None, :, None]
    return params


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sparse_triton_kernel_matches_controlled_reference(dtype):
    from adabmDCA.sampling_triton import _replica_metropolized_gibbs_sparse_steps_triton, sparse_coupling_layout

    # Two replicas with different graphs; the widest site has 2 of 10 neighbours.
    models = [sparse_potts(10, 5, [(0, 3), (3, 7), (1, 8)], seed=1, dtype=dtype),
              sparse_potts(10, 5, [(2, 9), (4, 5)], seed=2, dtype=dtype)]
    biases = torch.stack([m["bias"] for m in models])
    couplings = torch.stack([m["coupling_matrix"] for m in models])
    layout = sparse_coupling_layout(couplings)
    assert layout is not None and layout[2] == 2
    generator = torch.Generator(device="cuda").manual_seed(4)
    states = torch.randint(0, 5, (2, 9, 10), device="cuda", dtype=torch.int32, generator=generator)
    sites = torch.randint(0, 10, (60, 2), device="cuda", dtype=torch.int32, generator=generator)
    uniforms = torch.rand((60, 2, 9, 2), device="cuda", dtype=dtype, generator=generator)
    expected = controlled_reference(states, biases, couplings, sites, uniforms)
    result = states.clone()
    _replica_metropolized_gibbs_sparse_steps_triton(result, biases, layout, sites, uniforms, 1.0)
    assert torch.equal(result, expected)


@cuda
def test_sparse_triton_samplers_sample_exact_distributions():
    from adabmDCA.sampling_triton import (
        metropolized_gibbs_sampling_categorical_triton,
        metropolized_gibbs_sampling_replicas_triton,
        sparse_coupling_layout,
    )

    torch.manual_seed(12)
    models = [sparse_potts(5, 3, [(0, 2)], seed=7), sparse_potts(5, 3, [(1, 4)], seed=8)]
    couplings = torch.stack([m["coupling_matrix"] for m in models])
    assert sparse_coupling_layout(couplings) is not None
    states = torch.zeros(2, 40_000, 5, device="cuda", dtype=torch.int32)
    samples = metropolized_gibbs_sampling_replicas_triton(
        states, torch.stack([m["bias"] for m in models]), couplings, nsweeps=12)
    single = metropolized_gibbs_sampling_categorical_triton(states[0], models[0], nsweeps=12)
    for result, params in ((samples[0], models[0]), (samples[1], models[1]), (single, models[0])):
        _, exact = exact_distribution(params)
        np.testing.assert_allclose(empirical_distribution(result, 3), exact, atol=0.01)


@cuda
@pytest.mark.parametrize("symmetric", [False, True])
def test_sparse_triton_exchange_matches_the_one_hot_energies(symmetric):
    from adabmDCA.sampling_triton import exchange_log_acceptance_sparse_triton

    lower = sparse_potts(8, 4, [(0, 4), (2, 3)], seed=4, dtype=torch.float32)
    upper = sparse_potts(8, 4, [(0, 4), (1, 6), (2, 5)], seed=5, dtype=torch.float32)
    if not symmetric:
        upper["coupling_matrix"] = upper["coupling_matrix"] * (1 + 0.3 * torch.rand_like(upper["coupling_matrix"]))
    x, y = torch.randint(0, 4, (2, 300, 8), device="cuda", dtype=torch.int32)

    def energy(states, params):
        return compute_energy(torch.nn.functional.one_hot(states.long(), 4).double(),
                              {key: value.double() for key, value in params.items()})

    expected = (energy(y, upper) - energy(y, lower) - energy(x, upper) + energy(x, lower)).clamp_max(0.0)
    result = exchange_log_acceptance_sparse_triton(lower, upper, x, y)
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-4)
    dense = random_potts(8, 4, device="cuda", dtype=torch.float32)
    assert exchange_log_acceptance_sparse_triton(dense, upper, x, y) is None


@cuda
def test_replica_sampler_falls_back_to_the_dense_kernel_on_full_graphs():
    from adabmDCA.sampling_triton import (
        metropolized_gibbs_sampling_replicas_categorical_triton,
        metropolized_gibbs_sampling_replicas_triton,
        sparse_coupling_layout,
        transpose_couplings_for_metropolized_gibbs,
    )

    models = [random_potts(6, 4, scale=0.5, seed=seed, device="cuda") for seed in (3, 4)]
    biases = torch.stack([m["bias"] for m in models])
    couplings = torch.stack([m["coupling_matrix"] for m in models])
    assert sparse_coupling_layout(couplings) is None
    states = torch.randint(0, 4, (2, 50, 6), device="cuda", dtype=torch.int32)
    torch.manual_seed(5)
    result = metropolized_gibbs_sampling_replicas_triton(states, biases, couplings, nsweeps=3)
    torch.manual_seed(5)
    transposed = torch.stack([transpose_couplings_for_metropolized_gibbs(m["coupling_matrix"]) for m in models])
    expected = metropolized_gibbs_sampling_replicas_categorical_triton(states, biases, transposed.contiguous(), nsweeps=3)
    assert torch.equal(result, expected)


@cuda
@pytest.mark.parametrize("method", ["gibbs", "metropolis"])
def test_sparse_gibbs_and_metropolis_match_their_dense_kernels(method, monkeypatch):
    from adabmDCA import sampling_triton as st

    sample = {"gibbs": st.gibbs_sampling_categorical_triton,
              "metropolis": st.metropolis_sampling_categorical_triton}[method]
    params = sparse_potts(12, 4, [(0, 5), (2, 9), (5, 11), (3, 7)], seed=6)
    states = torch.randint(0, 4, (200, 12), device="cuda", dtype=torch.int32)
    results = []
    for enabled in ("1", "0"):
        monkeypatch.setenv("ADABMDCA_SPARSE", enabled)
        fresh = {key: value.clone() for key, value in params.items()}  # no cached layout
        assert (st.sparse_coupling_layout(fresh["coupling_matrix"][None]) is None) == (enabled == "0")
        torch.manual_seed(9)
        results.append(sample(states, fresh, 5))
    assert torch.equal(results[0], results[1])
    if method == "gibbs":
        sites = torch.randint(0, 12, (40,), device="cuda", dtype=torch.int32)
        explicit = []
        for enabled in ("1", "0"):
            monkeypatch.setenv("ADABMDCA_SPARSE", enabled)
            fresh = {key: value.clone() for key, value in params.items()}
            torch.manual_seed(10)
            explicit.append(sample(states, fresh, 0, sites=sites))
        assert torch.equal(explicit[0], explicit[1])


@cuda
def test_sparse_metropolis_replicas_match_the_dense_kernel(monkeypatch):
    from adabmDCA import sampling_triton as st

    models = [sparse_potts(10, 5, [(0, 3), (3, 7)], seed=1), sparse_potts(10, 5, [(2, 9)], seed=2)]
    biases = torch.stack([m["bias"] for m in models])
    states = torch.randint(0, 5, (2, 100, 10), device="cuda", dtype=torch.int32)
    results = []
    for enabled in ("1", "0"):
        monkeypatch.setenv("ADABMDCA_SPARSE", enabled)
        couplings = torch.stack([m["coupling_matrix"] for m in models])
        torch.manual_seed(3)
        results.append(st.metropolis_sampling_replicas_categorical_triton(states, biases, couplings, nsweeps=4))
    assert torch.equal(results[0], results[1])


@cuda
@pytest.mark.parametrize("method", ["gibbs", "metropolis"])
def test_sparse_gibbs_and_metropolis_sample_exact_distributions(method):
    from adabmDCA import sampling_triton as st

    sample = {"gibbs": st.gibbs_sampling_categorical_triton,
              "metropolis": st.metropolis_sampling_categorical_triton}[method]
    params = sparse_potts(5, 3, [(0, 2), (1, 4)], seed=7)
    assert st.sparse_coupling_layout(params["coupling_matrix"][None]) is not None
    torch.manual_seed(13)
    samples = sample(torch.zeros(40_000, 5, device="cuda", dtype=torch.int32), params, 20)
    _, exact = exact_distribution(params)
    np.testing.assert_allclose(empirical_distribution(samples, 3), exact, atol=0.01)
