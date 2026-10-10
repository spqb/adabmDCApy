"""Metal samplers: reference trajectories, equilibrium, dispatch and cache safety.

Run on Apple Silicon with: python -m pytest -q tests/test_sampling_mps.py
Tests skip cleanly on hosts without MPS or torch.mps.compile_shader.
"""

from itertools import product

import pytest
import torch

from adabmDCA.mps_kernels import is_mps_available
from adabmDCA.mps_kernels import sampling as metal
from adabmDCA.mps_kernels.random import SOURCE as RANDOM_SOURCE
from adabmDCA.sampling import prepare_sampler
from adabmDCA.statmech import compute_energy

pytestmark = pytest.mark.skipif(not is_mps_available(), reason="Metal shaders unavailable")


def model(length, q, seed=9):
    rng = torch.Generator().manual_seed(seed)
    j = torch.randn(length, q, length, q, generator=rng) * 0.3
    j = (j + j.permute(2, 3, 0, 1)) / 2
    j[torch.arange(length), :, torch.arange(length), :] = 0
    return {"bias": torch.randn(length, q, generator=rng) * 0.4, "coupling_matrix": j}


def to_mps(params):
    return {key: value.to("mps") for key, value in params.items()}


def reference(method, x, params, sites, uniforms, beta):
    x = x.long().clone()
    rows = torch.arange(len(x))
    length, q = params["bias"].shape
    j = params["coupling_matrix"].permute(0, 2, 3, 1).double()
    for t, site in enumerate(sites.tolist()):
        field = params["bias"][site].double() + j[site][torch.arange(length), x].sum(1)
        old = x[:, site].clone()
        if method == "metropolis":
            proposed = (uniforms[t, :, 0] * q).long().clamp_max(q - 1)
            probability = ((field[rows, proposed] - field[rows, old]) * beta).clamp_max(0).exp()
            accepted = uniforms[t, :, 1] < probability
        else:
            weights = torch.softmax(beta * field, 1)
            proposal = weights.clone()
            if method == "metropolized_gibbs":
                proposal[rows, old] = 0
            total = proposal.sum(1)
            proposed = (proposal.cumsum(1) <= uniforms[t, :, :1] * total[:, None]).sum(1).clamp_max(q - 1)
            accepted = torch.ones(len(x), dtype=torch.bool)
            if method == "metropolized_gibbs":
                excluded = weights.clone()
                excluded[rows, proposed] = 0
                accepted = (total > 0) & (uniforms[t, :, 1] * excluded.sum(1) < total)
        x[:, site] = torch.where(accepted, proposed, old)
    return x


@pytest.mark.parametrize("method", metal.METHODS)
@pytest.mark.parametrize("q", [2, 5, 21, 32])
@pytest.mark.parametrize("beta", [0.0, 0.7, 1.5])
def test_explicit_randomness_matches_reference(method, q, beta):
    length, n, steps = 7, 37, 17
    p = model(length, q)
    rng = torch.Generator().manual_seed(10)
    start = torch.randint(q, (n, length), generator=rng, dtype=torch.int32)
    sites = torch.randint(length, (steps,), generator=rng, dtype=torch.int32)
    uniforms = torch.rand(steps, n, 2, generator=rng)
    expected = reference(method, start, p, sites, uniforms, beta)
    pm = to_mps(p)
    result = start.to("mps")[None]
    metal._run(method, result, pm["bias"][None], metal._layout(pm["coupling_matrix"], method)[None], steps, beta,
               sites=sites.to("mps")[None], uniforms=uniforms.to("mps")[None])
    torch.testing.assert_close(result[0].cpu().long(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("method", metal.METHODS)
@pytest.mark.parametrize("beta", [0.0, 0.6, 1.4])
def test_stationary_distribution_matches_enumeration(method, beta):
    length, q, n = 3, 3, 40000
    p = model(length, q)
    all_states = torch.tensor(list(product(range(q), repeat=length)))
    energy = compute_energy(torch.nn.functional.one_hot(all_states, q).float(), p)
    exact = torch.softmax(-beta * energy, 0)
    torch.manual_seed(13)
    final = metal.sample_categorical(method, torch.zeros(n, length, dtype=torch.int32, device="mps"),
                                     to_mps(p), 30, beta).cpu().long()
    index = (final * q ** torch.arange(length - 1, -1, -1)).sum(1)
    empirical = torch.bincount(index, minlength=q**length).float() / n
    assert float((exact - empirical).abs().sum() / 2) < 0.025


@pytest.mark.parametrize("method", metal.METHODS)
def test_reproducible_nonmutating_and_rng_restore(method):
    p = to_mps(model(11, 5))
    x = torch.zeros(43, 11, dtype=torch.int32, device="mps")
    original = x.clone()
    torch.manual_seed(41)
    rng = torch.mps.get_rng_state()
    first = metal.sample_categorical(method, x, p, 3)
    torch.mps.set_rng_state(rng)
    second = metal.sample_categorical(method, x, p, 3)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(x, original)
    third = metal.sample_categorical(method, x, p, 3)
    assert not torch.equal(first, third)


@pytest.mark.parametrize("method", metal.METHODS)
def test_single_site_extreme_bias_and_one_state(method):
    for q in (1, 2, 21):
        p = {"bias": torch.full((1, q), -1000., device="mps"),
             "coupling_matrix": torch.zeros(1, q, 1, q, device="mps")}
        p["bias"][0, 0] = 1000
        x = torch.zeros(100, 1, device="mps", dtype=torch.int32)
        torch.testing.assert_close(metal.sample_categorical(method, x, p, 5), x)


def test_explicit_sites_empty_batches_and_zero_sweeps():
    p = to_mps(model(5, 3))
    x = torch.zeros(33, 5, device="mps", dtype=torch.int32)
    result = metal.sample_categorical("gibbs", x, p, 0, sites=torch.tensor([1, 3, 1]))
    torch.testing.assert_close(result[:, [0, 2, 4]], x[:, [0, 2, 4]])
    assert bool((result[:, [1, 3]] != 0).any())
    before = torch.mps.get_rng_state()
    assert metal.sample_categorical("gibbs", x[:0], p, 5).shape == (0, 5)
    torch.testing.assert_close(metal.sample_categorical("gibbs", x, p, 0), x)
    torch.testing.assert_close(before, torch.mps.get_rng_state())
    with pytest.raises(ValueError, match="within the sequence"):
        metal.sample_categorical("gibbs", x, p, 1, sites=torch.tensor([5]))


def test_layout_cache_is_rebuilt_after_parameter_update():
    p = to_mps(model(5, 3))
    j = p["coupling_matrix"]
    initial = metal._layout(j)
    assert metal._layout(j) is initial
    j.add_(0.5)
    updated = metal._layout(j)
    assert updated is not initial
    torch.testing.assert_close(updated.cpu(), j.cpu().permute(0, 2, 3, 1))


def test_noncontiguous_tensors_and_storage_offsets():
    p = to_mps(model(7, 5))
    p["bias"] = p["bias"].t().contiguous().t()
    p["coupling_matrix"] = p["coupling_matrix"].permute(2, 3, 0, 1)
    x = torch.zeros(7, 31, dtype=torch.int64, device="mps").t()[2:]
    torch.manual_seed(42)
    result = metal.sample_categorical("gibbs", x, p, 2)
    torch.manual_seed(42)
    expected = metal.sample_categorical("gibbs", x.contiguous(), {k: v.contiguous() for k, v in p.items()}, 2)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("method", metal.METHODS)
def test_onehot_dispatch_and_fallback(method, monkeypatch):
    p = to_mps(model(5, 3))
    x = torch.zeros(19, 5, dtype=torch.int32, device="mps")
    chains = torch.nn.functional.one_hot(x.long(), 3).float()
    sampler = prepare_sampler(method, torch.device("mps"))
    assert sampler.__name__ == f"{method}_sampling_mps"
    torch.manual_seed(7)
    expected = metal.sample_categorical(method, x, p, 2)
    torch.manual_seed(7)
    actual = sampler(chains, p, 2)
    torch.testing.assert_close(actual.argmax(-1).to(torch.int32), expected)
    torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual[..., 0]))
    monkeypatch.setenv("ADABMDCA_MPS", "0")
    assert not is_mps_available()
    fallback = prepare_sampler(method, torch.device("mps"))
    assert isinstance(fallback, torch.jit.ScriptFunction)
    torch.testing.assert_close(fallback(chains, p, 0), chains)


def test_large_alphabet_uses_torch_fallback():
    p = to_mps(model(2, 33))
    chains = torch.nn.functional.one_hot(torch.zeros(3, 2, dtype=torch.long, device="mps"), 33).float()
    result = prepare_sampler("gibbs", torch.device("mps"))(chains, p, 1)
    assert result.shape == chains.shape
    torch.testing.assert_close(result.sum(-1), torch.ones(3, 2, device="mps"))


@pytest.mark.parametrize("method", metal.METHODS)
def test_multiple_launches_match_reference(method):
    length, q, n, steps = 3, 3, 17, metal._STEPS_PER_LAUNCH + 7
    p = model(length, q)
    rng = torch.Generator().manual_seed(24)
    x = torch.randint(q, (n, length), generator=rng, dtype=torch.int32)
    sites = torch.randint(length, (steps,), generator=rng, dtype=torch.int32)
    uniforms = torch.rand(steps, n, 2, generator=rng)
    pm = to_mps(p)
    result = x.to("mps")[None]
    metal._run(method, result, pm["bias"][None], metal._layout(pm["coupling_matrix"], method)[None], steps, 1.,
               sites=sites.to("mps")[None], uniforms=uniforms.to("mps")[None])
    torch.testing.assert_close(result[0].cpu().long(), reference(method, x, p, sites, uniforms, 1.))


@pytest.mark.parametrize("method", metal.METHODS)
@pytest.mark.parametrize("q", [5, 21])
@pytest.mark.parametrize("replicas", [1, 3])
@pytest.mark.parametrize("sparse", [False, True])
def test_tuned_launches_preserve_trajectory_and_rng(method, q, replicas, sparse, monkeypatch):
    # Exercise tuned launches, a full 512-site bank, and its final short tail.
    length, n = 64, 1024
    params = to_mps(model(length, q))
    if sparse:
        rows = torch.arange(length, device='mps')
        mask = torch.zeros(length, length, device='mps', dtype=torch.bool)
        for offset in range(1, 12):
            mask[rows, (rows + offset) % length] = mask[rows, (rows - offset) % length] = True
        params['coupling_matrix'] *= mask[:, None, :, None]
    h = params['bias'][None].expand(replicas, -1, -1).contiguous()
    j = metal._layout(params['coupling_matrix'], method)[None].expand(replicas, -1, -1, -1, -1).contiguous()
    layout = metal.sparse_coupling_layout(params['coupling_matrix'][None].expand(replicas, -1, -1, -1, -1).contiguous()) if sparse else None
    initial = torch.zeros(replicas, n, length, device='mps', dtype=torch.int32)
    state = torch.mps.get_rng_state()
    actual = metal._run(method, initial.clone(), h, j, 529, .7, sparse=layout)
    after = torch.mps.get_rng_state()
    torch.mps.set_rng_state(state)
    monkeypatch.setattr(metal, '_steps_per_launch', lambda *args: 512)
    expected = metal._run(method, initial.clone(), h, j, 529, .7, sparse=layout)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(after, torch.mps.get_rng_state())


def test_philox_matches_random123_known_answer():
    # Random123's Philox4x32-10 zero-counter, zero-key known-answer vector.
    library = torch.mps.compile_shader(RANDOM_SOURCE + r'''
    kernel void answer(device uint* out, uint lane [[thread_position_in_grid]]) {
        out[lane] = philox(uint4(0), uint2(0))[lane];
    }
    ''')
    result = torch.empty(4, dtype=torch.int32, device="mps")
    library.answer(result, threads=4)
    assert [x & 0xffffffff for x in result.cpu().tolist()] == [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]


def test_sampling_inside_inference_mode():
    with torch.inference_mode():
        p = to_mps(model(5, 3))
        x = torch.zeros(17, 5, dtype=torch.int32, device="mps")
        result = metal.sample_categorical("gibbs", x, p, 2)
        assert result.shape == x.shape


def test_pcd_training_dispatches_to_metal_and_computes_validation(monkeypatch):
    from adabmDCA import train_model
    from adabmDCA.alignment import Alignment

    rng = torch.Generator().manual_seed(52)
    rows = torch.randint(3, (80, 6), generator=rng)
    alignment = Alignment(tuple(map(str, range(80))), tuple("".join("ABC"[x] for x in row) for row in rows))
    calls = []
    original = metal._run

    def record(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(metal, "_run", record)
    result = train_model(alignment, validation_path=alignment, alphabet="ABC", device="mps", n_chains=64,
                         n_sweeps=2, max_gradient_steps=3, target_pearson=0.99999, no_reweighting=True, seed=42)
    assert calls and set(calls) == {"metropolized_gibbs"}
    assert result.gradient_steps == 3
    assert "Pearson_val" in result.final_metrics
    assert all(torch.isfinite(value).all() for value in result.model.params.values())


def test_long_sequence_uses_bounded_threadgroup_memory():
    length, q = 1025, 2
    p = model(length, q)
    rng = torch.Generator().manual_seed(91)
    x = torch.randint(q, (9, length), generator=rng, dtype=torch.int32)
    sites = torch.tensor([33, 1024], dtype=torch.int32)
    uniforms = torch.rand(2, 9, 2, generator=rng)
    pm = to_mps(p)
    result = x.to("mps")[None]
    metal._run("gibbs", result, pm["bias"][None], metal._layout(pm["coupling_matrix"])[None], 2, 0.2,
               sites=sites.to("mps")[None], uniforms=uniforms.to("mps")[None])
    torch.testing.assert_close(result[0].cpu().long(), reference("gibbs", x, p, sites, uniforms, 0.2))
