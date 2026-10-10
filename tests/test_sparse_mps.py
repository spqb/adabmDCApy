"""Sparse Metal layouts, controlled trajectories, exchanges and dispatch."""

import pytest
import torch

from adabmDCA.mps_kernels import exchange_log_acceptance, is_mps_available
from adabmDCA.mps_kernels import sampling as metal
from adabmDCA.mps_kernels.sparse import sparse_coupling_layout
from adabmDCA.statmech import compute_energy
from tests.test_sampling_mps import model, reference, to_mps

pytestmark = pytest.mark.skipif(not is_mps_available(), reason='Metal shaders unavailable')


def sparse_model(length, q, seed=9, profile=False):
    params = model(length, q, seed)
    mask = torch.zeros(length, q, length, q, dtype=torch.bool)
    # Disconnected sites, degrees 0..2, and partially activated state blocks.
    if not profile:
        for i in range(0, length - 2, 3):
            for j in (i + 1, i + 2):
                mask[i, 0, j, -1] = mask[j, -1, i, 0] = True
    params['coupling_matrix'] *= mask
    return params


@pytest.mark.parametrize('q', [2, 5, 21, 32])
@pytest.mark.parametrize('method', metal.METHODS)
@pytest.mark.parametrize('profile', [False, True])
def test_controlled_sparse_trajectories(q, method, profile):
    length, n, steps = 13, 43, 529
    params = sparse_model(length, q, profile=profile)
    rng = torch.Generator().manual_seed(15)
    x = torch.randint(q, (n, length), generator=rng, dtype=torch.int32)
    sites = torch.randint(length, (steps,), generator=rng, dtype=torch.int32)
    uniforms = torch.rand(steps, n, 2, generator=rng)
    expected = reference(method, x, params, sites, uniforms, beta=0.7)
    pm = to_mps(params)
    layout = sparse_coupling_layout(pm['coupling_matrix'][None], source=pm['coupling_matrix'])
    assert layout is not None
    actual = x.to('mps')[None]
    metal._run(method, actual, pm['bias'][None], layout[1], steps, 0.7,
               sites=sites.to('mps')[None], uniforms=uniforms.to('mps')[None], sparse=layout)
    torch.testing.assert_close(actual[0].cpu().long(), expected, atol=0, rtol=0)


@pytest.mark.parametrize('method', metal.METHODS)
def test_sparse_stacked_replicas_and_rng(method, monkeypatch):
    models = [sparse_model(13, 5, seed=seed, profile=(seed == 0)) for seed in (0, 1, 2)]
    h = torch.stack([p['bias'] for p in models]).to('mps')
    j = torch.stack([p['coupling_matrix'] for p in models]).to('mps')
    x = torch.zeros(3, 70, 13, device='mps', dtype=torch.int32)
    rng = torch.mps.get_rng_state()
    sparse = metal.sample_replicas(method, x, h, j, 3)
    torch.mps.set_rng_state(rng)
    monkeypatch.setenv('ADABMDCA_SPARSE', '0')
    dense = metal.sample_replicas(method, x, h, j, 3)
    torch.testing.assert_close(sparse, dense, atol=0, rtol=0)
    assert not bool(x.any())


@pytest.mark.parametrize('q', [2, 5, 21])
@pytest.mark.parametrize('profile', [False, True])
def test_sparse_exchange_different_graphs_matches_energy(q, profile, monkeypatch):
    lower, upper = sparse_model(13, q, profile=profile), sparse_model(13, q, seed=1)
    # Move upper graph so the two lists have different neighbours and degrees.
    upper['coupling_matrix'] = upper['coupling_matrix'].roll((1, 1), dims=(0, 2))
    x, y = torch.randint(q, (43, 13)), torch.randint(q, (43, 13))
    delta = {key: (upper[key] - lower[key]).double() for key in lower}
    hot = lambda value: torch.nn.functional.one_hot(value, q).double()
    expected = (compute_energy(hot(y), delta) - compute_energy(hot(x), delta)).clamp_max(0)
    pm0, pm1 = to_mps(lower), to_mps(upper)
    actual = exchange_log_acceptance(pm0, pm1, x.to('mps'), y.to('mps'))
    torch.testing.assert_close(actual.cpu().double(), expected, atol=3e-6, rtol=3e-6)
    monkeypatch.setenv('ADABMDCA_SPARSE', '0')
    dense = exchange_log_acceptance(pm0, pm1, x.to('mps'), y.to('mps'))
    torch.testing.assert_close(actual, dense, atol=3e-6, rtol=3e-6)


def test_layout_contents_cache_updates_and_dense_fallback(monkeypatch):
    j = to_mps(sparse_model(13, 5))['coupling_matrix']
    first = sparse_coupling_layout(j[None], source=j)
    assert sparse_coupling_layout(j[None], source=j) is first
    neighbours, blocks, width = first
    assert width == 2
    for i in range(13):
        for slot, neighbour in enumerate(neighbours[0, i].cpu().tolist()):
            if neighbour >= 0:
                torch.testing.assert_close(blocks[0, i, slot], j[i, :, neighbour, :].T)
            else:
                assert not bool(blocks[0, i, slot].any())
    j[0, 1, 1, 1] = j[1, 1, 0, 1] = 9
    updated = sparse_coupling_layout(j[None], source=j)
    assert updated is not first
    assert float(updated[1][0, 0, 0, 1, 1]) == 9
    # Remove an entire edge; both the neighbours and its values must be rebuilt.
    j[0, :, 1, :] = j[1, :, 0, :] = 0
    removed = sparse_coupling_layout(j[None], source=j)
    assert removed is not updated
    assert 1 not in removed[0][0, 0].cpu().tolist()
    monkeypatch.setenv('ADABMDCA_SPARSE', '0')
    assert sparse_coupling_layout(j[None], source=j) is None
    monkeypatch.delenv('ADABMDCA_SPARSE')
    j.fill_(0.1)
    assert sparse_coupling_layout(j[None], source=j) is None


def test_inference_noncontiguous_and_mixed_dense_stack():
    with torch.inference_mode():
        j = to_mps(sparse_model(13, 5))['coupling_matrix'].permute(2, 3, 0, 1)
        first = sparse_coupling_layout(j[None], source=j)
        j.mul_(2)
        second = sparse_coupling_layout(j[None], source=j)
        torch.testing.assert_close(second[1], 2 * first[1])
    j = torch.stack([sparse_model(13, 5)['coupling_matrix'], model(13, 5)['coupling_matrix']]).to('mps')
    assert sparse_coupling_layout(j) is None


def test_ptt_sparse_dispatch_in_place_updates_and_archive(tmp_path, monkeypatch):
    from adabmDCA import PTTSampler
    from adabmDCA.mps_kernels import exchange as exchange_module

    target = to_mps(sparse_model(13, 5))
    anchor = {k: v.clone() for k, v in target.items()}
    anchor['coupling_matrix'].zero_()
    sampler = PTTSampler(anchor, tokens='ABCDE', n_chains=200, seed=8)
    sampler.models = [anchor, {k: 0.5 * v for k, v in target.items()}, target]
    sampler.chains.append(sampler.chains[-1].clone())
    sampler._reset_lineage()
    sampler.acceptance = [1., 1.]
    sampler.total_models = 3
    packed_before = sampler._stacked_replica_params([1, 2])
    j = target['coupling_matrix']
    j[0, 1, 1, 1] = j[1, 1, 0, 1] = 2
    target['bias'].add_(0.1)
    packed_after = sampler._stacked_replica_params([1, 2])
    assert packed_before[1] is not packed_after[1]
    torch.testing.assert_close(packed_after[1][-1], j)
    torch.testing.assert_close(packed_after[0][-1], target['bias'])
    calls = []

    def tracked(couplings, **kwargs):
        result = sparse_coupling_layout(couplings, **kwargs)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(metal, 'sparse_coupling_layout', tracked)
    monkeypatch.setattr(exchange_module, 'sparse_coupling_layout', tracked)
    sampler.advance(rounds=3, local_sweeps=3)
    assert len(calls) >= 6 and all(calls)
    archive = sampler.save_archive(tmp_path / 'sparse.h5')
    restored = PTTSampler.from_archive(archive, device='mps', mode='resume')
    torch.testing.assert_close(sampler.advance(rounds=2, local_sweeps=3),
                               restored.advance(rounds=2, local_sweeps=3), atol=0, rtol=0)


def test_ptt_inference_stack_is_not_stale():
    from adabmDCA import PTTSampler

    with torch.inference_mode():
        params = to_mps(sparse_model(13, 5, profile=True))
        sampler = PTTSampler(params, tokens='ABCDE', n_chains=8)
        first = sampler._stacked_replica_params([0, 1])
        sampler.models[-1]['bias'].add_(0.5)
        second = sampler._stacked_replica_params([0, 1])
        torch.testing.assert_close(second[0][-1], first[0][-1] + 0.5)


@pytest.mark.parametrize('method', metal.METHODS)
def test_sparse_stationary_distribution(method):
    from itertools import product

    length, q, n = 5, 3, 40000
    params = model(length, q)
    mask = torch.zeros(length, length, dtype=torch.bool)
    for i, j in ((0, 1), (2, 4)):
        mask[i, j] = mask[j, i] = True
    params['coupling_matrix'] *= mask[:, None, :, None]
    pm = to_mps(params)
    assert sparse_coupling_layout(pm['coupling_matrix'][None], source=pm['coupling_matrix']) is not None
    all_states = torch.tensor(list(product(range(q), repeat=length)))
    energy = compute_energy(torch.nn.functional.one_hot(all_states, q).float(), params)
    torch.manual_seed(17)
    final = metal.sample_categorical(method, torch.zeros(n, length, device='mps', dtype=torch.int32), pm, 25).cpu()
    index = (final * q ** torch.arange(length - 1, -1, -1)).sum(1)
    observed = torch.bincount(index, minlength=q**length).float() / n
    assert float((observed - (-energy).softmax(0)).abs().sum() / 2) < 0.04
