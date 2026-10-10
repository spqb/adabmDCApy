"""Seven-model CPU generation fusion: exact movement, RNG and narrow dispatch."""

from itertools import product

import pytest
import torch

pytest.importorskip('numba')

from adabmDCA import PTTConfig, PTTSampler
from adabmDCA.numba_kernels import is_numba_available
from adabmDCA.numba_kernels.swaps import swap_and_permute, swaps_available
from tests.test_numba_kernels import random_potts

pytestmark = pytest.mark.skipif(not is_numba_available(), reason='Requires Numba')


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('n', [1, 129, 2000])
@pytest.mark.parametrize('enabled', list(product([False, True], repeat=3)))
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_movement_matches_reference(n, enabled, dtype):
    generator = torch.Generator().manual_seed(19)
    x = torch.randint(5, (2, n, 7), generator=generator, dtype=torch.int32)
    ids = torch.arange(2 * n).reshape(2, n) + 2**40
    values = {'birth': ids - 2**41, 'reached': (ids % 2).bool(),
              'memory': torch.randn(2, n, generator=generator, dtype=dtype)}
    log_a = -torch.rand(n, generator=generator, dtype=torch.float64)
    log_u = torch.rand(n, generator=generator).log()
    log_a[0] = log_u[0]  # Strict comparison, including float32/float64 promotion.
    orders = tuple(torch.randperm(n, generator=generator) for _ in range(2))
    mask = log_u < log_a
    kwargs = {key: tuple(value.unbind(0)) if use else None for (key, value), use in zip(values.items(), enabled)}
    result = swap_and_permute(tuple(x.unbind(0)), tuple(ids.unbind(0)), log_a, log_u, orders, **kwargs)
    for key, original in [('chains', x), ('lineage', ids), *values.items()]:
        if key in kwargs and kwargs[key] is None:
            assert result[key] is None
            continue
        m = mask[:, None] if original.ndim == 3 else mask
        expected = torch.stack([torch.where(m, original[1], original[0])[orders[0]],
                                torch.where(m, original[0], original[1])[orders[1]]])
        torch.testing.assert_close(result[key], expected, atol=0, rtol=0)
    assert result['accepted'] == int(mask.sum())


def ladder(replicas=7, *, dtype=torch.float32, sparse=False, mode='generate'):
    params = random_potts(13, 5, dtype, seed=13)
    if sparse:
        mask = torch.zeros(13, 13, dtype=torch.bool)
        for i in range(0, 12, 2):
            mask[i, i + 1] = mask[i + 1, i] = True
        params['coupling_matrix'] *= mask[:, None, :, None]
    anchor = {'bias': params['bias'].clone(), 'coupling_matrix': torch.zeros_like(params['coupling_matrix'])}
    sampler = PTTSampler(anchor, tokens='ABCDE', n_chains=129, seed=24,
                         config=PTTConfig(max_replicas=max(2, replicas - 1)))
    sampler.models = [{'bias': anchor['bias'].clone(), 'coupling_matrix': params['coupling_matrix'] * (k / (replicas - 1))}
                      for k in range(replicas)]
    sampler.models[0] = {key: value.clone() for key, value in anchor.items()}
    sampler.ptt_checkpoints = [{'step': k, 'params': p} for k, p in enumerate(sampler.models[:-1])]
    sampler.model_version = replicas - 1
    sampler.total_models = replicas
    sampler.prepare_sampling_ladder(129)
    sampler.mode = mode
    sampler.lineage += 2**40
    sampler.birth -= 2**40
    sampler.reached_top = (sampler.lineage % 2).bool()
    sampler.lag_memory = torch.randn(replicas, 129, generator=torch.Generator().manual_seed(53), dtype=dtype)
    return sampler


def assert_same(actual, expected):
    for a, b in zip(actual.chains, expected.chains):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for key in ('lineage', 'birth', 'reached_top', 'lag_memory', '_rng'):
        a, b = getattr(actual, key), getattr(expected, key)
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert actual.acceptance == expected.acceptance
    assert actual.rounds == expected.rounds and actual.local_sweeps == expected.local_sweeps


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('n_threads', [1, 4])
def test_fixed_seven_model_trajectory_and_rng(dtype, sparse, n_threads, monkeypatch):
    torch.set_num_threads(n_threads)
    sampler = ladder(dtype=dtype, sparse=sparse)
    reference = sampler.fork()
    global_rng = torch.get_rng_state().clone()
    sampler.advance(rounds=3, local_sweeps=2, return_samples=False)
    monkeypatch.setenv('ADABMDCA_NUMBA_SWAPS', '0')
    reference.advance(rounds=3, local_sweeps=2, return_samples=False)
    assert_same(sampler, reference)
    assert torch.equal(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize('replicas,mode', [(2, 'train'), (3, 'train'), (7, 'train'),
                                         (2, 'generate'), (3, 'generate'), (6, 'generate'), (8, 'generate')])
def test_other_ladders_do_not_dispatch(replicas, mode, monkeypatch):
    from adabmDCA.numba_kernels import swaps

    sampler = ladder(replicas, mode=mode)
    assert not swaps_available(sampler, sampler.birth, sampler.reached_top, sampler.lag_memory)

    def forbidden(*args, **kwargs):
        pytest.fail('Fusion must only run for seven-model generation')

    monkeypatch.setattr(swaps, 'swap_and_permute', forbidden)
    sampler.advance(return_samples=False)


def test_dispatch_disabled_and_steering_reservoir_excluded(monkeypatch):
    sampler = ladder()
    assert swaps_available(sampler, sampler.birth, sampler.reached_top, sampler.lag_memory)
    monkeypatch.setenv('ADABMDCA_NUMBA_SWAPS', '0')
    assert not swaps_available(sampler, sampler.birth, sampler.reached_top, sampler.lag_memory)
    monkeypatch.delenv('ADABMDCA_NUMBA_SWAPS')
    monkeypatch.setenv('ADABMDCA_NUMBA', '0')
    assert not swaps_available(sampler, sampler.birth, sampler.reached_top, sampler.lag_memory)
    monkeypatch.delenv('ADABMDCA_NUMBA')
    sampler.steering = object()
    assert not swaps_available(sampler, sampler.birth, sampler.reached_top, sampler.lag_memory)
    sampler.steering = None
    sampler.reservoir = sampler.chains[0]
    assert not swaps_available(sampler, sampler.birth, sampler.reached_top, sampler.lag_memory)


def test_partial_rounds_keep_exact_acceptance(monkeypatch):
    sampler = ladder(sparse=True)
    reference = sampler.fork()
    sampler.advance(rounds=7, until=lambda: sampler.rounds >= 2, return_samples=False)
    monkeypatch.setenv('ADABMDCA_NUMBA_SWAPS', '0')
    reference.advance(rounds=2, return_samples=False)
    assert_same(sampler, reference)


def test_strided_populations_and_metadata(monkeypatch):
    sampler = ladder(sparse=True)
    sampler.chains = [x.T.contiguous().T for x in sampler.chains]
    for key in ('lineage', 'birth', 'reached_top', 'lag_memory'):
        setattr(sampler, key, getattr(sampler, key).T.contiguous().T)
    reference = sampler.fork()
    sampler.advance(rounds=2, return_samples=False)
    monkeypatch.setenv('ADABMDCA_NUMBA_SWAPS', '0')
    reference.advance(rounds=2, return_samples=False)
    assert_same(sampler, reference)


def test_movement_accepts_inference_tensors():
    with torch.inference_mode():
        chains = torch.arange(42, dtype=torch.int32).reshape(2, 7, 3)
        ids = torch.arange(14).reshape(2, 7)
        orders = (torch.arange(6, -1, -1), torch.arange(7))
        result = swap_and_permute(tuple(chains.unbind(0)), tuple(ids.unbind(0)),
                                  torch.zeros(7, dtype=torch.float64), torch.full((7,), -1.), orders)
        assert result['accepted'] == 7
        assert torch.equal(result['chains'][0], chains[1, orders[0]])
        assert torch.equal(result['chains'][1], chains[0, orders[1]])


def test_public_generation_matches_reference(monkeypatch):
    from adabmDCA.alignment import Alignment
    from adabmDCA.api.ptt import sample_ptt_sequences

    sampler = ladder(sparse=True)
    options = {'source': sampler, 'n_sequences': 21, 'local_sweeps': 1,
               'reference_fasta': Alignment(('a', 'b'), ('A' * 13, 'E' * 13)),
               'seed': 11, 'no_reweighting': True, 'max_rounds': 500, 'n_measure': 21}
    actual = sample_ptt_sequences(**options)
    monkeypatch.setenv('ADABMDCA_NUMBA_SWAPS', '0')
    expected = sample_ptt_sequences(**options)
    assert len(actual.ptt_diagnostics['models']) == 7
    assert actual.sequences == expected.sequences
    torch.testing.assert_close(torch.as_tensor(actual.energies), torch.as_tensor(expected.energies), atol=0, rtol=0)
