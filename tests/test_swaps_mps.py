"""Fused pair movement: metadata, sequential ladders, RNG and archives."""

from itertools import product

import pytest
import torch

from adabmDCA import PTTConfig, PTTSampler
from adabmDCA.mps_kernels import is_mps_available
from adabmDCA.mps_kernels.swaps import acceptance_rates, swap_and_permute
from tests.test_sampling_mps import model, to_mps

pytestmark = pytest.mark.skipif(not is_mps_available(), reason='Requires Metal shaders')


@pytest.mark.parametrize('n', [1, 7, 129, 2000])
@pytest.mark.parametrize('enabled', list(product([False, True], repeat=3)))
def test_swap_and_permute_matches_tensor_reference(n, enabled):
    generator = torch.Generator().manual_seed(19)
    length = 7
    x = torch.randint(5, (2, n, length), generator=generator, dtype=torch.int32)
    ids = torch.arange(2 * n).reshape(2, n) + 2**40
    values = {'birth': ids - 2**41, 'reached': (ids % 2).bool(),
              'memory': torch.randn(2, n, generator=generator)}
    log_a = -torch.rand(n, generator=generator)
    log_u = torch.rand(n, generator=generator).log()
    # Include exact threshold equality: the comparison is strict.
    log_u[0] = log_a[0]
    orders = tuple(torch.randperm(n, generator=generator) for _ in range(2))
    mask = log_u < log_a
    kwargs = {key: tuple(value.to('mps').unbind(0)) if use else None
              for (key, value), use in zip(values.items(), enabled)}
    counts = torch.zeros(3, 2, device='mps', dtype=torch.int32)
    result = swap_and_permute(tuple(x.to('mps').unbind(0)), tuple(ids.to('mps').unbind(0)),
                              log_a.to('mps'), log_u.to('mps'), tuple(o.to('mps') for o in orders), counts, 3, **kwargs)
    for key, original in [('chains', x), ('lineage', ids), *values.items()]:
        if key in kwargs and kwargs[key] is None:
            assert result[key] is None
            continue
        m = mask[:, None] if original.ndim == 3 else mask
        expected = torch.stack([torch.where(m, original[1], original[0])[orders[0]],
                                torch.where(m, original[0], original[1])[orders[1]]])
        torch.testing.assert_close(result[key].cpu(), expected, atol=0, rtol=0)
    assert int(counts[1, 1]) == int(mask.sum())
    assert int(counts.sum()) == int(mask.sum())


@pytest.mark.parametrize('n', [1, 7, 129, 2000])
def test_acceptance_means_match_reference_round_accumulation(n):
    generator = torch.Generator().manual_seed(14)
    masks = (torch.rand(23, 6, n, generator=generator) < .63).to('mps')
    counts = masks.sum(-1).int()
    expected = torch.zeros(6, dtype=torch.float64)
    for row in masks[:17]:
        expected += row.float().mean(-1).cpu()
    actual = torch.tensor(acceptance_rates(counts, 17, n), dtype=torch.float64)
    torch.testing.assert_close(actual, expected / 17, atol=0, rtol=0)


def ladder(replicas, *, mode, sparse=False, reservoir=False):
    length, q, n = 13, 5, 129
    params = model(length, q)
    if sparse:
        mask = torch.zeros(length, length, dtype=torch.bool)
        for i in range(0, length - 1, 2):
            mask[i, i + 1] = mask[i + 1, i] = True
        params['coupling_matrix'] *= mask[:, None, :, None]
    anchor = to_mps(params)
    anchor['coupling_matrix'] = torch.zeros_like(anchor['coupling_matrix'])
    sampler = PTTSampler(anchor, tokens='ABCDE', n_chains=n, seed=24,
                         config=PTTConfig(max_replicas=max(2, replicas - 1)))
    sampler.models = [{'bias': anchor['bias'].clone(), 'coupling_matrix': params['coupling_matrix'].to('mps') * (k / (replicas - 1))}
                      for k in range(replicas)]
    sampler.models[0] = {key: value.clone() for key, value in anchor.items()}
    generator = torch.Generator().manual_seed(53)
    sampler.chains = [torch.randint(q, (n, length), generator=generator, dtype=torch.int32).to('mps') for _ in range(replicas)]
    sampler._reset_lineage()
    sampler.acceptance = [1.] * (replicas - 1)
    sampler.total_models = replicas
    sampler.mode = mode
    sampler.start_birth_tracking()
    sampler.birth += 2**40
    sampler.reached_top = (sampler.lineage % 2).bool()
    sampler.lag_memory = torch.randn(replicas, n, generator=generator).to('mps')
    if reservoir:
        sampler.active_start = 1
        sampler.reservoir = sampler.chains[0].repeat(2, 1)
        sampler.acceptance = [1.] * (replicas - 2)
    return sampler


@pytest.mark.parametrize('replicas,mode,reservoir', [(2, 'train', False), (3, 'train', False), (7, 'generate', False),
                                                   (3, 'train', True), (7, 'generate', True)])
@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('sweeps', [1, 2])
def test_ladder_trajectory_and_all_metadata_match_reference(replicas, mode, sparse, reservoir, sweeps, monkeypatch):
    sampler = ladder(replicas, mode=mode, sparse=sparse, reservoir=reservoir)
    reference = sampler.fork()
    global_rng = torch.mps.get_rng_state().clone()
    sampler.advance(rounds=3, local_sweeps=sweeps, return_samples=False)
    monkeypatch.setenv('ADABMDCA_MPS_SWAPS', '0')
    reference.advance(rounds=3, local_sweeps=sweeps, return_samples=False)
    for actual, expected in zip(sampler.chains, reference.chains):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for key in ('lineage', 'birth', 'reached_top', 'lag_memory', '_rng'):
        torch.testing.assert_close(getattr(sampler, key), getattr(reference, key), atol=0, rtol=0)
    assert sampler.acceptance == reference.acceptance
    assert sampler.rounds == reference.rounds and sampler.local_sweeps == reference.local_sweeps
    assert torch.equal(torch.mps.get_rng_state(), global_rng)


def test_fixed_ladder_archive_resumes_exactly(tmp_path):
    sampler = ladder(7, mode='generate', sparse=True)
    sampler.advance(rounds=2, local_sweeps=2, return_samples=False)
    # Save a legal training archive, then exercise its fixed numerical ladder.
    sampler.mode = 'train'
    archive = sampler.save_archive(tmp_path / 'swaps.h5')
    restored = PTTSampler.from_archive(archive, device='mps', mode='resume')
    sampler.advance(rounds=3, local_sweeps=2, return_samples=False)
    restored.advance(rounds=3, local_sweeps=2, return_samples=False)
    for actual, expected in zip(sampler.chains, restored.chains):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(sampler.lineage, restored.lineage, atol=0, rtol=0)
    assert torch.equal(sampler._rng, restored._rng)


def test_until_callback_uses_completed_acceptance_rows(monkeypatch):
    sampler = ladder(3, mode='train')
    reference = sampler.fork()
    sampler.advance(rounds=7, local_sweeps=1, until=lambda: sampler.rounds >= 2, return_samples=False)
    monkeypatch.setenv('ADABMDCA_MPS_SWAPS', '0')
    reference.advance(rounds=2, local_sweeps=1, return_samples=False)
    assert sampler.acceptance == reference.acceptance
    assert torch.equal(sampler._rng, reference._rng)


def test_inference_swaps_work():
    with torch.inference_mode():
        sampler = ladder(3, mode='train')
        sampler.advance(rounds=2, local_sweeps=1, return_samples=False)
        assert sampler.chains[-1].shape == (129, 13)


def test_public_generation_uses_seven_fixed_models_and_matches_reference(monkeypatch):
    from adabmDCA.alignment import Alignment
    from adabmDCA.api.ptt import sample_ptt_sequences

    sampler = ladder(7, mode='generate', sparse=True)
    sampler.model_version = 6
    sampler.ptt_checkpoints = [{'step': k, 'params': p} for k, p in enumerate(sampler.models[:-1])]
    reference = Alignment(('a', 'b'), ('A' * 13, 'E' * 13))
    options = {'source': sampler, 'n_sequences': 21, 'local_sweeps': 1, 'reference_fasta': reference,
               'seed': 11, 'no_reweighting': True, 'max_rounds': 500, 'n_measure': 21}
    actual = sample_ptt_sequences(**options)
    monkeypatch.setenv('ADABMDCA_MPS_SWAPS', '0')
    expected = sample_ptt_sequences(**options)
    assert len(actual.ptt_diagnostics['models']) == 7
    assert actual.sequences == expected.sequences
    torch.testing.assert_close(torch.as_tensor(actual.energies), torch.as_tensor(expected.energies), atol=0, rtol=0)
