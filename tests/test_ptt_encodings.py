"""Categorical profile draws, endpoint encoding ownership and unused outputs."""

import pytest
import torch

from adabmDCA import PTTSampler
from adabmDCA.ptt import sampler as sampler_module
from adabmDCA.sampling import sampling_profile, sampling_profile_categorical
from tests.test_ptt import profile

DEVICES = ['cpu', pytest.param('mps', marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Requires MPS'))]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('beta', [0., .7, 1.5])
def test_profile_draws_preserve_legacy_rng_and_states(device, beta):
    generator = torch.Generator().manual_seed(53)
    bias = torch.randn(5, 7, generator=generator).T.to(device)
    get_rng = torch.mps.get_rng_state if device == 'mps' else torch.get_rng_state
    set_rng = torch.mps.set_rng_state if device == 'mps' else torch.set_rng_state
    initial = get_rng()
    logits = beta * bias[None].expand(257, -1, -1)
    indices = torch.multinomial(logits.reshape(-1, 5).softmax(-1), 1).squeeze(-1)
    expected = torch.nn.functional.one_hot(indices, 5).reshape(257, 7, 5).to(bias.dtype)
    after = get_rng()
    set_rng(initial)
    actual = sampling_profile_categorical({'bias': bias}, 257, beta)
    torch.testing.assert_close(actual, expected.argmax(-1), atol=0, rtol=0)
    assert torch.equal(after, get_rng())
    set_rng(initial)
    torch.testing.assert_close(sampling_profile({'bias': bias}, 257, beta), expected, atol=0, rtol=0)
    assert torch.equal(after, get_rng())


def make_sampler(device):
    return PTTSampler({k: v.float().to(device) for k, v in profile().items()}, tokens='AB', n_chains=64, seed=3)


@pytest.mark.parametrize('device', DEVICES)
def test_endpoint_cache_reuses_encoding_and_invalidates_mutations(device, monkeypatch):
    sampler = make_sampler(device)
    original = sampler_module._one_hot
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(sampler_module, '_one_hot', counted)
    first = sampler.endpoint_samples(copy=False)
    assert sampler.endpoint_samples(copy=False) is first
    assert len(calls) == 1
    copied = sampler.endpoint_samples()
    copied.zero_()
    assert sampler.endpoint_samples(copy=False) is first
    assert bool(first.any())
    sampler.chains[-1][0, 0] = 1 - sampler.chains[-1][0, 0]
    second = sampler.endpoint_samples(copy=False)
    assert second is not first
    torch.testing.assert_close(second.argmax(-1).int(), sampler.chains[-1])
    # Even accidental mutation of a borrowed encoding must not poison reads.
    second.zero_()
    rebuilt = sampler.endpoint_samples(copy=False)
    torch.testing.assert_close(rebuilt.sum(-1), torch.ones(64, 2, device=device))
    sampler.chains[-1] = sampler.chains[-1].clone()
    assert sampler.endpoint_samples(copy=False) is not rebuilt
    assert len(calls) == 4
    if device == 'cpu':
        sampler.models[-1]['bias'] = sampler.models[-1]['bias'].double()
        assert sampler.endpoint_samples(copy=False).dtype == torch.float64


@pytest.mark.parametrize('device', DEVICES)
def test_endpoint_cache_is_excluded_from_forks_and_recovery(device):
    sampler = make_sampler(device)
    cached = sampler.endpoint_samples(copy=False)
    checkpoint = sampler.capture_state()
    assert '_endpoint_samples_cache' not in checkpoint
    fork = sampler.fork()
    assert fork._endpoint_samples_cache is None
    assert fork.endpoint_samples(copy=False) is not cached
    fork.chains[-1].zero_()
    torch.testing.assert_close(sampler.endpoint_samples(copy=False), cached, atol=0, rtol=0)
    sampler.restore_state(checkpoint)
    assert sampler._endpoint_samples_cache is None
    torch.testing.assert_close(sampler.endpoint_samples(copy=False), cached, atol=0, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_endpoint_inference_tensors_are_not_cached(device):
    with torch.inference_mode():
        sampler = make_sampler(device)
        first = sampler.endpoint_samples(copy=False)
        assert sampler._endpoint_samples_cache is None
        sampler.chains[-1].zero_()
        second = sampler.endpoint_samples(copy=False)
        assert second is not first
        torch.testing.assert_close(second.argmax(-1).int(), sampler.chains[-1])


@pytest.mark.parametrize('device', DEVICES)
def test_advance_without_output_preserves_state_and_rng(device, monkeypatch):
    sampler = make_sampler(device)
    expected = sampler.fork()
    samples = expected.advance(rounds=3, local_sweeps=2)

    def unused(*args, **kwargs):
        pytest.fail('Advancing without output must not construct endpoint samples')

    monkeypatch.setattr(sampler, 'endpoint_samples', unused)
    assert sampler.advance(rounds=3, local_sweeps=2, return_samples=False) is None
    for actual, reference in zip(sampler.chains, expected.chains):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    assert torch.equal(sampler._rng, expected._rng)
    assert sampler.acceptance == expected.acceptance
    assert samples.shape == (64, 2, 2)


@pytest.mark.parametrize('device', DEVICES)
def test_internal_mixing_skips_endpoint_outputs(device, monkeypatch):
    from tests.test_ptt_mixing import fast_config

    sampler = PTTSampler({k: v.float().to(device) for k, v in profile().items()}, tokens='AB', n_chains=64,
                         seed=3, config=fast_config())
    calls = []
    original = PTTSampler.advance

    def counted(self, **kwargs):
        calls.append(kwargs.get('return_samples', True))
        return original(self, **kwargs)

    monkeypatch.setattr(PTTSampler, 'advance', counted)
    sampler.estimate_mixing_time(local_sweeps=1, max_rounds=32)
    assert calls and not any(calls)
