"""Apple GPU PTT correctness, precision, RNG and checkpoint regression gates."""

from dataclasses import replace
from itertools import product

import pytest
import torch

from adabmDCA import PTTSampler, train_model
from adabmDCA.exceptions import InputValidationError
from adabmDCA.mps_kernels import exchange_log_acceptance, sample_replicas
from adabmDCA.ptt import bridge_increment
from adabmDCA.ptt.mixing import replica_autocorrelation
from adabmDCA.ptt.optim import _PTTOptimizer
from adabmDCA.statmech import compute_energy
from tests.test_ptt import coupled, profile, states, training_case

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple MPS GPU")


def mps(params):
    return {key: value.float().to('mps') for key, value in params.items()}


@pytest.mark.parametrize('length,q', [(136, 5), (96, 21)])
@pytest.mark.parametrize('sparse', [False, True])
def test_directional_scores_match_float64_energy_and_covariance(length, q, sparse):
    generator = torch.Generator().manual_seed(63)
    # Noncontiguous inputs and weak, signed directions exercise the matrix
    # path with dense and partially activated coupling blocks.
    samples = torch.nn.functional.one_hot(torch.randint(q, (519, length), generator=generator), q).float()[::2]
    h = (torch.randn(q, length, generator=generator) * .002).T
    j = torch.randn(length, q, length, q, generator=generator) * .001
    j = (j + j.permute(2, 3, 0, 1)) / 2
    j[torch.arange(length), :, torch.arange(length), :] = 0
    if sparse:
        j *= (torch.rand(length, q, length, q, generator=generator) > .9)
    j = j.permute(2, 3, 0, 1)
    expected = torch.stack(_PTTOptimizer._directional_score_components(samples.double(), h.double(), j.double()))
    actual = torch.stack(_PTTOptimizer._directional_score_components(samples.to('mps'), h.to('mps'), j.to('mps'))).cpu().double()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.sum(0), -compute_energy(samples.double().contiguous(),
                               {'bias': h.double().contiguous(), 'coupling_matrix': j.double().contiguous()}),
                               atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(torch.cov(actual), torch.cov(expected), atol=2e-8, rtol=2e-5)


def test_directional_scores_bound_large_population_intermediate():
    # Cross the 32 MiB projection boundary even with a small coupling matrix.
    q, n = 32, 270000
    generator = torch.Generator().manual_seed(14)
    samples = torch.nn.functional.one_hot(torch.randint(q, (n, 1), generator=generator), q).float()
    h = torch.randn(1, q, generator=generator) * .01
    j = torch.randn(1, q, 1, q, generator=generator) * .01
    actual = _PTTOptimizer._directional_score_components(samples.to('mps'), h.to('mps'), j.to('mps'))
    indices = samples.argmax(-1)[:, 0]
    expected = (h[0, indices], .5 * j[0, indices, 0, indices])
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result.cpu(), reference, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('q', [2, 5, 21])
@pytest.mark.parametrize('n', [0, 1, 7, 129])
def test_exchange_matches_float64_hamiltonian(q, n):
    generator = torch.Generator().manual_seed(2)
    models = []
    length = 7
    for _ in range(2):
        h = torch.randn(length, q, generator=generator) / 3
        j = torch.randn(length, q, length, q, generator=generator) / 10
        j = (j + j.permute(2, 3, 0, 1)) / 2
        j[torch.arange(length), :, torch.arange(length), :] = 0
        models.append({'bias': h, 'coupling_matrix': j})
    x, y = [torch.randint(q, (n, length), generator=generator) for _ in range(2)]
    delta = {key: (models[1][key] - models[0][key]).double() for key in models[0]}
    hot = lambda value: torch.nn.functional.one_hot(value, q).double()
    reference = ((compute_energy(hot(y), delta) - compute_energy(hot(x), delta)).clamp_max(0)
                 if n else torch.empty(0, dtype=torch.float64))
    actual = exchange_log_acceptance(*map(mps, models), x.to('mps'), y.to('mps')).cpu().double()
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('method', ['gibbs', 'metropolis', 'metropolized_gibbs'])
def test_replicas_follow_distinct_models(method):
    models = [mps(profile()), mps(coupled())]
    biases = torch.stack([p['bias'] for p in models])
    couplings = torch.stack([p['coupling_matrix'] for p in models])
    population = torch.zeros(2, 12000, 2, dtype=torch.int32, device='mps')
    result = sample_replicas(method, population, biases, couplings, nsweeps=20)
    assert not bool(population.any())
    for row, params in zip(result.cpu(), [profile(), coupled()]):
        observed = torch.bincount(row[:, 0] * 2 + row[:, 1], minlength=4).double() / len(row)
        expected = (-compute_energy(states(), params)).softmax(0)
        torch.testing.assert_close(observed, expected, atol=0.025, rtol=0)


@pytest.mark.parametrize('method', ['gibbs', 'metropolis', 'metropolized_gibbs'])
def test_distribution_diagnostics_rng_and_archive(method, tmp_path):
    global_mps = torch.mps.get_rng_state().clone()
    global_cpu = torch.get_rng_state().clone()
    sampler = PTTSampler(mps(profile()), tokens='AB', n_chains=12000, sampler=method, seed=6)
    assert sampler.transition_target(mps(coupled()), local_sweeps=3)[0]
    sampler.advance(rounds=20, local_sweeps=2)
    assert torch.equal(global_mps, torch.mps.get_rng_state())
    assert torch.equal(global_cpu, torch.get_rng_state())
    observed = sampler.chains[-1].cpu()
    counts = torch.bincount(observed[:, 0] * 2 + observed[:, 1], minlength=4).double() / len(observed)
    energy = compute_energy(states(), coupled())
    probability = (-energy).softmax(0)
    torch.testing.assert_close(counts, probability, atol=0.025, rtol=0)
    log_z = float(torch.logsumexp(-energy, 0))
    assert sampler.partition_estimate().log_z == pytest.approx(log_z, abs=0.03)
    assert sampler.partition_estimate(estimator='bar').log_z == pytest.approx(log_z, abs=0.03)
    assert sampler.entropy()['entropy'] == pytest.approx(float((energy * probability).sum()) + log_z, abs=0.04)
    sampler.ladder_statistics(states().float().to('mps'), torch.ones(4, device='mps'))
    sampler.ladder_health(bootstrap=3)
    updated = sampler.endpoint_params()
    updated['bias'] += 2
    assert bridge_increment(sampler.models[-1], updated, sampler.chains[-1]) == pytest.approx(4, abs=1e-5)
    sampler._record_endpoint_update(updated)
    assert sampler.lag_memory.dtype == torch.float32
    path = sampler.save_archive(tmp_path / 'mps.h5')
    restored = PTTSampler.from_archive(path, device='mps', mode='resume')
    fork = sampler.fork()
    expected = sampler.advance(rounds=3, local_sweeps=2)
    torch.testing.assert_close(expected, restored.advance(rounds=3, local_sweeps=2), atol=0, rtol=0)
    torch.testing.assert_close(expected, fork.advance(rounds=3, local_sweeps=2), atol=0, rtol=0)
    torch.testing.assert_close(sampler.lag_memory, restored.lag_memory, atol=0, rtol=0)
    inspected = PTTSampler.from_archive(path, device='cpu', mode='inspect')
    assert inspected.lag_memory.dtype == torch.float32
    with pytest.raises(InputValidationError, match='Changing PTT backend'):
        PTTSampler.from_archive(path, device='cpu', mode='resume')
    moved = PTTSampler.from_archive(path, device='cpu', mode='generate', seed=8)
    moved.advance(rounds=2)


@pytest.mark.parametrize('model,optimizer,activation', [(m, o, a) for m, o, a in product(['bmDCA', 'eaDCA', 'edgeDCA'], ['sgd', 'adaptive'], ['fixed', 'adaptive'])
                                                    if a == 'fixed' or m == 'eaDCA'])
def test_training(model, optimizer, activation, tmp_path):
    path, config = training_case.__wrapped__(tmp_path)
    config = replace(config, device='mps', dtype='float32', model_type=model,
                     ptt=replace(config.ptt, optimizer=optimizer, activation=activation, activation_significance=0))
    result = train_model(path, config=config, output_dir=tmp_path / 'output')
    assert all(value.device.type == 'mps' and torch.isfinite(value).all() for value in result.model.params.values())
    backend = result.ptt_sampler
    assert backend.device.type == 'mps'
    if optimizer == 'adaptive':
        samples = backend.endpoint_samples()
        names, values = _PTTOptimizer._lag_observables(backend, samples)
        assert len(names) == len(values)
        assert values.dtype == torch.float32
        assert torch.isfinite(_PTTOptimizer._lags(backend, values)).all()


def test_mixing_diagnostic_on_cpu():
    labels = torch.randint(3, (64, 3, 12))
    torch.testing.assert_close(replica_autocorrelation(labels.to('mps')), replica_autocorrelation(labels))


def test_disabled_metal_backend(monkeypatch):
    monkeypatch.setenv('ADABMDCA_MPS', '0')
    sampler = PTTSampler(mps(profile()), tokens='AB', n_chains=32, seed=2)
    sampler.advance(rounds=2, local_sweeps=2)
    assert sampler._replica_kernel is None
    assert torch.isfinite(sampler.endpoint_samples()).all()


def test_training_exact_resume_and_generation(tmp_path):
    from adabmDCA.api.ptt import estimate_ptt_entropy, sample_ptt_sequences

    path, config = training_case.__wrapped__(tmp_path)
    config = replace(config, device='mps', dtype='float32')
    full = train_model(path, validation_path=path, config=config)
    first = train_model(path, validation_path=path, config=replace(config, max_epochs=1), output_dir=tmp_path / 'first')
    resumed = train_model(path, validation_path=path, config=config, ptt_resume=first.artifacts['ptt_archive'])
    assert full.gradient_steps == resumed.gradient_steps == 3
    torch.testing.assert_close(full.chains, resumed.chains, atol=0, rtol=0)
    for key in full.model.params:
        torch.testing.assert_close(full.model.params[key], resumed.model.params[key], atol=0, rtol=0)
    assert full.partition_estimate == resumed.partition_estimate
    sampled = sample_ptt_sequences(source=full.ptt_sampler, n_sequences=80, local_sweeps=2,
                                   max_rounds=100, seed=7, reference_fasta=path, no_reweighting=True)
    assert len(sampled.sequences) == 80
    entropy = estimate_ptt_entropy(model=full.ptt_sampler, device='mps', n_sweeps=2, seed=7)
    assert torch.isfinite(torch.tensor(entropy.entropy))


def test_lag_memory_tracks_exchanges():
    sampler = PTTSampler(mps(profile()), tokens='AB', n_chains=64, seed=3)
    sampler.models.append(mps(coupled()))
    sampler.chains.append(sampler.chains[-1].clone())
    sampler._reset_lineage()
    sampler.acceptance = [1., 1.]
    sampler.lag_memory = sampler.lineage.float() + 1
    sampler.advance(rounds=5, local_sweeps=3)
    carried = sampler.lag_memory == sampler.lineage.float() + 1
    assert bool((carried | (sampler.lag_memory == 0)).all())
    assert bool(carried[1:].any())
    assert bool((sampler.lag_memory[0] == 0).all())


def test_unsupported_alphabet_falls_back():
    q = 33
    params = {'bias': torch.zeros(2, q, device='mps'), 'coupling_matrix': torch.zeros(2, q, 2, q, device='mps')}
    sampler = PTTSampler(params, tokens=''.join(chr(65 + i) for i in range(q)), n_chains=8, seed=3)
    sampler.models.append({k: v.clone() for k, v in params.items()})
    sampler.chains.append(sampler.chains[-1].clone())
    sampler._reset_lineage()
    sampler.acceptance = [1., 1.]
    sampler.advance(rounds=2, local_sweeps=2)
    assert sampler.endpoint_samples().shape == (8, 2, q)


@pytest.mark.parametrize('steering_input', ['onehot', 'sequences'])
def test_steered_ptt_sampling(steering_input, tmp_path):
    from adabmDCA.api.ptt import sample_ptt_sequences
    from tests.test_steering import SEQUENCES, coupled, exact, onehot_potential, string_potential, total_variation

    reference = tmp_path / 'reference.fasta'
    reference.write_text(''.join(f'>s{i}\n{s}\n' for i, s in enumerate(SEQUENCES)))
    parameters = mps(coupled())
    anchor = {k: v.clone() for k, v in parameters.items()}
    anchor['coupling_matrix'].zero_()
    sampler = PTTSampler(anchor, tokens='AB', n_chains=2000, seed=1)
    assert sampler.transition_target(parameters, local_sweeps=2)[0]
    result = sample_ptt_sequences(source=sampler, n_sequences=4000, reference_fasta=reference,
                                  local_sweeps=2, max_rounds=500, steering_strength=0.8,
                                  steering_potential=onehot_potential if steering_input == 'onehot' else string_potential,
                                  steering_input=steering_input, seed=7, no_reweighting=True)
    probability, log_z = exact(0.8)
    _, unsteered_z = exact(0)
    assert total_variation(list(result.sequences), probability) < 0.045
    assert result.steering['log_z_ratio'] == pytest.approx(log_z - unsteered_z, abs=0.06)


@pytest.mark.parametrize('length,q', [(136, 5), (96, 21)])
def test_exchange_family_sized_models(length, q):
    generator = torch.Generator().manual_seed(8)
    models = []
    for _ in range(2):
        h = torch.randn(length, q, generator=generator) / 10
        j = torch.randn(length, q, length, q, generator=generator) / 100
        j = (j + j.permute(2, 3, 0, 1)) / 2
        j[torch.arange(length), :, torch.arange(length), :] = 0
        models.append({'bias': h, 'coupling_matrix': j})
    x, y = [torch.randint(q, (17, length), generator=generator) for _ in range(2)]
    delta = {key: (models[1][key] - models[0][key]).double() for key in models[0]}
    hot = lambda value: torch.nn.functional.one_hot(value, q).double()
    expected = (compute_energy(hot(y), delta) - compute_energy(hot(x), delta)).clamp_max(0)
    actual = exchange_log_acceptance(*map(mps, models), x.to('mps'), y.to('mps'))
    torch.testing.assert_close(actual.cpu().double(), expected, atol=5e-6, rtol=5e-6)


def test_cancellation_between_local_sweeps():
    from adabmDCA.exceptions import OperationCancelledError

    sampler = PTTSampler(mps(profile()), tokens='AB', n_chains=32)
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(OperationCancelledError):
        sampler.advance(rounds=2, local_sweeps=10, is_cancelled=cancelled)
    assert sampler.local_sweeps == 1
