"""Numerical and algorithm regressions for PTT mixing and reservoir recovery."""

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from adabmDCA import PTTConfig, PTTSampler, train_model
from adabmDCA.ptt.mixing import (
    MixingEstimate,
    exponential_autocorrelation_time,
    integrated_autocorrelation_time,
    process_replica_experiment,
    replica_autocorrelation,
)
from adabmDCA.ptt.sampler import _replica_positions_by_lineage
from tests import test_ptt as helpers

coupled, profile = helpers.coupled, helpers.profile
single_thread, training_case = helpers.single_thread, helpers.training_case


def test_fft_matches_direct_non_circular_autocorrelation():
    rng = torch.Generator().manual_seed(19)
    indices = torch.randint(0, 3, (37, 3, 20), generator=rng)
    x = indices.double().reshape(37, -1) - 1
    expected = torch.stack([(x[:37 - lag] * x[lag:]).sum(0).mean() for lag in range(37 // 2)])
    expected /= expected[0].clone()
    torch.testing.assert_close(replica_autocorrelation(indices), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("tau", [2.0, 12.0, 40.0])
def test_exponential_fit_and_integrated_window(tau):
    correlation = torch.exp(-torch.arange(2000, dtype=torch.float64) / tau)
    original = correlation.clone()
    assert exponential_autocorrelation_time(correlation) == pytest.approx(tau, rel=1e-3)
    exact_integrated = 0.5 + 1 / np.expm1(1 / tau)
    assert integrated_autocorrelation_time(correlation) == pytest.approx(exact_integrated, rel=0.01)
    assert torch.equal(original, correlation)


def test_replica_experiment_recovers_known_two_state_relaxation():
    rng = torch.Generator().manual_seed(71)
    tau = 12.0
    flip_probability = (1 - np.exp(-1 / tau)) / 2
    flips = torch.rand(8192, 64, generator=rng) < flip_probability
    labels = flips.long().cumsum(0) % 2
    indices = torch.stack((labels, 1 - labels), dim=1)
    tau_int, tau_exp, correlation = process_replica_experiment(indices)
    assert tau_int == pytest.approx(tau, rel=0.15)
    assert tau_exp == pytest.approx(tau, rel=0.25)
    assert correlation[0] == 1


def test_lineage_recording_survives_population_slot_permutations():
    rng = torch.Generator().manual_seed(72)
    tau = 12.0
    n_chains = 64
    flip_probability = (1 - np.exp(-1 / tau)) / 2
    lineage = torch.arange(2 * n_chains).reshape(2, n_chains)
    trajectories = []
    slot_labels = []
    for _ in range(8192):
        swap = torch.rand(n_chains, generator=rng) < flip_probability
        saved = lineage[0, swap].clone()
        lineage[0, swap] = lineage[1, swap]
        lineage[1, swap] = saved
        lineage[0] = lineage[0, torch.randperm(n_chains, generator=rng)]
        lineage[1] = lineage[1, torch.randperm(n_chains, generator=rng)]
        trajectories.append(_replica_positions_by_lineage(lineage))
        slot_labels.append(lineage // n_chains)

    tau_int, tau_exp, _ = process_replica_experiment(torch.stack(trajectories))
    assert tau_int == pytest.approx(tau, rel=0.15)
    assert tau_exp == pytest.approx(tau, rel=0.25)
    # The previous fixed-slot recording loses identity at every permutation.
    old_tau_int, _, _ = process_replica_experiment(torch.stack(slot_labels))
    assert old_tau_int == 1


def fast_config(**kwargs):
    # These tests exercise the replica-index autocorrelation experiment.
    kwargs = {"mixing_method": "autocorrelation", **kwargs}
    return PTTConfig(mixing_chains=20, mixing_thermalization_rounds=2,
                     mixing_initial_rounds=100, mixing_max_rounds=500, **kwargs)


def test_trwa_preserves_populations_rng_and_reports_work():
    sampler = PTTSampler(profile(), tokens="AB", n_chains=30, config=fast_config())
    before = sampler.capture_state()
    result = sampler.estimate_mixing_time()
    assert result.converged
    assert result.tau_int == result.tau_exp == 1
    assert result.rounds == 100
    assert result.local_sweeps == 102
    assert sampler.local_sweeps == before["local_sweeps"] + result.local_sweeps
    assert torch.equal(sampler._rng, before["_rng"])
    assert all(torch.equal(x, y) for x, y in zip(sampler.chains, before["chains"]))


def test_mixing_and_reservoir_emit_bounded_work_updates():
    sampler = PTTSampler(profile(), tokens="AB", n_chains=40,
                         config=fast_config(reservoir_size=80))
    updates = []
    sampler.models.append(sampler.endpoint_params())
    sampler.chains.append(sampler.chains[-1].clone())
    sampler.total_models += 1
    sampler._reset_lineage()
    sampler.acceptance = [1.0, 1.0]
    assert sampler.refresh_reservoir(on_progress=lambda stage, current, total, **details:
                                     updates.append((stage, current, total, details)))
    assert ("mixing_warmup", 2, 2) in [(stage, current, total) for stage, current, total, _ in updates]
    assert ("mixing_measure", 100, 100) in [(stage, current, total) for stage, current, total, _ in updates]
    assert any(stage == "reservoir_warmup" and current == total for stage, current, total, _ in updates)
    assert ("reservoir_collect", 80, 80) in [(stage, current, total) for stage, current, total, _ in updates]
    assert all(0 <= current <= total for _, current, total, _ in updates)


def test_non_mixing_replica_labels_exceed_budget(monkeypatch):
    monkeypatch.setattr(
        "adabmDCA.ptt.sampler._prepare_exchange_kernel",
        lambda device: lambda lo, up, x, y: torch.full((len(x),), -1000.0, device=x.device),
    )
    sampler = PTTSampler(profile(), tokens="AB", n_chains=20, config=fast_config())
    result = sampler.estimate_mixing_time()
    assert not result.converged
    assert result.status == "budget_exceeded"
    assert result.required_rounds > sampler.config.mixing_max_rounds
    assert result.acceptance == (0.0,)


def test_explicit_mixing_budget_excludes_warmup(monkeypatch):
    sampler = PTTSampler(
        profile(), tokens="AB", n_chains=20,
        config=replace(fast_config(), mixing_thermalization_rounds=7),
    )
    monkeypatch.setattr(
        "adabmDCA.ptt.sampler.process_replica_experiment",
        lambda indices: (100.0, 100.0, torch.ones(1)),
    )

    result = sampler.estimate_mixing_time(max_rounds=25)

    assert result.status == "budget_exceeded"
    assert result.rounds == 25
    assert result.local_sweeps == 7 + 25


def test_unbounded_mixing_extends_past_training_budget(monkeypatch):
    sampler = PTTSampler(profile(), tokens="AB", n_chains=20, config=fast_config())
    monkeypatch.setattr("adabmDCA.ptt.sampler.process_replica_experiment",
                        lambda indices: (30.0, 30.0, torch.ones(1)))
    result = sampler.estimate_mixing_time(bounded=False)
    assert result.converged
    assert result.rounds == 600 > sampler.config.mixing_max_rounds


def test_unbounded_mixing_can_be_cancelled(monkeypatch):
    from adabmDCA.exceptions import OperationCancelledError

    sampler = PTTSampler(profile(), tokens="AB", n_chains=20, config=fast_config())
    monkeypatch.setattr("adabmDCA.ptt.sampler.process_replica_experiment",
                        lambda indices: (30.0, 30.0, torch.ones(1)))
    completed = []
    with pytest.raises(OperationCancelledError):
        sampler.estimate_mixing_time(
            bounded=False, is_cancelled=lambda: bool(completed and completed[-1] >= 550),
            on_progress=lambda stage, current, total, **details: completed.append(current),
        )


def test_partition_estimate_has_no_ess_diagnostic(training_case):
    data, config = training_case
    result = train_model(data, config=config)
    assert result.gradient_steps == 3
    assert set(result.history["bias_learning_rate"]) == {config.learning_rate}
    assert result.ptt_sampler.training_state["recovery_events"] == []
    assert not hasattr(result.partition_estimate, "bridge_ess")


def test_low_acceptance_with_fast_mixing_keeps_update(monkeypatch):
    sampler = PTTSampler(profile(), tokens="AB", n_chains=30, config=fast_config())
    original = PTTSampler.advance
    def low_acceptance(self, **kwargs):
        samples = original(self, **kwargs)
        self.acceptance = [0.05] * (self.n_active - 1)
        return samples
    monkeypatch.setattr(PTTSampler, "advance", low_acceptance)
    assert sampler.transition_target(coupled())[0]
    assert sampler.last_mixing["status"] == "converged"
    assert sampler.temporary is not None


@pytest.mark.parametrize("full_sampler", [False, True])
def test_reservoir_compression_normalizer_and_restart(tmp_path, full_sampler):
    sampler = PTTSampler(profile(), tokens="AB", n_chains=100, seed=8,
                         config=fast_config(full_sampler=full_sampler, reservoir_size=1000))
    # Identical replicas provide an exactly normalized compression case.
    sampler.models.append(sampler.endpoint_params())
    sampler.total_models += 1
    sampler.chains.append(sampler.chains[-1].clone())
    sampler._reset_lineage()
    sampler.acceptance = [1.0, 1.0]
    exact = sampler.partition_estimate().log_z
    assert sampler.refresh_reservoir()
    assert sampler.active_start == int(full_sampler) and sampler.n_active == 2
    assert sampler.reservoir.shape == (1000, 2)
    assert sampler.reservoir.dtype == torch.int32
    assert sampler.partition_estimate().log_z == pytest.approx(exact)
    # A later refresh can use either the old reservoir or the entire history.
    sampler.models.append(sampler.endpoint_params())
    sampler.total_models += 1
    sampler.chains.append(sampler.chains[-1].clone())
    sampler._reset_lineage()
    sampler.acceptance = [1.0, 1.0]
    assert sampler.refresh_reservoir()
    assert sampler.active_start == 2 * int(full_sampler) and sampler.n_active == 2
    assert sampler.partition_estimate().log_z == pytest.approx(exact)
    if full_sampler:
        active = sampler.estimate_mixing_time()
        complete = sampler.estimate_mixing_time(full_ladder=True)
        assert active.replicas == 2 and not active.full_ladder
        assert complete.replicas == 4 and complete.full_ladder
        assert sampler.active_start == 2
    else:
        with pytest.raises(ValueError, match="history was not retained"):
            sampler.estimate_mixing_time(full_ladder=True)
    path = sampler.save_archive(tmp_path / "compressed.h5")
    resumed = PTTSampler.from_archive(path, mode="resume")
    torch.testing.assert_close(sampler.draw(250), resumed.draw(250), rtol=0, atol=0)
    torch.testing.assert_close(sampler.reservoir, resumed.reservoir, rtol=0, atol=0)


def test_temporary_and_held_snapshots_survive_archives(tmp_path):
    sampler = PTTSampler(profile(), tokens="AB", n_chains=30, config=fast_config(max_replicas=3))
    sampler.acceptance = [0.4]
    sampler.update_replica_chain()
    resumed = PTTSampler.from_archive(sampler.save_archive(tmp_path / "temporary.h5"), mode="resume")
    for obj in (sampler, resumed):
        obj.models[-1] = coupled()
        obj.model_version = 1
        obj.acceptance = [0.2]
        obj.update_replica_chain()
    torch.testing.assert_close(sampler.held["chains"], resumed.held["chains"], rtol=0, atol=0)
    resumed = PTTSampler.from_archive(sampler.save_archive(tmp_path / "held.h5"), mode="resume")
    for obj in (sampler, resumed):
        obj.acceptance[-1] = 0.2
        obj.update_replica_chain()
    torch.testing.assert_close(sampler.draw(40), resumed.draw(40), rtol=0, atol=0)


def test_compressed_coupled_anchor_preserves_distribution_and_normalizer(tmp_path):
    from adabmDCA.statmech import compute_energy

    sampler = PTTSampler(profile(), tokens="AB", n_chains=4000, seed=12,
                         config=fast_config(reservoir_size=8000))
    sampler.models[-1] = coupled()
    endpoint = coupled()
    endpoint["bias"] *= 0.5
    sampler.models.append(endpoint)
    sampler.chains.append(sampler.chains[-1].clone())
    sampler.total_models += 1
    sampler._reset_lineage()
    sampler.acceptance = [1.0, 1.0]
    sampler.advance(rounds=30)
    assert sampler.refresh_reservoir()
    assert torch.count_nonzero(sampler.models[0]["coupling_matrix"])
    resumed = PTTSampler.from_archive(sampler.save_archive(tmp_path / "coupled_anchor.h5"), mode="resume")
    samples = resumed.draw(12000, spacing=3).argmax(-1)
    energies = compute_energy(helpers.states(), endpoint)
    probabilities = (-energies).softmax(0)
    observed = torch.bincount(samples[:, 0] * 2 + samples[:, 1], minlength=4).double() / len(samples)
    torch.testing.assert_close(observed, probabilities, atol=0.03, rtol=0)
    assert resumed.partition_estimate().log_z == pytest.approx(torch.logsumexp(-energies, 0).item(), abs=0.04)


@pytest.mark.parametrize("resume_before_failure", [False, True])
def test_mixing_recovery_searches_older_checkpoints_and_halves_once(
    training_case, tmp_path, monkeypatch, resume_before_failure
):
    data, config = training_case
    original_transition = PTTSampler.transition_target
    original_mixing = PTTSampler.estimate_mixing_time
    failures = []
    probes = []

    def transition(self, params, **kwargs):
        if self.model_version == 2 and not failures:
            failures.append(3)
            self.last_failure = {"reason": "mixing_budget_exceeded", "model_version": 3,
                                 "mixing": asdict(MixingEstimate(2000, 2000, 100, 40000, "budget_exceeded")),
                                 "events": []}
            return False, 7
        return original_transition(self, params, **kwargs)

    def mixing(self, **kwargs):
        probes.append(self.model_version)
        if failures and self.model_version == 2:
            result = MixingEstimate(2000, 2000, 100, 40000, "budget_exceeded", local_sweeps=7)
            self.last_mixing = asdict(result)
            self.local_sweeps += 7
            return result
        return original_mixing(self, **kwargs)

    monkeypatch.setattr(PTTSampler, "transition_target", transition)
    monkeypatch.setattr(PTTSampler, "estimate_mixing_time", mixing)
    kwargs = {}
    if resume_before_failure:
        first = train_model(data, config=replace(config, max_epochs=2), output_dir=tmp_path / "before_failure")
        kwargs["ptt_resume"] = first.artifacts["ptt_archive"]
    result = train_model(data, config=config, output_dir=tmp_path / "recover", **kwargs)
    assert result.gradient_steps == 3
    assert probes == [0, 2, 1]
    events = result.ptt_sampler.training_state["recovery_events"]
    assert len(events) == 1 and events[0]["restored_step"] == 1
    assert result.history["bias_learning_rate"] == [0.01, 0.005, 0.005, 0.005]
    saved = PTTSampler.from_archive(result.artifacts["ptt_archive"], mode="resume")
    assert saved.training_state["history"]["Epochs"] == [0, 1, 2, 3]
    assert saved.training_state["recovery_events"] == events
    resumed = train_model(data, config=replace(config, max_epochs=4), ptt_resume=result.artifacts["ptt_archive"])
    assert resumed.history["bias_learning_rate"][-1] == 0.005


@pytest.mark.parametrize("kwargs", [
    {"target_replicas": 3}, {"mixing_initial_rounds": 7}, {"mixing_max_rounds": 20},
    {"mixing_thermalization_rounds": -1}, {"mixing_window_factor": float("nan")}, {"reservoir_size": 0},
    {"mixing_method": "unknown"},
])
def test_invalid_mixing_and_reservoir_configuration(kwargs):
    with pytest.raises(ValueError):
        PTTConfig(**kwargs)


def test_development_archive_formats_are_rejected_with_conversion_hint(training_case, tmp_path):
    import json

    import h5py

    from adabmDCA.api import load_model
    from adabmDCA.api.exceptions import InputValidationError

    data, config = training_case
    result = train_model(data, config=replace(config, max_epochs=1), output_dir=tmp_path / "old")
    path = result.artifacts["ptt_archive"]
    with h5py.File(path, "r+") as archive:
        metadata = json.loads(archive.attrs["metadata"])
        metadata["schema_version"] = 7
        archive.attrs["metadata"] = json.dumps(metadata)
    for mode in ("generate", "resume"):
        with pytest.raises(InputValidationError, match="format 7.*no longer read"):
            PTTSampler.from_archive(path, mode=mode)
    with pytest.raises(InputValidationError, match="format 7"):
        load_model(path, device="cpu")


def test_reference_sampling_estimator_matches_reference_trace():
    # Golden trace evaluated directly with ptt_paper._process_experiment.
    generator = torch.Generator().manual_seed(42)
    labels = torch.zeros(500, 2, 100, dtype=torch.int64)
    labels[0, 1] = 1
    for t in range(1, len(labels)):
        swap = torch.rand(100, generator=generator) < 0.04
        labels[t, 0] = torch.where(swap, labels[t - 1, 1], labels[t - 1, 0])
        labels[t, 1] = 1 - labels[t, 0]
    tau_int, tau_exp, correlation = process_replica_experiment(labels, reference=True)
    assert tau_int == pytest.approx(11.147520065307617, rel=1e-6)
    assert tau_exp == pytest.approx(9.743298781036707, rel=1e-5)
    assert correlation[0] == 0.5


def test_reference_mixing_uses_and_advances_production_populations():
    sampler = PTTSampler(profile(), tokens="AB", n_chains=31, config=fast_config())
    before = sampler.local_sweeps
    rng = sampler._rng.clone()
    result = sampler.estimate_mixing_time(in_place=True, reference=True, bounded=False)
    assert result.converged
    assert len(sampler.chains[0]) == 31
    assert sampler.rounds == 102
    assert sampler.local_sweeps - before == result.local_sweeps == 102
    assert not torch.equal(rng, sampler._rng)


def test_mixing_fit_longer_than_the_trajectory_extends_instead_of_failing(monkeypatch):
    from adabmDCA import PTTConfig, PTTSampler
    from tests.test_ptt import profile

    calls = []

    def fake_fit(indices, **kwargs):
        calls.append(len(indices))
        # First fit: an exponential time far beyond the trajectory, as a flat correlation produces.
        return (1.0, 3.6e6 if len(calls) == 1 else 2.0, torch.zeros(len(indices)))

    monkeypatch.setattr("adabmDCA.ptt.sampler.process_replica_experiment", fake_fit)
    config = PTTConfig(mixing_method="autocorrelation", mixing_chains=10, mixing_thermalization_rounds=0, mixing_initial_rounds=8, mixing_max_rounds=64)
    sampler = PTTSampler(profile(), tokens="AB", n_chains=10, config=config)
    result = sampler.estimate_mixing_time()
    assert result.status == "converged" and calls[0] == 8 and calls[1] == 16
