"""The lag monitoring and pauses of the 'adaptive' PTT optimizer, and the lag memory they rely on."""

import json
from dataclasses import replace

import pytest
import torch

from adabmDCA import PTTConfig, PTTSampler, train_model
from adabmDCA.api import TrainingConfig
from adabmDCA.exceptions import InputValidationError
from adabmDCA.ptt.optim import _PTTOptimizer
from adabmDCA.ptt.training import _LAG_PAUSE_COOLDOWN
from adabmDCA.statmech import compute_energy
from adabmDCA.stats import get_freq_single_point, get_freq_two_points
from tests import test_ptt as helpers

single_thread = helpers.single_thread
training_case = helpers.training_case


def exact_samples(params, n, seed=0):
    """Categorical samples drawn exactly from a two-site binary model."""
    energies = compute_energy(helpers.states(), params)
    generator = torch.Generator().manual_seed(seed)
    index = torch.multinomial((-energies).softmax(0), n, replacement=True, generator=generator)
    return helpers.states().argmax(-1)[index].to(torch.int32)


def lag_sampler(n_chains, endpoint):
    config = PTTConfig(equilibration_rounds=2, mixing_thermalization_rounds=10, mixing_chains=20)
    sampler = PTTSampler(helpers.profile(), tokens="AB", n_chains=n_chains, config=config, seed=0)
    sampler.models[-1] = endpoint
    sampler.chains = [exact_samples(helpers.profile(), n_chains, seed=1), exact_samples(endpoint, n_chains, seed=2)]
    return sampler


def test_lag_memory_follows_configurations_through_exchanges_and_permutations():
    middle = {key: 0.5 * value for key, value in helpers.coupled().items()}
    sampler = PTTSampler(helpers.profile(), tokens="AB", n_chains=64, seed=3)
    sampler.models = [helpers.profile(), middle, helpers.coupled()]
    sampler.chains = [exact_samples(model, 64, seed=k) for k, model in enumerate(sampler.models)]
    sampler._reset_lineage()
    sampler.acceptance = [1.0, 1.0]
    tag = sampler.lineage.double() + 1.0
    sampler.lag_memory = tag.clone()
    sampler.advance(rounds=5)
    # Each value either still belongs to its configuration (identified by its
    # lineage) or was cleared because the configuration was redrawn at rung 0.
    carried = sampler.lag_memory == sampler.lineage.double() + 1.0
    assert bool((carried | (sampler.lag_memory == 0)).all())
    # Some configurations kept their value after moving to another replica.
    moved = (sampler.lineage // 64) != torch.arange(3).unsqueeze(1)
    assert bool((carried & moved).any())
    assert bool((carried & ~moved).any()) and not torch.equal(sampler.lag_memory, tag)
    # Rung 0 is redrawn every round: its configurations are new.
    assert bool((sampler.lag_memory[0] == 0).all())


def test_lag_memory_gives_the_linear_response_lag():
    endpoint = helpers.coupled()
    sampler = lag_sampler(200_000, endpoint)
    direction = {"bias": torch.tensor([[0.0, 1.0], [0.5, -0.5]], dtype=torch.float64),
                 "coupling_matrix": torch.zeros(2, 2, 2, 2, dtype=torch.float64)}
    direction["coupling_matrix"][0, 1, 1, 1] = direction["coupling_matrix"][1, 1, 0, 1] = 1.0
    updated = {key: endpoint[key] + 0.05 * direction[key] for key in endpoint}
    sampler._record_endpoint_update(updated)
    # Chains still follow the old model: their excess over the new equilibrium,
    # for the indicator of state B at site 0, is Cov(m, memory) to linear order.
    observable = (sampler.chains[-1][:, 0] == 1).double()
    estimate = float(((observable - observable.mean()) * sampler.lag_memory[-1]).mean())

    def mean_observable(params):
        probabilities = (-compute_energy(helpers.states(), params)).softmax(0)
        return float(probabilities[helpers.states().argmax(-1)[:, 0] == 1].sum())

    exact = mean_observable(endpoint) - mean_observable(updated)
    assert abs(exact) > 0.005
    assert estimate == pytest.approx(exact, rel=0.05)
    assert bool((sampler.lag_memory[0] == 0).all())


def gradient_inputs(sampler):
    samples = sampler.endpoint_samples()
    target = helpers.states()[[0, 3]]
    mask = torch.ones_like(sampler.models[-1]["coupling_matrix"])
    for site in range(mask.shape[0]):
        mask[site, :, site, :] = 0.0
    return {
        "fi": get_freq_single_point(target), "fij": get_freq_two_points(target),
        "pi": get_freq_single_point(samples), "pij": get_freq_two_points(samples),
        "params": sampler.endpoint_params(), "mask": mask, "l2_regularization": 0.0,
    }


def test_adaptive_optimizer_measures_the_drift_lag_without_changing_the_step():
    sampler = lag_sampler(4000, helpers.coupled())
    config = TrainingConfig(ptt=PTTConfig(trust_radius=1e-3))
    optimizer = _PTTOptimizer(config)
    inputs = gradient_inputs(sampler)
    candidate = optimizer.step(**inputs, sampler=sampler)
    # Without memory, nothing lags.
    assert optimizer.lag_diagnostics["lag"] == 0.0 and not optimizer.wants_pause(sampler)
    rates = dict(optimizer.rates)

    # Chains behind along the drift by 0.3 standard deviations.
    samples = sampler.endpoint_samples()
    drift = compute_energy(samples, {key: sampler.models[-1][key] - sampler.models[-2][key]
                                     for key in ("bias", "coupling_matrix")})
    z = (drift - drift.mean()) / drift.std(unbiased=False)
    sampler.lag_memory = torch.zeros(2, 4000, dtype=torch.float64)
    sampler.lag_memory[-1] = 0.3 * z
    assert optimizer.current_lags(sampler)["drift"] == pytest.approx(0.3, rel=1e-6)
    assert optimizer.wants_pause(sampler)
    lagging = optimizer.step(**inputs, sampler=sampler)
    assert optimizer.lag_diagnostics["observable"] == "drift"
    assert optimizer.lag_diagnostics["lag"] == pytest.approx(0.3, rel=1e-6)
    # The lag triggers pauses; it never changes the trust-region step.
    assert optimizer.rates == rates
    for key in candidate:
        torch.testing.assert_close(lagging[key], candidate[key], rtol=0, atol=0)
    optimizer.pause_cooldown = _LAG_PAUSE_COOLDOWN
    assert not optimizer.wants_pause(sampler)
    optimizer.step(**inputs, sampler=sampler)
    assert optimizer.pause_cooldown == _LAG_PAUSE_COOLDOWN - 1
    restored = _PTTOptimizer(config, optimizer.state_dict())
    assert restored.pause_cooldown == optimizer.pause_cooldown
    assert restored.lag_diagnostics == optimizer.lag_diagnostics


def test_lag_memory_survives_checkpoint_insertion_and_replacement():
    sampler = lag_sampler(32, helpers.coupled())
    sampler.lag_memory = torch.ones(2, 32, dtype=torch.float64)
    sampler.acceptance = [0.3]
    sampler.update_replica_chain()
    assert sampler.temporary is not None
    sampler.acceptance = [0.1]
    sampler.update_replica_chain()
    assert len(sampler.models) == 3 and sampler.lag_memory.shape == (3, 32)
    sampler.acceptance = [1.0, 0.1]
    sampler.update_replica_chain()
    assert sampler.lag_memory is not None and sampler.lag_memory.shape == (len(sampler.chains), 32)


def test_adaptive_optimizer_resume_is_exact(training_case, tmp_path):
    data, base = training_case
    config = replace(base, max_epochs=3)
    complete = train_model(data, config=config)
    partial = train_model(data, config=replace(config, max_epochs=1), output_dir=tmp_path / "adaptive")
    resumed = train_model(data, config=config, ptt_resume=partial.artifacts["ptt_archive"])
    for key in complete.model.params:
        torch.testing.assert_close(complete.model.params[key], resumed.model.params[key], rtol=0, atol=0)
    for key in ("ptt_lag_drift", "ptt_lag_drift_tail", "ptt_predicted_kl"):
        assert complete.history[key] == resumed.history[key]
    assert complete.history["ptt_lag_drift"][-1] is not None
    saved = PTTSampler.from_archive(partial.artifacts["ptt_archive"], mode="resume")
    assert saved.lag_memory is not None and saved.lag_memory.shape == (len(saved.chains), len(saved.chains[0]))
    assert saved.optimizer_state["kind"] == "adaptive"


def test_forced_ladder_update_inserts_then_replaces_and_stays_archivable(tmp_path):
    sampler = lag_sampler(32, helpers.coupled())
    sampler.lag_memory = torch.zeros(2, 32, dtype=torch.float64)
    sampler.model_version = 5
    assert sampler.force_ladder_update()
    # No stored snapshot: the current endpoint is stored and inserted below it, and flagged.
    assert len(sampler.models) == 3 and sampler.held is not None and sampler.temporary is None
    assert sampler.ptt_checkpoints[-1]["step"] == 5
    kinds = [(event["kind"], event.get("reason")) for event in sampler.events]
    assert kinds[-3:] == [("snapshot_stored", "lag"), ("checkpoint_flagged", "lag"), ("snapshot_inserted", "lag")]
    # Nothing is inserted twice at the same model version; with a held snapshot the next update replaces.
    sampler.held, held = None, sampler.held
    assert sampler.force_ladder_update() and len(sampler.models) == 3
    sampler.held = held
    assert sampler.force_ladder_update()
    assert sampler.events[-1]["kind"] in ("snapshot_replaced", "reservoir_refreshed", "mixing")
    assert sampler.lag_memory.shape == (len(sampler.chains), 32)
    sampler.save_archive(tmp_path / "forced.h5")
    PTTSampler.from_archive(tmp_path / "forced.h5", mode="inspect")


def test_equilibrate_for_lag_keeps_parameters_and_stops_when_relaxed():
    sampler = lag_sampler(32, helpers.coupled())
    endpoint = sampler.endpoint_params()
    checks = []

    def relaxed(trial):
        checks.append(trial.rounds)
        return len(checks) > 2

    healthy, rounds, work = sampler.equilibrate_for_lag(relaxed, max_rounds=100, chunk=4)
    assert healthy and rounds == 8 and work > 0
    for key in endpoint:
        torch.testing.assert_close(sampler.models[-1][key], endpoint[key], rtol=0, atol=0)
    healthy, rounds, _ = sampler.equilibrate_for_lag(lambda trial: False, max_rounds=7, chunk=4)
    assert healthy and rounds == 7


def test_pause_response_equilibrates_before_updating_and_resumes_exactly(training_case, tmp_path, monkeypatch):
    # A lag above tolerance at model version 1 triggers one pause that never relaxes (budget of 5 rounds).
    monkeypatch.setattr(_PTTOptimizer, "current_lag", lambda self, sampler: 1.0 if sampler.model_version == 1 else 0.0)
    data, base = training_case
    ptt = replace(base.ptt, lag_pause_rounds=5)
    config = replace(base, ptt=ptt, max_epochs=3)
    complete = train_model(data, config=config, output_dir=tmp_path / "complete")
    events = [json.loads(line) for line in (tmp_path / "complete" / "events.jsonl").read_text().splitlines()]
    pauses = [event for event in events if event["event"] == "ptt_lag_pause"]
    assert [(event["step"], event["rounds"]) for event in pauses] == [(1, 5)]
    assert any(event.get("reason") == "lag" for event in events if event["event"].startswith("ptt_snapshot"))
    log = (tmp_path / "complete" / "adabmDCA.log").read_text()
    assert "pause: " in log and "5 extra rounds" in log
    assert complete.ptt_sampler.training_state["pauses"] == 1
    partial = train_model(data, config=replace(config, max_epochs=1), output_dir=tmp_path / "partial")
    resumed = train_model(data, config=config, ptt_resume=partial.artifacts["ptt_archive"])
    for key in complete.model.params:
        torch.testing.assert_close(complete.model.params[key], resumed.model.params[key], rtol=0, atol=0)
    assert resumed.ptt_sampler.training_state["pauses"] == 1


def test_drift_tails_are_monitored():
    # Twenty independent sites: the drift is close to Gaussian and has populated tails.
    profile = {"bias": torch.zeros(20, 2, dtype=torch.float64), "coupling_matrix": torch.zeros(20, 2, 20, 2, dtype=torch.float64)}
    sampler = PTTSampler(profile, tokens="AB", n_chains=4000, config=PTTConfig(), seed=0)
    generator = torch.Generator().manual_seed(0)
    sampler.models[-1] = {"bias": torch.randn(20, 2, generator=generator, dtype=torch.float64),
                          "coupling_matrix": profile["coupling_matrix"].clone()}
    names, z = _PTTOptimizer._lag_observables(sampler, sampler.endpoint_samples())
    assert names == ["drift", "drift_low", "drift_high"]
    torch.testing.assert_close(z.mean(1), torch.zeros(len(z), dtype=torch.float64), atol=1e-10, rtol=0)
    torch.testing.assert_close(z.std(1, unbiased=False), torch.ones(len(z), dtype=torch.float64), atol=1e-10, rtol=0)


def test_adaptive_optimizer_lag_configuration_and_cli():
    from adabmDCA.scripts.train import create_parser

    parsed = create_parser().parse_args([
        "--data", "unused.fasta", "--strategy", "ptt", "--ptt-lag-tolerance", "0.3", "--ptt-lag-horizon", "50",
        "--ptt-lag-pause-rounds", "40",
    ])
    assert (parsed.ptt_lag_tolerance, parsed.ptt_lag_horizon, parsed.ptt_lag_pause_rounds) == (0.3, 50, 40)
    for bad in ({"lag_tolerance": 0.0}, {"lag_horizon": 0}, {"lag_pause_rounds": 0}, {"trust_radius": 0.0}):
        with pytest.raises(InputValidationError):
            PTTConfig(**bad)
