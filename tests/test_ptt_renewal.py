"""Population-renewal equilibration and multi-pass exchanges for PTT generation."""

import io

import numpy as np
import pytest
import torch

from adabmDCA import PTTSampler
from adabmDCA.api import sample_sequences
from adabmDCA.exceptions import InputValidationError
from adabmDCA.statmech import compute_energy
from tests import test_ptt as helpers

coupled, profile, states = helpers.coupled, helpers.profile, helpers.states
single_thread = helpers.single_thread


def generation_ladder(n_chains, *, endpoint=None):
    """Two-model generation ladder: exact profile at rung 0, ``endpoint`` on top."""
    backend = PTTSampler(profile(), tokens="AB", n_chains=n_chains)
    if endpoint is not None:
        backend.models[-1] = endpoint
        backend.model_version = 1
    backend.prepare_sampling_ladder(n_chains)
    return backend


def reject_all_exchanges(lower, upper, lower_chains, upper_chains):
    return torch.full((len(lower_chains),), -torch.inf, dtype=torch.float64)


def test_birth_rounds_follow_configurations_and_rung_zero_redraws():
    backend = generation_ladder(64, endpoint=coupled())
    assert torch.equal(backend.birth, torch.full((2, 64), -1))
    snapshot = {int(i): int(b) for i, b in zip(backend.lineage.reshape(-1), backend.birth.reshape(-1))}
    backend.advance(rounds=1)
    # Rung 0 is redrawn in round 0; configurations swapped up keep their birth.
    assert torch.equal(backend.birth[0], torch.zeros(64, dtype=torch.int64))
    for i, b in zip(backend.lineage[1], backend.birth[1]):
        assert int(b) == snapshot[int(i)]
    backend.advance(rounds=3)
    assert torch.equal(backend.birth[0], torch.full((64,), 3))
    assert int(backend.birth.max()) == backend.rounds - 1


def test_always_accepted_exchange_renews_in_two_rounds():
    # Identical models accept every exchange: the endpoint receives the rung-0
    # population of the previous round, which is fresh from the second round on.
    backend = generation_ladder(40)
    result = backend.measure_renewal(max_rounds=50, tolerance=0.01)
    assert result.converged
    assert result.warmup_rounds == 2
    assert result.renewal_rounds == 2
    assert result.stationary_rounds == result.chunk_rounds == 2
    assert result.warmup_ladder_old == (0.5, 0.0)
    assert result.warmup_endpoint_fresh == (0.0, 1.0)
    assert result.trapped_fraction == 0.0
    assert result.rounds == backend.rounds == 4
    assert result.acceptance == pytest.approx((1.0,))


def test_trapped_endpoint_exhausts_budget_and_reports_fraction():
    backend = generation_ladder(30, endpoint=coupled())
    backend._exchange_kernel = reject_all_exchanges
    result = backend.measure_renewal(max_rounds=25, tolerance=0.05)
    assert result.status == "budget_exceeded"
    assert result.warmup_rounds is None and result.renewal_rounds is None
    assert result.trapped_fraction == 1.0
    assert result.old_fraction_by_model == pytest.approx((0.0, 1.0))
    assert len(result.warmup_ladder_old) == 25 and result.stationary_ladder_old == ()
    assert backend.events[-1]["kind"] == "renewal"


def test_ladder_old_fraction_never_increases():
    backend = generation_ladder(200, endpoint=coupled())
    result = backend.measure_renewal(max_rounds=500, tolerance=0.01)
    assert result.converged
    for history in (result.warmup_ladder_old, result.stationary_ladder_old):
        assert np.all(np.diff(history) <= 0)
        assert history[-1] * 2 * 200 <= 0.01 * 200
    # Stationary chunks end independently of the birth labels.
    assert result.stationary_rounds % result.chunk_rounds == 0
    assert result.renewal_rounds <= result.stationary_rounds


def test_renewal_requires_generation_ladder_and_valid_tolerance():
    with pytest.raises(InputValidationError, match="prepare_sampling_ladder"):
        PTTSampler(profile(), tokens="AB", n_chains=8).measure_renewal()
    backend = generation_ladder(8)
    for tolerance in (0.0, 1.0, True):
        with pytest.raises(InputValidationError, match="tolerance"):
            backend.measure_renewal(tolerance=tolerance)


def test_each_round_attempts_every_adjacent_exchange_once():
    backend = generation_ladder(16, endpoint=coupled())
    calls = []
    kernel = backend._exchange_kernel

    def counted(*args):
        calls.append(1)
        return kernel(*args)

    backend._exchange_kernel = counted
    backend.advance(rounds=3)
    assert len(calls) == 3
    assert 0.0 < backend.acceptance[0] <= 1.0


def test_until_stops_advance_after_the_requested_round():
    backend = generation_ladder(8)
    seen = []
    backend.advance(rounds=10, until=lambda: seen.append(1) or len(seen) == 3)
    assert backend.rounds == 3
    assert backend.acceptance == pytest.approx([1.0])


def test_renewal_sampling_matches_exact_endpoint_distribution(tmp_path):
    data = tmp_path / "reference.fasta"
    data.write_text(">a\nAA\n>b\nBB\n")
    backend = generation_ladder(8, endpoint=coupled())
    result = sample_sequences(model=backend, reference_fasta=data, n_sequences=20_000, collect_diagnostics=True,
                              ptt_local_sweeps=1)
    diagnostics = result.ptt_diagnostics
    assert diagnostics["mixing_method"] == "renewal" and diagnostics["converged"]
    assert diagnostics["mixing"]["status"] == "converged"
    assert set(diagnostics["renewal_history"]) == {
        "warmup_ladder_old", "warmup_endpoint_fresh", "stationary_ladder_old", "stationary_endpoint_fresh",
    }
    assert not {"ladder_repair", "ladder_repairs", "interpolated_models", "sweep_boost"} & set(diagnostics)
    assert result.mixing_history["warmup_rounds"][0] >= 1
    index = {"AA": 0, "AB": 1, "BA": 2, "BB": 3}
    counts = np.bincount([index[s] for s in result.sequences], minlength=4) / len(result.sequences)
    exact = (-compute_energy(states(), coupled())).softmax(0).numpy()
    np.testing.assert_allclose(counts, exact, atol=0.02)

    artifacts = result.save_bundle(tmp_path / "out", label="ptt")
    artifacts.update(result.save_diagnostic_plots(tmp_path / "out", label="ptt"))
    assert artifacts["ptt_renewal_plot"].read_bytes().startswith(b"\x89PNG")
    assert artifacts["ptt_renewal_log"].read_text().startswith("phase,round,ladder_old,endpoint_fresh")


def test_budget_exceeded_renewal_retains_samples_with_warning(tmp_path, monkeypatch):
    data = tmp_path / "reference.fasta"
    data.write_text(">a\nAA\n>b\nBB\n")
    backend = generation_ladder(8, endpoint=coupled())
    monkeypatch.setattr("adabmDCA.ptt.sampler._prepare_exchange_kernel", lambda device: reject_all_exchanges)
    result = sample_sequences(model=backend, reference_fasta=data, n_sequences=6, ptt_max_rounds=12,
                              collect_diagnostics=True)
    assert len(result.sequences) == 6
    assert result.ptt_diagnostics["mixing"]["status"] == "budget_exceeded"
    assert result.ptt_diagnostics["mixing"]["trapped_fraction"] == 1.0
    assert any("may be trapped" in warning for warning in result.warnings)
    artifacts = result.save_diagnostic_plots(tmp_path / "plots")
    assert artifacts["ptt_renewal_plot"].is_file()


def test_renewal_progress_renderer_reports_fractions_and_summary():
    from adabmDCA.api.results import SamplingProgress
    from adabmDCA.scripts.sample import _SamplingProgressRenderer

    stream = io.StringIO()
    renderer = _SamplingProgressRenderer(diagnostic=True, stream=stream)
    renderer(SamplingProgress("ptt_renewal_warmup", 100, 20_000, details={
        "ladder_old": 0.125, "endpoint_fresh": 0.75, "threshold": 0.005, "phase_round": 100, "block_rounds": None,
        "decay_rounds": 50.0, "predicted_round": 261.0, "first_decay_rounds": 20.0,
    }))
    renderer(SamplingProgress("ptt_renewal_stationary", 330, 20_000, details={
        "ladder_old": 0.01, "endpoint_fresh": 0.99, "threshold": 0.005, "phase_round": 70, "block_rounds": 50,
        "decay_rounds": 40.0, "predicted_round": 120.0, "first_decay_rounds": 35.0,
    }))
    renderer(SamplingProgress("ptt_renewal", 340, 340, details={
        "status": "converged", "warmup_rounds": 120, "renewal_rounds": 180,
        "trapped_fraction": 0.004, "acceptance": [0.25, 0.5],
    }))
    renderer.close()
    output = stream.getvalue()
    assert ("PTT diagnostic | warmup | round 100 | ladder old 0.1250 | endpoint fresh 0.7500 | "
            "predicted renewal round 261") in output
    assert "PTT warmup | G 1.2e-01 -> 5.0e-03 | decay ~50 rounds | renewal ~ round 261" in output
    # The stationary phase stops at the first block end after the predicted renewal.
    assert "PTT stationary | G 1.0e-02 -> 5.0e-03 | decay ~40 rounds | stop ~ round 150" in output
    assert output.count("decays more and more slowly (decay time 20 -> 50 rounds)") == 1
    assert "PTT renewal | converged | warmup 120 rounds | stationary renewal 180 rounds" in output
    assert "trapped endpoint fraction 0.0040 | acceptance [0.2500, 0.5000]" in output


# --- Population renewal as the PTT training mixing method ------------------------------------------


def renewal_config(**kwargs):
    from adabmDCA import PTTConfig

    return PTTConfig(mixing_method="renewal", mixing_chains=40, mixing_max_rounds=400, **kwargs)


def test_reservoir_emissions_are_births():
    backend = PTTSampler(profile(), tokens="AB", n_chains=16, config=renewal_config())
    backend.reservoir = backend.chains[0].repeat(3, 1)
    backend.start_birth_tracking()
    snapshot = {int(i): int(b) for i, b in zip(backend.lineage[1], backend.birth[1])}
    backend.advance(rounds=1)
    # Rung 0 is not redrawn with a reservoir: its population was emitted by the reservoir in round 0.
    assert torch.equal(backend.birth[0], torch.zeros(16, dtype=torch.int64))
    assert all(int(b) in (-1, 0) for b in backend.birth[1])
    assert set(snapshot.values()) == {-1}


def test_renewal_training_check_uses_a_separate_population_and_reports_work():
    backend = PTTSampler(profile(), tokens="AB", n_chains=24, config=renewal_config())
    backend.models[-1] = coupled()
    chains = [c.clone() for c in backend.chains]
    rng, before = backend._rng.clone(), backend.local_sweeps
    result = backend.estimate_mixing_time(local_sweeps=1)
    assert result.method == "renewal" and result.converged
    assert result.tau_int is None and result.tau_exp is None
    assert result.warmup_rounds >= 1 and result.renewal_rounds >= 1
    assert result.rounds == result.required_rounds
    assert result.local_sweeps == backend.local_sweeps - before > 0
    assert all(torch.equal(a, b) for a, b in zip(chains, backend.chains))
    assert torch.equal(rng, backend._rng)
    assert backend.last_mixing["method"] == "renewal" and backend.last_mixing["observable"] == "birth_round"
    assert backend.events[-1]["kind"] == "mixing"
    assert backend.birth is None
    # Generation keeps an explicit autocorrelation path regardless of the archived method.
    with pytest.raises(InputValidationError, match="measure_renewal"):
        backend.estimate_mixing_time(in_place=True, method="renewal")


def test_renewal_training_check_fails_when_the_ladder_cannot_renew(monkeypatch):
    # The check runs on a fork, which prepares its own exchange kernel.
    monkeypatch.setattr("adabmDCA.ptt.sampler._prepare_exchange_kernel", lambda device: reject_all_exchanges)
    backend = PTTSampler(profile(), tokens="AB", n_chains=24, config=renewal_config())
    backend.models[-1] = coupled()
    result = backend.estimate_mixing_time(local_sweeps=1)
    assert result.status == "budget_exceeded" and not result.converged
    assert result.trapped_fraction == 1.0
    assert result.rounds == 400


def test_reservoir_spaced_by_renewal_samples_exact_distribution():
    """Refresh a three-model full ladder; the reservoir must sample the collection model exactly."""
    config = renewal_config(max_replicas=2, target_replicas=2, reservoir_size=20_000, renewal_tolerance=0.01)
    backend = PTTSampler(profile(), tokens="AB", n_chains=2000, config=config)
    middle = coupled()
    middle["coupling_matrix"] *= 0.5
    backend.models = [profile(), middle, coupled()]
    backend.chains.append(backend.chains[-1].clone())
    backend.total_models = 3
    backend._reset_lineage()
    assert backend.refresh_reservoir(local_sweeps=1)
    event = backend.events[-1]
    assert event["kind"] == "reservoir_refreshed"
    assert event["renewal_warmup_rounds"] >= 1 and event["renewal_spacing_rounds"] >= 1
    assert event["min_batch_fresh"] >= 0.99
    assert backend.n_active == 2 and len(backend.reservoir) == 20_000
    assert backend.birth is None
    x = states()
    codes = backend.reservoir[:, 0].long() * 2 + backend.reservoir[:, 1].long()
    counts = np.bincount(codes.numpy(), minlength=4) / len(codes)
    exact = (-compute_energy(x, middle)).softmax(0).numpy()
    np.testing.assert_allclose(counts, exact, atol=0.015)


def test_renewal_training_logs_resumes_exactly_and_pins_the_method(tmp_path):
    from dataclasses import replace

    from adabmDCA import train_model

    data, config = _training_case(tmp_path)
    config = replace(config, ptt=replace(config.ptt, mixing_method="renewal", mixing_max_rounds=400), max_epochs=3)
    complete = train_model(data, config=config)
    checks = [event for event in complete.ptt_sampler.events if event["kind"] == "mixing"]
    assert checks and {event["method"] for event in checks} == {"renewal"}
    assert any(event["warmup_rounds"] is not None for event in checks)
    assert all("trapped_fraction" in event for event in checks)
    partial = train_model(data, config=replace(config, max_epochs=1), output_dir=tmp_path / "renewal")
    resumed = train_model(data, config=config, ptt_resume=partial.artifacts["ptt_archive"])
    torch.testing.assert_close(complete.model.params["coupling_matrix"],
                               resumed.model.params["coupling_matrix"], rtol=0, atol=0)
    with pytest.raises(Exception, match="configuration differ from the archive in settings: ptt"):
        train_model(data, config=replace(config, ptt=replace(config.ptt, mixing_method="autocorrelation")),
                    ptt_resume=partial.artifacts["ptt_archive"])


def _training_case(tmp_path):
    from adabmDCA import PTTConfig
    from adabmDCA.api import TrainingConfig

    path = tmp_path / "data.fasta"
    sequences = ["AAA", "AAB", "ABA", "BAA", "BBB", "BAB"]
    path.write_text("".join(f">s{i}\n{seq}\n" for i, seq in enumerate(sequences)))
    config = TrainingConfig(
        alphabet="AB", device="cpu", dtype="float64", n_chains=80, n_sweeps=1, max_epochs=3,
        target_pearson=0.999999, no_reweighting=True, pseudocount=0.1,
        ptt=PTTConfig(equilibration_rounds=2, initialization_rounds=2, swaps=1,
                      mixing_thermalization_rounds=2, mixing_chains=20),
        checkpoint_interval=1,
    )
    return path, config


def test_training_cli_maps_renewal_options():
    from adabmDCA.scripts.train import create_parser

    parsed = create_parser().parse_args([
        "--strategy", "ptt", "--data", "unused.fasta", "--ptt-mixing-method", "renewal", "--ptt-renewal-tolerance", "0.05",
    ])
    assert parsed.ptt_mixing_method == "renewal" and parsed.ptt_renewal_tolerance == 0.05
    defaults = create_parser().parse_args(["--strategy", "ptt", "--data", "unused.fasta"])
    assert defaults.ptt_mixing_method is None


def test_exchange_passes_are_not_an_option():
    from adabmDCA.scripts.sample import create_parser

    with pytest.raises(SystemExit):
        create_parser().parse_args(["-p", "model.h5", "-o", "out", "--strategy", "ptt", "--ptt-exchange-passes", "5"])
    with pytest.raises(TypeError):
        sample_sequences(model="model.h5", ptt=True, ptt_exchange_passes=5)


def test_ladder_modification_options_are_gone():
    from adabmDCA.scripts.sample import create_parser

    for removed in (["--ptt-interpolate", "0:1"], ["--ptt-sweep-boost", "1:2"], ["--ptt-repair-ladder"],
                    ["--ptt-repair-max-models", "2"], ["--ptt-repair-check-rounds", "50"]):
        with pytest.raises(SystemExit):
            create_parser().parse_args(["-p", "model.h5", "-o", "out", "--ngen", "10", "--strategy", "ptt", *removed])
    with pytest.raises(TypeError):
        PTTSampler(profile(), tokens="AB", n_chains=8).prepare_sampling_ladder(8, interpolate={0: 1})


def test_renewal_forecast_fits_the_exponential_decay_of_old_configurations():
    from adabmDCA.ptt.mixing import renewal_forecast

    rounds = np.arange(1, 301)
    ladder_old = 0.8 * np.exp(-rounds / 40.0)
    forecast = renewal_forecast(ladder_old[:150], threshold=1e-4)
    assert forecast.decay_rounds == pytest.approx(40.0, rel=1e-6)
    assert forecast.predicted_round == pytest.approx(40.0 * np.log(0.8 / 1e-4), rel=1e-6)
    assert renewal_forecast(ladder_old[:10], threshold=1e-4) is None  # too few usable rounds
    flat = renewal_forecast(np.full(100, 0.3), threshold=1e-4)
    assert flat.decay_rounds is None and flat.predicted_round is None


def test_stationary_renewal_stops_at_short_block_ends():
    backend = generation_ladder(200, endpoint=coupled())
    events = []
    result = backend.measure_renewal(max_rounds=2000, tolerance=0.01, blocks_per_warmup=4,
                                     on_progress=lambda stage, done, total, **details: events.append(details))
    assert result.converged
    assert result.chunk_rounds == -(-result.warmup_rounds // 4)
    assert result.stationary_rounds % result.chunk_rounds == 0
    assert result.renewal_rounds <= result.stationary_rounds < result.renewal_rounds + result.chunk_rounds
    # This ladder renews within a few rounds: too few to fit a decay time.
    assert result.warmup_decay_rounds is None
    assert all(details["threshold"] == pytest.approx(0.01 / 2) for details in events)
    assert {details["block_rounds"] for details in events if "block_rounds" in details} >= {result.chunk_rounds}


def test_warmup_only_renewal_converges_at_the_warmup():
    backend = generation_ladder(200, endpoint=coupled())
    result = backend.measure_renewal(max_rounds=2000, tolerance=0.01, stationary=False)
    assert result.converged and result.warmup_rounds is not None
    assert result.renewal_rounds is None and result.stationary_rounds == 0 and result.stationary_ladder_old == ()
    assert result.trapped_fraction <= 0.01
