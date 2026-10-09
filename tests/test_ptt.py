"""Correctness gates for the experimental full-ladder PTT implementation."""

import json
from dataclasses import replace
from itertools import product

import h5py
import numpy as np
import pytest
import torch

from adabmDCA import PTTConfig, PTTSampler, train_model
from adabmDCA.api import TrainingConfig, sample_sequences
from adabmDCA.api.ptt import estimate_ptt_entropy
from adabmDCA.exceptions import ConvergenceError, InputValidationError
from adabmDCA.ptt import bridge_increment
from adabmDCA.ptt.optim import _PTTOptimizer
from adabmDCA.statmech import compute_energy, compute_entropy, compute_log_likelihood, exchange_log_acceptance
from adabmDCA.stats import get_freq_two_points


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def profile(dtype=torch.float64):
    return {
        "bias": torch.tensor([[0.2, -0.3], [-0.7, 0.4]], dtype=dtype),
        "coupling_matrix": torch.zeros(2, 2, 2, 2, dtype=dtype),
    }


def coupled():
    params = profile()
    block = torch.tensor([[0.8, -0.2], [-0.2, 0.5]], dtype=torch.float64)
    params["coupling_matrix"][0, :, 1, :] = block
    params["coupling_matrix"][1, :, 0, :] = block.T
    return params


def states():
    return torch.nn.functional.one_hot(torch.tensor(list(product(range(2), repeat=2))), 2).double()


def test_energy_bridge_identity_and_exchange_detailed_balance():
    x = states()
    lower, upper = profile(), coupled()
    e0, e1 = compute_energy(x, lower), compute_energy(x, upper)
    manual = torch.tensor([-0.2 + 0.7 - 0.8, -0.2 - 0.4 + 0.2, 0.3 + 0.7 + 0.2, 0.3 - 0.4 - 0.5], dtype=torch.float64)
    torch.testing.assert_close(e1, manual)
    z0, z1 = torch.logsumexp(-e0, 0), torch.logsumexp(-e1, 0)
    log_p0 = -e0 - z0
    torch.testing.assert_close(z0 + torch.logsumexp(log_p0 + e0 - e1, 0), z1)
    for a, b in product(range(4), repeat=2):
        forward = exchange_log_acceptance(lower, upper, x[a : a + 1], x[b : b + 1]).exp()
        reverse = exchange_log_acceptance(lower, upper, x[b : b + 1], x[a : a + 1]).exp()
        torch.testing.assert_close(torch.exp(-e0[a] - e1[b]) * forward, torch.exp(-e0[b] - e1[a]) * reverse)
    assert torch.equal(exchange_log_acceptance(lower, lower, x, x.flip(0)), torch.zeros(4).double())


@pytest.mark.parametrize("kernel", ["gibbs", "metropolis"])
def test_fixed_ladder_distribution_logz_entropy_and_gauge(kernel):
    sampler = PTTSampler(profile(), tokens="AB", n_chains=12000, sampler=kernel, seed=6)
    accepted, _ = sampler.transition_target(coupled(), local_sweeps=3)
    assert accepted
    sampler.advance(rounds=30, local_sweeps=2)
    exact_energies = compute_energy(states(), coupled())
    exact_z = torch.logsumexp(-exact_energies, 0).item()
    observed = sampler.chains[-1]
    counts = torch.bincount(observed[:, 0] * 2 + observed[:, 1], minlength=4).double() / len(observed)
    torch.testing.assert_close(counts, (-exact_energies).softmax(0), atol=0.025, rtol=0)
    assert abs(sampler.partition_estimate().log_z - exact_z) < 0.025
    exact_entropy = float((exact_energies * (-exact_energies).softmax(0)).sum()) + exact_z
    assert abs(sampler.entropy()["entropy"] - exact_entropy) < 0.04
    reference_weights = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    statistics = sampler.ladder_statistics(states(), reference_weights)
    assert len(statistics) == 2
    assert statistics[-1]["entropy"] == pytest.approx(exact_entropy, abs=0.04)
    exact_ll = -float((exact_energies * reference_weights / reference_weights.sum()).sum()) - exact_z
    assert statistics[-1]["log_likelihood"] == pytest.approx(exact_ll, abs=0.025)
    assert statistics[-1]["log_likelihood_per_residue"] == statistics[-1]["log_likelihood"] / 2
    shifted = sampler.endpoint_params()
    shifted["bias"] += 2.0
    increment = bridge_increment(sampler.models[-1], shifted, sampler.chains[-1])
    assert increment == pytest.approx(4.0)


def test_anchor_rng_fork_and_independent_storage():
    sampler = PTTSampler(profile(), tokens="AB", n_chains=20, seed=9)
    assert sampler.partition_estimate().log_z == pytest.approx(float(torch.logsumexp(profile()["bias"], -1).sum()))
    fork = sampler.fork()
    global_rng = torch.get_rng_state().clone()
    torch.testing.assert_close(sampler.advance(rounds=3), fork.advance(rounds=3), rtol=0, atol=0)
    assert torch.equal(global_rng, torch.get_rng_state())
    fork.chains[-1].zero_()
    assert sampler.chains[-1].sum() > 0
    assert sampler.models[0]["bias"].data_ptr() != sampler.models[-1]["bias"].data_ptr()


def test_ptt_uses_prepared_local_sampler(monkeypatch):
    prepared = []
    calls = []

    def fake_prepare(name, device):
        prepared.append((name, device.type))

        def kernel(chains, params, nsweeps, beta=1.0):
            calls.append((len(chains), nsweeps, beta))
            return chains.clone()

        return kernel

    # The PyTorch fallback, used on CPU without the Numba kernels.
    monkeypatch.setenv("ADABMDCA_NUMBA", "0")
    monkeypatch.setattr("adabmDCA.ptt.kernels.prepare_sampler", fake_prepare)
    sampler = PTTSampler(profile(), tokens="AB", n_chains=12, sampler="metropolis")
    sampler.advance(rounds=2, local_sweeps=2)

    assert prepared == [("metropolis", "cpu")]
    assert calls == [(12, 1, 1.0)] * 4


def test_ptt_runtime_populations_are_categorical_and_phases_are_timed():
    sampler = PTTSampler(profile(), tokens="AB", n_chains=12, sampler="metropolis")
    assert all(chains.dtype == torch.int32 and chains.shape == (12, 2) for chains in sampler.chains)
    sampler.advance(rounds=2)
    assert set(sampler.last_advance_timing) == {
        "local_sampling_seconds", "exchange_seconds", "permutation_seconds",
    }
    assert all(value > 0 for value in sampler.last_advance_timing.values())
    samples = sampler.endpoint_samples()
    assert samples.shape == (12, 2, 2)
    torch.testing.assert_close(samples.sum(-1), torch.ones_like(samples[..., 0]))


def test_archive_roundtrip_hash_validation_and_atomic_failure(tmp_path, monkeypatch):
    sampler = PTTSampler(profile(), tokens="AB", n_chains=32)
    assert sampler.transition_target(coupled())[0]
    path = sampler.save_archive(tmp_path / "ptt.h5")
    resumed = PTTSampler.from_archive(path, mode="resume")
    torch.testing.assert_close(sampler.draw(70), resumed.draw(70), rtol=0, atol=0)
    original = path.read_bytes()
    monkeypatch.setattr("adabmDCA.ptt.archive.os.replace", lambda *args: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError, match="interrupted"):
        sampler.save_archive(path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    with h5py.File(path, "r+") as archive:
        archive["replicas/1/bias"][0, 0] += 1
    with pytest.raises(InputValidationError, match="hash mismatch"):
        PTTSampler.from_archive(path)


def test_archive_allows_cuda_cpu_float32_partition_roundoff(tmp_path):
    sampler = PTTSampler(profile(torch.float32), tokens="AB", n_chains=32)
    path = sampler.save_archive(tmp_path / "ptt.h5")
    with h5py.File(path, "r+") as archive:
        metadata = json.loads(archive.attrs["metadata"])
        metadata["rng_device"] = "cuda:0"
        metadata["partition"]["log_z"] += 1.2e-5
        archive.attrs["metadata"] = json.dumps(metadata)
    PTTSampler.from_archive(path, mode="inspect")

    with h5py.File(path, "r+") as archive:
        metadata = json.loads(archive.attrs["metadata"])
        metadata["partition"]["log_z"] += 0.1
        archive.attrs["metadata"] = json.dumps(metadata)
    with pytest.raises(InputValidationError, match="partition provenance mismatch"):
        PTTSampler.from_archive(path, mode="inspect")


def test_mixing_failure_leaves_state_and_rng_untouched(monkeypatch):
    from adabmDCA.ptt.mixing import MixingEstimate

    config = PTTConfig(
        target_acceptance=0.99, min_acceptance=0.98, equilibration_rounds=2, max_replicas=2
    )
    sampler = PTTSampler(profile(), tokens="AB", n_chains=40, config=config)
    previous = sampler.fork()
    candidate = coupled()
    candidate["bias"] *= -1000
    def failed_mixing(self, **kwargs):
        from dataclasses import asdict
        result = MixingEstimate(2000.0, 2000.0, 100, 40000, "budget_exceeded", (0.0,), 10)
        self.last_mixing = asdict(result)
        self.local_sweeps += 10
        return result

    monkeypatch.setattr(PTTSampler, "estimate_mixing_time", failed_mixing)
    accepted, _ = sampler.transition_target(candidate)
    assert not accepted
    assert sampler.model_version == 0
    assert torch.equal(sampler._rng, previous._rng)
    for actual, saved in zip(sampler.chains, previous.chains):
        assert torch.equal(actual, saved)


@pytest.fixture
def training_case(tmp_path):
    path = tmp_path / "data.fasta"
    sequences = ["AAA", "AAB", "ABA", "BAA", "BBB", "BAB"]
    path.write_text("".join(f">s{i}\n{seq}\n" for i, seq in enumerate(sequences)))
    config = TrainingConfig(
        alphabet="AB",
        device="cpu",
        dtype="float64",
        n_chains=80,
        n_sweeps=1,
        max_epochs=3,
        target_pearson=0.999999,
        no_reweighting=True,
        pseudocount=0.1,
        ptt=PTTConfig(equilibration_rounds=2, initialization_rounds=2, swaps=1,
                      mixing_thermalization_rounds=2, mixing_chains=20),
        checkpoint_interval=1,
    )
    return path, config


def test_ptt_sgd_keeps_both_rates_fixed():
    config = TrainingConfig(learning_rate=0.1, ptt=PTTConfig(optimizer="sgd"))
    optimizer = _PTTOptimizer(config)
    params = profile()
    mask = torch.ones_like(params["coupling_matrix"])
    fi = torch.ones_like(params["bias"])
    fij = torch.ones_like(params["coupling_matrix"])
    zero_i = torch.zeros_like(fi)
    zero_ij = torch.zeros_like(fij)
    for _ in range(2):
        updated = optimizer.step(fi=fi, fij=fij, pi=zero_i, pij=zero_ij, params=params, mask=mask,
                                 l2_regularization=0.0)
        assert optimizer.rates == {"bias": 0.1, "coupling_matrix": 0.1}
    torch.testing.assert_close(updated["bias"], params["bias"] + 0.1)
    torch.testing.assert_close(updated["coupling_matrix"], params["coupling_matrix"] + 0.1)
    assert optimizer.trust_diagnostics is None and optimizer.lag_diagnostics is None
    restored = _PTTOptimizer(config, optimizer.state_dict())
    assert restored.rates == optimizer.rates


def test_ptt_directional_score_matches_energy_convention():
    samples = states()
    direction = coupled()
    bias_scores, coupling_scores = _PTTOptimizer._directional_score_components(
        samples, direction["bias"], direction["coupling_matrix"]
    )
    torch.testing.assert_close(bias_scores + coupling_scores, -compute_energy(samples, direction))


def test_ptt_trust_region_uses_separate_scales_with_cross_covariance():
    scales, ridge = _PTTOptimizer._solve_trust_region(
        (1.0, 0.8, 4.0), (1.0, 1.0), trust_radius=0.05
    )
    bias_scale, coupling_scale = scales
    assert 0 < coupling_scale < bias_scale < 1
    predicted_kl = 0.5 * (
        (1.0 + ridge) * bias_scale**2
        + 2 * 0.8 * bias_scale * coupling_scale
        + (4.0 + ridge) * coupling_scale**2
    )
    assert predicted_kl == pytest.approx(0.05)

    config = TrainingConfig(ptt=PTTConfig(trust_radius=1e-5))
    optimizer = _PTTOptimizer(config)
    params = profile()
    samples = states()
    fi = torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float64)
    pi = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64)
    fij = get_freq_two_points(samples)
    gradients = {"bias": fi - pi, "coupling_matrix": fij - 0.01 * params["coupling_matrix"]}
    optimizer._adapt(gradients, samples, l2_regularization=0.01)
    diagnostics = optimizer.trust_diagnostics
    assert {"bias_scale", "coupling_scale", "fisher_cross"} <= diagnostics.keys()
    assert diagnostics["predicted_kl"] <= config.ptt.trust_radius * (1 + 1e-12)
    assert optimizer.rates["bias"] == pytest.approx(
        optimizer.nominal_rates["bias"] * diagnostics["bias_scale"]
    )
    assert optimizer.rates["coupling_matrix"] == pytest.approx(
        optimizer.nominal_rates["coupling_matrix"] * diagnostics["coupling_scale"]
    )
    restored = _PTTOptimizer(config, optimizer.state_dict())
    assert restored.rates == optimizer.rates
    assert restored.nominal_rates == optimizer.nominal_rates
    assert restored.trust_diagnostics == optimizer.trust_diagnostics


def test_ptt_trust_region_keeps_rates_positive_inside_the_trust_region():
    # Strongly correlated groups with a weak field gain: the box optimum drops
    # the field step, and a wide trust radius leaves the KL constraint inactive.
    scales, _ = _PTTOptimizer._solve_trust_region((1.0, 0.9, 1.0), (0.1, 1.0), trust_radius=1.0)
    assert scales[1] == pytest.approx(1.0)
    assert scales[0] == pytest.approx(0.0, abs=1e-11) and scales[0] > 0


def test_ptt_optimizer_configuration_and_cli_are_ptt_specific(tmp_path):
    from adabmDCA.scripts.train import create_parser, run

    parsed = create_parser().parse_args([
        "--data", "unused.fasta", "--strategy", "ptt", "--ptt-optimizer", "adaptive", "--ptt-trust-radius", "0.002",
    ])
    assert parsed.ptt_optimizer == "adaptive"
    assert parsed.ptt_trust_radius == 0.002
    assert create_parser().parse_args([
        "--data", "unused.fasta", "--strategy", "ptt", "--ptt-optimizer", "sgd",
    ]).ptt_optimizer == "sgd"
    data = tmp_path / "data.fasta"
    data.write_text(">a\nAA\n>b\nBB\n")
    with pytest.raises(InputValidationError, match="PTT tuning"):
        run(create_parser().parse_args([
            "--data", str(data), "--alphabet", "AB", "--strategy", "pcd", "--ptt-optimizer", "sgd",
        ]))
    with pytest.raises(InputValidationError, match="optimizer"):
        PTTConfig(optimizer="unknown")


def test_ptt_metrics_resume_and_consumers(training_case, tmp_path, monkeypatch):
    data, config = training_case
    full = train_model(data, validation_path=data, config=config)
    first = train_model(data, validation_path=data, config=replace(config, max_epochs=1), output_dir=tmp_path / "run")
    # Named alphabets and their explicit token strings are already checked by
    # the archive token-order guard and must not make numerical resume fail.
    alias_archive = PTTSampler.from_archive(first.artifacts["ptt_archive"], mode="resume")
    alias_archive.training_state["settings"]["alphabet"] = "equivalent-token-alias"
    alias_path = alias_archive.save_archive(tmp_path / "alias.h5")
    resumed = train_model(data, validation_path=data, config=config, ptt_resume=alias_path)
    assert full.gradient_steps == resumed.gradient_steps == 3
    torch.testing.assert_close(full.chains, resumed.chains, rtol=0, atol=0)
    for key in full.model.params:
        torch.testing.assert_close(full.model.params[key], resumed.model.params[key], rtol=0, atol=0)
    assert full.partition_estimate == resumed.partition_estimate
    for key in full.history:
        if key != "Time" and not key.startswith("ptt_time_"):
            np.testing.assert_equal(full.history[key], resumed.history[key])
    assert "ESS" not in full.final_metrics
    assert full.final_metrics["logZ_method"] == "ptt_bridge"
    assert full.final_metrics["Entropy"] == pytest.approx(
        compute_entropy(full.chains, full.model.params, full.partition_estimate.log_z)
    )
    from adabmDCA.api.input_loading import load_training_inputs

    inputs = load_training_inputs(data, validation=data, config=config, device=torch.device("cpu"), dtype=torch.float64)
    fi_val, fij_val = inputs.validation.get_frequencies(pseudocount=1 / inputs.validation.get_effective_size())
    assert full.final_metrics["LL_val"] == pytest.approx(
        compute_log_likelihood(fi_val, fij_val, full.model.params, full.partition_estimate.log_z)
    )
    assert full.to_dict()["data"]["partition_estimate"]["method"] == "ptt_bridge"
    assert "log_weight=" not in first.artifacts["chains"].read_text()
    archive_bytes = first.artifacts["ptt_archive"].read_bytes()
    generated = sample_sequences(
        model=first.artifacts["ptt_archive"], ptt=True, n_sequences=173, n_sweeps=1000, reference_fasta=data, alphabet="AB", device="cpu"
    )
    entropy = estimate_ptt_entropy(model=first.artifacts["ptt_archive"], n_sweeps=1)
    assert len(generated.sequences) == 173
    assert generated.sampler == "ptt"
    assert entropy.entropy == pytest.approx(entropy.mean_energy + entropy.log_z)
    assert first.artifacts["ptt_archive"].read_bytes() == archive_bytes
    from adabmDCA.io import load_params

    exported = load_params(str(first.artifacts["params"]), tokens="AB", device=torch.device("cpu"), dtype=torch.float64)
    for key in exported:
        torch.testing.assert_close(exported[key], first.model.params[key], atol=1e-5, rtol=1e-5)


def test_ptt_sgd_resume_preserves_group_rates(training_case, tmp_path):
    data, base = training_case
    config = replace(base, ptt=replace(base.ptt, optimizer="sgd"), max_epochs=3)
    complete = train_model(data, config=config)
    partial = train_model(data, config=replace(config, max_epochs=1), output_dir=tmp_path / "sgd")
    resumed = train_model(data, config=config, ptt_resume=partial.artifacts["ptt_archive"])
    torch.testing.assert_close(complete.model.params["bias"], resumed.model.params["bias"], rtol=0, atol=0)
    torch.testing.assert_close(complete.model.params["coupling_matrix"],
                               resumed.model.params["coupling_matrix"], rtol=0, atol=0)
    assert complete.history["bias_learning_rate"] == resumed.history["bias_learning_rate"]
    assert complete.history["coupling_learning_rate"] == resumed.history["coupling_learning_rate"]
    assert set(complete.history["bias_learning_rate"]) == {config.learning_rate}
    assert "ptt_lag_drift" not in complete.history
    saved = PTTSampler.from_archive(partial.artifacts["ptt_archive"], mode="resume")
    assert saved.optimizer_state["kind"] == "sgd"


def test_ptt_adaptive_resume_preserves_trust_region_state(training_case, tmp_path):
    data, base = training_case
    config = replace(base, learning_rate=0.02, ptt=replace(base.ptt, trust_radius=0.001), max_epochs=3)
    complete = train_model(data, config=config)
    partial = train_model(data, config=replace(config, max_epochs=1), output_dir=tmp_path / "adaptive")
    resumed = train_model(data, config=config, ptt_resume=partial.artifacts["ptt_archive"])

    torch.testing.assert_close(complete.model.params["bias"], resumed.model.params["bias"], rtol=0, atol=0)
    torch.testing.assert_close(
        complete.model.params["coupling_matrix"], resumed.model.params["coupling_matrix"], rtol=0, atol=0
    )
    for key in ("bias_learning_rate", "coupling_learning_rate", "ptt_predicted_kl"):
        assert complete.history[key] == resumed.history[key]
    saved = PTTSampler.from_archive(partial.artifacts["ptt_archive"], mode="resume")
    assert saved.optimizer_state["kind"] == "adaptive"
    assert saved.optimizer_state["trust_diagnostics"] is not None
    assert saved.optimizer_state["nominal_rates"] == {"bias": 0.02, "coupling_matrix": 0.02}


def test_metrics_use_injected_ptt_normalizer(training_case, monkeypatch):
    original = PTTSampler.partition_estimate
    monkeypatch.setattr(PTTSampler, "partition_estimate", lambda self: replace(original(self), log_z=42.0))
    data, config = training_case
    result = train_model(data, validation_path=data, config=config)
    assert set(result.history["logZ"]) == {42.0}
    assert result.final_metrics["Entropy"] == pytest.approx(compute_entropy(result.chains, result.model.params, 42.0))
    from adabmDCA.api.input_loading import load_training_inputs

    inputs = load_training_inputs(data, config=config, device=torch.device("cpu"), dtype=torch.float64)
    fi, fij = inputs.training.get_frequencies(pseudocount=0.1)
    assert result.final_metrics["LL_train"] == pytest.approx(compute_log_likelihood(fi, fij, result.model.params, 42.0))


def reject_for_mixing(self, *args, **kwargs):
    self.last_failure = {"reason": "mixing_budget_exceeded", "model_version": self.model_version + 1,
                         "mixing": {"tau_int": None, "tau_exp": None, "rounds": 100, "required_rounds": 100,
                                    "status": "budget_exceeded", "method": "renewal", "warmup_rounds": 30,
                                    "renewal_rounds": None, "trapped_fraction": 0.03,
                                    "model_version": self.model_version + 1, "acceptance": [0.05, 0.4]},
                         "events": []}
    return False, 7


def test_bounded_mixing_recovery_and_restored_archive(training_case, tmp_path, monkeypatch):
    data, config = training_case
    monkeypatch.setattr(PTTSampler, "transition_target", reject_for_mixing)
    with pytest.raises(ConvergenceError, match=r"link 1 of 2 \(acceptance 0\.050, below the target"):
        train_model(data, config=config, output_dir=tmp_path / "failure")
    saved = PTTSampler.from_archive(tmp_path / "failure/ptt.h5", mode="resume")
    assert saved.model_version == saved.training_state["gradient_steps"] == 0
    assert len(saved.training_state["recovery_events"]) == config.ptt.max_recoveries
    assert saved.training_state["learning_rate"] == config.learning_rate / 2 ** config.ptt.max_recoveries
    assert saved.training_state["sweeps"] >= 7 * config.ptt.max_recoveries
    assert saved.training_state["history"]["Epochs"] == [0]
    log = (tmp_path / "failure/adabmDCA.log").read_text()
    assert "mixing budget exceeded" in log
    assert f"stopped after {config.ptt.max_recoveries} recoveries" in log
    assert "3.0% of the endpoint configurations were never replaced" in log
    events = [json.loads(line) for line in (tmp_path / "failure/events.jsonl").read_text().splitlines()]
    recoveries = [event for event in events if event["event"] == "ptt_recovery"]
    # Each recovery halves the rates; the one after the last halving exhausts the budget.
    assert len(recoveries) == config.ptt.max_recoveries + 1
    assert {event["reason"] for event in recoveries} == {"mixing_budget_exceeded"}


def test_recovery_halves_both_ptt_optimizer_rates(training_case, tmp_path, monkeypatch):
    data, base = training_case
    config = replace(base, learning_rate=0.016, ptt=replace(base.ptt, optimizer="sgd"))
    monkeypatch.setattr(PTTSampler, "transition_target", reject_for_mixing)
    with pytest.raises(ConvergenceError, match="replicas failed to mix again"):
        train_model(data, config=config, output_dir=tmp_path / "two_rates")
    saved = PTTSampler.from_archive(tmp_path / "two_rates/ptt.h5", mode="resume")
    rates = saved.training_state["learning_rates"]
    assert rates["bias"] == pytest.approx(0.016 / 2 ** config.ptt.max_recoveries)
    assert rates["coupling_matrix"] == pytest.approx(0.016 / 2 ** config.ptt.max_recoveries)
    assert saved.optimizer_state["rates"] == rates


@pytest.mark.parametrize(
    "kwargs", [{"model_type": "edDCA"}, {"dtype": "bfloat16"}, {"model_type": "eaDCA", "dtype": "bfloat16"}]
)
def test_unsupported_training_modes(kwargs):
    with pytest.raises(InputValidationError):
        TrainingConfig(ptt=PTTConfig(), **kwargs)


def test_ptt_chain_default_matches_standard_training():
    from adabmDCA.scripts.train import create_parser

    standard = TrainingConfig()
    ptt = TrainingConfig(ptt=PTTConfig())
    assert standard.n_chains == ptt.n_chains == 2_000
    assert ptt.n_sweeps == 10
    assert ptt.sampler == "metropolized_gibbs"
    assert ptt.max_epochs == 50_000
    assert ptt.ptt.optimizer == "adaptive"
    assert ptt.ptt.mixing_method == "renewal"
    parsed = create_parser().parse_args(["--strategy", "ptt", "--data", "unused.fasta"])
    assert parsed.nchains == 2_000
    assert parsed.nsweeps == 10
    assert parsed.sampler == "metropolized_gibbs"
    assert parsed.nepochs == 50_000


def test_ptt_swaps_factor_cli_name_and_help_order():
    from adabmDCA.scripts.train import create_parser

    parser = create_parser()
    parsed = parser.parse_args([
        "--strategy", "ptt", "--data", "unused.fasta", "--ptt-swaps-factor", "7",
    ])
    assert parsed.ptt_swaps == 7
    with pytest.raises(SystemExit):
        parser.parse_args(["--strategy", "ptt", "--data", "unused.fasta", "--ptt-swaps", "7"])
    help_text = parser.format_help()
    assert help_text.index("Sequence reweighting arguments:") < help_text.index(
        "Parallel Trajectory Tempering arguments:"
    )
    assert "Parallel Trajectory Tempering" in help_text
    for expected in ("Metropolized", "10 sweeps per exchange round", "2000 chains", "50000 updates"):
        assert expected in " ".join(help_text.split())


def test_cli_training_sampling_entropy(training_case, tmp_path):
    from adabmDCA.scripts.sample import create_parser as sample_parser
    from adabmDCA.scripts.sample import run as sample_run
    from adabmDCA.scripts.td_integration import create_parser as entropy_parser
    from adabmDCA.scripts.td_integration import run as entropy_run
    from adabmDCA.scripts.train import create_parser as train_parser
    from adabmDCA.scripts.train import run as train_run

    data, _ = training_case
    result = train_run(
        train_parser().parse_args(
            [
                "-d",
                str(data),
                "-o",
                str(tmp_path / "cli"),
                "--strategy", "ptt",
                "--alphabet",
                "AB",
                "--nchains",
                "24",
                "--nsweeps",
                "1",
                "--nepochs",
                "1",
                "--ptt-equilibration-rounds",
                "1",
                "--device",
                "cpu",
            ]
        )
    )
    archive = str(result.artifacts["ptt_archive"])
    generated = sample_run(
        sample_parser().parse_args(
            ["-p", archive, "-o", str(tmp_path), "--strategy", "ptt", "--ngen", "17", "--data", str(data), "--max_nsweeps", "1000",
             "--device", "cpu", "--ptt-stationary"]
        )
    )
    assert len(generated.sequences) == 17
    artifacts = generated.save_bundle(tmp_path / "generated")
    import pandas as pd

    ladder_log = pd.read_csv(artifacts["ptt_log"])
    assert len(ladder_log) == len(result.ptt_sampler.models)
    assert {"entropy", "log_likelihood", "log_z", "model_id"}.issubset(ladder_log.columns)
    assert artifacts["samples"].read_text().count(">sequence ") == 17
    mixing_log = pd.read_csv(artifacts["mixing_log"])
    assert mixing_log["warmup_rounds"][0] > 0
    assert mixing_log["renewal_rounds"][0] > 0
    renewal_log = pd.read_csv(artifacts["ptt_renewal_log"])
    assert set(renewal_log["phase"]) == {"warmup", "stationary"}
    for _, phase in renewal_log.groupby("phase"):
        assert phase["ladder_old"].is_monotonic_decreasing
    entropy = entropy_run(
        entropy_parser().parse_args(["-p", archive, "--strategy", "ptt", "--nsweeps", "1", "--device", "cpu", "-o", str(tmp_path)])
    )
    assert entropy.method == "ptt_bar"


def test_ordered_three_replica_exchange_preserves_product_target():
    x = states()
    models = [profile(), coupled(), coupled()]
    models[-1]["bias"] = models[-1]["bias"] * -0.8
    combinations = list(product(range(4), repeat=3))
    lookup = {indices: i for i, indices in enumerate(combinations)}
    log_probability = torch.tensor(
        [
            -sum(compute_energy(x[j : j + 1], model).item() for j, model in zip(indices, models))
            for indices in combinations
        ],
        dtype=torch.float64,
    )
    probability = log_probability.softmax(0)
    transition = torch.eye(len(combinations), dtype=torch.float64)
    for k in range(2):
        pair_transition = torch.zeros_like(transition)
        for i, indices in enumerate(combinations):
            j0, j1 = indices[k : k + 2]
            accept = exchange_log_acceptance(models[k], models[k + 1], x[j0 : j0 + 1], x[j1 : j1 + 1]).exp().item()
            swapped = list(indices)
            swapped[k], swapped[k + 1] = swapped[k + 1], swapped[k]
            pair_transition[i, i] += 1 - accept
            pair_transition[i, lookup[tuple(swapped)]] += accept
        transition = transition @ pair_transition
    torch.testing.assert_close(probability @ transition, probability, rtol=1e-12, atol=1e-14)


def test_identical_exchange_swaps_both_populations_and_lineage(monkeypatch):
    sampler = PTTSampler(profile(), tokens="AB", n_chains=12, sampler="metropolis")
    base = sampler.chains[0].clone()
    lineage = sampler.lineage.clone()
    monkeypatch.setattr("adabmDCA.ptt.sampler._profile_states", lambda *args: base.clone())
    sampler._local_kernel = lambda chains, params, nsweeps: chains.clone()
    monkeypatch.setattr("adabmDCA.ptt.sampler.torch.randperm", lambda n, **kwargs: torch.arange(n, device=kwargs.get("device")))
    sampler.advance()
    assert sampler.acceptance == [1.0]
    # Exchanges precede local moves; the exact profile then refreshes slot 0.
    assert torch.equal(sampler.chains[0], base)
    assert torch.equal(sampler.chains[1], base)
    assert torch.equal(sampler.lineage, lineage.flip(0))


def test_extreme_finite_exchange_log_acceptance():
    x = states()
    lower, upper = profile(), profile()
    upper["bias"][0] = torch.tensor([10000.0, -10000.0])
    log_a = exchange_log_acceptance(lower, upper, x[:1], x[2:3])
    assert torch.isfinite(log_a).all()
    assert log_a.item() == 0.0
    reverse = exchange_log_acceptance(lower, upper, x[2:3], x[:1])
    assert reverse.item() < -19000


def test_temporary_and_held_snapshot_procedure():
    sampler = PTTSampler(
        profile(),
        tokens="AB",
        n_chains=2000,
        config=PTTConfig(equilibration_rounds=4, max_replicas=3),
        seed=3,
    )
    temporary = sampler.endpoint_params()
    sampler.acceptance = [0.4]
    assert sampler.update_replica_chain()
    assert sampler.temporary is not None and sampler.held is None
    assert len(sampler.models) == 2
    sampler.models[-1] = coupled()
    sampler.model_version += 1
    held = sampler.endpoint_params()
    sampler.acceptance = [0.2]
    assert sampler.update_replica_chain(local_sweeps=2)
    assert len(sampler.models) == 3
    assert sampler.ladder_version == sampler.model_version == 1
    assert torch.equal(sampler.models[-2]["coupling_matrix"], temporary["coupling_matrix"])
    assert sampler.temporary is None and sampler.held is not None
    assert sampler.models[-2]["bias"].data_ptr() != sampler.models[-1]["bias"].data_ptr()
    sampler.models[-1] = profile()
    sampler.model_version += 1
    sampler.acceptance[-1] = 0.2
    assert sampler.update_replica_chain()
    assert sampler.held is None
    assert len(sampler.models) == 3
    assert torch.equal(sampler.models[-2]["coupling_matrix"], held["coupling_matrix"])


def test_endpoint_loader_does_not_read_sampler_arrays(tmp_path, monkeypatch):
    from adabmDCA import load_model

    sampler = PTTSampler(profile(), tokens="AB", n_chains=10)
    path = sampler.save_archive(tmp_path / "ptt.h5")
    monkeypatch.setattr(PTTSampler, "from_archive", lambda *a, **kw: pytest.fail("Should only read endpoint"))
    with h5py.File(path, "r+") as archive:
        del archive["rng"]
        del archive["replicas/0/chains"]
    model = load_model(path, device="cpu")
    assert model.tokens == "AB"
    torch.testing.assert_close(model.params["bias"], profile()["bias"])
    with pytest.raises(InputValidationError, match="alphabet"):
        load_model(path, alphabet="BA", device="cpu")


def test_validation_early_stop_resume_and_progress(training_case, tmp_path, monkeypatch):
    data, config = training_case
    events = []
    monkeypatch.setattr("adabmDCA.ptt.training.get_correlation_two_points", lambda **kwargs: (0.5, 1.0))
    stopped = train_model(
        data, config=replace(config, target_pearson=0.0), output_dir=tmp_path / "stopped", progress=events.append
    )
    assert stopped.gradient_steps == 0
    assert stopped.history["Epochs"] == [0]
    assert events[0].partition_estimate["method"] == "ptt_bridge"
    with pytest.raises(InputValidationError, match="configuration"):
        train_model(data, config=replace(config, learning_rate=0.04), ptt_resume=stopped.artifacts["ptt_archive"])
    with pytest.raises(InputValidationError, match="warm starts"):
        train_model(data, config=config, initial_params_path=stopped.artifacts["params"])
    with pytest.raises(InputValidationError, match="seed"):
        PTTSampler.from_archive(stopped.artifacts["ptt_archive"], mode="resume", seed=9)


def test_ptt_stage_progress_reports_work_without_extra_metric_events(training_case, tmp_path):
    data, config = training_case
    config = replace(config, ptt=replace(config.ptt, mixing_method="autocorrelation"))
    metrics, stages = [], []
    result = train_model(data, config=config, output_dir=tmp_path / "stages",
                         progress=metrics.append, stage_progress=stages.append)
    assert len(metrics) == len(result.history["Epochs"])
    assert [(e.current, e.total) for e in stages if e.stage == "ptt_equilibration" and e.kind == "progress"] == [
        (0, 2), (1, 2), (2, 2),
    ]
    assert any(e.stage == "mixing_warmup" and e.current == e.total == 2 for e in stages)
    assert any(e.stage == "mixing_measure" and e.current == e.total == 100 for e in stages)
    assert any(e.stage == "ptt_mixing" and e.details.get("status") == "converged" for e in stages)
    starts = [e for e in stages if e.stage == "ptt_checkpoint_start"]
    done = [e for e in stages if e.stage == "ptt_checkpoint_done"]
    assert starts and len(starts) == len(done)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["bias"].fill_(float("nan")),
        lambda p: p["bias"][0, 0].fill_(float("-inf")),
        lambda p: p["coupling_matrix"][0, 0, 1, 1].fill_(1),
    ],
)
def test_invalid_parameters_fail_before_sampling(change):
    params = profile()
    change(params)
    with pytest.raises(InputValidationError):
        PTTSampler(params, tokens="AB", n_chains=4)


def test_generation_freezes_models_and_rejects_conflicting_overrides(tmp_path):
    sampler = PTTSampler(profile(), tokens="AB", n_chains=8, sampler="gibbs")
    path = sampler.save_archive(tmp_path / "ptt.h5")
    data = tmp_path / "reference.fasta"
    data.write_text(">a\nAA\n>b\nBB\n")
    frozen = PTTSampler.from_archive(path)
    with pytest.raises(InputValidationError, match="freezes"):
        frozen.transition_target(coupled())
    for kwargs in ({"sampler": "metropolis"}, {"dtype": "float32"}, {"alphabet": "BA"}, {"beta": 0.5}):
        with pytest.raises(InputValidationError):
            sample_sequences(model=path, ptt=True, n_sequences=4, n_sweeps=1000, reference_fasta=data, device="cpu", **kwargs)
    result = sample_sequences(model=path, ptt=True, n_sequences=4, n_sweeps=1000, reference_fasta=data, device="cpu")
    assert result.sampling_dtype == "float64"
    assert result.model.tokens == "AB"


def test_learning_rate_floor_is_not_crossed(training_case, tmp_path, monkeypatch):
    data, config = training_case
    config = replace(config, ptt=replace(config.ptt, min_learning_rate=0.007))
    attempted = []

    def reject(self, *args, **kwargs):
        attempted.append(1)
        return reject_for_mixing(self, *args, **kwargs)

    monkeypatch.setattr(PTTSampler, "transition_target", reject)
    with pytest.raises(ConvergenceError):
        train_model(data, config=config, output_dir=tmp_path / "floor")
    assert len(attempted) == 1
    with pytest.raises(ConvergenceError):
        train_model(data, config=config, ptt_resume=tmp_path / "floor/ptt.h5")
    assert len(attempted) == 1


def test_trial_cancellation_preserves_last_committed_archive(training_case, tmp_path, monkeypatch):
    from adabmDCA.exceptions import OperationCancelledError

    data, config = training_case

    def cancel(*args, **kwargs):
        raise OperationCancelledError("cancelled in trial")

    monkeypatch.setattr(PTTSampler, "transition_target", cancel)
    with pytest.raises(OperationCancelledError):
        train_model(data, config=config, output_dir=tmp_path / "cancelled")
    saved = PTTSampler.from_archive(tmp_path / "cancelled/ptt.h5", mode="resume")
    assert saved.training_state["gradient_steps"] == saved.model_version == 0
    assert saved.training_state["history"]["logZ_method"] == ["ptt_bridge"]
    assert "status: cancelled" in (tmp_path / "cancelled/adabmDCA.log").read_text()


def test_ptt_log_remains_readable_by_existing_plotter(training_case, tmp_path):
    from adabmDCA.plot_training_log import parse_training_log

    data, config = training_case
    result = train_model(data, config=config, output_dir=tmp_path / "logged")
    _, history = parse_training_log(result.artifacts["log"])
    assert len(history["Epochs"]) == len(result.history["Epochs"])
    assert "ESS" not in history
    np.testing.assert_allclose(history["Entropy"], result.history["Entropy"], rtol=1e-5)


def test_generation_reuses_in_place_mixing_and_flagged_ladder(tmp_path, monkeypatch):
    from adabmDCA.ptt.mixing import MixingEstimate

    backend = PTTSampler(profile(), tokens="AB", n_chains=8, config=PTTConfig(full_sampler=True))
    backend.models.append(profile())
    backend.chains.append(backend.chains[-1].clone())
    backend.total_models = 3
    backend.model_version = 2
    backend.ptt_checkpoints.append({"step": 1, "params": profile()})
    backend.active_start = 1
    backend.reservoir = backend.chains[0].clone()
    backend._reset_lineage()
    data = tmp_path / "reference.fasta"
    data.write_text(">a\nAA\n>b\nBB\n")
    calls = []
    advance = PTTSampler.advance

    def measure(self, **kwargs):
        assert kwargs["bounded"] is True
        assert kwargs["max_rounds"] == 20_000
        assert kwargs["in_place"] is True
        assert kwargs["reference"] is True
        assert len(self.chains[0]) == 17
        assert self.active_start == 0
        assert self.reservoir is None
        return MixingEstimate(1.2, 2.2, 100, 44, "converged")

    def counted_advance(self, **kwargs):
        calls.append(kwargs["rounds"])
        return advance(self, **kwargs)

    monkeypatch.setattr(PTTSampler, "estimate_mixing_time", measure)
    monkeypatch.setattr(PTTSampler, "advance", counted_advance)
    result = sample_sequences(model=backend, reference_fasta=data, n_sequences=17, n_sweeps=0,
                              ptt_mixing_method="autocorrelation")
    assert calls == []
    assert result.ptt_diagnostics["equilibration_rounds"] == 0
    assert result.ptt_diagnostics["selected_updates"] == [0, 1, 2]
    assert len(result.ptt_diagnostics["models"]) == 3
    assert result.ptt_diagnostics["likelihood_basis"] == "weighted_reference_data"
    assert backend.active_start == 1 and backend.reservoir is not None
    assert backend.rounds == 0


def test_generation_retains_endpoint_when_mixing_is_unresolved(tmp_path, monkeypatch):
    from adabmDCA.ptt.mixing import MixingEstimate

    backend = PTTSampler(profile(), tokens="AB", n_chains=8)
    with pytest.raises(InputValidationError, match="--data"):
        sample_sequences(model=backend, n_sequences=4)
    data = tmp_path / "reference.fasta"
    data.write_text(">a\nAA\n>b\nBB\n")
    def unresolved(self, *args, **kwargs):
        self.mixing_correlation = torch.tensor([1.0, 0.7, 0.4, 0.2])
        return MixingEstimate(None, None, 100, 200, "budget_exceeded")

    monkeypatch.setattr(PTTSampler, "estimate_mixing_time", unresolved)
    monkeypatch.setattr(PTTSampler, "advance", lambda *args, **kwargs: pytest.fail("Endpoint batch needs no spacing"))
    result = sample_sequences(
        model=backend, reference_fasta=data, n_sequences=4, collect_diagnostics=True,
        ptt_mixing_method="autocorrelation",
    )
    assert len(result.sequences) == 4
    assert result.ptt_diagnostics["mixing"]["status"] == "budget_exceeded"
    assert result.mixing_history["spacing_rounds"] == [1]
    assert any("samples were retained" in warning for warning in result.warnings)
    artifacts = result.save_diagnostic_plots(tmp_path / "plots")
    assert all(path.is_file() for path in artifacts.values())


def test_flagged_checkpoints_survive_compression_archive_and_rollback(tmp_path):
    backend = PTTSampler(profile(), tokens="AB", n_chains=12,
                         config=PTTConfig(mixing_thermalization_rounds=2, reservoir_size=24))
    backend.model_version = 1
    backend.acceptance = [0.4]
    assert backend.update_replica_chain()
    assert [p["step"] for p in backend.ptt_checkpoints] == [0]
    backend.model_version = 2
    backend.acceptance = [0.2]
    assert backend.update_replica_chain()
    flagged = backend.endpoint_params()
    assert [p["step"] for p in backend.ptt_checkpoints] == [0, 2]
    point = backend.capture_state()
    backend.model_version = 3
    backend.acceptance[-1] = 0.2
    assert backend.update_replica_chain()
    assert len(backend.models) == 2 and backend.reservoir is not None
    archive = backend.save_archive(tmp_path / "flagged.h5")
    with h5py.File(archive) as handle:
        assert list(handle["ptt_checkpoints"]) == ["0", "2"]
        assert handle["ptt_checkpoints/2"].attrs["flag"] == "ptt"
    loaded = PTTSampler.from_archive(archive)
    loaded.prepare_sampling_ladder(7)
    assert loaded.sampling_steps == [0, 2, 3]
    assert loaded.reservoir is None
    torch.testing.assert_close(loaded.models[1]["bias"], flagged["bias"])
    assert all(len(chains) == 7 for chains in loaded.chains)
    backend.ptt_checkpoints.append({"step": 3, "params": backend.endpoint_params()})
    backend.restore_state(point)
    assert [p["step"] for p in backend.ptt_checkpoints] == [0, 2]


def test_ladder_without_flagged_models_cannot_be_sampled(tmp_path):
    backend = PTTSampler(profile(), tokens="AB", n_chains=8)
    loaded = PTTSampler.from_archive(backend.save_archive(tmp_path / "archive.h5"))
    assert loaded.ptt_checkpoints_complete
    # A ladder whose flagged models are missing cannot recover their arrays
    # from the selection events alone.
    loaded.ptt_checkpoints_complete = False
    with pytest.raises(InputValidationError, match="lacks the saved models"):
        loaded.prepare_sampling_ladder(8)


def test_resume_into_the_same_folder_rewrites_history_and_appends_events(training_case, tmp_path):
    data, config = training_case
    folder = tmp_path / "run"
    partial = train_model(data, config=replace(config, max_epochs=1), output_dir=folder)
    # Rows after the saved checkpoint would be lost in a crash; the archive's history is authoritative.
    with open(folder / "history.csv", "a", encoding="utf-8") as handle:
        handle.write("99,0,0,0,0\n")
    train_model(data, config=config, output_dir=folder, ptt_resume=partial.artifacts["ptt_archive"])
    steps = [line.split(",")[0] for line in (folder / "history.csv").read_text().splitlines()[1:]]
    assert steps == ["0", "1", "2", "3"]
    events = [json.loads(line)["event"] for line in (folder / "events.jsonl").read_text().splitlines()]
    assert events.count("resumed") == 1 and events.index("resumed") > events.index("checkpoint_saved")
    log = (folder / "adabmDCA.log").read_text()
    assert log.count("adabmDCA training log") == 2 and "resumed from step 1" in log


def test_validation_plateau_compares_window_medians_and_ignores_spikes():
    from adabmDCA.ptt.training import validation_plateau

    rising = [0.01 * step for step in range(40)]
    assert validation_plateau(rising, 10, 0.0) is None
    assert validation_plateau(rising[:19], 10, 0.0) is None  # fewer than two windows
    flat = rising + [0.4] * 20
    # Equal medians: no gain, but no loss either; a positive minimum gain stops.
    assert validation_plateau(flat, 10, 0.0) is None
    assert validation_plateau(flat, 10, 1e-3) == pytest.approx(0.0)
    # A single-update spike in the last window does not move its median.
    spiked = rising + [0.39] * 20
    spiked[-3] = -5.0
    assert validation_plateau(spiked, 10, 0.0) is None
    assert validation_plateau(rising + [0.39] * 10 + [0.385] * 10, 10, 0.0) == pytest.approx(-0.005)


def test_validation_stop_ends_training_on_the_validation_plateau(training_case, tmp_path):
    data, config = training_case
    ptt = replace(config.ptt, validation_stop=True, validation_window=1, validation_min_gain=10.0)
    stopped = train_model(data, validation_path=data, output_dir=tmp_path / "plateau",
                          config=replace(config, ptt=ptt, max_epochs=10, target_pearson=0.0))
    # The target Pearson (0 here) is not a stop criterion; two one-update windows are.
    assert stopped.gradient_steps == 2 and stopped.stop_reason == "validation_plateau" and stopped.converged
    events = [json.loads(line) for line in (tmp_path / "plateau" / "events.jsonl").read_text().splitlines()]
    assert [event["window"] for event in events if event["event"] == "ptt_validation_plateau"] == [1]
    assert "validation plateau" in (tmp_path / "plateau" / "adabmDCA.log").read_text()
    with pytest.raises(InputValidationError, match="validation set"):
        train_model(data, config=replace(config, ptt=ptt))


def test_validation_stop_cli_and_configuration():
    from adabmDCA.scripts.train import create_parser

    parsed = create_parser().parse_args([
        "--data", "unused.fasta", "--strategy", "ptt", "--ptt-validation-stop", "--ptt-validation-window", "50",
        "--ptt-validation-min-gain", "1e-4",
    ])
    assert (parsed.ptt_validation_stop, parsed.ptt_validation_window, parsed.ptt_validation_min_gain) == (True, 50, 1e-4)
    assert create_parser().parse_args(["--data", "unused.fasta", "--strategy", "ptt"]).ptt_validation_stop is None
    for bad in ({"validation_window": 0}, {"validation_min_gain": float("nan")}, {"validation_stop": "yes"}):
        with pytest.raises(InputValidationError):
            PTTConfig(**bad)


def test_training_cli_plots_the_training_curves_of_a_ptt_run(tmp_path, capsys):
    from adabmDCA.scripts.train import create_parser, main

    data = tmp_path / "data.fasta"
    data.write_text("".join(f">s{i}\n{sequence}\n" for i, sequence in enumerate(("AAB", "ABA", "BAB", "BBA", "AAA", "BBB"))))
    args = create_parser().parse_args([
        "--data", str(data), "--val", str(data), "--alphabet", "AB", "--strategy", "ptt", "--output", str(tmp_path / "run"),
        "--label", "tiny", "--device", "cpu", "--nchains", "16", "--nsweeps", "1", "--nepochs", "3",
        "--ptt-initialization-rounds", "2", "--ptt-mixing-thermalization-rounds", "2", "--ptt-mixing-chains", "10",
        "--no_reweighting", "--no-progress", "--plot-training-logs",
    ])
    assert main(args) == 0
    folder = tmp_path / "run" / "tiny_plots"
    names = {path.name for path in folder.glob("*.png")}
    for panel in ("overview", "pearson", "slope", "loglikelihood", "entropy", "ladder", "acceptance",
                  "learning_rates", "kl", "lag"):
        assert f"tiny_{panel}.png" in names
    assert "tiny_density.png" not in names
    assert f"training plots: {folder}" in capsys.readouterr().out






def test_failed_renewal_names_configurations_only_local_moves_can_free():
    sampler = PTTSampler(profile(), tokens="AB", n_chains=200, seed=1)
    upper = {key: value.clone() for key, value in profile().items()}
    upper["bias"][0, 0] += 12.0  # a deep basin (site 0 in state A) the lower model lacks
    sampler.models[1] = upper
    sampler.chains[0][:, 0] = 1
    sampler.chains[1][:, 0] = 1
    sampler.chains[1][:20, 0] = 0
    sampler.start_birth_tracking()
    found = sampler._immobile_old_configurations(sampler.rounds, 0.01)
    assert found == [{"replica": 1, "chains": 200, "old": 200, "immobile": 20, "blocking": True}]
    assert not sampler._immobile_old_configurations(sampler.rounds, 0.2)[0]["blocking"]


def test_failure_message_recommends_more_sweeps_only_for_immobile_configurations():
    from adabmDCA.ptt.training import _mixing_failure_summary

    config = PTTConfig()
    mixing = {"model_version": 9, "rounds": 100, "method": "renewal", "warmup_rounds": 30, "acceptance": [0.3, 0.6],
              "immobile": [{"replica": 1, "chains": 200, "old": 25, "immobile": 20, "blocking": True}]}
    assert "--nsweeps 50" in _mixing_failure_summary(mixing, config, 5)
    mixing["immobile"] = [{"replica": 1, "chains": 200, "old": 25, "immobile": 1, "blocking": False}]
    message = _mixing_failure_summary({**mixing, "acceptance": [0.05, 0.6]}, config, 5)
    assert "link 1 of 2" in message and "--nsweeps" not in message
    reservoir = {"model_version": 9, "acceptance": [0.3, 0.6], "status": "budget_exceeded",
                 "reservoir_renewal": {"warmup_rounds": None, "immobile": [
                     {"replica": 1, "chains": 200, "old": 25, "immobile": 20, "blocking": True}]}}
    message = _mixing_failure_summary(reservoir, config, 10)
    assert message.startswith("At step 9 a new reservoir could not be collected") and "--nsweeps 100" in message

