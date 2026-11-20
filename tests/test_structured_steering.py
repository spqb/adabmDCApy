from itertools import product
import json

import numpy as np
import pytest
from scipy.special import logsumexp
import torch

from adabmDCA import DCAModel, HammingBias, HammingWindow, PTTSampler, SteeringConfig, WeightedPotential, sample_sequences
from adabmDCA.anchor.anchor import TreeLowRankPotts
from adabmDCA.exceptions import InputValidationError
from adabmDCA.statmech import compute_energy
from adabmDCA.steering.kernels import PottsKernel
from adabmDCA.steering.support import ExactProfile, Support


def small_model(length=4):
    generator = torch.Generator().manual_seed(123)
    return DCAModel({
        "bias": torch.randn(length, 3, generator=generator, dtype=torch.float64) * 0.15,
        "coupling_matrix": torch.randn(length, 3, length, 3, generator=generator, dtype=torch.float64) * 0.2,
    }, alphabet="-AB")


def enumerated(model, support):
    states = np.array(list(product(range(support.q), repeat=support.length)), dtype=np.int32)
    states = states[support.contains(states)]
    onehot = torch.nn.functional.one_hot(torch.from_numpy(states).long(), support.q).double()
    return states, compute_energy(onehot, model.params).numpy()


def test_constrained_profile_matches_enumeration():
    model = small_model()
    support = Support(model.tokens, 4, "AB", (HammingWindow("AAAA", 2, 2), HammingWindow("ABAB", 1, 3)))
    states, energies = enumerated(model, support)
    np.testing.assert_allclose(PottsKernel(model.params, support).energy(states), energies, atol=1e-12)
    fields = model.params["bias"].numpy()
    exact = ExactProfile(fields, support)
    log_weights = fields[np.arange(4), states].sum(axis=1)
    assert exact.log_z == pytest.approx(logsumexp(log_weights))
    draws = exact.draw(20_000, np.random.default_rng(351))
    assert support.contains(draws).all()
    counts = (draws[:, None, :] == states).all(axis=2).mean(axis=0)
    np.testing.assert_allclose(counts, np.exp(log_weights - exact.log_z), atol=0.014)
    impossible = Support(model.tokens, 4, "AB", (HammingWindow("----", 0, 1),))
    with pytest.raises(InputValidationError, match="empty support"):
        ExactProfile(fields, impossible)


@pytest.mark.parametrize("ptt", [False, True])
def test_cold_constrained_weighted_sampling_and_entropy(ptt):
    model = small_model()
    windows = (HammingWindow("AAAA", 2, 2), HammingWindow("BBAA", 0, 4))
    config = SteeringConfig(
        allowed_tokens="AB", windows=windows, distance_biases=(HammingBias("BABA", 0.23),),
        potentials=(WeightedPotential(lambda x: 0.5 * (x[:, 0] == x[:, 1]), weight=0.8, input="categorical"),),
        chains=128, min_rounds=64, min_ess=24, bridge_steps=1, ladder_steps=2, entropy=ptt,
    )
    beta = 1.5
    support = Support(model.tokens, 4, "AB", windows)
    states, energies = enumerated(model, support)
    cost = 0.23 * (states != [2, 1, 2, 1]).sum(axis=1) + 0.4 * (states[:, 0] == states[:, 1])
    reduced = beta * energies + cost
    log_z = logsumexp(-reduced)
    probabilities = np.exp(-reduced - log_z)
    result = sample_sequences(model=model, steering=config, n_sequences=8192, seed=82, device="cpu",
                              ptt=ptt, beta=beta, n_sweeps=256, ptt_max_rounds=256)
    sequences = ["".join(model.tokens[token] for token in row) for row in states]
    assert set(result.sequences) == set(sequences)
    counts = np.array([result.sequences.count(seq) / len(result.sequences) for seq in sequences])
    np.testing.assert_allclose(counts, probabilities, atol=0.04 if ptt else 0.065)
    costs = dict(zip(sequences, cost))
    np.testing.assert_allclose(result.steering_potentials, [costs[seq] for seq in result.sequences])
    if ptt:
        estimate = result.steering["entropy"]
        assert estimate["log_z"] == pytest.approx(log_z, abs=0.04)
        assert estimate["entropy_nats"] == pytest.approx(log_z + probabilities @ reduced, abs=0.04)


@pytest.mark.parametrize("restriction", ["none", "alphabet", "window", "windows"])
def test_archive_anchor_stays_on_ladder(tmp_path, restriction):
    model = small_model()
    allowed = None if restriction == "none" else "AB"
    windows = {"window": (HammingWindow("AAAA", 1, 3),),
               "windows": (HammingWindow("AAAA", 1, 3), HammingWindow("ABAB", 0, 2))}.get(restriction, ())
    anchor = TreeLowRankPotts(
        model.params["bias"],
        torch.empty(4, 3, 0, dtype=torch.float64),
        torch.tensor([[0, 1], [1, 2], [2, 3]]),
        0.25 * torch.eye(3, dtype=torch.float64).repeat(3, 1, 1),
    )
    params = anchor.potts_params(torch.float64)
    backend = PTTSampler(params, tokens="-AB", n_chains=16, anchor=anchor, seed=47)
    path = backend.save_archive(tmp_path / "anchor.h5")
    result = sample_sequences(
        model=path, ptt=True, beta=1.3, seed=16, device="cpu", n_sequences=2048,
        ptt_max_rounds=96, steering=SteeringConfig(
            allowed_tokens=allowed, windows=windows,
            chains=64, min_rounds=32, min_ess=18, entropy=True, bridge_steps=1, ladder_steps=2,
        ),
    )
    summary = result.steering
    # Windows restrict the anchor itself: no profile and no bridge below it.
    assert summary["exact_archive_anchor"]
    assert summary["ladder"][0] == "archive anchor"
    assert "exact profile" not in summary["ladder"]
    support = Support("-AB", 4, allowed, windows)
    _, energies = enumerated(DCAModel(params, alphabet="-AB"), support)
    assert summary["entropy"]["anchor_log_z"] == pytest.approx(logsumexp(-energies))
    if allowed is not None:
        assert all("-" not in sequence for sequence in result.sequences)
    assert summary["entropy"]["log_z"] == pytest.approx(logsumexp(-1.3 * energies), abs=0.055)


def test_short_warmup_reports_budget():
    result = sample_sequences(
        model=small_model(), steering=SteeringConfig(chains=4), n_sequences=9, n_sweeps=3,
    )
    assert result.steering["status"] == "warmup_budget_exhausted"
    assert len(result.sequences) == 9



@pytest.mark.parametrize("ptt", [False, True])
def test_steering_history_tracks_rounds_and_serializes(ptt, tmp_path):
    config = SteeringConfig(
        allowed_tokens="AB", windows=(HammingWindow("AAAA", 1, 3),),
        potentials=(WeightedPotential(lambda states: 2.0 * (states[:, 0] == 2), input="categorical"),),
        chains=8, min_rounds=6, check_every=2, min_ess=4, production_spacing=2,
        pilot_rounds=2, bridge_steps=1, ladder_steps=2, entropy=ptt,
    )
    result = sample_sequences(
        model=small_model(), steering=config, n_sequences=17, ptt=ptt,
        n_sweeps=12, ptt_max_rounds=12, device="cpu", seed=37,
    )
    summary = result.steering
    history = summary["history"]
    warmup, production = history["warmup"], history["production"]
    assert [row["round"] for row in warmup] == list(range(summary["pilot_rounds"] + 1, warmup[-1]["round"] + 1))
    assert [row["round"] for row in production] == [warmup[-1]["round"] + n for n in (2, 4, 6)]
    assert [row["retained"] for row in production] == [8, 8, 1]
    assert summary["trace"]["potts_energy"] == [row["observables"]["potts_energy"] for row in warmup]
    assert production[0]["observables"]["potts_energy"] == pytest.approx(result.energies[:8].mean())
    assert len(production[0]["group_means"]["distance_1"]) == config.groups
    assert history["checks"][-1]["observables"] == summary["observables"]
    for row in (*warmup, *production):
        assert len(row["block_acceptance"]) == len(history["rungs"])
        assert len(row["swap_acceptance"]) == len(history["rungs"]) - 1
        assert row["old_configurations_per_chain"] == sum(row["old_configurations_by_rung"]) / config.chains
    assert production[-1]["potential_evaluations"] == summary["potential_evaluations"]
    if ptt:
        assert history["pilot"]
        assert all(len(block["swap_acceptance"]) == len(block["ladder"]) - 1 for block in history["pilot"])
        assert warmup[0]["block_acceptance"][0] is None
        assert [point["batches"] for point in history["entropy"]] == [1, 2, 3]
        assert history["entropy"][-1]["entropy_nats"] == summary["entropy"]["entropy_nats"]
    else:
        assert not history["pilot"] and not history["entropy"]
    saved = json.loads(result.to_json(tmp_path / "result.json").read_text())
    assert saved["data"]["steering"]["history"]["production"] == production


def test_temperature_only_dispatch():
    result = sample_sequences(model=small_model(), ptt=True, beta=1.4, n_sequences=8, ptt_max_rounds=4)
    assert result.beta == 1.4
    assert result.steering["ladder"][-1] == "steering 1"
    assert not np.any(result.steering_potentials)


def test_torch_fallback_respects_support(monkeypatch):
    import adabmDCA.steering.kernels as kernels

    monkeypatch.setattr(kernels, "njit", None)
    config = SteeringConfig(allowed_tokens="AB", windows=(HammingWindow("AAAA", 2, 2),), chains=4)
    result = sample_sequences(model=small_model(), steering=config, n_sequences=8, n_sweeps=4)
    assert all(sequence.count("A") == sequence.count("B") == 2 for sequence in result.sequences)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("windows", [(), (HammingWindow("AAAA", 1, 3), HammingWindow("BABA", 1, 3))])
def test_fused_cuda_keeps_group_schedules_and_hard_support(windows):
    pytest.importorskip("triton")
    model = small_model()
    params = {name: tensor.float() for name, tensor in model.params.items()}
    support = Support("-AB", 4, "AB", windows)
    rng = np.random.default_rng(48)
    initial = ExactProfile(params["bias"].numpy(), support).draw(31, rng)
    groups = np.array_split(np.arange(len(initial)), 4)
    schedules = [rng.integers(0, 4, 71) for _ in groups]
    uniforms = [rng.uniform(1e-5, 1 - 1e-5, (len(group), len(sites))) for group, sites in zip(groups, schedules)]
    offsets = rng.normal(0, 0.2, size=(4, 3))
    cpu = PottsKernel(params, support)
    gpu = PottsKernel({name: value.cuda() for name, value in params.items()}, support)
    assert gpu.cuda
    actual = gpu.propose_groups(initial, 1.3, offsets, groups, schedules, uniforms)
    expected = cpu.propose_groups(initial, 1.3, offsets, groups, schedules, uniforms)
    assert support.contains(actual).all()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(gpu.energy(actual), cpu.energy(expected), atol=1e-6)


def test_wall_clock_budget_returns_single_snapshot(monkeypatch):
    import adabmDCA.steering.sampling as sampling

    tick = iter(range(10_000))
    monkeypatch.setattr(sampling, "perf_counter", lambda: next(tick))
    result = sample_sequences(
        model=small_model(), steering=SteeringConfig(chains=8, production_spacing=0, warmup_seconds=4),
        n_sequences=8, n_sweeps=20, device="cpu",
    )
    assert result.steering["status"] == "warmup_time_exhausted"
    assert result.steering["production_rounds"] == 0
    assert len(result.sequences) == 8
    assert result.steering["history"]["production"][0]["retained"] == 8


def test_snapshot_requires_enough_chains():
    with pytest.raises(InputValidationError, match="final snapshot"):
        sample_sequences(
            model=small_model(), steering=SteeringConfig(chains=4, production_spacing=0), n_sequences=5,
        )


def test_ptt_seeds_upper_rungs_and_counts_seeds_as_old(monkeypatch):
    import adabmDCA.steering.sampling as sampling

    tick = iter(range(10_000))
    monkeypatch.setattr(sampling, "perf_counter", lambda: next(tick))
    seeds = ("AAAB", "AABA", "ABAA", "BAAA", "AABB", "ABAB", "ABBA", "BAAB")
    config = SteeringConfig(
        allowed_tokens="AB", windows=(HammingWindow("AAAA", 1, 2),),
        chains=8, production_spacing=0, bridge_steps=1, ladder_steps=2,
        pilot_rounds=0, warmup_seconds=0.5,
    )
    options = dict(model=small_model(), steering=config, n_sequences=8, beta=2, ptt=True)
    result = sample_sequences(**options, initial_sequences=seeds)
    for start in range(0, 8, 2):
        assert set(result.sequences[start:start+2]) == set(seeds[start:start+2])
    assert result.steering["initialization"] == "initial_sequences"
    snapshot = result.steering["history"]["production"][0]
    assert snapshot["old_configurations_by_rung"] == [8] * len(result.steering["ladder"])
    with pytest.raises(InputValidationError, match="alphabet and hard windows"):
        sample_sequences(**options, initial_sequences=(*seeds[:-1], "AAAA"))
