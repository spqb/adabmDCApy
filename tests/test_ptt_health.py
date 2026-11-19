"""Bidirectional reweighting estimators and ladder-health diagnostics."""

import math

import numpy as np
import pytest
import torch

from adabmDCA import PTTSampler
from adabmDCA.api import sample_sequences
from adabmDCA.exceptions import InputValidationError
from adabmDCA.ptt.health import (
    age_permutation_test,
    bar,
    classify_trapping,
    crooks_slope,
    exp_forward,
    exp_reverse,
    mobility,
    pair_health,
)
from adabmDCA.statmech import compute_energy
from tests import test_ptt as helpers

coupled, profile, states = helpers.coupled, helpers.profile, helpers.states
single_thread = helpers.single_thread


def gaussian_pair(mean, std, n_lower, n_upper, seed=0, upper_shift=0.0):
    """W ~ N(mean, std^2) under p_k implies W ~ N(mean - std^2, std^2) under p_{k+1}; dF = mean - std^2 / 2."""
    generator = torch.Generator().manual_seed(seed)
    lower = mean + std * torch.randn(n_lower, generator=generator, dtype=torch.float64)
    upper = mean - std**2 + upper_shift + std * torch.randn(n_upper, generator=generator, dtype=torch.float64)
    return lower, upper, mean - std**2 / 2


@pytest.mark.parametrize("n_lower, n_upper", [(20_000, 20_000), (30_000, 10_000)])
def test_estimators_recover_exact_free_energy_for_consistent_populations(n_lower, n_upper):
    lower, upper, exact = gaussian_pair(3.0, 1.2, n_lower, n_upper)
    assert float(bar(lower, upper)) == pytest.approx(exact, abs=0.02)
    assert float(exp_forward(lower)) == pytest.approx(exact, abs=0.05)
    assert float(exp_reverse(upper)) == pytest.approx(exact, abs=0.05)
    slope, error, bins = crooks_slope(lower, upper)
    assert bins >= 10 and abs(slope + 1.0) < 4 * error


def test_bar_is_batched_and_bounded_by_poor_overlap():
    lower, upper, exact = gaussian_pair(0.0, 2.5, 2000, 2000, seed=3)
    batch = bar(torch.stack([lower, lower]), torch.stack([upper, upper]))
    assert torch.allclose(batch, bar(lower, upper).expand(2))
    health = pair_health(lower, upper, generator=torch.Generator().manual_seed(1))
    # Forward weights are heavy-tailed at this overlap (exact ESS exp(-6.25) ~ 0.002; a finite sample
    # misses the extreme tail and reports more). BAR stays the most precise estimate.
    assert health["ess_forward"] < 0.1
    assert health["dF_bar_error"] < health["dF_forward_error"]
    assert abs(health["dF_bar"] - exact) < 4 * health["dF_bar_error"]


def test_inconsistent_populations_break_crooks_and_bracketing():
    lower, upper, _ = gaussian_pair(1.0, 1.0, 20_000, 20_000, seed=5, upper_shift=1.0)
    health = pair_health(lower, upper, generator=torch.Generator().manual_seed(2))
    assert abs(health["hysteresis"]) > 10 * max(health["dF_forward_error"], health["dF_reverse_error"])
    # The histogram ratio stays linear with slope -1 but its offset no longer matches dF; a population
    # with the wrong shape, not only the wrong location, bends the slope.
    wide = upper.mean() + 2.0 * (upper - upper.mean())
    slope, error, _ = crooks_slope(lower, wide)
    assert abs(slope + 1.0) > 5 * error


def test_mobility_marks_configurations_that_cannot_swap_and_matches_dense_computation():
    down, up = mobility(torch.zeros(3, dtype=torch.float64), torch.tensor([-50.0, 0.0], dtype=torch.float64))
    assert down[0] < 1e-20 and down[1] == 1.0
    torch.testing.assert_close(up, torch.full((3,), 0.5, dtype=torch.float64))
    generator = torch.Generator().manual_seed(0)
    lower = torch.randn(1500, generator=generator, dtype=torch.float64)
    upper = torch.randn(3001, generator=generator, dtype=torch.float64) - 1.0
    dense = torch.exp((upper[:, None] - lower[None, :]).clamp_max(0.0))
    down, up = mobility(lower, upper)
    torch.testing.assert_close(down, dense.mean(1))
    torch.testing.assert_close(up, dense.mean(0))
    ages = torch.arange(3001, dtype=torch.float64)
    health = pair_health(lower, upper - 12.0 * (ages > 2900), ages=ages, generator=generator)
    assert health["immobile_down_fraction"] == pytest.approx(100 / 3001, abs=1e-3)
    assert health["immobile_mean_age"] > health["mobile_mean_age"]


def ladder(n_chains=12_000):
    sampler = PTTSampler(profile(), tokens="AB", n_chains=n_chains, seed=4)
    accepted, _ = sampler.transition_target(coupled(), local_sweeps=3)
    assert accepted
    sampler.advance(rounds=30, local_sweeps=2)
    return sampler


def test_bar_partition_entropy_and_ladder_statistics_match_exact_values():
    sampler = ladder()
    exact_energies = compute_energy(states(), coupled())
    exact_z = torch.logsumexp(-exact_energies, 0).item()
    bar_estimate = sampler.partition_estimate("bar")
    assert bar_estimate.method == "ptt_bar" and abs(bar_estimate.log_z - exact_z) < 0.02
    assert sampler.partition_estimate().method == "ptt_bridge"
    with pytest.raises(InputValidationError, match="estimator"):
        sampler.partition_estimate("mbar")
    entropy = sampler.entropy()
    exact_entropy = float((exact_energies * (-exact_energies).softmax(0)).sum()) + exact_z
    assert entropy["method"] == "ptt_bar" and entropy["entropy"] == pytest.approx(exact_entropy, abs=0.04)
    rows = sampler.ladder_statistics(states(), torch.ones(4, dtype=torch.float64))
    assert rows[-1]["log_z_estimator"] == "bar"
    assert rows[-1]["log_z"] == pytest.approx(bar_estimate.log_z)
    assert rows[-1]["log_z_forward"] == pytest.approx(sampler.partition_estimate().log_z)
    health = sampler.ladder_health(bootstrap=50)
    assert health["log_z_bar"] == pytest.approx(bar_estimate.log_z)
    assert abs(health["log_z_bar"] - exact_z) < 4 * health["log_z_bar_error"] + 0.01
    pair = health["pairs"][0]
    assert (pair["lower"], pair["upper"]) == (0, 1)
    assert pair["immobile_down_fraction"] == 0.0
    # The random-pairing acceptance implied by the populations matches what the sampler measured.
    assert pair["mean_acceptance"] == pytest.approx(sampler.acceptance[0], abs=0.05)


def test_generation_reports_ladder_health_with_plot_and_log(tmp_path):
    data = tmp_path / "reference.fasta"
    data.write_text(">a\nAA\n>b\nBB\n")
    backend = PTTSampler(profile(), tokens="AB", n_chains=8)
    backend.models[-1] = coupled()
    backend.model_version = 1
    result = sample_sequences(model=backend, reference_fasta=data, n_sequences=4000, ptt_local_sweeps=1,
                              collect_diagnostics=True)
    health = result.ptt_diagnostics["ladder_health"]
    assert len(health["pairs"]) == 1 and math.isfinite(health["log_z_bar"])
    assert "immobile_mean_age" not in health["pairs"][0] or health["pairs"][0]["immobile_down_fraction"] > 0
    rows = result.ptt_diagnostics["models"]
    assert rows[-1]["log_z_estimator"] == "bar" and rows[-1]["log_z"] == pytest.approx(health["log_z_bar"])
    artifacts = result.save_bundle(tmp_path / "out", label="ptt")
    artifacts.update(result.save_diagnostic_plots(tmp_path / "out", label="ptt"))
    assert artifacts["ptt_ladder_health_plot"].read_bytes().startswith(b"\x89PNG")
    log = artifacts["ptt_ladder_health_log"].read_text()
    assert log.startswith("dF_forward,") and "immobile_down_fraction" in log
    # Per-configuration ages and mobilities, to re-run the age test without sampling again.
    saved = np.load(artifacts["ptt_ages"])
    assert {"ages_0", "ages_1", "mobility_down_0", "rounds", "old_age", "tolerance"} <= set(saved.files)
    assert saved["ages_1"].shape == saved["mobility_down_0"].shape == (health["n_chains"],)
    immobile = saved["mobility_down_0"] < saved["immobile_threshold"]
    assert int(immobile.sum()) == health["pairs"][0]["immobile_count"]
    assert "ptt_configurations" not in result.to_dict()["data"]
    exact_z = torch.logsumexp(-compute_energy(states(), coupled()), 0).item()
    assert abs(health["log_z_bar"] - exact_z) < 4 * health["log_z_bar_error"] + 0.02
    assert np.isfinite(result.ptt_diagnostics["final_pearson"])


def test_age_test_is_calibrated_when_immobility_ignores_age():
    generator = torch.Generator().manual_seed(1)
    ages = (-300 * torch.rand(4000, generator=generator, dtype=torch.float64).log()).floor()
    p_values = []
    for seed in range(40):
        mask = torch.zeros(4000, dtype=torch.bool)
        mask[torch.randperm(4000, generator=torch.Generator().manual_seed(seed))[:30]] = True
        p_values.append(age_permutation_test(ages, mask, draws=400, generator=generator)["p_value"])
    # Under the null the p-values are uniform: about 10% below 0.1, none extreme.
    assert 0 < sum(p < 0.1 for p in p_values) <= 10 and min(p_values) > 1e-3


def test_age_test_detects_old_immobile_configurations_and_widens_for_small_groups():
    ages = torch.arange(10_000, dtype=torch.float64) % 600
    old = ages >= 590  # The oldest 1.7%.
    test = age_permutation_test(ages, old, draws=2000, generator=torch.Generator().manual_seed(0))
    assert test["p_value"] == pytest.approx(1 / 2001) and test["ratio"] > 1.9
    few = torch.zeros(10_000, dtype=torch.bool)
    few[:6] = True
    many = torch.zeros(10_000, dtype=torch.bool)
    many[:600] = True
    q95 = [age_permutation_test(ages, mask, draws=2000, generator=torch.Generator().manual_seed(0))["null_q95"]
           for mask in (few, many)]
    assert q95[0] > q95[1] > 1.0
    with pytest.raises(ValueError):
        age_permutation_test(ages, torch.zeros(10_000, dtype=torch.bool))


@pytest.mark.parametrize("p, old, expected", [
    (None, None, "none"), (0.5, 500, "noise"), (1e-4, 500, "blocking"), (1e-4, 5, "slow"), (1e-4, None, "slow"),
])
def test_trapping_needs_significance_before_counting_old_configurations(p, old, expected):
    pairs = [{"immobile_age_p": p, "immobile_old": old}, {"immobile_age_p": 0.9}]
    classify_trapping(pairs, alpha=0.01, tolerance=0.01, n_chains=10_000)
    assert pairs[0]["trapping"] == expected
    assert pairs[0]["trapping_level"] == pytest.approx(0.01 / (2 if p is not None else 1))


def test_failed_renewal_reports_old_immobile_configurations_and_red_title(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from adabmDCA.plot import plot_ptt_ladder_health

    data = tmp_path / "reference.fasta"
    data.write_text(">a\nAA\n>b\nBB\n")
    backend = PTTSampler(profile(), tokens="AB", n_chains=64)
    backend.models[-1] = coupled()
    backend.model_version = 1
    result = sample_sequences(model=backend, reference_fasta=data, n_sequences=64, ptt_local_sweeps=1,
                              ptt_max_rounds=1, ptt_renewal_tolerance=0.01, collect_diagnostics=True)
    mixing = result.ptt_diagnostics["mixing"]
    health = result.ptt_diagnostics["ladder_health"]
    assert mixing["status"] == "budget_exceeded" and mixing["rounds"] == 1 and mixing["n_chains"] == 64
    assert health["old_age"] is not None and health["tolerance"] == 0.01
    assert all(pair["trapping"] in ("none", "noise", "slow", "blocking") for pair in health["pairs"])
    figure = plt.figure()
    plot_ptt_ladder_health(figure, health, mixing=mixing)
    assert "renewal NOT reached within 1 rounds" in figure._suptitle.get_text()
    plt.close(figure)
