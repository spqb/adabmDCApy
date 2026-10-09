"""CDE-based local scoring and equilibrium-sample slope fitting."""

import numpy as np
import pytest
import torch

from adabmDCA import DCAModel, PTTSampler, score_sequences
from adabmDCA.api.sampling import sample_sequences
from adabmDCA.api.scoring import fit_local_lambda


def test_local_score_uses_summed_cde_and_configured_lambda(tmp_path):
    params = {
        "bias": torch.zeros(3, 2, dtype=torch.float64),
        "coupling_matrix": torch.zeros(3, 2, 3, 2, dtype=torch.float64),
    }
    model = DCAModel(params, alphabet="AB")
    result = score_sequences(["AAA", "ABB"], model=model, local_lambda=2.0)

    expected_entropy = 3 * np.log(2)
    np.testing.assert_allclose(result.cde_sum, expected_entropy)
    np.testing.assert_allclose(result.local_free_energies, result.energies - 2 * expected_entropy)
    assert result.local_lambda == 2.0
    artifacts = result.save_bundle(tmp_path)
    assert "cde_sum,local_free_energy" in artifacts["csv"].read_text()
    assert "local_free_energy:" in artifacts["fasta"].read_text()


def test_score_without_lambda_omits_local_free_energy(tmp_path):
    params = {
        "bias": torch.zeros(2, 2),
        "coupling_matrix": torch.zeros(2, 2, 2, 2),
    }
    model = DCAModel(params, alphabet="AB")
    result = score_sequences(["AA"], model=model, local_lambda=None)

    assert result.cde_sum is None
    assert result.local_free_energies is None
    artifacts = result.save_bundle(tmp_path)
    assert "local_free_energy" not in artifacts["csv"].read_text()
    assert "local_free_energy" not in artifacts["fasta"].read_text()


def test_lambda_regression_recovers_slope_with_intercept():
    entropy = np.array([0.2, 0.7, 1.1, 1.8])
    energy = -4.0 + 1.7 * entropy
    fit = fit_local_lambda(energy, entropy)

    assert fit["lambda"] == pytest.approx(1.7)
    assert fit["intercept"] == pytest.approx(-4.0)
    assert fit["r_squared"] == pytest.approx(1.0)
    assert fit["n_samples"] == 4


def test_sampling_fit_records_cde_and_regression():
    bias = torch.tensor([[0.8, -0.2], [0.1, -0.5], [0.4, -0.3]])
    couplings = torch.zeros(3, 2, 3, 2)
    couplings[0, 0, 1, 0] = couplings[1, 0, 0, 0] = 0.7
    model = DCAModel({"bias": bias, "coupling_matrix": couplings}, alphabet="AB")
    result = sample_sequences(
        model=model, n_sequences=32, n_sweeps=8, sampler="gibbs", seed=5,
    )

    assert result.cde_sum.shape == (32,)
    assert result.local_lambda_fit is not None
    expected = fit_local_lambda(result.energies, result.cde_sum)
    assert result.local_lambda_fit["lambda"] == pytest.approx(expected["lambda"])
    assert "cde_sum" in result.to_dataframe()


def test_sampling_always_fits_at_nonunit_beta():
    bias = torch.tensor([[0.8, -0.2], [0.1, -0.5], [0.4, -0.3]])
    couplings = torch.zeros(3, 2, 3, 2)
    couplings[0, 0, 1, 0] = couplings[1, 0, 0, 0] = 0.7
    model = DCAModel({"bias": bias, "coupling_matrix": couplings}, alphabet="AB")
    result = sample_sequences(model=model, n_sequences=32, n_sweeps=4, beta=0.5, seed=9)
    assert result.local_lambda_fit is not None
    assert result.beta == 0.5


def test_ptt_sampling_fit_uses_converged_endpoint_samples(tmp_path):
    bias = torch.tensor([[0.4, -0.2], [-0.1, 0.5]], dtype=torch.float64)
    profile = {"bias": bias, "coupling_matrix": torch.zeros(2, 2, 2, 2, dtype=torch.float64)}
    endpoint = {key: value.clone() for key, value in profile.items()}
    endpoint["coupling_matrix"][0, 0, 1, 0] = 0.5
    endpoint["coupling_matrix"][1, 0, 0, 0] = 0.5
    backend = PTTSampler(profile, tokens="AB", n_chains=64, seed=7)
    backend.models[-1] = endpoint
    backend.model_version = 1
    backend.prepare_sampling_ladder(64)
    data = tmp_path / "reference.fasta"
    data.write_text(">a\nAA\n>b\nAB\n>c\nBA\n>d\nBB\n")

    result = sample_sequences(
        model=backend, reference_fasta=data, n_sequences=64,
        ptt_local_sweeps=1, ptt_max_rounds=500, collect_diagnostics=True,
    )

    assert result.ptt_diagnostics["converged"]
    assert result.local_lambda_fit is not None
    assert result.local_lambda_fit["lambda"] == pytest.approx(
        fit_local_lambda(result.energies, result.cde_sum)["lambda"]
    )
    artifacts = result.save_diagnostic_plots(tmp_path / "plots")
    assert artifacts["energy_cde_plot"].read_bytes().startswith(b"\x89PNG")


def test_uniform_samples_keep_output_when_lambda_is_unidentifiable():
    params = {"bias": torch.zeros(2, 2), "coupling_matrix": torch.zeros(2, 2, 2, 2)}
    model = DCAModel(params, alphabet="AB")
    result = sample_sequences(model=model, n_sequences=8, n_sweeps=1)

    assert len(result.sequences) == 8
    assert result.local_lambda_fit is None
    assert np.allclose(result.cde_sum, 2 * np.log(2))
    assert "all sampled CDE sums are equal" in result.warnings[0]


def test_energy_cde_plot_displays_slope_and_r_squared():
    import matplotlib.pyplot as plt

    from adabmDCA.plot import plot_energy_cde_scatter

    entropy = np.array([0.2, 0.7, 1.1, 1.8])
    energy = -4.0 + 1.7 * entropy
    figure, axis = plt.subplots()
    try:
        plot_energy_cde_scatter(axis, entropy, energy, fit_local_lambda(energy, entropy))
        assert len(axis.lines) == 1
        assert "\\lambda" in axis.texts[0].get_text()
        assert "R^2" in axis.texts[0].get_text()
    finally:
        plt.close(figure)
