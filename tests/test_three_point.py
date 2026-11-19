"""Three-point correlations on triplets without a strong pair, with a noise threshold."""

import numpy as np
import pytest
import torch

from adabmDCA.stats import compare_three_point_correlations


def xor_alignment(n, length, seed, xor=True, duplicate=False):
    """Fair binary sites; site 2 is the XOR of sites 0 and 1, and optionally site 3 copies site 4."""
    generator = torch.Generator().manual_seed(seed)
    data = torch.randint(2, (n, length), generator=generator)
    if xor:
        data[:, 2] = data[:, 0] ^ data[:, 1]
    if duplicate:
        data[:, 3] = data[:, 4]
    return data


def selected_sites(result):
    return {tuple(sorted(int(site) for site in triplet[:, 0])) for triplet in result["triplets"]}


def test_xor_triplet_is_found_without_any_pair_correlation():
    data = xor_alignment(4000, 8, seed=0)
    independent = xor_alignment(4000, 8, seed=1, xor=False)
    result = compare_three_point_correlations(data, independent, q=2, gap_index=None, n_seeds=500)
    # Pairwise correlations of an XOR are zero, so only the three-point correlation reveals it.
    assert selected_sites(result) == {(0, 1, 2)}
    assert np.allclose(np.abs(result["data"]), 0.125, atol=0.01)
    assert np.all(np.abs(result["samples"]) < 0.02)
    assert result["n_resolved"] == len(result["data"]) == 8  # Two states per site: 2^3 combinations.
    assert np.all(result["z"] >= 5)


def test_samples_with_the_same_structure_agree_and_noise_ceiling_is_high():
    data = xor_alignment(4000, 8, seed=0)
    same = xor_alignment(4000, 8, seed=2)
    result = compare_three_point_correlations(data, same, q=2, gap_index=None, n_seeds=500)
    assert result["pearson"] > 0.99 and result["noise_ceiling"] > 0.99
    assert result["slope"] == pytest.approx(1.0, abs=0.05)
    assert result["r2"] > 0.98 and result["noise_ceiling_r2"] > 0.98


def test_r2_about_identity_penalizes_scale_unlike_pearson():
    from adabmDCA.stats import r2_about_identity

    reference = torch.linspace(-1, 1, 50)
    assert r2_about_identity(reference, reference) == pytest.approx(1.0)
    # Half the scale: perfectly correlated, but far from the diagonal.
    assert r2_about_identity(reference, 0.5 * reference) == pytest.approx(0.75)
    assert r2_about_identity(reference, torch.zeros(50)) == pytest.approx(0.0)


def test_triplets_with_a_strong_pair_are_excluded():
    data = xor_alignment(4000, 8, seed=0, duplicate=True)
    result = compare_three_point_correlations(data, data.clone(), q=2, gap_index=None, n_seeds=500)
    # Sites 3 and 4 are identical: any triplet holding both is a two-point correlation in disguise.
    assert all(not {3, 4} <= set(sites) for sites in selected_sites(result))


def test_pure_noise_resolves_nothing_and_reports_it():
    data = xor_alignment(2000, 10, seed=3, xor=False)
    samples = xor_alignment(2000, 10, seed=4, xor=False)
    result = compare_three_point_correlations(data, samples, q=2, gap_index=None, n_seeds=500)
    assert result["n_candidates"] > 0 and result["n_resolved"] == 0
    assert result["pearson"] is None and len(result["data"]) == 0


def test_gap_state_and_rare_states_are_excluded_and_one_hot_samples_accepted():
    data = xor_alignment(4000, 8, seed=0)
    samples = torch.nn.functional.one_hot(xor_alignment(4000, 8, seed=2), 2)
    # Treating state 0 as the gap removes every triplet that uses it.
    result = compare_three_point_correlations(data, samples, q=2, gap_index=0, n_seeds=500)
    # Only state 1 at sites 0, 1, 2 remains: one triplet, too few for a Pearson correlation.
    assert result["n_resolved"] == 1 and result["pearson"] is None
    rare = compare_three_point_correlations(data, samples, q=2, gap_index=None, n_seeds=500,
                                            frequency_range=(0.6, 0.95))
    assert rare["n_candidates"] == 0
