"""Reference checks for weighted Potts statistics."""

from itertools import combinations
from math import isnan

import pytest
import torch
from torch.nn.functional import one_hot

from adabmDCA.stats import (
    extract_Cij_from_freq,
    generate_unique_triplets,
    get_correlation_two_points,
    get_covariance_matrix,
    get_freq_single_point,
    get_freq_three_points,
    get_freq_two_points,
)


def test_pseudocounted_pair_frequencies_are_moments_of_the_same_distribution():
    data = one_hot(torch.tensor([[0, 0], [1, 1]]), 2)
    weights = torch.tensor([9., 1.], dtype=torch.float64)
    pc = 0.4
    fi = get_freq_single_point(data, weights, pc)
    fij = get_freq_two_points(data.float(), weights, pc)
    torch.testing.assert_close(fi, weights.new_tensor([[0.74, 0.26]]).expand(2, 2))
    for i in range(2):
        torch.testing.assert_close(fij[i, :, i, :], fi[i].diag())
    torch.testing.assert_close(fij[1, :, 0, :], weights.new_tensor([[0.64, 0.1], [0.1, 0.16]]))
    covariance = get_covariance_matrix(data, weights, pc).reshape(2, 2, 2, 2)
    torch.testing.assert_close(covariance.sum(-1), weights.new_zeros((2, 2, 2)))
    assert get_freq_two_points(data).shape == (2, 2, 2, 2)


@pytest.mark.parametrize("weights", [[1., -1.], [float("inf"), 1.]])
def test_invalid_weight_total_fails_explicitly(weights):
    data = one_hot(torch.tensor([[0, 1], [1, 0]]), 2)
    with pytest.raises(ValueError, match="finite, nonzero sum"):
        get_freq_single_point(data, torch.tensor(weights))
    with pytest.raises(ValueError, match="finite, nonzero sum"):
        get_freq_two_points(data, torch.tensor(weights))


def test_weighted_third_moment_uses_the_weighted_mean():
    data = one_hot(torch.tensor([[0, 0, 0], [1, 1, 1]]), 2)
    weights = torch.tensor([0.9, 0.1], dtype=torch.float64)
    natural, generated = get_freq_three_points(data, data, weights=weights, triplets=torch.tensor([[0, 1, 2]]))
    sign = weights.new_tensor([-1., 1.])
    expected = 0.072 * torch.einsum("a,b,c->abc", sign, sign, sign).flatten()
    torch.testing.assert_close(natural, expected)
    torch.testing.assert_close(generated, torch.zeros_like(generated))


def test_triplet_sampling_and_empty_selection():
    all_triplets = generate_unique_triplets(7, 100)
    assert set(map(tuple, all_triplets.tolist())) == set(combinations(range(7), 3))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        selected = generate_unique_triplets(2000, 100)
        torch.manual_seed(21)
        assert torch.equal(selected, generate_unique_triplets(2000, 100))
    assert len(torch.unique(selected, dim=0)) == 100
    assert (selected.diff(dim=1) > 0).all()
    assert generate_unique_triplets(2, 10).shape == (0, 3)
    data = one_hot(torch.tensor([[0, 1], [1, 0]]), 2)
    natural, generated = get_freq_three_points(data, data)
    assert natural.shape == generated.shape == (0,)


def test_masked_correlation_uses_only_selected_entries_and_preserves_double_precision():
    fi = torch.full((2, 3), 1/3, dtype=torch.float64)
    fij = torch.full((2, 3, 2, 3), 1/9, dtype=fi.dtype)
    pij = fij.clone()
    mask = torch.zeros_like(fij, dtype=torch.bool)
    mask[1, :, 0, 0] = True
    x = fi.new_tensor([0.01, 0.01 + 1e-10, 0.01 + 2e-10])
    y = -2 * x
    fij[mask] += x
    pij[mask] += y
    selected_x, selected_y = extract_Cij_from_freq(fij, pij, fi, fi, mask)
    torch.testing.assert_close(selected_x, x)
    torch.testing.assert_close(selected_y, y)
    pearson, slope = get_correlation_two_points(fij, pij, fi, fi, mask)
    assert pearson == pytest.approx(-1)
    assert slope == pytest.approx(2)
    assert all(map(isnan, get_correlation_two_points(fij, pij, fi, fi, torch.zeros_like(mask))))
