"""Regression coverage for graph selection and KL boundary cases."""

import pytest
import torch

from adabmDCA.graph import (
    compute_density,
    compute_Dkl_decimation,
    compute_Dkl_edge_activation,
    compute_Dkl_element_activation,
    compute_sym_Dkl,
    select_inactive_elements,
    update_mask_decimation,
)


def test_density_counts_only_off_site_entries():
    assert compute_density(torch.ones(1, 2, 1, 2, dtype=torch.bool)) == 0
    assert compute_density(torch.ones(2, 2, 2, 2, dtype=torch.bool)) == 1


def test_kl_activation_keeps_zero_and_infinite_boundary_values():
    target = torch.zeros(2, 2, 2, 2)
    target[0, 0, 1, 0] = target[1, 0, 0, 0] = 1
    assert compute_Dkl_element_activation(target, target)[0, 0, 1, 0].item() == 0
    assert compute_Dkl_edge_activation(target, target)[0, 1].item() == 0
    model = torch.zeros_like(target)
    model[0, 1, 1, 0] = model[1, 0, 0, 1] = 1
    scores = compute_Dkl_element_activation(target, model)
    assert scores[0, 0, 1, 0].isposinf()
    assert compute_Dkl_edge_activation(target, model)[0, 1].isposinf()
    mask, _, count = select_inactive_elements(scores, torch.zeros_like(scores, dtype=torch.bool), 0.01)
    assert count == 1 and mask[0, 0, 1, 0] and mask[1, 0, 0, 0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pruning_kl_matches_reweighted_bernoulli_distribution(dtype):
    p = torch.tensor([0., 1., 0.25, 0.25, 0.37, 0.5], dtype=dtype)
    j = torch.tensor([-100., 100., -100., 100., 2.3, 0.001], dtype=dtype)
    params = {"coupling_matrix": j}
    prior = torch.stack([p, 1 - p], dim=-1).double()
    log_prior = prior.log()
    log_posterior = (log_prior - torch.stack([j, torch.zeros_like(j)], dim=-1)).log_softmax(-1)
    posterior = log_posterior.exp()
    forward = torch.where(prior > 0, prior * (log_prior - log_posterior), 0).sum(-1)
    reverse = torch.where(posterior > 0, posterior * (log_posterior - log_prior), 0).sum(-1)
    torch.testing.assert_close(compute_Dkl_decimation(params, p).double(), forward, atol=1e-9, rtol=2e-6)
    torch.testing.assert_close(compute_sym_Dkl(params, p).double(), forward + reverse, atol=1e-9, rtol=2e-6)


def test_pruning_breaks_ties_by_unique_pair_and_guarantees_progress():
    mask = torch.zeros(2, 2, 2, 2, dtype=torch.bool)
    mask[0, :, 1, :] = mask[1, :, 0, :] = True
    scores = torch.zeros_like(mask, dtype=torch.float32)
    pruned = update_mask_decimation(mask.clone(), scores, 0.01)
    expected = mask.clone()
    expected[0, 0, 1, 0] = expected[1, 0, 0, 0] = False
    assert torch.equal(pruned, expected)
    assert torch.equal(update_mask_decimation(pruned.clone(), scores, 0), pruned)
    assert not update_mask_decimation(pruned, scores, 1).any()
