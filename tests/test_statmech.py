"""Regression checks for thermodynamic helpers and their tensor inputs."""

import math
from itertools import product

import pytest
import torch

from adabmDCA.statmech import (
    compute_energy,
    compute_logZ_exact,
    couplings_are_symmetric,
    enumerate_states,
    exchange_log_acceptance,
    iterate_tap,
)


def test_energy_accepts_empty_and_noncontiguous_batches():
    states = enumerate_states(3, 2).double()
    params = {"bias": torch.arange(6, dtype=states.dtype).reshape(2, 3).T,
              "coupling_matrix": torch.ones((3, 2, 3, 2), dtype=states.dtype)}
    strided_states = states.transpose(1, 2).contiguous().transpose(1, 2)
    assert not strided_states.is_contiguous()
    expected = -(states * params["bias"]).sum((1, 2)) - 4.5
    torch.testing.assert_close(compute_energy(strided_states, params), expected)
    assert compute_energy(states[:0], params).shape == (0,)


def test_exact_partition_with_generated_and_supplied_state_batches():
    generator = torch.Generator().manual_seed(19)
    bias = torch.randn(3, 2, dtype=torch.float64, generator=generator)
    couplings = torch.randn(3, 2, 3, 2, dtype=torch.float64, generator=generator)
    params = {"bias": bias, "coupling_matrix": couplings}
    states = enumerate_states(3, 2)
    labels = list(product(range(2), repeat=3))
    assert states.argmax(-1).tolist() == [list(row) for row in labels]
    score = [sum(float(bias[i, a[i]]) for i in range(3))
             + 0.5 * sum(float(couplings[i, a[i], j, a[j]]) for i in range(3) for j in range(3))
             for a in labels]
    expected = math.log(sum(map(math.exp, score)))
    assert compute_logZ_exact(params=params, batch_size=3) == pytest.approx(expected, abs=1e-13)
    assert compute_logZ_exact(states, params, batch_size=3) == pytest.approx(expected, abs=1e-13)


def test_inference_tensor_symmetry_is_rechecked_after_mutation():
    with torch.inference_mode():
        couplings = torch.zeros(2, 2, 2, 2)
        assert couplings_are_symmetric(couplings)
        couplings[0, 0, 1, 0] = 1
        assert not couplings_are_symmetric(couplings)


def test_exchange_keeps_small_differences_between_large_absolute_energies():
    bias = torch.tensor([[1e20, 1e20], [0., 0.]], dtype=torch.float64)
    lower = {"bias": bias, "coupling_matrix": torch.zeros(2, 2, 2, 2, dtype=bias.dtype)}
    upper = {**lower, "bias": bias.clone()}
    upper["bias"][1, 1] = 1
    states = torch.nn.functional.one_hot(torch.tensor([[0, 0], [0, 1]]), 2).to(bias)
    assert exchange_log_acceptance(lower, upper, states[:1], states[1:]).item() == -1


def test_tap_folds_same_site_diagonal_into_bias_and_honors_sweep_limit():
    mag = torch.tensor([[[0.8, 0.2]]], dtype=torch.float64)
    bias = torch.tensor([[0.5, 0.]], dtype=mag.dtype)
    couplings = torch.tensor([2., 10., 12., 0.], dtype=mag.dtype).reshape(1, 2, 1, 2)
    params = {"bias": bias, "coupling_matrix": couplings}
    expected = (bias + bias.new_tensor([[1., 0.]])).softmax(-1).unsqueeze(0)
    torch.testing.assert_close(iterate_tap(mag, params), expected)
    assert torch.equal(iterate_tap(mag, params, max_iter=0), mag)
    with pytest.warns(RuntimeWarning, match="after 1 sweeps"):
        torch.testing.assert_close(iterate_tap(mag, params, max_iter=1), expected)
