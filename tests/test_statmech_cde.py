"""Exact checks for per-site context-dependent entropy."""

import pytest
import torch
from torch.nn.functional import one_hot

from adabmDCA import get_cde
from adabmDCA.statmech import compute_energy


def _reference_cde(sequence: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Enumerate every candidate state and calculate its conditional entropy."""
    length, states = sequence.shape
    values = []
    for site in range(length):
        candidates = sequence.repeat(states, 1, 1)
        candidates[:, site] = torch.eye(states, dtype=sequence.dtype)
        log_probabilities = torch.log_softmax(-compute_energy(candidates, params), dim=0)
        values.append(-(log_probabilities.exp() * log_probabilities).sum())
    return torch.stack(values)


def test_get_cde_is_log_q_for_uniform_conditionals():
    params = {
        "bias": torch.zeros(4, 3, dtype=torch.float64),
        "coupling_matrix": torch.zeros(4, 3, 4, 3, dtype=torch.float64),
    }
    sequence = one_hot(torch.tensor([0, 1, 2, 0]), num_classes=3)

    actual = get_cde(sequence, params)

    assert actual.shape == (4,)
    torch.testing.assert_close(actual, torch.full((4,), torch.log(torch.tensor(3.0)).item(), dtype=torch.float64))


def test_get_cde_matches_exact_conditionals_for_one_sequence_and_batch():
    params = {
        "bias": torch.tensor([[0.2, -0.4], [0.3, 0.1], [-0.1, 0.5]], dtype=torch.float64),
        "coupling_matrix": torch.arange(36, dtype=torch.float64).reshape(3, 2, 3, 2) / 17,
    }
    first = one_hot(torch.tensor([0, 1, 0]), num_classes=2).to(torch.float64)
    second = one_hot(torch.tensor([1, 0, 1]), num_classes=2).to(torch.float64)

    expected = torch.stack([_reference_cde(first, params), _reference_cde(second, params)])
    actual = get_cde(torch.stack([first, second]), params)

    assert actual.shape == (2, 3)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(get_cde(first, params), expected[0])


def test_get_cde_rejects_non_one_hot_sequences():
    params = {
        "bias": torch.zeros(2, 2),
        "coupling_matrix": torch.zeros(2, 2, 2, 2),
    }
    with pytest.raises(ValueError, match="one-hot"):
        get_cde(torch.tensor([[1.0, 1.0], [0.0, 1.0]]), params)
