"""Regression coverage for sequence encodings and sequence weights."""

import numpy as np
import pytest
import torch

from adabmDCA import one_hot
from adabmDCA import fasta


def test_public_one_hot_checks_index_range_and_preserves_dtype():
    values = torch.tensor([[1, 2], [0, 1]], dtype=torch.int32)
    expected = torch.nn.functional.one_hot(values.long(), 3).double()
    torch.testing.assert_close(one_hot(values, dtype=torch.float64), expected)
    with pytest.raises(ValueError, match="Class indices"):
        one_hot(torch.tensor([-1, 1]), 3)
    with pytest.raises(ValueError, match="Class indices"):
        one_hot(torch.tensor([3]), 3)
    assert one_hot(values[:0], 3).shape == (0, 2, 3)


def test_sequence_encoding_supports_iterables_and_empty_integer_output():
    np.testing.assert_array_equal(fasta.encode_sequence(iter(["AB", "BA"]), "AB"), [[0, 1], [1, 0]])
    assert fasta.encode_sequence("", "AB").dtype == np.int64


def test_decode_validates_numeric_indices_and_one_hot_rows():
    valid = torch.tensor([[[1., 0.], [0., 1.]]], dtype=torch.bfloat16)
    np.testing.assert_array_equal(fasta.decode_sequence(valid, "AB"), ["AB"])
    assert fasta.decode_sequence([0., 1.], "AB") == "AB"
    for invalid in ([-1], [2], [0.75], [np.nan], [[[0, 0], [1, 0]]], [[[1, 1]]]):
        with pytest.raises(ValueError):
            fasta.decode_sequence(invalid, "AB")


def test_weights_preserve_strict_threshold_and_support_one_hot():
    data = torch.tensor([[0, 1], [0, 2], [0, 1], [1, 2]])
    expected = torch.tensor([0.5, 1., 0.5, 1.], dtype=torch.float64)
    torch.testing.assert_close(fasta.compute_weights(data.numpy(), th=0.5, dtype=torch.float64), expected)
    torch.testing.assert_close(fasta.compute_weights(one_hot(data, 3, dtype=torch.bool), th=0.5), expected.float())
    torch.testing.assert_close(fasta.compute_weights(data, th=1 - 1e-10), expected.float())
    torch.testing.assert_close(fasta.compute_weights(data, th=1), torch.ones(len(data)))
    assert fasta.compute_weights(data[:0]).shape == (0,)
    with pytest.raises(ValueError, match="at least one site"):
        fasta.compute_weights(data[:, :0])


def test_tiled_weight_paths_agree_with_exact_match_counts(monkeypatch):
    generator = torch.Generator().manual_seed(7)
    data = torch.randint(0, 3, (21, 8), generator=generator)
    data[1::4] = data[0:-1:4]
    threshold = 0.5
    identity = (data[:, None] == data[None]).sum(-1) / data.shape[1]
    expected = (identity > threshold).sum(1).double().reciprocal()
    monkeypatch.setattr(fasta, "_WEIGHT_TILE_ELEMENTS", 32)
    torch.testing.assert_close(fasta.compute_weights(data, th=threshold, dtype=torch.float64), expected)
    monkeypatch.setattr(fasta, "_WEIGHT_ENCODING_ELEMENTS", 4)
    sparse_labels = data * 1_000_000 - 2_000_000
    torch.testing.assert_close(fasta.compute_weights(sparse_labels, th=threshold, dtype=torch.float64), expected)
