import unittest
from importlib.util import find_spec

import numpy as np
import pytest

HAS_TORCH = find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for DCA contact-map tests")
class DcaContactTests(unittest.TestCase):
    def test_mf_contact_map_is_finite_symmetric_and_zero_diagonal(self):
        import torch
        from torch.nn.functional import one_hot

        from adabmDCA.dca import get_mf_contact_map

        tokens = "AB-"
        categorical = torch.tensor(
            [
                [0, 0, 1, 2],
                [0, 1, 1, 2],
                [1, 0, 0, 2],
                [1, 1, 0, 2],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
            ],
            dtype=torch.int64,
        )
        data = one_hot(categorical, num_classes=len(tokens)).to(torch.float64)

        contact_map = get_mf_contact_map(data=data, tokens=tokens)

        self.assertEqual(contact_map.shape, (categorical.shape[1], categorical.shape[1]))
        self.assertTrue(np.isfinite(contact_map).all())
        np.testing.assert_allclose(contact_map, contact_map.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(contact_map), 0.0, atol=1e-10)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_sequence_identity_accepts_strided_integer_one_hot_and_checks_shape():
    import torch
    from torch.nn.functional import one_hot

    from adabmDCA.dca import get_seqid, get_seqid_stats

    data = one_hot(torch.tensor([[0, 2], [1, 2]]), 3)
    strided = data.transpose(1, 2).contiguous().transpose(1, 2)
    assert get_seqid(strided, data[0]).tolist() == [2, 1]
    mean, std = get_seqid_stats(strided, data[0])
    assert mean.item() == pytest.approx(1.5)
    assert std.item() == pytest.approx(2**-0.5)
    with pytest.raises(ValueError, match="same length and state count"):
        get_seqid(data, data.transpose(1, 2))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_contact_scores_ignore_antisymmetric_interactions():
    import torch

    from adabmDCA.dca import get_contact_map

    g = torch.Generator().manual_seed(42)
    bias = torch.zeros((4, 3), dtype=torch.float64)
    raw = torch.randn((4, 3, 4, 3), dtype=bias.dtype, generator=g)
    symmetric = raw + raw.permute(2, 3, 0, 1)
    antisymmetric = raw - raw.permute(2, 3, 0, 1)
    actual = get_contact_map({"bias": bias, "coupling_matrix": symmetric + antisymmetric}, "A-B")
    expected = get_contact_map({"bias": bias, "coupling_matrix": symmetric}, "A-B")
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_array_equal(get_contact_map({"bias": bias, "coupling_matrix": antisymmetric}, "A-B"), 0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_mf_map_matches_inverse_of_covariance_with_gap_in_middle():
    import torch
    from torch.nn.functional import one_hot

    from adabmDCA.dca import get_contact_map, get_mf_contact_map

    q, length = 3, 3
    data = one_hot(torch.tensor([[0, 0, 2], [1, 1, 2], [2, 2, 0], [0, 2, 0]]), q).double()
    weights = torch.tensor([0.1, 0.3, 0.4, 0.2], dtype=data.dtype)
    uniform = one_hot(torch.cartesian_prod(*[torch.arange(q)] * length), q).double()
    mix = torch.cat([weights * 0.5, weights.new_full((len(uniform),), 0.5 / len(uniform))])
    states = [0, 2]
    reduced = torch.cat([data, uniform])[:, :, states].reshape(len(mix), -1)
    centered = reduced - mix @ reduced
    inverse = -torch.linalg.inv((centered.T * mix) @ centered).reshape(length, 2, length, 2)
    couplings = data.new_zeros((length, q, length, q))
    for a, token_a in enumerate(states):
        for b, token_b in enumerate(states):
            couplings[:, token_a, :, token_b] = inverse[:, a, :, b]
    expected = get_contact_map({"bias": data.new_zeros((length, q)), "coupling_matrix": couplings}, "A-B")
    np.testing.assert_allclose(get_mf_contact_map(data, "A-B", weights=weights), expected, atol=1e-12)
    np.testing.assert_allclose(get_mf_contact_map(data.bfloat16(), "A-B", weights=weights), expected, atol=1e-12)
    with pytest.raises(ValueError, match="singular.*pseudocount"):
        get_mf_contact_map(data[:1], "A-B", pseudo_count=0)


if __name__ == "__main__":
    unittest.main()
