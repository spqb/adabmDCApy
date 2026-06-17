import unittest
from importlib.util import find_spec

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
