import unittest
from importlib.util import find_spec
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch


HAS_TORCH = find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for high-level API tests")
class HighLevelApiTests(unittest.TestCase):
    def setUp(self):
        import torch

        from adabmDCA import DCAModel

        self.params = {
            "bias": torch.zeros(2, 3),
            "coupling_matrix": torch.zeros(2, 3, 2, 3),
        }
        self.model = DCAModel(self.params, alphabet="AB-")

    def test_top_level_convenience_api(self):
        import numpy as np

        from adabmDCA import compute_energies

        energies = compute_energies(["AB", "A-"], model=self.model)

        self.assertIsInstance(energies, np.ndarray)
        np.testing.assert_allclose(energies, [0.0, 0.0])

    def test_model_object_reuses_configuration(self):
        result = self.model.score_sequences("AB")

        self.assertEqual(result.sequences, ("AB",))
        self.assertEqual(result.model.tokens, "AB-")
        self.assertEqual(result.model.length, 2)
        self.assertEqual(result.model.schema_version, "1.0")

    def test_saved_model_can_be_loaded_through_public_api(self):
        from adabmDCA import load_model
        from adabmDCA.io import save_params

        with TemporaryDirectory() as directory:
            path = Path(directory) / "params.dat"
            save_params(str(path), self.params, tokens="AB-")
            loaded = load_model(path, alphabet="AB-", device="cpu")

        self.assertEqual(loaded.metadata.length, 2)
        self.assertEqual(loaded.metadata.num_states, 3)
        self.assertEqual(loaded.compute_energies("AB").shape, (1,))

    def test_length_mismatch_raises_structured_error(self):
        from adabmDCA import ModelCompatibilityError

        with self.assertRaises(ModelCompatibilityError) as context:
            self.model.compute_energies("A")

        self.assertEqual(context.exception.code, "model_compatibility_error")
        self.assertEqual(context.exception.details["model_length"], 2)

    def test_mutation_scan_exposes_both_position_conventions(self):
        result = self.model.scan_mutations("AB")

        self.assertEqual(len(result.mutations), 4)
        self.assertEqual(result.mutations[0].position, 0)
        self.assertEqual(result.mutations[0].position_1based, 1)
        self.assertEqual(result.mutations[0].label, "A0B")

    def test_low_level_energy_function_remains_available(self):
        import torch

        from adabmDCA import compute_energy

        data = torch.nn.functional.one_hot(
            torch.tensor([[0, 1]]), num_classes=3
        ).float()
        energies = compute_energy(data, self.params)
        torch.testing.assert_close(energies, torch.zeros(1))

    def test_sampling_is_reproducible(self):
        first = self.model.sample(4, n_sweeps=2, seed=7)
        second = self.model.sample(4, n_sweeps=2, seed=7)

        self.assertEqual(first, second)

    def test_sampling_supports_cancellation(self):
        from adabmDCA import OperationCancelledError, sample_sequences

        with self.assertRaises(OperationCancelledError):
            sample_sequences(
                model=self.model,
                n_sequences=2,
                n_sweeps=2,
                is_cancelled=lambda: True,
            )

    def test_training_returns_reusable_in_memory_model(self):
        from adabmDCA import train_model

        progress_events = []
        with TemporaryDirectory() as directory:
            fasta = Path(directory) / "tiny.fasta"
            fasta.write_text(
                ">s1\nAA\n>s2\nAB\n>s3\nBA\n>s4\nBB\n",
                encoding="utf-8",
            )
            result = train_model(
                fasta,
                model_type="bmDCA",
                alphabet="AB-",
                n_chains=8,
                n_sweeps=1,
                max_epochs=1,
                target_pearson=0.1,
                device="cpu",
                seed=3,
                progress=progress_events.append,
            )

        self.assertEqual(result.model.metadata.length, 2)
        self.assertEqual(result.num_sequences, 4)
        self.assertEqual(len(result.history["Epochs"]), 1)
        self.assertEqual(len(progress_events), 1)
        self.assertEqual(progress_events[0].epoch, 1)
        self.assertIn("Pearson", progress_events[0].metrics)

    def test_auto_device_prefers_cuda_then_mps_then_cpu(self):
        from adabmDCA.utils import get_device

        with (
            patch("adabmDCA.utils.torch.cuda.is_available", return_value=True),
            patch("adabmDCA.utils.torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(get_device("auto", message=False).type, "cuda")

        with (
            patch("adabmDCA.utils.torch.cuda.is_available", return_value=False),
            patch("adabmDCA.utils.torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(get_device("auto", message=False).type, "mps")

        with (
            patch("adabmDCA.utils.torch.cuda.is_available", return_value=False),
            patch("adabmDCA.utils.torch.backends.mps.is_available", return_value=False),
        ):
            self.assertEqual(get_device("auto", message=False).type, "cpu")


if __name__ == "__main__":
    unittest.main()
