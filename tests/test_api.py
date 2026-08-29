import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
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

        data = torch.nn.functional.one_hot(torch.tensor([[0, 1]]), num_classes=3).float()
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
        self.assertEqual(result.gradient_steps, 1)
        self.assertEqual(result.structure_steps, 0)
        self.assertEqual(result.sweeps, 1)
        self.assertIn(result.stop_reason, {"target_pearson", "max_gradient_steps"})
        self.assertEqual(result.final_metrics["Epochs"], 1)

    def test_edge_dca_training_respects_max_epochs(self):
        import torch

        from adabmDCA.training import train_edgeDCA

        fi = torch.full((2, 2), 0.5)
        fij = torch.full((2, 2, 2, 2), 0.25)
        params = {
            "bias": torch.zeros(2, 2),
            "coupling_matrix": torch.zeros(2, 2, 2, 2),
        }
        mask = torch.zeros_like(params["coupling_matrix"], dtype=torch.bool)
        chains = torch.nn.functional.one_hot(torch.zeros((4, 2), dtype=torch.int64), num_classes=2).float()
        sampler_calls = []

        class RecordingCheckpoint:
            def __init__(self):
                self.logged_epochs = []

            def log(self, record):
                self.logged_epochs.append(record["Epochs"])

            def check(self, updates):
                return updates == self.max_epochs

            def save(self, **kwargs):
                pass

        checkpoint = RecordingCheckpoint()

        def sampler(*, chains, params, nsweeps):
            sampler_calls.append(nsweeps)
            return chains

        with (
            patch("adabmDCA.training.get_freq_single_point", return_value=fi),
            patch("adabmDCA.training.get_freq_two_points", return_value=fij),
            patch("adabmDCA.training.get_correlation_two_points", return_value=(0.0, 0.0)),
            patch("adabmDCA.training.compute_log_likelihood", return_value=0.0),
            patch("adabmDCA.training.compute_entropy", return_value=torch.tensor(0.0)),
            patch("adabmDCA.training.compute_density", return_value=0.0),
            patch(
                "adabmDCA.training.update_params_edge_activation",
                side_effect=lambda **kwargs: ((0, 1), kwargs["mask"], kwargs["params"]),
            ),
            patch("adabmDCA.training._update_logZ_edge_activation", return_value=0.0),
        ):
            _, _, _, history = train_edgeDCA(
                sampler=sampler,
                fi_target=fi,
                fij_target=fij,
                fi_pseudocounted=fi,
                fij_pseudocounted=fij,
                params=params,
                mask=mask,
                chains=chains,
                target_pearson=0.95,
                nsweeps=1,
                max_epochs=2,
                pseudo_count=0.1,
                checkpoint=checkpoint,
                progress_bar=False,
            )

        self.assertEqual(history["Epochs"], [1, 2])
        self.assertEqual(sampler_calls, [1, 1])
        self.assertEqual(checkpoint.logged_epochs, [1, 2])

    def test_training_accepts_an_authoritative_config_object(self):
        import torch

        from adabmDCA import TrainingConfig, load_model, train_model

        with TemporaryDirectory() as directory:
            fasta = Path(directory) / "tiny.fasta"
            fasta.write_text(
                ">s1\nAA\n>s2\nAB\n>s3\nBA\n>s4\nBB\n",
                encoding="utf-8",
            )
            config = TrainingConfig(
                alphabet="AB-",
                n_chains=8,
                n_sweeps=1,
                max_epochs=1,
                target_pearson=0.1,
                device="cpu",
                seed=3,
                checkpoint_interval=1,
            )

            result = train_model(
                fasta,
                config=config,
                output_dir=Path(directory) / "model",
            )
            log_text = result.artifacts["log"].read_text(encoding="utf-8")
            params_text = result.artifacts["params"].read_text(encoding="utf-8")
            reloaded = load_model(result.artifacts["params"], alphabet="AB-", device="cpu")

            records = [line.split() for line in params_text.splitlines()]
            self.assertTrue(records)
            self.assertTrue(all(parts[0] in {"J", "h"} for parts in records))
            self.assertTrue(all(int(parts[1]) < int(parts[2]) for parts in records if parts[0] == "J"))
            torch.testing.assert_close(reloaded.params["bias"], result.model.params["bias"])
            torch.testing.assert_close(
                reloaded.params["coupling_matrix"],
                result.model.params["coupling_matrix"],
            )

        self.assertEqual(result.model.tokens, "AB-")
        self.assertEqual(result.gradient_steps, 1)
        self.assertIs(result.config, config)
        self.assertIn("checkpoint interval:", log_text)

    def test_training_accepts_an_in_memory_alignment_and_weights(self):
        from adabmDCA import Alignment, TrainingConfig, train_model

        alignment = Alignment(
            ("s1", "bad", "s2", "duplicate"),
            ("AA", "AX", "BB", "AA"),
        )
        result = train_model(
            alignment,
            weights_path=[1.0, 2.0, 3.0, 4.0],
            config=TrainingConfig(
                alphabet="AB-",
                n_chains=8,
                n_sweeps=1,
                max_epochs=1,
                target_pearson=0.1,
                device="cpu",
            ),
        )

        self.assertEqual(result.num_sequences, 2)
        self.assertEqual(result.effective_sequences, 4.0)
        self.assertEqual(result.input_report["dropped_indices"], [1])
        self.assertEqual(result.input_report["duplicate_indices"], [3])

    def test_scoring_accepts_an_in_memory_alignment(self):
        from adabmDCA import Alignment, score_sequences

        result = score_sequences(
            model=self.model,
            fasta_path=Alignment(("first", "second"), ("AB", "A-")),
        )

        self.assertEqual(result.names, ("first", "second"))
        self.assertEqual(result.sequences, ("AB", "A-"))

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
