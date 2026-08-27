import argparse
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "adabmDCA.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class CliTests(unittest.TestCase):
    def test_global_help_lists_commands(self):
        for flag in ("-h", "--help", "help"):
            with self.subTest(flag=flag):
                result = run_cli(flag)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage: adabmDCA <command> [options]", result.stdout)
                self.assertIn("train", result.stdout)
                self.assertIn("preprocess", result.stdout)

    def test_global_version_flag_succeeds(self):
        result = run_cli("--version")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout.strip(), r"adabmDCA version: \d+\.\d+\.\d+$")

    def test_package_import_is_lightweight(self):
        result = subprocess.run(
            [sys.executable, "-c", "import adabmDCA; print(adabmDCA.__version__)"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout.strip(), r"^\d+\.\d+\.\d+$")

    def test_unknown_command_reports_available_commands(self):
        result = run_cli("unknown")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid command 'unknown'", result.stdout)
        self.assertIn("train", result.stdout)
        self.assertIn("plot-training-log", result.stdout)

    def test_command_help_does_not_require_runtime_dependencies(self):
        commands = [
            "train",
            "sample",
            "contacts",
            "energies",
            "dms",
            "DMS",
            "entropy",
            "reintegrate",
            "profmark",
            "plot-training-log",
            "plot_training_log",
            "preprocess",
        ]

        for command in commands:
            with self.subTest(command=command):
                result = run_cli(command, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_all_compute_commands_default_to_auto_device(self):
        from adabmDCA.parser import (
            add_args_contacts,
            add_args_dms,
            add_args_energies,
            add_args_profmark,
            add_args_sample,
            add_args_tdint,
            add_args_train,
        )

        builders = (
            add_args_train,
            add_args_sample,
            add_args_contacts,
            add_args_energies,
            add_args_dms,
            add_args_tdint,
            add_args_profmark,
        )
        for builder in builders:
            with self.subTest(builder=builder.__name__):
                parser = builder(argparse.ArgumentParser())
                self.assertEqual(parser.get_default("device"), "auto")

    def test_invalid_sampler_is_rejected_by_parser(self):
        result = run_cli(
            "sample",
            "--path_params",
            "params.dat",
            "--output",
            "out",
            "--ngen",
            "10",
            "--sampler",
            "not-a-sampler",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_train_help_documents_edge_dca_pseudocount_default(self):
        result = run_cli("train", "--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0.1 for edgeDCA", result.stdout)

    def test_train_help_uses_validation_flag(self):
        result = run_cli("train", "--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-v", result.stdout)
        self.assertIn("--val VAL", result.stdout)
        self.assertIn("validation", result.stdout)
        self.assertIn("metrics are computed", result.stdout)
        self.assertNotIn("--test", result.stdout)

    def test_train_help_documents_progress_opt_out(self):
        result = run_cli("train", "--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--no-progress", result.stdout)

    def test_train_help_exposes_unambiguous_step_limits(self):
        result = run_cli("train", "--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-gradient-steps", result.stdout)
        self.assertIn("--max-structure-steps", result.stdout)

    def test_train_renders_structured_progress_by_default(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            fasta = workspace / "tiny.fasta"
            fasta.write_text(
                ">s1\nAA\n>s2\nAB\n>s3\nBA\n>s4\nBB\n",
                encoding="utf-8",
            )
            result = run_cli(
                "train",
                "--data",
                str(fasta),
                "--output",
                str(workspace / "model"),
                "--alphabet",
                "AB-",
                "--nchains",
                "8",
                "--nsweeps",
                "1",
                "--nepochs",
                "1",
                "--target",
                "0.99",
                "--device",
                "cpu",
                "--seed",
                "3",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Pearson", result.stderr)
        self.assertIn("Step 1/1", result.stderr)

    def test_plot_training_log_missing_file_reports_parser_error(self):
        result = run_cli("plot-training-log", "missing.log")

        self.assertEqual(result.returncode, 2)
        self.assertIn("Log file 'missing.log' not found", result.stderr)

    def test_structured_application_errors_are_rendered_without_traceback(self):
        result = run_cli(
            "energies",
            "--data",
            "missing.fasta",
            "--path_params",
            "missing.dat",
            "--output",
            "unused-output",
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Error [model_load_error]", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
