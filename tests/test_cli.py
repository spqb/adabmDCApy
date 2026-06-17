import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "adabmDCA.cli", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class CliTests(unittest.TestCase):
    def test_package_import_is_lightweight(self):
        result = subprocess.run(
            [sys.executable, "-c", "import adabmDCA; print(adabmDCA.__version__)"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
        ]

        for command in commands:
            with self.subTest(command=command):
                result = run_cli(command, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

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
        self.assertIn("-v VAL, --val VAL", result.stdout)
        self.assertIn("validation", result.stdout)
        self.assertIn("metrics are computed", result.stdout)
        self.assertNotIn("--test", result.stdout)

    def test_plot_training_log_missing_file_reports_parser_error(self):
        result = run_cli("plot-training-log", "missing.log")

        self.assertEqual(result.returncode, 2)
        self.assertIn("Log file 'missing.log' not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
