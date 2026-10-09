import argparse
import io
import subprocess
import sys
import time
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
    def test_strategy_defaults_and_choices_across_scripts(self):
        from adabmDCA.scripts.sample import create_parser as sample_parser
        from adabmDCA.scripts.td_integration import create_parser as entropy_parser
        from adabmDCA.scripts.train import create_parser as train_parser

        cases = (
            (train_parser(), ["-d", "alignment.fasta"], []),
            (sample_parser(), ["-p", "model.h5", "-o", "output", "--ngen", "10"], []),
            (entropy_parser(), ["-p", "model.h5"], ["-d", "alignment.fasta", "-t", "target.fasta"]),
        )
        for parser, required, pcd_required in cases:
            with self.subTest(parser=parser.prog):
                self.assertEqual(parser.parse_args(required).strategy, "ptt")
                self.assertEqual(
                    parser.parse_args([*required, "--strategy", "pcd", *pcd_required]).strategy, "pcd"
                )
                with self.assertRaises(SystemExit):
                    parser.parse_args([*required, "--strategy", "invalid"])
                with self.assertRaises(SystemExit):
                    parser.parse_args([*required, "--ptt"])

    def test_ptt_entropy_accepts_explicit_nchains_without_overriding_archive(self):
        from unittest.mock import patch

        from adabmDCA.exceptions import InputValidationError
        from adabmDCA.scripts.td_integration import create_parser, run

        parser = create_parser()
        self.assertIn("ignored with --strategy ptt", " ".join(parser.format_help().split()))
        args = parser.parse_args([
            "-p", "ptt.h5", "-o", "entropy_estimation", "--nchains", "2000", "-l", "ptt", "--strategy", "ptt"
        ])
        with patch("adabmDCA.scripts._frontend.resolve_alphabet"), \
                patch("adabmDCA.api.ptt.estimate_ptt_entropy") as estimate:
            run(args)
        self.assertEqual(estimate.call_args.kwargs["model"], "ptt.h5")
        self.assertEqual(estimate.call_args.kwargs["label"], "ptt")
        self.assertNotIn("n_chains", estimate.call_args.kwargs)

        incompatible = parser.parse_args(["-p", "ptt.h5", "--strategy", "ptt", "--sampler", "gibbs"])
        with patch("adabmDCA.scripts._frontend.resolve_alphabet"), self.assertRaises(InputValidationError):
            run(incompatible)

    def test_ptt_sampling_defaults_to_ten_local_sweeps(self):
        from adabmDCA.scripts.sample import create_parser

        args = create_parser().parse_args(["--strategy", "ptt", "-p", "model.h5", "-o", "out", "--ngen", "20"])
        self.assertEqual(args.ptt_local_sweeps, 10)
        self.assertEqual(args.ptt_max_rounds, 20_000)

    def test_ptt_sampling_diagnostic_prints_taus_and_all_acceptance_rates(self):
        from adabmDCA.api.results import SamplingProgress
        from adabmDCA.scripts.sample import _SamplingProgressRenderer

        stream = io.StringIO()
        renderer = _SamplingProgressRenderer(diagnostic=True, stream=stream)
        renderer(SamplingProgress("ptt_mixing_measure", 0, 100))
        renderer(SamplingProgress(
            "ptt_mixing_measure", 100, 487,
            details={
                "tau_int": 12.25,
                "tau_exp": 24.31,
                "required_rounds": 487,
                "acceptance": [0.25123, 0.31789, 0.44211],
            },
        ))
        renderer.close()

        output = stream.getvalue()
        self.assertIn("PTT diagnostic | rounds 100 -> 487", output)
        self.assertIn("tau_int 12.250 | tau_exp 24.310", output)
        self.assertIn("acceptance [0.2512, 0.3179, 0.4421]", output)

    def test_ptt_sampling_saves_endpoint_fasta_and_ladder_diagnostics(self):
        import json

        import pandas as pd
        import torch

        from adabmDCA import PTTConfig, PTTSampler

        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            reference = workspace / "data.fasta"
            reference.write_text(">a\nAA\n>b\nAB\n>c\nBA\n>d\nBB\n")
            params = {"bias": torch.zeros(2, 2, dtype=torch.float64),
                      "coupling_matrix": torch.zeros(2, 2, 2, 2, dtype=torch.float64)}
            backend = PTTSampler(params, tokens="AB", n_chains=20,
                                 config=PTTConfig(mixing_thermalization_rounds=2))
            archive = backend.save_archive(workspace / "ptt.h5")
            original = archive.read_bytes()
            result = run_cli("sample", "--strategy", "ptt", "-p", str(archive), "--data", str(reference),
                             "-o", str(workspace / "generated"), "--ngen", "23", "--label", "check",
                             "--max_nsweeps", "1", "--device", "cpu", "--plot", "--no_reweighting")
            self.assertEqual(result.returncode, 0, result.stderr)
            output = workspace / "generated"
            self.assertEqual((output / "check_samples.fasta").read_text().count(">sequence "), 23)
            ladder = pd.read_csv(output / "logs" / "check_ptt.log")
            self.assertEqual(len(ladder), 2)
            self.assertTrue((abs(ladder["entropy"] - 2 * torch.log(torch.tensor(2.)).item()) < 1e-6).all())
            self.assertTrue((abs(ladder["entropy"] + ladder["log_likelihood"]) < 1e-10).all())
            summary = json.loads((output / "check_sampling.json").read_text())["data"]
            self.assertEqual(summary["ptt_diagnostics"]["mixing"]["status"], "converged")
            self.assertEqual(summary["ptt_diagnostics"]["mixing_method"], "renewal")
            self.assertNotIn("exchange_passes", summary["ptt_diagnostics"])
            self.assertEqual(summary["ptt_diagnostics"]["equilibration_rounds"], 0)
            self.assertIn("renewal warmup", result.stdout)
            self.assertIn("stationary renewal: not run (add --ptt-stationary", result.stdout)
            # 23 sequences from 20 chains: the second batch follows one warmup-length renewal time.
            mixing = summary["ptt_diagnostics"]["mixing"]
            self.assertFalse(mixing["stationary"])
            self.assertEqual(summary["ptt_diagnostics"]["spacing_rounds"], mixing["warmup_rounds"])
            self.assertIn("trapped endpoint fraction", result.stdout)
            self.assertIn("final Pearson", result.stdout)
            self.assertIn("final_pearson", summary["ptt_diagnostics"])
            renewal_log = pd.read_csv(output / "logs" / "check_ptt_renewal.log")
            self.assertEqual(set(renewal_log["phase"]), {"warmup"})
            for filename in (
                "check_ptt_renewal.png", "check_cij_scatter.png",
                "check_pca_1_2.png", "check_pca_3_4.png",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            self.assertEqual(original, archive.read_bytes())

    def test_ptt_stage_status_carries_current_pearson(self):
        from adabmDCA.api.results import TrainingProgress
        from adabmDCA.scripts.train import _PTTProgressRenderer
        from adabmDCA.training_control import StageProgress

        stream = io.StringIO()
        renderer = _PTTProgressRenderer(target_pearson=0.95, max_steps=100, stream=stream)
        renderer.on_stage(StageProgress("ptt_equilibration", "start", 0))
        renderer.on_stage(StageProgress("ptt_equilibration", "progress", 0, 2, 2))
        renderer(TrainingProgress(epoch=1, metrics={"Pearson": 0.72}, gradient_steps=1))
        renderer.on_stage(StageProgress("mixing_measure", "progress", 1, 5, 5))
        renderer.on_stage(StageProgress("ptt_pearson_updated", "event", 0, details={"pearson": 0.61}))
        renderer.on_stage(StageProgress("reservoir_collect", "progress", 0, 2, 2))
        renderer.close()

        output = stream.getvalue()
        self.assertIn("Pearson ?/0.950 | initializing replicas 2/2 rounds", output)
        self.assertIn("step 1/100 | Pearson 0.720/0.950 | measuring mixing 5/5 rounds", output)
        self.assertIn("step 0/100 | Pearson 0.610/0.950 | collecting reservoir 2/2 samples", output)

    def test_ptt_interactive_checkpoint_status_keeps_time_moving(self):
        from adabmDCA.scripts.train import _PTTProgressRenderer
        from adabmDCA.training_control import StageProgress

        class Terminal(io.StringIO):
            def isatty(self):
                return True

        stream = Terminal()
        renderer = _PTTProgressRenderer(target_pearson=0.95, max_steps=100, stream=stream)
        renderer.on_stage(StageProgress("ptt_checkpoint_start", "event", 5))
        time.sleep(1.1)
        renderer.on_stage(StageProgress("ptt_checkpoint_done", "event", 5))
        renderer.close()
        output = stream.getvalue()
        self.assertGreaterEqual(output.count("saving checkpoint"), 2)
        self.assertIn("checkpoint saved", output)
        self.assertIn("\r\x1b[2K", output)

    def test_ptt_progress_is_concise_by_default_and_detailed_on_request(self):
        from adabmDCA.api.results import TrainingProgress
        from adabmDCA.scripts.train import _PTTProgressRenderer
        from adabmDCA.training_control import StageProgress
        from adabmDCA.training_log import history_columns

        metrics = {
            "Pearson": 0.72,
            "Pearson_val": 0.70,
            "Sweeps": 1200.0,
            "Time": 65.0,
            "bias_learning_rate": 0.005,
            "coupling_learning_rate": 0.004,
            "ptt_replicas": 3,
            "ptt_acceptance": 0.21,
            "ptt_acceptance_rates": (0.21, 0.38),
            "ptt_predicted_kl": 0.0021,
            "ptt_lag_drift": 0.12,
            "ptt_lag_drift_tail": 0.05,
            "LL_train": -0.612,
            "LL_val": -0.645,
        }
        event = TrainingProgress(epoch=12, metrics=metrics, gradient_steps=12)
        pause = StageProgress("ptt_lag_pause", "start", 12, details={
            "model_version": 12, "rounds": 0, "healthy": True, "lag_before": 0.28, "lag_after": 0.07,
            "ladder_action": "insert", "trigger_drift": 0.27, "trigger_drift_tail": 0.28,
        })
        mixing = StageProgress("ptt_mixing", "start", 12, details={
            "status": "converged", "method": "renewal", "warmup_rounds": 40, "renewal_rounds": 20,
            "trapped_fraction": 0.0, "rounds": 60,
        })
        columns = history_columns(model_type="bmDCA", ptt_optimizer="adaptive", validation=True)

        concise_stream = io.StringIO()
        concise = _PTTProgressRenderer(
            target_pearson=0.95, max_steps=100, stream=concise_stream
        )
        concise.set_columns(columns)
        concise(event)
        concise.on_stage(mixing)
        concise.on_stage(pause)
        concise.close()
        concise_output = concise_stream.getvalue()
        self.assertIn("step 12/100 | Pearson 0.720/0.950 | optimizing | elapsed", concise_output)
        self.assertIn("»    12  pause: drift tail lag 0.28 → 0.07, inserted a snapshot, 0 extra rounds", concise_output)
        self.assertNotIn("» ", concise_output.replace("»    12  pause", ""))
        self.assertNotIn("pearson", concise_output)

        diagnostic_stream = io.StringIO()
        diagnostic = _PTTProgressRenderer(
            target_pearson=0.95, max_steps=100, diagnostic=True, stream=diagnostic_stream
        )
        diagnostic.set_columns(columns)
        diagnostic(event)
        diagnostic.on_stage(mixing)
        diagnostic.close()
        lines = diagnostic_stream.getvalue().splitlines()
        self.assertEqual(lines[0].split(), ["step", "sweeps", "time", "pearson", "val", "ll/L", "ll_val",
                                            "reps", "acc", "lr_h", "lr_J", "lag", "tail"])
        self.assertEqual(lines[1].split(), ["12", "1.2k", "0:01:05", "0.7200", "0.7000", "-0.612", "-0.645",
                                            "3", "0.21", "0.005", "0.004", "0.12", "0.05"])
        self.assertIn("»    12  mixing check converged: renewal warmup 40 rounds, stationary renewal 20 rounds",
                      diagnostic_stream.getvalue())

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
            "split-data",
            "plot-training-log",
            "plot_training_log",
            "preprocess",
        ]

        for command in commands:
            with self.subTest(command=command):
                result = run_cli(command, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_energies_local_lambda_defaults_to_none(self):
        from adabmDCA.scripts.energies import create_parser

        required = ["-d", "alignment.fasta", "-p", "model.h5", "-o", "out"]
        parser = create_parser()
        self.assertIsNone(parser.parse_args(required).local_lambda)
        self.assertEqual(parser.parse_args([*required, "--local-lambda", "1.5"]).local_lambda, 1.5)
        self.assertIn("sampling script", parser.format_help())

    def test_energies_cli_omits_local_free_energy_without_lambda(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np

        from adabmDCA.scripts.energies import create_parser, main

        args = create_parser().parse_args(["-d", "alignment.fasta", "-p", "model.h5", "-o", "out"])
        result = SimpleNamespace(
            sequences=("AA",), model=SimpleNamespace(length=2),
            energies=np.array([0.0]), local_free_energies=None,
            save_bundle=lambda *_args, **_kwargs: {},
        )
        with patch("adabmDCA.scripts._frontend.resolve_alphabet"), \
                patch("adabmDCA.scripts.energies.run", return_value=result), \
                patch("adabmDCA.scripts._frontend.print_completion") as completion:
            self.assertEqual(main(args), 0)
        self.assertNotIn("mean local free energy", completion.call_args.kwargs["metrics"])

    def test_all_compute_commands_default_to_auto_device(self):
        from adabmDCA.parser import (
            add_args_contacts,
            add_args_dms,
            add_args_energies,
            add_args_sample,
            add_args_split_data,
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
            add_args_split_data,
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

    def test_sample_help_and_parser_accept_bfloat16(self):
        result = run_cli("sample", "--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bfloat16", result.stdout)
        self.assertIn("--plot", result.stdout)

    def test_sample_plot_option_writes_all_diagnostics(self):
        import torch

        from adabmDCA.io import save_params

        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = workspace / "params.dat"
            reference = workspace / "reference.fasta"
            output = workspace / "samples"
            params = {
                "bias": torch.zeros(2, 3),
                "coupling_matrix": torch.zeros(2, 3, 2, 3),
            }
            save_params(str(model), params, tokens="AB-")
            reference.write_text(
                ">s1\nAA\n>s2\nAB\n>s3\nA-\n>s4\nBA\n>s5\nBB\n>s6\nB-\n>s7\n-A\n>s8\n-B\n",
                encoding="utf-8",
            )

            completed = run_cli(
                "sample",
                "--strategy", "pcd",
                "--path_params",
                str(model),
                "--data",
                str(reference),
                "--output",
                str(output),
                "--ngen",
                "8",
                "--nmeasure",
                "8",
                "--nmix",
                "1",
                "--max_nsweeps",
                "2",
                "--alphabet",
                "AB-",
                "--device",
                "cpu",
                "--no_reweighting",
                "--plot",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Generating sequences", completed.stderr)
            for filename in (
                "autocorrelation.png",
                "pearson_sampling.png",
                "cij_scatter.png",
                "pca_1_2.png",
                "pca_3_4.png",
            ):
                self.assertTrue((output / filename).is_file(), filename)

    def test_train_help_documents_edge_dca_pseudocount_default(self):
        result = run_cli("train", "--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0.1 for edgeDCA", result.stdout)

    def test_train_help_uses_validation_flag(self):
        result = run_cli("train", "--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-v", result.stdout)
        self.assertIn("--validation VAL", result.stdout)
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
                "--strategy", "pcd",
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
        self.assertIn("/0.9900", result.stderr)
        self.assertIn("Optimization | Step 1/1", result.stderr)

    def test_reintegrate_accepts_progress_renderer_by_default(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            natural = workspace / "natural.fasta"
            experimental = workspace / "experimental.fasta"
            adjustments = workspace / "adjustments.txt"
            natural.write_text(
                ">n1\nAA\n>n2\nAB\n>n3\nBA\n>n4\nBB\n",
                encoding="utf-8",
            )
            experimental.write_text(">e1\nA-\n>e2\nB-\n", encoding="utf-8")
            adjustments.write_text("-1\n1\n", encoding="utf-8")

            result = run_cli(
                "reintegrate",
                "--data",
                str(natural),
                "--reint",
                str(experimental),
                "--adj",
                str(adjustments),
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
                "--no_reweighting",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Reintegrated training completed successfully.", result.stdout)

    def test_train_no_progress_keeps_transient_output_quiet(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            fasta = workspace / "tiny.fasta"
            fasta.write_text(
                ">s1\nAA\n>s2\nAB\n>s3\nBA\n>s4\nBB\n",
                encoding="utf-8",
            )
            result = run_cli(
                "train",
                "--strategy", "pcd",
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
                "--no-progress",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Pearson", result.stderr)
        self.assertIn("Training completed successfully.", result.stdout)

    def test_ptt_terminal_reports_stages_with_current_pearson(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            fasta = workspace / "tiny.fasta"
            fasta.write_text(">s1\nAAA\n>s2\nAAB\n>s3\nABA\n>s4\nBBB\n", encoding="utf-8")
            base = (
                "train", "--data", str(fasta), "--alphabet", "AB", "--strategy", "ptt",
                "--nchains", "20", "--nsweeps", "1", "--nepochs", "1",
                "--ptt-initialization-rounds", "2", "--ptt-mixing-thermalization-rounds", "2",
                "--ptt-mixing-method", "autocorrelation",
                "--device", "cpu", "--no_reweighting",
            )
            shown = run_cli(*base, "--output", str(workspace / "shown"))
            quiet = run_cli(*base, "--output", str(workspace / "quiet"), "--no-progress")

        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("initializing replicas 2/2 rounds", shown.stderr)
        self.assertIn("mixing warmup 2/2 rounds", shown.stderr)
        self.assertIn("measuring mixing 100/100 rounds", shown.stderr)
        self.assertIn("mixing check converged", shown.stderr)
        self.assertIn("checkpoint saved", shown.stderr)
        self.assertIn("Pearson ?/", shown.stderr)
        # The Pearson of an untrained toy model can have either sign.
        self.assertRegex(shown.stderr, r"Pearson -?[01]\.\d{3}/")
        self.assertNotIn("\x1b[", shown.stderr)
        self.assertNotIn("Pearson 0.0000/", shown.stderr)
        self.assertEqual(quiet.returncode, 0, quiet.stderr)
        self.assertEqual(quiet.stderr, "")
        self.assertIn("Training completed successfully.", quiet.stdout)

    def test_plot_training_log_missing_file_reports_parser_error(self):
        result = run_cli("plot-training-log", "missing.csv")

        self.assertEqual(result.returncode, 2)
        self.assertIn("History file 'missing.csv' not found", result.stderr)

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


def test_ptt_status_shows_the_validation_plateau_test_instead_of_the_pearson_target():
    from adabmDCA.api.results import TrainingProgress
    from adabmDCA.ptt.training import validation_window_gain
    from adabmDCA.scripts.train import _PTTProgressRenderer

    plain = _PTTProgressRenderer(target_pearson=0.95, max_steps=100, stream=io.StringIO())
    plain(TrainingProgress(epoch=1, metrics={"Pearson": 0.72, "LL_val": -0.9}, gradient_steps=1))
    assert "Pearson 0.720/0.950" in plain._status
    plain.close()

    renderer = _PTTProgressRenderer(target_pearson=0.95, max_steps=100, stream=io.StringIO())
    renderer.set_validation_stop(window=2, min_gain=0.0)
    values = [-0.95, -0.90, -0.80, -0.75, -0.74]
    for step, value in enumerate([-1.2, *values]):
        renderer(TrainingProgress(epoch=step, metrics={"Pearson": 0.97, "LL_val": value}, gradient_steps=step))
        if step == 3:
            assert "plateau check from step 4" in renderer._status
    assert "/0.950" not in renderer._status
    assert "Pearson 0.970 | val LL -0.7400" in renderer._status
    # Same statistic as the trainer's plateau test, which skips the initial row.
    assert f"plateau gain {validation_window_gain(values, 2):+.1e} (stops < 0)" in renderer._status
    # A recovery that restores step 2 rolls back the later updates.
    renderer(TrainingProgress(epoch=3, metrics={"Pearson": 0.9, "LL_val": -1.0}, gradient_steps=3))
    assert "val LL -1.0000" in renderer._status
    assert "plateau check from step 4" in renderer._status
    renderer.close()
