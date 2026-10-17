import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from adabmDCA import diagnose_run
from adabmDCA.exceptions import InputLoadError

ROOT = Path(__file__).resolve().parents[1]


def write_ptt_run(directory, *, updates=60, lr_bias=0.01, kl=0.001, events=(), summary=None):
    directory.mkdir()
    header = "step,sweeps,time_s,pearson,slope,ll_train,acceptance_min,lr_bias,lr_coupling,kl,"
    header += "lag_drift,lag_drift_tail,pearson_val,ll_val"
    rows = [header]
    for step in range(updates):
        rows.append(
            f"{step},{10 * step},{step},{0.5 + step / 200},1,-1.5,0.4,{lr_bias},0.01,{kl},"
            f"0.01,0.02,{0.5 + step / 250},{-1.6 + step / 1000}"
        )
    (directory / "history.csv").write_text("\n".join(rows) + "\n")
    lines = [{"event": "ptt_optimization", "step": 0}, *events]
    (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in lines))
    if summary is not None:
        (directory / "training.json").write_text(json.dumps({"data": summary}))


class DiagnoseRunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_healthy_ptt_run(self):
        run = self.tmp / "run"
        write_ptt_run(run, summary={"stop_reason": "validation_plateau", "converged": True, "warnings": []})

        report = diagnose_run(run)

        self.assertEqual(report["verdict"], "healthy")
        self.assertEqual(report["training_kind"], "ptt")
        self.assertEqual(report["training"]["updates"], 59)
        self.assertEqual([finding["code"] for finding in report["findings"]], ["unchecked_model"])

    def test_trapped_configurations_and_large_learning_rate(self):
        run = self.tmp / "run"
        failed_check = {
            "event": "ptt_mixing",
            "step": 40,
            "status": "budget_exceeded",
            "immobile": [{"replica": 2, "immobile": 61, "blocking": True}],
        }
        recoveries = [{"event": "ptt_recovery", "step": 40}] * 3
        write_ptt_run(
            run, lr_bias=1e-14, kl=0.01, events=[failed_check, *recoveries],
            summary={"stop_reason": None, "converged": False},
        )

        report = diagnose_run(run)

        self.assertEqual(report["verdict"], "failed")
        failures = {finding["code"]: finding for finding in report["findings"]}
        self.assertEqual(failures["trapped_configurations"]["failure"], "F1")
        self.assertIn("--nsweeps 100", failures["trapped_configurations"]["action"])
        self.assertEqual(failures["learning_rate_too_large"]["failure"], "F5")
        self.assertEqual(report["training"]["blocking_immobile_configurations"], 61)

    def test_short_runs_do_not_trigger_rate_based_findings(self):
        run = self.tmp / "run"
        write_ptt_run(run, updates=20, lr_bias=1e-14, kl=0.01, summary={"stop_reason": "validation_plateau"})

        codes = {finding["code"] for finding in diagnose_run(run)["findings"]}

        self.assertNotIn("learning_rate_too_large", codes)

    def test_sampling_check_compares_with_training(self):
        run = self.tmp / "run"
        write_ptt_run(run, summary={"stop_reason": "validation_plateau", "converged": True})
        sampling = self.tmp / "sampling.json"
        sampling.write_text(json.dumps({"data": {
            "ptt_diagnostics": {
                "converged": True,
                "final_pearson": 0.70,
                "mixing": {"status": "converged", "trapped_fraction": 0.0},
            },
            "distance_comparison": {
                "summary": {"generated_to_natural": {"median": 0.3, "identical": 0.01}},
                "privet": {"n_flagged": 2, "n_memorized": 1},
            },
        }}))

        report = diagnose_run(run, sampling_json=sampling)

        codes = {finding["code"] for finding in report["findings"]}
        self.assertLessEqual({"sampling_pearson_drop", "memorization"}, codes)
        self.assertNotIn("unchecked_model", codes)
        self.assertAlmostEqual(report["sampling"]["pearson_drop"], 0.095, places=3)
        self.assertEqual(report["verdict"], "needs_attention")

    def test_running_and_missing_runs(self):
        run = self.tmp / "run"
        write_ptt_run(run)
        self.assertEqual(diagnose_run(run)["verdict"], "running")

        labeled = self.tmp / "labeled"
        labeled.mkdir()
        (labeled / "x_history.csv").write_text("step\n")
        with self.assertRaisesRegex(InputLoadError, "pass the matching label"):
            diagnose_run(labeled)
        self.assertEqual(diagnose_run(labeled, label="x")["verdict"], "running")

    def test_manually_halted_run_is_still_diagnosed(self):
        import os

        run = self.tmp / "run"
        write_ptt_run(run, events=[{"event": "ptt_mixing", "step": 50, "status": "converged"}])
        old = 3600
        for path in run.iterdir():
            os.utime(path, (path.stat().st_atime - old, path.stat().st_mtime - old))

        report = diagnose_run(run)

        codes = {finding["code"] for finding in report["findings"]}
        self.assertEqual(report["verdict"], "incomplete")
        self.assertIn("halted", codes)
        self.assertNotIn("stalled_or_stopped", codes)
        self.assertEqual(report["training"]["updates"], 59)

    def test_cli_prints_verdict_and_json(self):
        run = self.tmp / "run"
        write_ptt_run(run, summary={"stop_reason": "max_gradient_steps", "converged": False})

        text = subprocess.run(
            [sys.executable, "-m", "adabmDCA.cli", "diagnose", str(run)],
            cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout
        self.assertIn("Verdict: needs_attention", text)
        self.assertIn("--ptt-resume", text)

        output = subprocess.run(
            [sys.executable, "-m", "adabmDCA.cli", "diagnose", str(run), "--json"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout
        report = json.loads(output[output.index("{"):])
        self.assertEqual(report["training"]["stop_reason"], "max_gradient_steps")


if __name__ == "__main__":
    unittest.main()
