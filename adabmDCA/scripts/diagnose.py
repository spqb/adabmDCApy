"""Command-line adapter for diagnosing a training run."""

from __future__ import annotations

import argparse
import json

MARKS = {"error": "✗", "warning": "!", "info": "·"}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose a training run from its history, events, summary and log, and "
            "optionally the sampling.json of a check of the finished model."
        )
    )
    parser.add_argument("run_dir", type=str, help="The training output folder (train -o).")
    parser.add_argument("-l", "--label", type=str, default=None, help="The run's label (train -l), if any.")
    parser.add_argument(
        "-s", "--sampling", type=str, default=None,
        help="sampling.json written by 'adabmDCA sample --plot' on this model, to check the finished model.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    return parser


def run(args):
    """Diagnose the run selected by ``args`` and return the report."""
    from adabmDCA.api.diagnostics import diagnose_run

    return diagnose_run(args.run_dir, label=args.label, sampling_json=args.sampling)


def main(args=None) -> int:
    args = create_parser().parse_args() if args is None else args
    report = run(args)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    training = report["training"]
    print(f"Verdict: {report['verdict']}  ({report['training_kind'].upper()} run in {report['run_dir']})")
    summary = [
        f"{name}={training[name]}"
        for name in ("stop_reason", "updates", "final_pearson", "best_ll_val", "failed_mixing_checks", "recoveries")
        if training.get(name) is not None
    ]
    if summary:
        print("  " + ", ".join(summary))
    if report.get("sampling"):
        sampling = report["sampling"]
        print("  sampling: " + ", ".join(f"{key}={value}" for key, value in sampling.items() if value is not None))
    for finding in report["findings"]:
        failure = f" [{finding['failure']}: {finding['failure_name']}]" if finding.get("failure") else ""
        print(f"{MARKS[finding['severity']]} {finding['severity']}{failure}: {finding['message']}")
        if finding.get("action"):
            print(f"    → {finding['action']}")
    if report.get("log_tail"):
        print("Last log lines:")
        for line in report["log_tail"]:
            print(f"    {line}")
    print(f"Guide: {report['guide']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
