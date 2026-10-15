"""Turn the files of a training run (and optionally a sampling check) into a verdict.

The rules follow the "Diagnose a run" guide (docs/algorithms/diagnostics.md): each
finding names the failure it points to (F1-F8) and the change the guide recommends,
written as CLI options. Thresholds are the guide's; they are heuristics, so findings say
what was measured.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from adabmDCA.exceptions import InputLoadError

GUIDE = "https://spqb.github.io/adabmDCApy/algorithms/diagnostics/"
FAILURES = {
    "F1": "a narrow basin traps configurations",
    "F2": "the chains fall behind a forming mode",
    "F3": "a ladder link is too wide",
    "F4": "the replicas relax too slowly",
    "F5": "the learning rate is too large",
    "F6": "good metrics, miscalibrated model",
    "F7": "slow or incomplete PTT sampling",
    "F8": "the initial mixing check fails",
    "PCD": "checking a PCD model",
}
SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
STALE_SECONDS = 15 * 60
# Fractions of updates are only meaningful over a run of some length.
MIN_UPDATES_FOR_RATES = 50


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_history(path: Path) -> dict[str, list[float | None]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = rows[0].keys() if rows else []
    return {name: [_float(row.get(name)) for row in rows] for name in columns}


def _values(history: dict[str, list[float | None]], name: str) -> list[float]:
    return [value for value in history.get(name, []) if value is not None]


def _read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a line being written
    return events


def _read_data(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return payload.get("data", payload) if isinstance(payload, dict) else {}


def _tail(path: Path, lines: int = 8) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


class _Findings:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        action: str | None = None,
        failure: str | None = None,
    ) -> None:
        item: dict[str, Any] = {"severity": severity, "code": code, "message": message}
        if action:
            item["action"] = action
        if failure:
            item["failure"] = failure
            item["failure_name"] = FAILURES[failure]
        self.items.append(item)

    def sorted(self) -> list[dict[str, Any]]:
        return sorted(self.items, key=lambda item: SEVERITY_ORDER[item["severity"]])


def _training_indicators(
    history: dict[str, list[float | None]], events: list[dict[str, Any]]
) -> dict[str, Any]:
    indicators: dict[str, Any] = {}
    steps = _values(history, "step")
    if steps:
        indicators["updates"] = int(steps[-1])
    for name in ("pearson", "pearson_val", "ll_train", "ll_val", "time_s", "replicas"):
        values = _values(history, name)
        if values:
            indicators[f"final_{name}"] = _round(values[-1])
    ll_val = _values(history, "ll_val")
    if ll_val:
        best = max(range(len(ll_val)), key=ll_val.__getitem__)
        indicators["best_ll_val"] = _round(ll_val[best])
        indicators["best_ll_val_position"] = _round(best / max(len(ll_val) - 1, 1), 3)
    acceptance = _values(history, "acceptance_min")
    if acceptance:
        indicators["min_acceptance"] = _round(min(acceptance), 3)
        indicators["fraction_acceptance_below_0.25"] = _round(
            sum(value < 0.25 for value in acceptance) / len(acceptance), 3
        )
        indicators["fraction_acceptance_below_0.1"] = _round(
            sum(value < 0.1 for value in acceptance) / len(acceptance), 3
        )
    lr_bias = _values(history, "lr_bias")
    if lr_bias:
        indicators["lr_bias_at_floor_fraction"] = _round(
            sum(value < 1e-10 for value in lr_bias) / len(lr_bias), 3
        )
    kl = _values(history, "kl")
    if kl:
        indicators["median_kl"] = float(f"{statistics.median(kl):.3g}")
        indicators["kl_at_cap_fraction"] = _round(sum(value >= 0.0099 for value in kl) / len(kl), 3)
    tail = _values(history, "lag_drift_tail")
    if tail:
        indicators["max_lag_drift_tail"] = _round(max(tail), 3)

    counts = Counter(event.get("event") for event in events)
    failed = [
        event
        for event in events
        if event.get("event") == "ptt_mixing" and event.get("status") not in (None, "converged")
    ]
    blocking = [
        entry
        for event in failed
        for entry in (
            event.get("immobile") or (event.get("reservoir_renewal") or {}).get("immobile") or []
        )
        if isinstance(entry, dict) and entry.get("blocking")
    ]
    refreshes = [event for event in events if event.get("event") == "ptt_reservoir_refreshed"]
    indicators["events"] = dict(counts)
    indicators["failed_mixing_checks"] = len(failed)
    indicators["blocking_immobile_configurations"] = max(
        (int(entry.get("immobile", 0)) for entry in blocking), default=0
    )
    indicators["recoveries"] = counts["ptt_recovery"]
    indicators["lag_pauses"] = counts["ptt_lag_pause"]
    indicators["failed_pause_refreshes"] = sum(
        1
        for event in events
        if event.get("event") == "ptt_lag_pause" and event.get("healthy") is False
    )
    indicators["reservoir_refreshes"] = len(refreshes)
    warmups = [
        event["renewal_warmup_rounds"]
        for event in refreshes
        if isinstance(event.get("renewal_warmup_rounds"), (int, float))
    ]
    if warmups:
        indicators["reservoir_warmup_rounds"] = warmups[-6:]
    if events:
        indicators["last_event"] = {
            key: events[-1].get(key) for key in ("event", "step", "utc") if key in events[-1]
        }
    return indicators


def _check_training(
    findings: _Findings,
    summary: dict[str, Any] | None,
    indicators: dict[str, Any],
    *,
    ptt: bool,
    stale: bool,
    log_tail: list[str],
) -> None:
    if summary is None:
        last = indicators.get("last_event", {}).get("event", "")
        if not stale:
            findings.add(
                "info",
                "running",
                "No training summary yet and the run files changed recently: the run is "
                f"probably still going (last event: {last or 'none'}).",
                "Check again later: tail history.csv and grep '»' in the log.",
            )
        elif ptt and (
            last in ("ptt_mixing", "ptt_recovery", "ptt_lag_pause", "ptt_equilibration")
            or indicators.get("failed_mixing_checks")
        ):
            findings.add(
                "error",
                "stalled_or_stopped",
                "No training summary and no output for a while; the last event was "
                f"'{last}'. A run stuck in a mixing check or reservoir refresh is the "
                "signature of a trapping basin. See log_tail for an error message.",
                "Restart from scratch with 10x the local sweeps (--nsweeps 100); do not "
                "resume the archive.",
                "F1",
            )
        elif any("TrainingCancelled" in line for line in log_tail):
            findings.add(
                "info",
                "cancelled",
                "The run was cancelled; its last checkpoint is in run_dir.",
                "Resume from the checkpoint: --ptt-resume ptt.h5 for PTT runs, or -p "
                "params.dat.gz -c chains.fasta for PCD runs.",
            )
        else:
            findings.add(
                "warning",
                "no_summary",
                "No training summary and no recent output: the run ended without "
                "saving one (crash, kill or error). See log_tail.",
                "Resume from the checkpoint: --ptt-resume ptt.h5 for PTT runs, or -p "
                "params.dat.gz -c chains.fasta for PCD runs.",
            )
        if any("Restart the training with more local sweeps" in line for line in log_tail):
            findings.add(
                "error",
                "trapped_configurations",
                "The run stopped after repeated mixing failures caused by configurations "
                "that cannot swap toward the bottom of the ladder.",
                "Restart from scratch with --nsweeps 100. Do not resume.",
                "F1",
            )
        return

    stop_reason = summary.get("stop_reason")
    if summary.get("converged") is False:
        findings.add(
            "warning",
            "not_converged",
            f"The run ended on '{stop_reason}' before reaching its convergence criterion.",
            "If the curves were still improving, continue with a larger budget "
            "(--ptt-resume ptt.h5 for PTT runs, -p params.dat.gz -c chains.fasta for PCD).",
        )
    for warning in summary.get("warnings") or []:
        findings.add("info", "training_warning", str(warning))

    blocking = indicators.get("blocking_immobile_configurations", 0)
    failed = indicators.get("failed_mixing_checks", 0)
    if blocking:
        findings.add(
            "error" if indicators.get("recoveries", 0) >= 3 else "warning",
            "trapped_configurations",
            f"{failed} mixing check(s) failed with up to {blocking} configurations that "
            "cannot swap toward the bottom of the ladder.",
            "Restart from scratch with --nsweeps 100. Do not resume the archive.",
            "F1",
        )
    elif failed:
        findings.add(
            "warning",
            "mixing_failures",
            f"{failed} mixing check(s) failed without blocking immobile configurations: "
            "a link too wide (F3) or replicas relaxing too slowly (F4). The failure "
            "messages in the log say which.",
            "F3: a smaller --lr or a larger --ptt-target-acceptance. F4: more --nsweeps.",
            "F3",
        )
    if indicators.get("recoveries"):
        findings.add(
            "warning",
            "recoveries",
            f"{indicators['recoveries']} recovery rollback(s) halved the learning rates.",
            "Read the mixing failure before each recovery in the log or events.",
        )
    pauses = indicators.get("lag_pauses", 0)
    if indicators.get("failed_pause_refreshes") or pauses >= 3:
        findings.add(
            "warning",
            "repeated_pauses",
            f"{pauses} lag pause(s), {indicators.get('failed_pause_refreshes', 0)} with a "
            "failed reservoir refresh: the forming mode is too narrow for the lag estimate.",
            "Treat as F1: restart with --nsweeps 100.",
            "F1",
        )
    elif pauses:
        findings.add(
            "info",
            "lag_pause",
            f"{pauses} lag pause(s) let the chains catch up with a forming mode; handled.",
            "Nothing to do; check the finished model by sampling it.",
            "F2",
        )
    warmups = indicators.get("reservoir_warmup_rounds") or []
    if len(warmups) >= 3 and warmups[-1] >= 2 * warmups[0] and warmups[-1] > 50:
        findings.add(
            "warning",
            "growing_reservoir_warmup",
            f"Reservoir refresh warmups grew {warmups}: an early warning of a trapping basin.",
            "Watch for stalls; consider restarting with --nsweeps 100.",
            "F1",
        )
    long_enough = (indicators.get("updates") or 0) >= MIN_UPDATES_FOR_RATES
    if long_enough and (
        (indicators.get("lr_bias_at_floor_fraction") or 0) > 0.5
        or (indicators.get("median_kl") or 0) >= 0.0099
    ):
        findings.add(
            "warning",
            "learning_rate_too_large",
            f"lr_bias at its floor in {indicators.get('lr_bias_at_floor_fraction')} of the "
            f"updates, median kl {indicators.get('median_kl')} (cap 0.01): the trust region "
            "binds at every step.",
            "Go back to the default --lr 0.01.",
            "F5",
        )
    if long_enough and (indicators.get("fraction_acceptance_below_0.1") or 0) > 0.1:
        findings.add(
            "warning",
            "low_acceptance",
            "The minimum swap acceptance stayed below 0.1 in "
            f"{indicators['fraction_acceptance_below_0.1']} of the updates.",
            "Check for immobile configurations first (F1); otherwise a smaller --lr or a "
            "larger ptt target acceptance.",
            "F3",
        )
    if (indicators.get("max_lag_drift_tail") or 0) >= 0.2 and not pauses:
        findings.add(
            "info",
            "lag_peak",
            f"lag_drift_tail peaked at {indicators['max_lag_drift_tail']} (pause threshold "
            "0.25): the chains briefly trailed a forming mode.",
            failure="F2",
        )
    best, position = indicators.get("best_ll_val"), indicators.get("best_ll_val_position")
    final = indicators.get("final_ll_val")
    if None not in (best, final, position) and best - final > 0.01 and position < 0.8:
        findings.add(
            "warning",
            "validation_falling",
            f"Validation log-likelihood fell from its best {best} to {final} per site "
            f"over the last {round((1 - position) * 100)}% of the run.",
            "Keep the validation stop on (the default with -v), or use the checkpoint near "
            "the best value; a falling "
            "curve can also follow a trapped mode (F1).",
        )
    if not ptt:
        findings.add(
            "info",
            "pcd_model",
            "PCD training metrics come from chains that may not be at equilibrium.",
            "Resample it with sample --strategy pcd -d train.fasta --plot: the Pearson should "
            "settle on a plateau and a mixing time should be found.",
            "PCD",
        )


def _check_sampling(
    findings: _Findings,
    sampling: dict[str, Any],
    training_pearson: float | None,
) -> dict[str, Any]:
    indicators: dict[str, Any] = {}
    ptt = sampling.get("ptt_diagnostics") or None
    history = sampling.get("sampling_history") or {}
    pearsons = [value for value in map(_float, history.get("pearson") or []) if value is not None]
    final = _float((ptt or {}).get("final_pearson")) or (pearsons[-1] if pearsons else None)
    indicators["final_pearson"] = _round(final)
    if training_pearson is not None and final is not None:
        indicators["training_pearson"] = _round(training_pearson)
        indicators["pearson_drop"] = _round(training_pearson - final)
        if training_pearson - final > 0.01:
            findings.add(
                "warning",
                "sampling_pearson_drop",
                f"Independent samples reach Pearson {final:.3f} against {training_pearson:.3f} "
                "reported by training. Compare at equal sample size; healthy runs lose "
                "0.003-0.007.",
                "Retrain with more --nsweeps.",
                "F6",
            )
    if ptt:
        mixing = ptt.get("mixing") or {}
        indicators["ptt_converged"] = ptt.get("converged")
        indicators["ptt_status"] = mixing.get("status")
        if mixing.get("trapped_fraction") is not None:
            indicators["trapped_fraction"] = _round(_float(mixing.get("trapped_fraction")))
        if ptt.get("converged") is False:
            findings.add(
                "warning",
                "ptt_sampling_incomplete",
                f"PTT sampling did not renew within its budget (status "
                f"{mixing.get('status')}, trapped fraction {indicators.get('trapped_fraction')}).",
                "If G was still decaying, raise --ptt-max-rounds; otherwise retrain with more "
                "--nsweeps.",
                "F7",
            )
        pairs = (ptt.get("ladder_health") or {}).get("pairs") or []
        immobile = [_float(pair.get("immobile_down_fraction")) or 0.0 for pair in pairs]
        if immobile:
            indicators["max_immobile_down_fraction"] = _round(max(immobile))
            if max(immobile) >= 0.01:
                findings.add(
                    "info",
                    "trapped_links",
                    f"Up to {max(immobile):.1%} of a rung's configurations rarely swap down. "
                    "This does not mean bad samples, but a basin's share may be off.",
                    "Compare cluster shares with and without --ptt-stationary.",
                    "F6",
                )
    elif len(pearsons) >= 6:
        # A sustained decline after a peak, not noise around a plateau.
        tail = pearsons[-max(len(pearsons) // 3, 2) :]
        level = statistics.mean(tail)
        spread = statistics.pstdev(tail)
        peak = max(pearsons)
        if peak - level > max(0.01, 3 * spread) and level > 0.3:
            findings.add(
                "warning",
                "transient_peak",
                f"The sampling Pearson peaked at {peak:.3f} and settled at {level:.3f}: the "
                "model reproduces the data only during a transient of the sampler, so it "
                "was trained out of equilibrium.",
                "Retrain with PTT.",
                "PCD",
            )

    distances = sampling.get("distance_comparison") or {}
    summary = distances.get("summary") or {}
    to_train = summary.get("generated_to_natural") or {}
    held_out = summary.get("test_to_natural") or {}
    if to_train:
        indicators["generated_to_train_median"] = _round(_float(to_train.get("median")), 3)
        indicators["identical_to_train"] = _round(_float(to_train.get("identical")), 4)
    if held_out:
        indicators["held_out_to_train_median"] = _round(_float(held_out.get("median")), 3)
    privet = distances.get("privet") or {}
    for key in ("n_flagged", "n_memorized", "flagged_fraction"):
        if key in privet:
            indicators[f"privet_{key}"] = privet[key]
    if (privet.get("n_memorized") or 0) > 0 or (_float(to_train.get("identical")) or 0) > 0:
        findings.add(
            "warning",
            "memorization",
            f"{privet.get('n_memorized', 0)} generated sequence(s) flagged as memorized by "
            f"PRIVET; {to_train.get('identical', 0):.1%} identical to a training sequence.",
            "Inspect those sequences; consider more regularization or fewer epochs.",
        )
    elif (privet.get("n_flagged") or 0) > 0:
        findings.add(
            "info",
            "privet_flagged",
            f"PRIVET flagged {privet['n_flagged']} generated sequence(s) as unusually close "
            "to training sequences.",
        )
    generated = _float(to_train.get("median"))
    test = _float(held_out.get("median"))
    if generated is not None and test is not None and test > 0 and generated < test - 0.03:
        findings.add(
            "info",
            "closer_than_held_out",
            f"Generated sequences sit closer to the training set (median {generated:.3f}) "
            f"than held-out ones ({test:.3f}). Not proof of copying; a pile near zero is.",
        )
    for warning in sampling.get("warnings") or []:
        if "not a certificate of equilibrium" not in str(warning):
            findings.add("info", "sampling_warning", str(warning))
    return indicators


def diagnose_run(
    run_dir: str | Path,
    *,
    label: str | None = None,
    sampling_json: str | Path | None = None,
) -> dict[str, Any]:
    """Indicators and findings for a training run directory, and optionally a sampling check."""
    run = Path(run_dir).expanduser()
    prefix = f"{label}_" if label else ""
    files = {
        "history": run / f"{prefix}history.csv",
        "events": run / f"{prefix}events.jsonl",
        "summary": run / f"{prefix}training.json",
        "log": run / f"{label or 'adabmDCA'}.log",
    }
    if not files["history"].exists() and not files["summary"].exists():
        candidates = sorted(path.name for path in run.glob("*history.csv"))
        hint = f" Found: {', '.join(candidates)}; pass the matching label." if candidates else ""
        raise InputLoadError(
            f"No {files['history'].name} or {files['summary'].name} in {run}.{hint}",
            details={"run_dir": str(run), "label": label},
        )

    history = _read_history(files["history"]) if files["history"].exists() else {}
    events = _read_events(files["events"]) if files["events"].exists() else []
    summary = _read_data(files["summary"]) if files["summary"].exists() else None
    ptt = "acceptance_min" in history or any(
        str(event.get("event", "")).startswith("ptt_") for event in events
    )
    mtimes = [path.stat().st_mtime for path in files.values() if path.exists()]
    stale = not mtimes or time.time() - max(mtimes) > STALE_SECONDS
    log_tail = _tail(files["log"]) if files["log"].exists() else []

    findings = _Findings()
    indicators = _training_indicators(history, events)
    if summary is not None:
        indicators["stop_reason"] = summary.get("stop_reason")
        indicators["converged"] = summary.get("converged")
        indicators["gradient_steps"] = summary.get("gradient_steps")
    _check_training(findings, summary, indicators, ptt=ptt, stale=stale, log_tail=log_tail)

    sampling_indicators = None
    if sampling_json is not None:
        sampling = _read_data(Path(sampling_json).expanduser())
        training_pearson = indicators.get("final_pearson")
        sampling_indicators = _check_sampling(findings, sampling, training_pearson)
    elif summary is not None:
        findings.add(
            "info",
            "unchecked_model",
            "Training metrics alone cannot show a miscalibrated model.",
            (
                "Sample it with sample -p ptt.h5 -d train.fasta -v test.fasta --plot -o check, "
                if ptt
                else "Sample it with sample --strategy pcd -p params.dat.gz -d train.fasta -v test.fasta --plot -o check, "
            )
            + "then diagnose again with that sampling.json.",
            "F6",
        )

    ordered = findings.sorted()
    severities = {item["severity"] for item in ordered}
    if summary is None and not stale and "error" not in severities:
        verdict = "running"
    elif "error" in severities:
        verdict = "failed"
    elif summary is None:
        verdict = "incomplete"
    elif "warning" in severities:
        verdict = "needs_attention"
    else:
        verdict = "healthy"
    report: dict[str, Any] = {
        "verdict": verdict,
        "run_dir": str(run),
        "label": label,
        "training_kind": "ptt" if ptt else "pcd",
        "findings": ordered,
        "training": indicators,
        "files": {name: str(path) for name, path in files.items() if path.exists()},
        "guide": GUIDE,
    }
    if sampling_indicators is not None:
        report["sampling"] = sampling_indicators
    if summary is None or verdict in ("failed", "needs_attention"):
        report["log_tail"] = log_tail
    return report
