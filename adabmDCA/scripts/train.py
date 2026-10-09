"""Command-line adapter for DCA model training."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar

from adabmDCA.parser import add_args_train
from adabmDCA.ptt.config import DEFAULT_PTT_MAX_EPOCHS
from adabmDCA.training_config import DEFAULT_MAX_EPOCHS
from adabmDCA.training_log import (
    HEADER_EVERY,
    NOTABLE_EVENTS,
    describe_event,
    event_line,
    table_header,
    table_row,
)

if TYPE_CHECKING:
    from adabmDCA.api.results import TrainingProgress
    from adabmDCA.training_control import StageProgress


def create_parser() -> argparse.ArgumentParser:
    parser = add_args_train(argparse.ArgumentParser(description=("Train a DCA model. PTT (Parallel Trajectory Tempering) supports bmDCA, eaDCA and edgeDCA; "
                     "PCD (Persistent Contrastive Divergence) is faster and can work well for datasets that are "
                     "not too clustered. PTT: Béreux et al. (2026), arXiv:2607.27077, "
                     "https://doi.org/10.48550/arXiv.2607.27077."), allow_abbrev=False))
    parser.add_argument(
        "--plot-training-logs", action="store_true",
        help="At the end of training, plot the recorded curves into <output>/<label>_plots.",
    )
    return parser


class _TrainingProgressRenderer:
    """Render structured training events for an interactive terminal."""

    def __init__(self, *, model_type: str, target_pearson: float, max_steps: int) -> None:
        self._tracks_gradients = model_type == "bmDCA"
        self._target_pearson = target_pearson
        self._max_steps = max_steps
        self._bar = None

    def _start(self) -> None:
        if self._bar is not None:
            return
        from tqdm import tqdm

        self._bar = tqdm(
            total=self._target_pearson, colour="red", dynamic_ncols=True, leave=False, ascii="-#",
            bar_format="  {desc} [{bar}] Pearson {n:.4f}/{total:.4f} [{elapsed}]",
        )

    def __call__(self, event: TrainingProgress) -> None:
        self._start()
        completed = event.gradient_steps if self._tracks_gradients else event.structure_steps
        pearson = event.metrics.get("Pearson", 0.0)
        stage = event.stage.replace("_", " ").title()
        description = f"{stage} | Step {completed}/{self._max_steps}"
        if math.isfinite(pearson):
            self._bar.n = min(max(0.0, pearson), self._target_pearson)
        likelihood = event.metrics.get("LL_train")
        if likelihood is not None and math.isfinite(likelihood):
            description += f" | LL/L {likelihood:.3f}"
        density = event.metrics.get("Density")
        if density is not None and math.isfinite(density):
            description += f" | density {density:.2f}"
        self._bar.set_description(description)
        self._bar.refresh()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


class _PTTProgressRenderer:
    """One live PTT status line plus lasting milestones on the terminal."""

    _PHASES: ClassVar[dict[str, str]] = {
        "ptt_equilibration": "initializing replicas",
        "mixing_warmup": "mixing warmup",
        "mixing_measure": "measuring mixing",
        "replica_equilibration": "equilibrating replicas",
        "reservoir_warmup": "reservoir warmup",
        "reservoir_renewal": "reservoir renewal",
        "reservoir_collect": "collecting reservoir",
        "recovery_warmup": "recovery warmup",
    }

    def __init__(self, *, target_pearson: float, max_steps: int, diagnostic=False, stream=None,
                 structure_budget=False) -> None:
        self._target_pearson = target_pearson
        self._max_steps = max_steps
        # With validation stopping, the plateau test replaces the Pearson target:
        # (window, min_gain) of the test and the validation LL by update.
        self._validation_stop = None
        self._validation = {}
        # eaDCA budgets graph activations; the step counter still shows gradient updates.
        self._structure_budget = structure_budget
        self._structure = 0
        self._diagnostic = diagnostic
        self._stream = sys.stderr if stream is None else stream
        self._interactive = self._stream.isatty()
        self._started = time.monotonic()
        self._last_output = 0.0
        self._last_phase = None
        self._status = None
        self._metrics = None
        self._pearson = None
        self._step = 0
        self._columns = None
        self._rows = 0
        self._last_row = None
        self._event_since_row = False
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._heartbeat = None
        if self._interactive:
            self._heartbeat = threading.Thread(target=self._tick, name="ptt-terminal-status", daemon=True)
            self._heartbeat.start()

    def _tick(self) -> None:
        while not self._stop.wait(1.0):
            with self._lock:
                if self._status is not None:
                    self._write_status(force=True)

    def _elapsed(self) -> str:
        seconds = int(time.monotonic() - self._started)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _number(value, digits=3) -> str:
        return "?" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"

    def _position(self) -> str:
        if self._structure_budget:
            return f"graph {self._structure}/{self._max_steps} | step {self._step}"
        return f"step {self._step}/{self._max_steps}"

    def _metric_status(self) -> str:
        return f"PTT | {self._position()} | {self._pearson_status()}"

    def set_columns(self, columns) -> None:
        """The history quantities of this run, which define the diagnostic table."""
        self._columns = columns

    def set_validation_stop(self, window: int, min_gain: float) -> None:
        """Show the validation plateau test instead of the Pearson target."""
        self._validation_stop = (window, min_gain)

    @classmethod
    def _acceptance_status(cls, values) -> str:
        return "[" + ", ".join(cls._number(float(value), 2) for value in values) + "]"

    def _pearson_status(self) -> str:
        if self._validation_stop is None:
            return f"Pearson {self._number(self._pearson)}/{self._target_pearson:.3f}"
        from adabmDCA.ptt.training import validation_window_gain

        window, min_gain = self._validation_stop
        values = [self._validation[step] for step in sorted(self._validation)]
        gain = validation_window_gain(values, window)
        plateau = (f"plateau check from step {2 * window}" if gain is None
                   else f"plateau gain {gain:+.1e} (stops < {min_gain:g})")
        latest = values[-1] if values else None
        return f"Pearson {self._number(self._pearson)} | val LL {self._number(latest, 4)} | {plateau}"

    def _record_validation(self, event: TrainingProgress) -> None:
        """Keep the validation LL of each update, as the trainer's plateau test sees it."""
        if self._validation_stop is None or event.epoch == 0:
            return
        # A recovery rolls the history back: forget updates from the restored step on.
        for step in [step for step in self._validation if step >= event.epoch]:
            del self._validation[step]
        value = event.metrics.get("LL_val")
        if value is not None and math.isfinite(value):
            self._validation[event.epoch] = value

    def _write_status(self, *, force=False, phase=None) -> None:
        if self._status is None:
            return
        now = time.monotonic()
        phase_changed = phase is not None and phase != self._last_phase
        if not force and not phase_changed and now - self._last_output < (0.2 if self._interactive else 5.0):
            return
        line = f"{self._status} | elapsed {self._elapsed()}"
        if self._interactive:
            width = shutil.get_terminal_size((120, 20)).columns
            self._stream.write("\r\x1b[2K" + line[:max(1, width - 1)])
        else:
            self._stream.write(line + "\n")
        self._stream.flush()
        self._last_output = now
        if phase is not None:
            self._last_phase = phase

    def _set_status(self, line: str, *, phase: str, force=False) -> None:
        with self._lock:
            self._status = line
            self._write_status(force=force, phase=phase)

    def _lines(self, lines: list[str], *, live_status: str | None = None, phase: str | None = None) -> None:
        """Write lasting lines above the live status and restore that status."""
        with self._lock:
            if live_status is not None:
                self._status = live_status
            if self._interactive:
                self._stream.write("\r\x1b[2K")
            self._stream.writelines(line + "\n" for line in lines)
            self._stream.flush()
            self._last_output = 0.0
            if self._interactive or live_status is not None:
                self._write_status(force=True, phase=phase)

    def _event(self, step, message: str) -> None:
        self._event_since_row = True
        self._lines([event_line(step, message, self._columns or ())])

    def _table_row(self, event: TrainingProgress) -> list[str]:
        """The diagnostic table row of an update: every update while updates take a second
        or more, otherwise about one per second, and always right after an event."""
        now = time.monotonic()
        if self._columns is None or not (
            self._last_row is None or self._event_since_row or now - self._last_row >= 1.0
        ):
            return []
        record = {**event.metrics, "Epochs": event.epoch}
        lines = [table_header(self._columns)] if self._rows % HEADER_EVERY == 0 else []
        lines.append(table_row(record, self._columns))
        self._rows += 1
        self._last_row = now
        self._event_since_row = False
        return lines

    def __call__(self, event: TrainingProgress) -> None:
        with self._lock:
            self._metrics = event
            self._pearson = event.metrics.get("Pearson")
            self._step = event.gradient_steps
            self._structure = event.structure_steps
            self._record_validation(event)
            live_status = f"{self._metric_status()} | optimizing"
            rows = self._table_row(event) if self._diagnostic else []
            if rows:
                self._lines(rows, live_status=live_status, phase="optimization")
            else:
                self._set_status(live_status, phase="optimization")

    def on_stage(self, event: StageProgress) -> None:
        details = event.details
        self._step = event.gradient_steps
        if "pearson" in details:
            self._pearson = details["pearson"]
        if event.kind == "progress":
            step = f"{self._position()} | {self._pearson_status()}"
            if "probe_step" in details:
                step += f" | checking checkpoint {details['probe_step']}"
            phase = self._PHASES.get(event.stage, event.stage.replace("_", " "))
            unit = "samples" if event.stage == "reservoir_collect" else "rounds"
            live_status = f"PTT | {step} | {phase} {event.current}/{event.total} {unit}"
            diagnostic = []
            if self._diagnostic and "chains" in details:
                diagnostic.append(f"chains {details['chains']}")
            if self._diagnostic and "spacing_current" in details:
                diagnostic.append(f"spacing {details['spacing_current']}/{details['spacing_total']}")
            if self._diagnostic and details.get("tau_int") is not None:
                diagnostic.append(f"τint {details['tau_int']:.1f}")
            if self._diagnostic and details.get("tau_exp") is not None:
                diagnostic.append(f"τexp {details['tau_exp']:.1f}")
            if self._diagnostic and details.get("endpoint_fresh") is not None:
                diagnostic.append(f"old {details['ladder_old']:.3f} fresh {details['endpoint_fresh']:.3f}")
            if self._diagnostic and details.get("acceptance") is not None:
                diagnostic.append(f"acc {self._acceptance_status(details['acceptance'])}")
            report_diagnostic = (event.current in (0, event.total) or "tau_exp" in details
                                 or ("endpoint_fresh" in details and event.current % 100 == 0))
            if diagnostic and report_diagnostic:
                self._lines(["  · " + " | ".join(diagnostic)], live_status=live_status, phase=event.stage)
            else:
                self._set_status(live_status, phase=event.stage,
                                 force=event.current in (0, event.total) or "tau_exp" in details)
            return
        if event.kind == "event":
            if event.stage == "ptt_checkpoint_start":
                self._checkpoint_started = time.monotonic()
                self._set_status(f"{self._metric_status()} | saving checkpoint", phase="checkpoint", force=True)
            elif event.stage == "ptt_checkpoint_done":
                duration = time.monotonic() - self._checkpoint_started
                if self._diagnostic:
                    self._event(event.gradient_steps, f"checkpoint saved in {duration:.1f}s")
                else:
                    self._set_status(f"{self._metric_status()} | checkpoint saved", phase="checkpoint_done", force=True)
                self._set_status(f"{self._metric_status()} | optimizing", phase="optimization")
            return
        message = describe_event(event.stage, details)
        if event.stage == "ptt_mixing" and "status" not in details:
            self._set_status(f"{self._metric_status()} | starting mixing check", phase=event.stage, force=True)
            return
        if event.stage in ("ptt_optimization", "ptt_equilibration") or message is None:
            self._set_status(f"{self._metric_status()} | {message or event.stage.replace('_', ' ')}",
                             phase=event.stage)
            return
        failed = event.stage == "ptt_mixing" and details.get("status") != "converged"
        if self._diagnostic or event.stage in NOTABLE_EVENTS or failed:
            self._event(details.get("model_version", event.gradient_steps), message)
        self._set_status(f"{self._metric_status()} | {message.split(':')[0]}", phase=event.stage, force=True)

    def close(self) -> None:
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=2.0)
        with self._lock:
            if self._interactive and self._status is not None:
                self._stream.write("\r\x1b[2K")
                self._stream.flush()


def _print_initialization(args: argparse.Namespace, initialized) -> None:
    """Print requested options together with resolved input statistics."""
    from adabmDCA.scripts._frontend import print_configuration

    training = initialized.training
    validation = initialized.validation
    limits = initialized.config.limits
    values = {
        "input": training.source or "<memory>",
        "validation": None if validation is None else validation.source or "<memory>",
        "output": args.output,
        "model": initialized.config.model_type,
        "sampler": initialized.config.sampler,
        "alphabet": initialized.config.alphabet,
        "training sequences": training.retained_sequences,
        "sequence length": training.sequence_length,
        "alphabet states": training.num_states,
        "effective sequences (M_eff)": f"{training.effective_sequences:.6g}",
        "invalid sequences removed": training.removed_invalid,
        "duplicates removed": training.removed_duplicates,
        "validation sequences": None if validation is None else validation.retained_sequences,
        "validation effective sequences": None if validation is None else f"{validation.effective_sequences:.6g}",
        "target Pearson": initialized.config.target_pearson,
        "number of chains": initialized.n_chains,
        "sweeps per step": initialized.config.n_sweeps,
        "PTT optimizer": None if initialized.config.ptt is None else initialized.config.ptt.optimizer,
        "PTT mixing method": None if initialized.config.ptt is None else initialized.config.ptt.mixing_method,
        "maximum gradient steps": limits.max_gradient_steps,
        "maximum structure steps": limits.max_structure_steps,
        "effective pseudocount": f"{initialized.effective_pseudocount:.6g}",
        "device": initialized.device,
        "dtype": initialized.dtype,
    }
    print_configuration(values)


def run(args: argparse.Namespace, *, progress: Any = None, stage_progress: Any = None, on_initialized: Any = None):
    """Execute training from parsed CLI arguments and return its result."""
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)
    from dataclasses import fields

    from adabmDCA.exceptions import InputValidationError
    from adabmDCA.api.training import train_model
    from adabmDCA.ptt import PTTConfig

    ptt_options = {key: getattr(args, "ptt_" + key, None)
                   for key in (field.name for field in fields(PTTConfig))}
    ptt_options = {key: value for key, value in ptt_options.items() if value is not None}
    if args.strategy != "ptt" and (ptt_options or args.ptt_resume):
        raise InputValidationError("PTT tuning and resume options require --strategy ptt.")
    if args.strategy == "ptt" and args.val is not None and "validation_stop" not in ptt_options:
        # With a validation set, stop on its log-likelihood unless --no-ptt-validation-stop is given.
        ptt_options["validation_stop"] = True
    return train_model(
        args.data,
        model_type=args.model,
        ptt=PTTConfig(**ptt_options) if args.strategy == "ptt" else None,
        ptt_resume=args.ptt_resume,
        validation_path=args.val,
        weights_path=args.weights,
        output_dir=args.output,
        label=args.label,
        initial_params_path=args.path_params,
        initial_chains_path=args.path_chains,
        alphabet=args.alphabet,
        learning_rate=args.lr,
        n_sweeps=args.nsweeps,
        sampler=args.sampler,
        n_chains=args.nchains,
        target_pearson=args.target,
        max_epochs=args.nepochs,
        max_gradient_steps=args.max_gradient_steps,
        max_structure_steps=args.max_structure_steps,
        checkpoint_interval=args.checkpoint_interval,
        pseudocount=args.pseudocount,
        l2_regularization=args.l2_reg,
        seed=args.seed,
        clustering_seqid=args.clustering_seqid,
        no_reweighting=args.no_reweighting,
        activation_steps=args.gsteps,
        activation_fraction=args.factivate,
        target_density=args.density,
        decimation_rate=args.drate,
        device=args.device,
        dtype=args.dtype,
        use_wandb=args.wandb,
        progress=progress,
        stage_progress=stage_progress,
        on_initialized=on_initialized,
    )


def main(args: argparse.Namespace | None = None) -> int:
    args = create_parser().parse_args() if args is None else args
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)

    from adabmDCA.scripts._frontend import print_completion, print_header

    print_header(f"Training {args.model} model")
    epoch_limit = DEFAULT_PTT_MAX_EPOCHS if args.strategy == "ptt" and args.nepochs is DEFAULT_MAX_EPOCHS else args.nepochs
    max_steps = (
        args.max_gradient_steps or epoch_limit
        if args.model == "bmDCA" or (args.model == "edgeDCA" and args.strategy == "ptt")
        else args.max_structure_steps or epoch_limit
    )
    if args.no_progress:
        renderer = None
    elif args.strategy == "ptt":
        renderer = _PTTProgressRenderer(
            target_pearson=args.target,
            max_steps=max_steps,
            diagnostic=getattr(args, "diagnostic", False),
            structure_budget=args.model == "eaDCA",
        )
    else:
        renderer = _TrainingProgressRenderer(
            model_type=args.model,
            target_pearson=args.target,
            max_steps=max_steps,
        )
    def initialized(info) -> None:
        _print_initialization(args, info)
        if isinstance(renderer, _PTTProgressRenderer):
            from adabmDCA.training_log import columns_for_config

            renderer.set_columns(columns_for_config(info.config, validation=info.validation is not None))
            ptt = info.config.ptt
            if ptt is not None and ptt.validation_stop:
                renderer.set_validation_stop(ptt.validation_window, ptt.validation_min_gain)

    try:
        result = run(args, progress=renderer,
                     stage_progress=(renderer.on_stage if args.strategy == "ptt" and renderer is not None else None),
                     on_initialized=initialized)
    finally:
        if renderer is not None:
            renderer.close()

    serialized = result.save_bundle(args.output, label=args.label)
    artifacts = {**result.artifacts, **serialized}
    if getattr(args, "plot_training_logs", False):
        from adabmDCA.plot_training_log import plot_training_history

        plots = plot_training_history(artifacts["history"])
        artifacts["training plots"] = plots["overview"].parent
    print_completion("Training completed successfully.", metrics=_final_summary(result), artifacts=artifacts)
    return 0


def _final_summary(result) -> dict[str, Any]:
    """Key end-of-run figures; everything else is in training.json and history.csv."""
    from adabmDCA.training_log import format_cell

    final = result.final_metrics

    def available(key):
        value = final.get(key)
        return value is not None and math.isfinite(value)

    def pair(name, train_key, validation_key):
        if not available(train_key):
            return {}
        if not available(validation_key):
            return {name: f"{final[train_key]:.4f}"}
        return {f"{name} train / validation": f"{final[train_key]:.4f} / {final[validation_key]:.4f}"}

    sparse = result.config is not None and result.config.model_type != "bmDCA"
    summary = {
        "stop reason": result.stop_reason.replace("_", " ") if result.stop_reason else None,
        "gradient steps": result.gradient_steps,
        "structure steps": result.structure_steps if sparse else None,
        "sweeps": format_cell(result.sweeps, "count"),
        "time": format_cell(final.get("Time"), "clock"),
        **pair("Pearson", "Pearson", "Pearson_val"),
        **pair("log-likelihood per site", "LL_train", "LL_val"),
        "density": format_cell(final.get("Density"), "f4") if sparse else None,
    }
    sampler = result.ptt_sampler
    if sampler is not None:
        state = sampler.training_state or {}
        summary["replicas / checkpoint models"] = f"{sampler.n_active} / {sampler.total_models}"
        summary["pauses / recoveries"] = f"{state.get('pauses', 0)} / {len(state.get('recovery_events', []))}"
    return {key: value for key, value in summary.items() if value is not None}


if __name__ == "__main__":
    raise SystemExit(main())
