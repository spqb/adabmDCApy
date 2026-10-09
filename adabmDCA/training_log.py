"""What a training run records, and how it is shown.

Every run writes three files: ``history.csv`` (one row per update, for
analysis tools), ``events.jsonl`` (one JSON object per event) and the
readable log, a narrow progress table with events as one-line ``»`` entries.
The terminal ``--diagnostic`` output uses the same table and event lines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Phases of a run. Entering one is logged once; returning to the current
# phase after an event is not an event.
PHASES = frozenset({
    "ptt_equilibration", "ptt_optimization", "optimization", "activation", "decimation", "equilibration",
})
# Events worth a lasting terminal line even without --diagnostic.
NOTABLE_EVENTS = frozenset({
    "ptt_lag_pause", "ptt_snapshot_inserted", "ptt_snapshot_replaced", "ptt_reservoir_refreshed",
    "ptt_recovery", "ptt_learning_rate",
})
HEADER_EVERY = 50


@dataclass(frozen=True)
class Column:
    """One history quantity: its key in ``TrainingResult.history``, its CSV
    name, and, if it is part of the progress table, its label and format."""

    key: str
    name: str
    label: str | None = None
    style: str = "g"
    width: int = 8


def history_columns(*, model_type: str, ptt_optimizer: str | None, validation: bool) -> tuple[Column, ...]:
    """Quantities recorded at every update, in CSV order."""
    sparse = model_type != "bmDCA"
    ptt = ptt_optimizer is not None
    columns = [Column("Epochs", "step", "step", "int", 7)]
    if sparse:
        columns += [Column("Gradient_steps", "gradient_steps", "grad", "int", 7), Column("Stage", "stage")]
    columns += [
        Column("Sweeps", "sweeps", "sweeps", "count", 7),
        Column("Time", "time_s", "time", "clock", 8),
        Column("Pearson", "pearson", "pearson", "f4", 7),
        Column("Slope", "slope", None if ptt else "slope", "f3", 6),
    ]
    if validation:
        columns += [Column("Pearson_val", "pearson_val", "val", "f4", 7), Column("Slope_val", "slope_val")]
    if sparse:
        columns.append(Column("Density", "density", "density", "f4", 7))
    if ptt:
        columns.append(Column("LL_train", "ll_train", "ll/L", "f3", 6))
        if validation:
            columns.append(Column("LL_val", "ll_val", "ll_val", "f3", 6))
        columns += [
            Column("Entropy", "entropy"),
            Column("logZ", "logz"),
            Column("ptt_replicas", "replicas", "reps", "int", 4),
            Column("ptt_total_models", "models"),
            Column("ptt_acceptance", "acceptance_min", "acc", "f2", 4),
        ]
        if model_type == "edgeDCA":
            columns += [
                Column("edge_pseudocount", "edge_pseudocount", "pseudo", "g3", 7),
                Column("edge_i", "edge_i"),
                Column("edge_j", "edge_j"),
                Column("edge_new", "edge_new"),
                Column("ptt_predicted_kl", "kl"),
            ]
        else:
            if model_type == "eaDCA":
                columns += [
                    Column("Structure_steps", "structure_steps", "graph", "int", 5),
                    Column("steps_on_graph", "steps_on_graph"),
                    Column("activated_entries", "activated_entries"),
                    Column("active_entries", "active_entries"),
                ]
            columns += [
                Column("bias_learning_rate", "lr_bias", "lr_h", "g3", 7),
                Column("coupling_learning_rate", "lr_coupling", "lr_J", "g3", 7),
            ]
        if ptt_optimizer == "adaptive":
            columns += (
                ([] if model_type == "edgeDCA" else [Column("ptt_predicted_kl", "kl")]) + [
                Column("ptt_lag_drift", "lag_drift", "lag", "f2", 4),
                Column("ptt_lag_drift_tail", "lag_drift_tail", "tail", "f2", 4),
            ])
    return tuple(columns)


def columns_for_config(config: Any, *, validation: bool) -> tuple[Column, ...]:
    ptt = getattr(config, "ptt", None)
    return history_columns(model_type=config.model_type, ptt_optimizer=None if ptt is None else ptt.optimizer,
                           validation=validation)


def _scalar(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError, RuntimeError):
            return value
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def csv_cell(value: Any) -> str:
    """Plain CSV value: empty for missing, full precision for numbers."""
    value = _scalar(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.10g}" if math.isfinite(value) else ("nan" if math.isnan(value) else str(value))
    return str(value)


def csv_row(record: dict[str, Any], columns: tuple[Column, ...]) -> list[str]:
    return [csv_cell(record.get(column.key)) for column in columns]


def history_rows(history: dict[str, list[Any]], columns: tuple[Column, ...]) -> list[list[str]]:
    """CSV rows of an in-memory history, tolerating quantities absent from older runs."""
    count = len(history.get("Epochs", ()))
    return [
        [csv_cell(history[column.key][index]) if column.key in history and index < len(history[column.key]) else ""
         for column in columns]
        for index in range(count)
    ]


def _count(value: float) -> str:
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:.0f}"


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def format_cell(value: Any, style: str) -> str:
    value = _scalar(value)
    if not _finite(value):
        return "-" if value is None or isinstance(value, float) else str(value)
    if style == "int":
        return str(int(value))
    if style == "count":
        return _count(value)
    if style == "clock":
        return _clock(value)
    if style.startswith("f"):
        return f"{value:.{int(style[1:])}f}"
    if style.startswith("g"):
        return f"{value:.{int(style[1:])}g}" if len(style) > 1 else f"{value:.6g}"
    return str(value)


def table_columns(columns: tuple[Column, ...]) -> tuple[Column, ...]:
    return tuple(column for column in columns if column.label is not None)


def table_header(columns: tuple[Column, ...]) -> str:
    return "  ".join(f"{column.label:>{column.width}}" for column in table_columns(columns))


def table_row(record: dict[str, Any], columns: tuple[Column, ...]) -> str:
    return "  ".join(f"{format_cell(record.get(column.key), column.style):>{column.width}}"
                     for column in table_columns(columns))


def event_line(step: Any, message: str, columns: tuple[Column, ...]) -> str:
    """An event aligned under the step column of the progress table."""
    width = table_columns(columns)[0].width if table_columns(columns) else 7
    return f"»{step!s:>{width - 1}}  {message}"


def _number(value: Any, digits: int = 3) -> str:
    value = _scalar(value)
    return f"{value:.{digits}f}" if _finite(value) else "?"


def _rounds(value: Any) -> str:
    return "unresolved" if value is None else f"{value} rounds"


def _generic(stage: str, details: dict[str, Any]) -> str:
    shown = []
    for key, value in details.items():
        value = _scalar(value)
        if value is None or isinstance(value, (list, tuple, dict)):
            continue
        shown.append(f"{key} {value:.4g}" if isinstance(value, float) else f"{key} {value}")
    name = stage.removeprefix("ptt_").replace("_", " ")
    return f"{name}: {', '.join(shown)}" if shown else name


def describe_event(stage: str, details: dict[str, Any]) -> str | None:
    """One-line description of a training event, or ``None`` if it only marks a start."""
    if stage == "ptt_mixing":
        if "status" not in details:
            return None
        message = f"mixing check {str(details['status']).replace('_', ' ')}"
        reservoir = details.get("reservoir_renewal")
        immobile = (reservoir or details).get("immobile") or ()
        stuck = sum(entry.get("immobile", 0) for entry in immobile if entry.get("blocking"))
        suffix = f"; {stuck} configurations cannot swap toward the bottom (only local moves free them)" if stuck else ""
        if reservoir is not None:
            return (f"reservoir collection {str(details['status']).replace('_', ' ')}: renewal warmup "
                    f"{_rounds(reservoir.get('warmup_rounds'))}, spacing {_rounds(reservoir.get('spacing_rounds'))}"
                    f"{suffix}")
        if details.get("method") == "renewal":
            return (f"{message}: renewal warmup {_rounds(details.get('warmup_rounds'))}, "
                    f"stationary renewal {_rounds(details.get('renewal_rounds'))}, "
                    f"trapped endpoint fraction {_number(details.get('trapped_fraction'))}{suffix}")
        return (f"{message}: τint {_number(details.get('tau_int'), 1)}, τexp {_number(details.get('tau_exp'), 1)}, "
                f"{details.get('rounds')} measured / {details.get('required_rounds')} required rounds")
    if stage in ("ptt_snapshot_stored", "ptt_snapshot_inserted", "ptt_snapshot_replaced", "ptt_checkpoint_flagged"):
        action = ("checkpoint flagged" if stage == "ptt_checkpoint_flagged"
                  else "snapshot " + stage.removeprefix("ptt_snapshot_"))
        reason = " (pause)" if details.get("reason") == "lag" else ""
        acceptance = details.get("acceptance")
        swap = f", swap acceptance {_number(acceptance, 2)}" if _finite(_scalar(acceptance)) else ""
        return f"{action}{reason}{swap}"
    if stage == "ptt_reservoir_refreshed":
        message = (f"reservoir refreshed: {details.get('reservoir_size')} samples, "
                   f"{details.get('active_replicas')} active replicas")
        if details.get("renewal_spacing_rounds") is not None:
            message += (f", warmup {details.get('renewal_warmup_rounds')} rounds, "
                        f"spacing {details.get('renewal_spacing_rounds')} rounds, "
                        f"min batch fresh {_number(details.get('min_batch_fresh'))}")
        return message
    if stage == "ptt_lag_pause":
        triggers = {key.removeprefix("trigger_"): _scalar(value) for key, value in details.items()
                    if key.startswith("trigger_") and _finite(_scalar(value))}
        signal = max(triggers, key=triggers.get).replace("_", " ") if triggers else "lag"
        action = {"insert": "inserted a snapshot", "replace": "replaced the replica below"}.get(
            details.get("ladder_action"), "ladder unchanged")
        health = "" if details.get("healthy", True) else "; reservoir refresh failed, pause discarded"
        return (f"pause: {signal} lag {_number(details.get('lag_before'), 2)} → {_number(details.get('lag_after'), 2)}, "
                f"{action}, {details.get('rounds', 0)} extra rounds{health}")
    if stage == "ptt_validation_plateau":
        window = details.get("window")
        return (f"validation plateau: median validation log-likelihood per site of the last {window} updates "
                f"{details.get('gain'):+.1e} against the {window} before")
    if stage == "ptt_activation":
        message = (f"graph {details.get('structure_step')}: activated {details.get('activated')} couplings "
                   f"({details.get('active_entries')} active), density {_number(details.get('density_before'), 4)} "
                   f"→ {_number(details.get('density_after'), 4)}")
        if "significant" in details:
            message += (f"; {details['significant']} of {details.get('candidates')} candidates significant, "
                        f"first-step KL {details.get('predicted_kl', 0.0):.2g}/{details.get('kl_budget', 0.0):.2g}")
        return message
    if stage == "ptt_graph_converged":
        return (f"graph converged: no inactive coupling differs significantly from the data "
                f"({details.get('active_entries')} active)")
    if stage == "ptt_graph_complete":
        return f"graph complete: all {details.get('active_entries')} couplings active"
    if stage == "ptt_recovery":
        return "mixing budget exceeded; searching saved checkpoints"
    if stage == "ptt_learning_rate":
        return (f"restored step {details.get('restored_step')}; learning rates "
                f"h {details.get('bias_learning_rate', details.get('learning_rate')):.3g}, "
                f"J {details.get('coupling_learning_rate', details.get('learning_rate')):.3g}")
    if stage == "checkpoint_saved":
        return "checkpoint saved"
    if stage == "resumed":
        return f"resumed from step {details.get('step')}"
    if stage == "ptt_equilibration":
        return "initializing replicas"
    if stage == "ptt_optimization":
        return "optimizing"
    return _generic(stage, details)


def event_step(details: dict[str, Any], counters: Any, *, sparse: bool) -> int:
    """The update an event belongs to: the model version it concerns, else the current step."""
    if "model_version" in details and details["model_version"] is not None:
        return int(details["model_version"])
    if counters is None:
        return 0
    return int(counters.structure_steps if sparse else counters.gradient_steps)
