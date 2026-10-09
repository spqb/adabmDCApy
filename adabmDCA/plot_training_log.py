"""Plot the training curves of a run from its history table.

Usage:
    adabmDCA plot-training-log <history.csv | training log> [--output-dir <dir>]

The history table (``history.csv``, or ``<label>_history.csv``) is written
during training, one row per update. The run's log, ``events.jsonl`` and
``training.json`` next to it, when present, supply the target Pearson, the lag
tolerance and events (pauses, recoveries, the validation stop) marked on the
curves. Tables written before log format 4, with ``TrainingResult.history``
column names, are read too.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from adabmDCA.training_config import DEFAULT_TARGET_PEARSON

TRAIN = "#31688E"
VALIDATION = "#E76F51"
ACCENT = "#35B779"
SECOND = "#8E6BB8"
MUTED = "#7A7F85"
PAUSE = "#8E6BB8"
RECOVERY = "#C0392B"
STOP = "#263238"


def _history_names() -> dict[str, str]:
    """``TrainingResult.history`` keys of older tables mapped to ``history.csv`` names."""
    from adabmDCA.training_log import history_columns

    return {column.key: column.name
            for column in history_columns(model_type="eaDCA", ptt_optimizer="adaptive", validation=True)}


def _run_files(path: Path) -> dict[str, Path]:
    """The history table of a run and the files written next to it."""
    if path.suffix != ".csv":
        history = None
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("history:"):
                    history = path.parent / line.split(":", 1)[1].strip()
                    break
                if line.startswith("["):
                    break
        if history is None:
            raise ValueError(
                f"'{path}' does not name its history table; pass the run's history.csv "
                "(logs written before format 4 have none, but the run's history.csv does)."
            )
        path = history
    prefix = path.name[: -len("history.csv")] if path.name.endswith("history.csv") else ""
    stem = prefix[:-1] if prefix else "adabmDCA"
    return {
        "history": path,
        "log": path.parent / f"{stem}.log",
        "events": path.parent / f"{prefix}events.jsonl",
        "summary": path.parent / f"{prefix}training.json",
        "label": prefix[:-1] or "training",
    }


def _log_settings(path: Path) -> dict[str, str]:
    """``key: value`` settings from the metadata sections of a training log."""
    settings = {}
    if not path.is_file():
        return settings
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip() in ("[PROGRESS]", "[END]"):
                break
            if ":" in line and not line.startswith("["):
                key, value = (part.strip() for part in line.split(":", 1))
                settings.setdefault(key, value)
    return settings


def read_training_history(path: str | Path):
    """Read a run's history table and its context.

    Returns ``(history, context)``: a DataFrame with ``history.csv`` column
    names, and a dictionary with the run ``label``, ``model``,
    ``target_pearson``, ``lag_tolerance`` and ``events`` (a list of dicts).
    """
    import pandas as pd

    files = _run_files(Path(path))
    history = pd.read_csv(files["history"]).rename(columns=_history_names())
    settings = _log_settings(files["log"])
    context = {
        "label": files["label"],
        # Older tables recorded a density of 1 for bmDCA too.
        "model": settings.get("model") or (
            "sparse" if "density" in history and (history["density"].dropna() != 1).any() else "bmDCA"
        ),
        "target_pearson": float(settings.get("target_pearson", DEFAULT_TARGET_PEARSON)),
        "lag_tolerance": float(settings["lag_tolerance"]) if "lag_tolerance" in settings else None,
        "events": [],
    }
    if files["summary"].is_file():
        config = json.loads(files["summary"].read_text(encoding="utf-8"))["data"].get("config") or {}
        context["target_pearson"] = float(config.get("target_pearson", context["target_pearson"]))
        context["model"] = config.get("model_type", context["model"])
        if config.get("ptt"):
            context["lag_tolerance"] = float(config["ptt"].get("lag_tolerance", context["lag_tolerance"] or 0.25))
    if files["events"].is_file():
        with open(files["events"], encoding="utf-8") as handle:
            context["events"] = [json.loads(line) for line in handle if line.strip()]
    return history, context


def parse_training_log(log_path: str):
    """Metadata and history of a training log, under ``TrainingResult.history`` keys.

    Kept for ``adabmDCA.parse_log_file``. Format-4 logs are read through their
    history table; older logs from their progress tables.
    """
    import numpy as np

    keys = {name: key for key, name in _history_names().items()}
    wanted = ("Epochs", "Pearson", "Slope", "LL_train", "LL_val", "Pearson_val", "Slope_val", "Entropy", "Density",
              "Time", "Stage", "Gradient_steps", "Structure_steps", "Sweeps")
    metadata, section, lines = {}, "header", []
    with open(log_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            lines.append(line)
            if not line or line == "adabmDCA training log":
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().lower()
                continue
            if section == "progress" or ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            metadata[key if section == "header" else f"{section}.{key}"] = value
    data = {key: [] for key in wanted}
    if "history" in metadata and int(metadata.get("format_version", 0)) >= 4:
        history, _ = read_training_history(log_path)
        for name in history.columns:
            key = keys.get(name, name)
            if key in data:
                data[key] = history[name].tolist()
    else:
        aliases = {"Step": "Epochs", "Grad_steps": "Gradient_steps", "Struct_steps": "Structure_steps",
                   "Elapsed_s": "Time"}
        header = []
        for line in lines:
            if line.startswith("[") and line.endswith("]"):
                header = []
            elif line.startswith("Step "):
                header = line.split()
            elif header and len(line.split()) == len(header):
                values = line.split()
                try:
                    float(values[0])
                except ValueError:
                    continue
                for label, value in zip(header, values, strict=True):
                    key = aliases.get(label, label)
                    if key in data:
                        data[key].append(value if key == "Stage" else float(value))
    return metadata, {key: np.asarray(values, dtype=object if key == "Stage" else float)
                      for key, values in data.items()}


def _events(context, *names):
    return [event for event in context["events"] if event.get("event") in names]


def _mark(ax, context, *, pauses=False, recoveries=False):
    """Vertical markers for the validation stop and, optionally, pauses and recoveries."""
    for event in _events(context, "ptt_validation_plateau"):
        ax.axvline(event["step"], color=STOP, lw=1.0, ls="--", zorder=1)
    if pauses:
        for event in _events(context, "ptt_lag_pause"):
            ax.axvline(event["step"], color=PAUSE, lw=0.9, alpha=0.6, zorder=1)
    if recoveries:
        for event in _events(context, "ptt_recovery"):
            ax.axvline(event["step"], color=RECOVERY, lw=0.9, alpha=0.7, zorder=1)


def _line(ax, x, y, **kwargs):
    ax.plot(x, y, lw=kwargs.pop("lw", 1.4), **kwargs)


def _has(history, *names):
    return all(name in history and history[name].notna().any() for name in names)


def _panel_pearson(ax, history, context):
    x = history["step"]
    _line(ax, x, history["pearson"], color=TRAIN, label="training")
    if _has(history, "pearson_val"):
        _line(ax, x, history["pearson_val"], color=VALIDATION, label="validation")
    target = context["target_pearson"]
    ax.axhline(target, color=ACCENT, ls="--", lw=1.2, label=f"target {target:.2f}", zorder=1)
    _mark(ax, context)
    ax.set_ylabel("Pearson of connected correlations")
    return "Pearson correlation"


def _panel_slope(ax, history, context):
    x = history["step"]
    _line(ax, x, history["slope"], color=TRAIN, label="training")
    if _has(history, "slope_val"):
        _line(ax, x, history["slope_val"], color=VALIDATION, label="validation")
    ax.axhline(1.0, color=MUTED, ls="--", lw=1.1, label="ideal", zorder=1)
    _mark(ax, context)
    ax.set_ylabel("slope")
    return "Correlation slope"


def _panel_loglikelihood(ax, history, context):
    x = history["step"]
    _line(ax, x, history["ll_train"], color=TRAIN, label="training", lw=1.1)
    if _has(history, "ll_val"):
        _line(ax, x, history["ll_val"], color=VALIDATION, label="validation", lw=1.1)
    # Early updates fall steeply; show the part of training where the curves are compared.
    values = history[[name for name in ("ll_train", "ll_val") if _has(history, name)]].to_numpy()
    tail = values[len(values) // 10:]
    low, high = float(tail.min()), float(tail.max())
    pad = 0.08 * (high - low) if high > low else 0.01
    ax.set_ylim(low - pad, high + pad)
    _mark(ax, context, pauses=True, recoveries=True)
    ax.set_ylabel("log-likelihood per site")
    return "Log-likelihood per site"


def _panel_entropy(ax, history, context):
    _line(ax, history["step"], history["entropy"], color=ACCENT, label="entropy")
    _mark(ax, context)
    ax.set_ylabel("entropy (nats)")
    return "Model entropy"


def _panel_density(ax, history, context):
    _line(ax, history["step"], history["density"], color=ACCENT, label="graph density")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("fraction of couplings")
    return "Graph density"


def _panel_ladder(ax, history, context):
    x = history["step"]
    ax.step(x, history["replicas"], where="post", color=TRAIN, lw=1.4, label="active replicas")
    if _has(history, "models"):
        ax.step(x, history["models"], where="post", color=SECOND, lw=1.4, label="checkpoint models")
    _mark(ax, context, pauses=True, recoveries=True)
    ax.set_ylabel("count")
    return "PTT ladder"


def _panel_acceptance(ax, history, context):
    _line(ax, history["step"], history["acceptance_min"], color=TRAIN, lw=0.9, label="smallest swap acceptance")
    ax.axhline(0.25, color=MUTED, ls="--", lw=1.1, label="snapshot target 0.25", zorder=1)
    ax.set_ylim(-0.02, 1.02)
    _mark(ax, context, pauses=True, recoveries=True)
    ax.set_ylabel("acceptance")
    return "Swap acceptance"


def _panel_rates(ax, history, context):
    x = history["step"]
    for name, color, label in (("lr_bias", TRAIN, "fields"), ("lr_coupling", VALIDATION, "couplings")):
        values = history[name]
        ax.plot(x, values, color=color, lw=0.5, alpha=0.2)
        ax.plot(x, values.rolling(50, min_periods=1).mean(), color=color, lw=1.6, label=f"{label} (50-update mean)")
    ax.set_ylim(bottom=0)
    _mark(ax, context, recoveries=True)
    ax.set_ylabel("learning rate")
    return "Learning rates"


def _panel_kl(ax, history, context):
    x = history["step"]
    ax.plot(x, history["kl"], color=MUTED, lw=0.5, alpha=0.35)
    ax.plot(x, history["kl"].rolling(50, min_periods=1).mean(), color=STOP, lw=1.5, label="50-update mean")
    ax.set_ylim(bottom=0)
    _mark(ax, context)
    ax.set_ylabel("predicted KL per update")
    return "Update size"


def _panel_lag(ax, history, context):
    x = history["step"]
    _line(ax, x, history["lag_drift"], color=TRAIN, label="drift", lw=0.9)
    if _has(history, "lag_drift_tail"):
        _line(ax, x, history["lag_drift_tail"], color=SECOND, label="drift tail", lw=0.9)
    if context["lag_tolerance"] is not None and context["lag_tolerance"] < 10:
        ax.axhline(context["lag_tolerance"], color=MUTED, ls="--", lw=1.1,
                   label=f"pause at {context['lag_tolerance']:g}", zorder=1)
    _mark(ax, context, pauses=True)
    ax.set_ylabel("lag (population std)")
    return "Chains behind the model"


PANELS = (
    ("pearson", ("pearson",), _panel_pearson),
    ("slope", ("slope",), _panel_slope),
    ("loglikelihood", ("ll_train",), _panel_loglikelihood),
    ("entropy", ("entropy",), _panel_entropy),
    ("density", ("density",), _panel_density),
    ("ladder", ("replicas",), _panel_ladder),
    ("acceptance", ("acceptance_min",), _panel_acceptance),
    ("learning_rates", ("lr_bias", "lr_coupling"), _panel_rates),
    ("kl", ("kl",), _panel_kl),
    ("lag", ("lag_drift",), _panel_lag),
)


def plot_training_history(path: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    """Plot every quantity a run recorded; return the written files by panel name.

    ``path`` is the run's ``history.csv`` (or its format-4 log). Each panel is
    saved as ``<label>_<panel>.png`` and all of them together as
    ``<label>_overview.png``, by default in ``<label>_plots`` next to the table.
    """
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from adabmDCA.plot import _DIAGNOSTIC_TEXT, _style_diagnostic_axis

    history, context = read_training_history(path)
    if history.empty:
        raise ValueError(f"'{path}' contains no training updates.")
    label = context["label"]
    folder = Path(output_dir) if output_dir is not None else Path(_run_files(Path(path))["history"]).parent / f"{label}_plots"
    folder.mkdir(parents=True, exist_ok=True)
    sparse = context["model"] != "bmDCA"
    unit = "graph update" if sparse else "gradient update"
    panels = [(name, draw) for name, needed, draw in PANELS
              if _has(history, *needed) and (sparse or name != "density")]

    def finish(ax, title):
        _style_diagnostic_axis(ax)
        ax.set_title(title, color=_DIAGNOSTIC_TEXT, loc="left", fontsize=11, pad=8)
        ax.set_xlabel(unit)
        if ax.get_legend() is None and ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=9, loc="best")

    written = {}
    for name, draw in panels:
        fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=160)
        finish(ax, draw(ax, history, context))
        fig.tight_layout()
        written[name] = folder / f"{label}_{name}.png"
        fig.savefig(written[name], facecolor="white")
        plt.close(fig)
    columns = 2 if len(panels) > 1 else 1
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 3.6 * rows), dpi=130, squeeze=False)
    for ax, (name, draw) in zip(axes.flat, panels):
        finish(ax, draw(ax, history, context))
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    notes = []
    if _events(context, "ptt_validation_plateau"):
        notes.append("dashed line: validation stop")
    if _events(context, "ptt_lag_pause"):
        notes.append("purple lines: pauses")
    if _events(context, "ptt_recovery"):
        notes.append("red lines: mixing recoveries")
    model = context["model"] + (" with PTT" if _has(history, "replicas") else "")
    title = f"{label} · {model} · {int(history['step'].iloc[-1])} {unit}s"
    fig.suptitle(title + (f"  ({'; '.join(notes)})" if notes else ""), x=0.01, ha="left",
                 color=_DIAGNOSTIC_TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    written["overview"] = folder / f"{label}_overview.png"
    fig.savefig(written["overview"], facecolor="white")
    plt.close(fig)
    return written


def create_parser():
    parser = argparse.ArgumentParser(
        description="Plot the training curves of a run from its history table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("history", type=str,
                        help="The run's history.csv (or <label>_history.csv), or its training log.")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                        help="Directory for the plots (default: <label>_plots next to the history table).")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    if not os.path.exists(args.history):
        parser.error(f"History file '{args.history}' not found.")
    try:
        written = plot_training_history(args.history, args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    print(f"Plots saved to: {written['overview'].parent}")
    for path in written.values():
        print(f"  • {path.name}")


if __name__ == "__main__":
    main()
