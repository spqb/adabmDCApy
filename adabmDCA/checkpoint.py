"""Training checkpoints and the versioned human-readable training log."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import wandb

from adabmDCA.serialization import to_jsonable
from adabmDCA.io import save_chains, save_params
from adabmDCA.training_config import DEFAULT_CHECKPOINT_INTERVAL, TrainingConfig
from adabmDCA.training_log import (
    HEADER_EVERY,
    PHASES,
    columns_for_config,
    csv_row,
    describe_event,
    event_line,
    event_step,
    history_columns,
    history_rows,
    table_header,
    table_row,
)

LOG_FORMAT_VERSION = 4


def _display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _metadata_lines(values: Mapping[str, Any], indent: str = "") -> list[str]:
    width = max(len(str(name)) for name in values) + 1
    lines = []
    for name, value in values.items():
        if isinstance(value, Mapping) and value:
            lines.append(f"{indent}{name}:\n")
            lines += _metadata_lines(value, indent + "  ")
        else:
            lines.append(f"{indent}{name + ':':<{width + 1}} {_display(value)}\n")
    return lines


class Checkpoint:
    """Save model state and write the training history, events and readable log.

    ``history.csv`` gains one row per update and is rewritten whenever the
    history changes retroactively (resume, recovery). ``events.jsonl`` gains
    one JSON object per event. The readable log holds the run metadata, a
    narrow progress table with events as ``»`` lines, and the end summary.
    """

    def __init__(
        self,
        file_paths: dict[str, str],
        tokens: str,
        metadata: Mapping[str, Any],
        use_wandb: bool = False,
        config: TrainingConfig | None = None,
        resume: bool = False,
    ) -> None:
        self.file_paths = file_paths
        self.tokens = tokens
        self.config = config
        self.checkpt_interval = (
            config.resolved_checkpoint_interval
            if config is not None
            else int(metadata.get("checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL))
        )
        limits = config.limits if config is not None else None
        # ``check`` receives gradient steps from bmDCA and every PTT model (PTT
        # eaDCA included, whose graph budget is enforced by the training loop)
        # and structure steps from the other sparse PCD routines.
        self.max_epochs = None if limits is None else (
            limits.max_gradient_steps if config.model_type == "bmDCA" or config.ptt is not None else limits.max_structure_steps
        )
        validation = metadata.get("validation data") is not None
        self.columns = (columns_for_config(config, validation=validation) if config is not None
                        else history_columns(model_type="bmDCA", ptt_optimizer=None, validation=validation))
        self._events_use_structure_steps = (config is not None and config.model_type != "bmDCA"
                                            and config.ptt is None)
        self._counters = None
        self._phase = None
        self._rows = 0
        self._progress_started = False
        self._finished = False
        self.wandb = use_wandb
        if self.wandb:
            wandb.init(project="adabmDCA", config=dict(metadata))

        mode = "a" if resume else "w"
        with open(file_paths["log"], mode, encoding="utf-8") as handle:
            if resume:
                handle.write("\n")
            handle.write("adabmDCA training log\n")
            handle.write(f"format_version: {LOG_FORMAT_VERSION}\n")
            handle.write(f"created_utc: {datetime.now(timezone.utc).isoformat()}\n")
            handle.write(f"history: {Path(file_paths['history']).name}\n")
            handle.write(f"events: {Path(file_paths['events']).name}\n")
            for section, values in metadata.items():
                if not isinstance(values, Mapping) or not values:
                    continue
                handle.write(f"\n[{section.upper()}]\n")
                handle.writelines(_metadata_lines(values))
        with open(file_paths["history"], "w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(column.name for column in self.columns)
        if not resume:
            open(file_paths["events"], "w", encoding="utf-8").close()

    def bind_counters(self, counters: Any) -> None:
        """Share the controller's counters, which place events in the run."""
        self._counters = counters

    def _append_log(self, lines: list[str]) -> None:
        with open(self.file_paths["log"], "a", encoding="utf-8") as handle:
            if not self._progress_started:
                handle.write("\n[PROGRESS]\n")
                self._progress_started = True
            handle.writelines(line + "\n" for line in lines)

    def event(self, name: str, details: Mapping[str, Any]) -> None:
        """Record one event in ``events.jsonl`` and, if describable, as a ``»`` log line."""
        step = event_step(dict(details), self._counters, sparse=self._events_use_structure_steps)
        document = {"event": name, "step": step, "utc": datetime.now(timezone.utc).isoformat()}
        if self._counters is not None:
            document["sweeps"] = self._counters.sweeps
        document.update(to_jsonable({key: value for key, value in details.items() if key not in document}))
        with open(self.file_paths["events"], "a", encoding="utf-8") as handle:
            handle.write(json.dumps(document) + "\n")
        message = describe_event(name, dict(details))
        if message is not None:
            self._append_log([event_line(step, message, self.columns)])

    def begin_stage(self, stage: str, metadata: dict[str, Any]) -> None:
        """Record a phase change or an event; returning to the current phase is not recorded."""
        if stage in PHASES:
            if stage == self._phase and not metadata:
                return
            self._phase = stage
        self.event(stage, metadata)

    def resumed(self, step: int) -> None:
        self.event("resumed", {"step": step})

    def log(self, record: dict[str, Any]) -> None:
        """Write a record for direct callers."""
        self.log_with_context(record, None)

    def log_with_context(self, record: dict[str, Any], counters: Any | None) -> None:
        """Append one update to the history table and the progress table."""
        record = {key: value.item() if isinstance(value, torch.Tensor) else value for key, value in record.items()}
        if self.wandb:
            wandb.log(record)
        with open(self.file_paths["history"], "a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(csv_row(record, self.columns))
        lines = []
        if self._rows % HEADER_EVERY == 0:
            lines.append(table_header(self.columns))
        lines.append(table_row(record, self.columns))
        self._rows += 1
        self._append_log(lines)

    def rewrite_history(self, history: Mapping[str, list[Any]]) -> None:
        """Replace ``history.csv`` after the history changed retroactively (resume, recovery)."""
        temporary = Path(self.file_paths["history"]).with_suffix(".csv.tmp")
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(column.name for column in self.columns)
            writer.writerows(history_rows(dict(history), self.columns))
        os.replace(temporary, self.file_paths["history"])

    def finish(self, status: str, summary: Mapping[str, Any]) -> None:
        """Append exactly one terminal status section."""
        if self._finished:
            return
        with open(self.file_paths["log"], "a", encoding="utf-8") as handle:
            handle.write("\n[END]\n")
            handle.write(f"status: {status}\n")
            handle.writelines(f"{name}: {_display(value)}\n" for name, value in summary.items())
        self._finished = True

    def check(self, updates: int) -> bool:
        """Return whether this update requires a persisted checkpoint."""
        return (updates % self.checkpt_interval == 0) or (updates == self.max_epochs)

    def save(
        self,
        params: dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: torch.Tensor,
        ptt_sampler=None,
    ) -> None:
        """Save parameters and chains."""
        if ptt_sampler is not None:
            ptt_sampler.save_archive(self.file_paths["ptt_archive"])
        save_params(fname=self.file_paths["params"], params=params, mask=mask, tokens=self.tokens)
        save_chains(
            fname=self.file_paths["chains"],
            chains=chains.argmax(dim=-1),
            tokens=self.tokens,
        )
        self.event("checkpoint_saved", {})
