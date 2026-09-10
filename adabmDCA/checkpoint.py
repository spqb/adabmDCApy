"""Training checkpoints and the versioned human-readable training log."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import torch
import wandb

from adabmDCA.io import save_chains, save_params
from adabmDCA.training_config import DEFAULT_CHECKPOINT_INTERVAL, TrainingConfig

LOG_FORMAT_VERSION = 2
HISTORY_KEYS = (
    "Epochs", "Pearson", "Slope", "LL_train", "LL_val", "Pearson_val",
    "Slope_val", "ESS", "Entropy", "Density", "Time",
)
LOG_COLUMNS = (
    ("Epochs", "Step", 8),
    ("Stage", "Stage", 14),
    ("Gradient_steps", "Grad_steps", 12),
    ("Structure_steps", "Struct_steps", 13),
    ("Sweeps", "Sweeps", 12),
    ("Pearson", "Pearson", 11),
    ("Slope", "Slope", 11),
    ("LL_train", "LL_train", 12),
    ("LL_val", "LL_val", 12),
    ("Pearson_val", "Pearson_val", 13),
    ("Slope_val", "Slope_val", 12),
    ("ESS", "Chain_ESS_frac", 15),
    ("Entropy", "Entropy", 12),
    ("Density", "Density", 12),
    ("Time", "Elapsed_s", 12),
)


def _display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _cell(value: Any, width: int) -> str:
    if isinstance(value, torch.Tensor):
        value = value.item()
    rendered = f"{value:.6g}" if isinstance(value, float) else str(value)
    return f"{rendered:<{width}}"


class Checkpoint:
    """Save model state and write a version-2 training log."""

    def __init__(
        self,
        file_paths: dict[str, str],
        tokens: str,
        metadata: Mapping[str, Any],
        use_wandb: bool = False,
        config: TrainingConfig | None = None,
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
        self.max_epochs = None if limits is None else (
            limits.max_gradient_steps if config.model_type == "bmDCA" else limits.max_structure_steps
        )
        self.logs = {key: 0 if key == "Epochs" else 0.0 for key in HISTORY_KEYS}
        self._stage = "optimization"
        self._finished = False
        self.wandb = use_wandb
        if self.wandb:
            wandb.init(project="adabmDCA", config=dict(metadata))

        with open(file_paths["log"], "w", encoding="utf-8") as handle:
            handle.write("adabmDCA training log\n")
            handle.write(f"format_version: {LOG_FORMAT_VERSION}\n")
            handle.write(f"created_utc: {datetime.now(timezone.utc).isoformat()}\n")
            for section, values in metadata.items():
                if not isinstance(values, Mapping) or not values:
                    continue
                handle.write(f"\n[{section.upper()}]\n")
                width = max(len(str(name)) for name in values) + 1
                for name, value in values.items():
                    handle.write(f"{name + ':':<{width + 1}} {_display(value)}\n")

    def begin_stage(self, stage: str, metadata: dict[str, Any]) -> None:
        """Record a phase boundary and its progress-table header."""
        self._stage = stage
        with open(self.file_paths["log"], "a", encoding="utf-8") as handle:
            handle.write(f"\n[STAGE {stage.upper()}]\n")
            for name, value in metadata.items():
                handle.write(f"{name}: {_display(value)}\n")
            handle.write("\n")
            handle.write(" ".join(f"{label:<{width}}" for _, label, width in LOG_COLUMNS).rstrip() + "\n")

    def log(self, record: dict[str, Any]) -> None:
        """Write a record without lifecycle counters for direct callers."""
        self.log_with_context(record, None)

    def log_with_context(self, record: dict[str, Any], counters: Any | None) -> None:
        """Write one metrics record with stage and lifecycle counters."""
        for key, value in record.items():
            if key not in self.logs:
                raise ValueError(f"Key {key} not recognized.")
            self.logs[key] = value.item() if isinstance(value, torch.Tensor) else value
        row = {
            **self.logs,
            "Stage": self._stage,
            "Gradient_steps": 0 if counters is None else counters.gradient_steps,
            "Structure_steps": 0 if counters is None else counters.structure_steps,
            "Sweeps": 0 if counters is None else counters.sweeps,
        }
        if self.wandb:
            wandb.log(row)
        with open(self.file_paths["log"], "a", encoding="utf-8") as handle:
            handle.write(" ".join(_cell(row[key], width) for key, _, width in LOG_COLUMNS).rstrip() + "\n")

    def finish(self, status: str, summary: Mapping[str, Any]) -> None:
        """Append exactly one terminal status section."""
        if self._finished:
            return
        with open(self.file_paths["log"], "a", encoding="utf-8") as handle:
            handle.write("\n[END]\n")
            handle.write(f"status: {status}\n")
            for name, value in summary.items():
                handle.write(f"{name}: {_display(value)}\n")
        self._finished = True

    def check(self, updates: int) -> bool:
        """Return whether this update requires a persisted checkpoint."""
        return (updates % self.checkpt_interval == 0) or (updates == self.max_epochs)

    def save(
        self,
        params: dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: torch.Tensor,
        log_weights: torch.Tensor,
    ) -> None:
        """Save parameters and chains."""
        save_params(fname=self.file_paths["params"], params=params, mask=mask, tokens=self.tokens)
        save_chains(
            fname=self.file_paths["chains"],
            chains=chains.argmax(dim=-1),
            tokens=self.tokens,
            log_weights=log_weights,
        )
