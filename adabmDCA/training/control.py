"""Shared lifecycle primitives for DCA training routines.

The numerical algorithms live in :mod:`adabmDCA.training`.  This module owns
the algorithm-independent parts of a run: counters, limits, history,
progress reporting, cancellation, and checkpoint scheduling.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from adabmDCA._validation import validate_integer

HISTORY_KEYS = (
    "Epochs",
    "Pearson",
    "Slope",
    "Pearson_val",
    "Slope_val",
    "Density",
    "Time",
)
TrainingHistory = dict[str, list[Any]]


class StopReason(str, Enum):
    """Reason why a training strategy stopped."""

    TARGET_PEARSON = "target_pearson"
    TARGET_DENSITY = "target_density"
    MAX_GRADIENT_STEPS = "max_gradient_steps"
    MAX_STRUCTURE_STEPS = "max_structure_steps"
    VALIDATION_PLATEAU = "validation_plateau"
    GRAPH_CONVERGED = "graph_converged"
    CANCELLED = "cancelled"


class TrainingCancelled(RuntimeError):
    """Internal cancellation signal raised by the shared controller."""


@dataclass(frozen=True)
class TrainingLimits:
    """Step budgets of a training run; ``None`` means no limit.

    Attributes:
        max_gradient_steps: Largest number of accepted parameter updates.
        max_structure_steps: Largest number of graph activations or decimations.
    """

    max_gradient_steps: int | None = None
    max_structure_steps: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_gradient_steps", self.max_gradient_steps),
            ("max_structure_steps", self.max_structure_steps),
        ):
            if value is not None:
                validate_integer(name, value)


@dataclass
class TrainingCounters:
    """Progress counters of a training run.

    Attributes:
        gradient_steps: Accepted parameter updates.
        structure_steps: Graph activations or decimations.
        sweeps: Monte Carlo sweeps performed, including diagnostics.
        stage: Current training phase.
    """

    gradient_steps: int = 0
    structure_steps: int = 0
    sweeps: int = 0
    stage: str = "optimization"


@dataclass(frozen=True)
class TrainingMetrics:
    """Common metrics emitted after a meaningful training step."""

    pearson: Any
    slope: Any
    pearson_val: Any
    slope_val: Any
    density: Any
    elapsed_time: float
    ll_train: Any = None
    ll_val: Any = None
    entropy: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_record(self, epoch: int) -> dict[str, Any]:
        record = {
            "Epochs": epoch,
            "Pearson": self.pearson,
            "Slope": self.slope,
            "Pearson_val": self.pearson_val,
            "Slope_val": self.slope_val,
            "Density": self.density,
            "Time": self.elapsed_time,
        }
        if self.ll_train is not None:
            record["LL_train"] = self.ll_train
        if self.ll_val is not None:
            record["LL_val"] = self.ll_val
        if self.entropy is not None:
            record["Entropy"] = self.entropy
        return {**record, **self.extra}


class CheckpointStore(Protocol):
    """Persistence interface used by the training controller."""

    def log(self, record: dict[str, Any]) -> None: ...

    def check(self, updates: int) -> bool: ...

    def save(self, **snapshot: Any) -> None: ...


RecordObserver = Callable[[dict[str, Any], TrainingCounters], None]
CancellationHook = Callable[[], bool]


@dataclass(frozen=True)
class StageProgress:
    """Transient stage update; separate from committed metric records."""

    stage: str
    kind: str
    gradient_steps: int
    current: int | None = None
    total: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


StageObserver = Callable[[StageProgress], None]


class TrainingController:
    """Coordinate one training run without owning its numerical updates."""

    def __init__(
        self,
        *,
        limits: TrainingLimits | None = None,
        checkpoint: CheckpointStore | None = None,
        observer: RecordObserver | None = None,
        stage_observer: StageObserver | None = None,
        is_cancelled: CancellationHook | None = None,
    ) -> None:
        self.limits = limits or TrainingLimits()
        self.checkpoint = checkpoint
        self.observer = observer
        self.stage_observer = stage_observer
        self.is_cancelled = is_cancelled
        self.counters = TrainingCounters()
        if checkpoint is not None and hasattr(checkpoint, "bind_counters"):
            checkpoint.bind_counters(self.counters)
        self.history: TrainingHistory = {key: [] for key in HISTORY_KEYS}
        self.stop_reason: StopReason | None = None
        self._finalized = False
        self._saved_at = None

    def check_cancellation(self) -> None:
        if self.is_cancelled is not None and self.is_cancelled():
            self.stop_reason = StopReason.CANCELLED
            raise TrainingCancelled("Model training was cancelled by the caller.")

    def add_gradient_steps(self, count: int, *, sweeps_per_step: int = 0) -> None:
        self.counters.gradient_steps += count
        self.counters.sweeps += count * sweeps_per_step

    def add_structure_step(self, *, sweeps: int = 0) -> None:
        self.counters.structure_steps += 1
        self.counters.sweeps += sweeps

    def begin_stage(self, stage: str, **metadata: Any) -> None:
        """Publish a phase change without coupling algorithms to a renderer."""
        self.counters.stage = stage
        if self.checkpoint is not None:
            begin_stage = getattr(self.checkpoint, "begin_stage", None)
            if begin_stage is not None:
                begin_stage(stage, metadata)
        if self.stage_observer is not None:
            self.stage_observer(StageProgress(stage, "start", self.counters.gradient_steps, details=metadata))

    def report_stage_progress(self, stage: str, current: int, total: int, **details: Any) -> None:
        """Report completed work without modifying log rows or counters."""
        if self.stage_observer is not None:
            self.stage_observer(StageProgress(stage, "progress", self.counters.gradient_steps,
                                              current=current, total=total, details=details))

    def report_stage_event(self, stage: str, **details: Any) -> None:
        if self.stage_observer is not None:
            self.stage_observer(StageProgress(stage, "event", self.counters.gradient_steps, details=details))

    def save_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Persist an explicit phase-boundary snapshot when configured."""
        if self.checkpoint is not None:
            ptt_snapshot = snapshot.get("ptt_sampler") is not None
            if ptt_snapshot:
                self.report_stage_event("ptt_checkpoint_start")
            self._save(snapshot)
            if ptt_snapshot:
                self.report_stage_event("ptt_checkpoint_done")

    def _position(self) -> tuple[int, int, int, int]:
        counters = self.counters
        return counters.gradient_steps, counters.structure_steps, counters.sweeps, len(self.history["Epochs"])

    def _save(self, snapshot: Mapping[str, Any]) -> None:
        self.checkpoint.save(**dict(snapshot))
        self._saved_at = self._position()

    def remaining_gradient_steps(self) -> int | None:
        limit = self.limits.max_gradient_steps
        if limit is None:
            return None
        return max(0, limit - self.counters.gradient_steps)

    def gradient_limit_reached(self) -> bool:
        limit = self.limits.max_gradient_steps
        return limit is not None and self.counters.gradient_steps >= limit

    def structure_limit_reached(self) -> bool:
        limit = self.limits.max_structure_steps
        return limit is not None and self.counters.structure_steps >= limit

    def record(
        self,
        metrics: TrainingMetrics,
        *,
        epoch: int,
        snapshot: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append and publish exactly one metrics record."""
        self.check_cancellation()
        record = {
            **metrics.as_record(epoch),
            "Sweeps": self.counters.sweeps,
            "Gradient_steps": self.counters.gradient_steps,
            "Structure_steps": self.counters.structure_steps,
            "Stage": self.counters.stage,
        }
        existing_rows = len(self.history["Epochs"])
        for key, value in record.items():
            if key not in self.history:
                self.history[key] = [None] * existing_rows
            self.history[key].append(value)
        if self.checkpoint is not None:
            log_with_context = getattr(self.checkpoint, "log_with_context", None)
            if log_with_context is None:
                self.checkpoint.log(record)
            else:
                log_with_context(record, self.counters)
            if snapshot is not None and self.checkpoint.check(epoch):
                self._save(snapshot)
        if self.observer is not None:
            self.observer(record, self.counters)
        return record

    def history_changed(self) -> None:
        """Propagate a retroactive history change (resume, rollback) to the saved history."""
        rewrite = getattr(self.checkpoint, "rewrite_history", None)
        if rewrite is not None:
            rewrite(self.history)

    def finalize(self, snapshot: Mapping[str, Any] | None = None) -> None:
        """Persist the final state once, without duplicating a history row."""
        if self._finalized:
            return
        # A checkpoint saved at this very point already holds the final state.
        if self.checkpoint is not None and snapshot is not None and self._saved_at != self._position():
            self.save_snapshot(snapshot)
        self._finalized = True

    def set_stop_reason(self, reason: StopReason) -> StopReason:
        if self.stop_reason is None:
            self.stop_reason = reason
        return self.stop_reason
