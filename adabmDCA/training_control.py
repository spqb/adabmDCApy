"""Shared lifecycle primitives for DCA training routines.

The numerical algorithms live in :mod:`adabmDCA.training`.  This module owns
the algorithm-independent parts of a run: counters, limits, history,
progress reporting, cancellation, and checkpoint scheduling.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from adabmDCA.api.exceptions import InputValidationError

HISTORY_KEYS = (
    "Epochs",
    "Pearson",
    "Slope",
    "LL_train",
    "LL_val",
    "Pearson_val",
    "Slope_val",
    "ESS",
    "Entropy",
    "Density",
    "Time",
)


class StopReason(str, Enum):
    """Reason why a training strategy stopped."""

    TARGET_PEARSON = "target_pearson"
    TARGET_DENSITY = "target_density"
    MAX_GRADIENT_STEPS = "max_gradient_steps"
    MAX_STRUCTURE_STEPS = "max_structure_steps"
    CANCELLED = "cancelled"


class TrainingCancelled(RuntimeError):
    """Internal cancellation signal raised by the shared controller."""


@dataclass(frozen=True)
class TrainingLimits:
    """Independent budgets for numerical and graph-structure updates."""

    max_gradient_steps: int | None = None
    max_structure_steps: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_gradient_steps", self.max_gradient_steps),
            ("max_structure_steps", self.max_structure_steps),
        ):
            if value is not None and value < 1:
                raise InputValidationError(
                    f"{name} must be positive when provided.",
                    details={"name": name, "value": value},
                )


@dataclass
class TrainingCounters:
    """Monotonic counters shared by all training strategies."""

    gradient_steps: int = 0
    structure_steps: int = 0
    sweeps: int = 0


@dataclass(frozen=True)
class TrainingMetrics:
    """Common metrics emitted after a meaningful training step."""

    pearson: Any
    slope: Any
    ll_train: Any
    ll_val: Any
    pearson_val: Any
    slope_val: Any
    ess: Any
    entropy: Any
    density: Any
    elapsed_time: float

    def as_record(self, epoch: int) -> dict[str, Any]:
        return {
            "Epochs": epoch,
            "Pearson": self.pearson,
            "Slope": self.slope,
            "LL_train": self.ll_train,
            "LL_val": self.ll_val,
            "Pearson_val": self.pearson_val,
            "Slope_val": self.slope_val,
            "ESS": self.ess,
            "Entropy": self.entropy,
            "Density": self.density,
            "Time": self.elapsed_time,
        }


class CheckpointStore(Protocol):
    """Persistence interface used by the training controller."""

    def log(self, record: dict[str, Any]) -> None: ...

    def check(self, updates: int) -> bool: ...

    def save(self, **snapshot: Any) -> None: ...


RecordObserver = Callable[[dict[str, Any], TrainingCounters], None]
CancellationHook = Callable[[], bool]


class TrainingController:
    """Coordinate one training run without owning its numerical updates."""

    def __init__(
        self,
        *,
        limits: TrainingLimits | None = None,
        checkpoint: CheckpointStore | None = None,
        observer: RecordObserver | None = None,
        is_cancelled: CancellationHook | None = None,
    ) -> None:
        self.limits = limits or TrainingLimits()
        self.checkpoint = checkpoint
        self.observer = observer
        self.is_cancelled = is_cancelled
        self.counters = TrainingCounters()
        self.history: dict[str, list[Any]] = {key: [] for key in HISTORY_KEYS}
        self.stop_reason: StopReason | None = None
        self._finalized = False

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
        record = metrics.as_record(epoch)
        for key in HISTORY_KEYS:
            self.history[key].append(record[key])
        if self.checkpoint is not None:
            self.checkpoint.log(record)
            if snapshot is not None and self.checkpoint.check(epoch):
                self.checkpoint.save(**dict(snapshot))
        if self.observer is not None:
            self.observer(record, self.counters)
        return record

    def finalize(self, snapshot: Mapping[str, Any] | None = None) -> None:
        """Persist the final state once, without duplicating a history row."""
        if self._finalized:
            return
        if self.checkpoint is not None and snapshot is not None:
            self.checkpoint.save(**dict(snapshot))
        self._finalized = True

    def set_stop_reason(self, reason: StopReason) -> StopReason:
        if self.stop_reason is None:
            self.stop_reason = reason
        return self.stop_reason
