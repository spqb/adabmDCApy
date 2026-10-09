<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.training_control`
Shared lifecycle primitives for DCA training routines.

The numerical algorithms live in :mod:`adabmDCA.training`.  This module owns the algorithm-independent parts of a run: counters, limits, history, progress reporting, cancellation, and checkpoint scheduling.

**Global Variables**
---------------
- **HISTORY_KEYS**


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StopReason`
Reason why a training strategy stopped.





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingCancelled`
Internal cancellation signal raised by the shared controller.





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingLimits`
Step budgets of a training run; ``None`` means no limit.



**Attributes:**

 - <b>`max_gradient_steps`</b>:  Largest number of accepted parameter updates.
 - <b>`max_structure_steps`</b>:  Largest number of graph activations or decimations.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    max_gradient_steps: 'int | None' = None,
    max_structure_steps: 'int | None' = None
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingCounters`
Progress counters of a training run.



**Attributes:**

 - <b>`gradient_steps`</b>:  Accepted parameter updates.
 - <b>`structure_steps`</b>:  Graph activations or decimations.
 - <b>`sweeps`</b>:  Monte Carlo sweeps performed, including diagnostics.
 - <b>`stage`</b>:  Current training phase.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    gradient_steps: 'int' = 0,
    structure_steps: 'int' = 0,
    sweeps: 'int' = 0,
    stage: 'str' = 'optimization'
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingMetrics`
Common metrics emitted after a meaningful training step.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    pearson: 'Any',
    slope: 'Any',
    pearson_val: 'Any',
    slope_val: 'Any',
    density: 'Any',
    elapsed_time: 'float',
    ll_train: 'Any' = None,
    ll_val: 'Any' = None,
    entropy: 'Any' = None,
    extra: 'dict[str, Any]' = <factory>
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `as_record`

```python
as_record(epoch: 'int') → dict[str, Any]
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CheckpointStore`
Persistence interface used by the training controller.




---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check`

```python
check(updates: 'int') → bool
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log`

```python
log(record: 'dict[str, Any]') → None
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(**snapshot: 'Any') → None
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StageProgress`
Transient stage update; separate from committed metric records.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    stage: 'str',
    kind: 'str',
    gradient_steps: 'int',
    current: 'int | None' = None,
    total: 'int | None' = None,
    details: 'dict[str, Any]' = <factory>
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingController`
Coordinate one training run without owning its numerical updates.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    limits: 'TrainingLimits | None' = None,
    checkpoint: 'CheckpointStore | None' = None,
    observer: 'RecordObserver | None' = None,
    stage_observer: 'StageObserver | None' = None,
    is_cancelled: 'CancellationHook | None' = None
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L176"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_gradient_steps`

```python
add_gradient_steps(count: 'int', sweeps_per_step: 'int' = 0) → None
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_structure_step`

```python
add_structure_step(sweeps: 'int' = 0) → None
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L184"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `begin_stage`

```python
begin_stage(stage: 'str', **metadata: 'Any') → None
```

Publish a phase change without coupling algorithms to a renderer.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_cancellation`

```python
check_cancellation() → None
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `finalize`

```python
finalize(snapshot: 'Mapping[str, Any] | None' = None) → None
```

Persist the final state once, without duplicating a history row.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `gradient_limit_reached`

```python
gradient_limit_reached() → bool
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L269"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `history_changed`

```python
history_changed() → None
```

Propagate a retroactive history change (resume, rollback) to the saved history.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `record`

```python
record(
    metrics: 'TrainingMetrics',
    epoch: 'int',
    snapshot: 'Mapping[str, Any] | None' = None
) → dict[str, Any]
```

Append and publish exactly one metrics record.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remaining_gradient_steps`

```python
remaining_gradient_steps() → int | None
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `report_stage_event`

```python
report_stage_event(stage: 'str', **details: 'Any') → None
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `report_stage_progress`

```python
report_stage_progress(
    stage: 'str',
    current: 'int',
    total: 'int',
    **details: 'Any'
) → None
```

Report completed work without modifying log rows or counters.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_snapshot`

```python
save_snapshot(snapshot: 'Mapping[str, Any]') → None
```

Persist an explicit phase-boundary snapshot when configured.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L284"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_stop_reason`

```python
set_stop_reason(reason: 'StopReason') → StopReason
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `structure_limit_reached`

```python
structure_limit_reached() → bool
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
