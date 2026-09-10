<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.training_control`
Shared lifecycle primitives for DCA training routines.

The numerical algorithms live in :mod:`adabmDCA.training`.  This module owns the algorithm-independent parts of a run: counters, limits, history, progress reporting, cancellation, and checkpoint scheduling.

**Global Variables**
---------------
- **HISTORY_KEYS**


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StopReason`
Reason why a training strategy stopped.





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingCancelled`
Internal cancellation signal raised by the shared controller.





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingLimits`
Independent budgets for numerical and graph-structure updates.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    max_gradient_steps: 'int | None' = None,
    max_structure_steps: 'int | None' = None
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingCounters`
Monotonic counters shared by all training strategies.

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingMetrics`
Common metrics emitted after a meaningful training step.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    pearson: 'Any',
    slope: 'Any',
    ll_train: 'Any',
    ll_val: 'Any',
    pearson_val: 'Any',
    slope_val: 'Any',
    ess: 'Any',
    entropy: 'Any',
    density: 'Any',
    elapsed_time: 'float'
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `as_record`

```python
as_record(epoch: 'int') → dict[str, Any]
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CheckpointStore`
Persistence interface used by the training controller.




---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check`

```python
check(updates: 'int') → bool
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log`

```python
log(record: 'dict[str, Any]') → None
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(**snapshot: 'Any') → None
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingController`
Coordinate one training run without owning its numerical updates.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    limits: 'TrainingLimits | None' = None,
    checkpoint: 'CheckpointStore | None' = None,
    observer: 'RecordObserver | None' = None,
    is_cancelled: 'CancellationHook | None' = None
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_gradient_steps`

```python
add_gradient_steps(count: 'int', sweeps_per_step: 'int' = 0) → None
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_structure_step`

```python
add_structure_step(sweeps: 'int' = 0) → None
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `begin_stage`

```python
begin_stage(stage: 'str', **metadata: 'Any') → None
```

Publish a phase change without coupling algorithms to a renderer.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_cancellation`

```python
check_cancellation() → None
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `finalize`

```python
finalize(snapshot: 'Mapping[str, Any] | None' = None) → None
```

Persist the final state once, without duplicating a history row.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L170"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `gradient_limit_reached`

```python
gradient_limit_reached() → bool
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remaining_gradient_steps`

```python
remaining_gradient_steps() → int | None
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_snapshot`

```python
save_snapshot(snapshot: 'Mapping[str, Any]') → None
```

Persist an explicit phase-boundary snapshot when configured.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L210"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_stop_reason`

```python
set_stop_reason(reason: 'StopReason') → StopReason
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_control.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `structure_limit_reached`

```python
structure_limit_reached() → bool
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
