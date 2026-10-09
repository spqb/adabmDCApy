<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.checkpoint`
Training checkpoints and the versioned human-readable training log.

**Global Variables**
---------------
- **DEFAULT_CHECKPOINT_INTERVAL**
- **HEADER_EVERY**
- **PHASES**
- **LOG_FORMAT_VERSION**


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Checkpoint`
Save model state and write the training history, events and readable log.

``history.csv`` gains one row per update and is rewritten whenever the history changes retroactively (resume, recovery). ``events.jsonl`` gains one JSON object per event. The readable log holds the run metadata, a narrow progress table with events as ``»`` lines, and the end summary.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    file_paths: 'dict[str, str]',
    tokens: 'str',
    metadata: 'Mapping[str, Any]',
    use_wandb: 'bool' = False,
    config: 'TrainingConfig | None' = None,
    resume: 'bool' = False
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `begin_stage`

```python
begin_stage(stage: 'str', metadata: 'dict[str, Any]') → None
```

Record a phase change or an event; returning to the current phase is not recorded.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bind_counters`

```python
bind_counters(counters: 'Any') → None
```

Share the controller's counters, which place events in the run.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check`

```python
check(updates: 'int') → bool
```

Return whether this update requires a persisted checkpoint.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `event`

```python
event(name: 'str', details: 'Mapping[str, Any]') → None
```

Record one event in ``events.jsonl`` and, if describable, as a ``»`` log line.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L184"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `finish`

```python
finish(status: 'str', summary: 'Mapping[str, Any]') → None
```

Append exactly one terminal status section.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log`

```python
log(record: 'dict[str, Any]') → None
```

Write a record for direct callers.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_with_context`

```python
log_with_context(record: 'dict[str, Any]', counters: 'Any | None') → None
```

Append one update to the history table and the progress table.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `resumed`

```python
resumed(step: 'int') → None
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `rewrite_history`

```python
rewrite_history(history: 'Mapping[str, list[Any]]') → None
```

Replace ``history.csv`` after the history changed retroactively (resume, recovery).

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L198"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(
    params: 'dict[str, Tensor]',
    mask: 'Tensor',
    chains: 'Tensor',
    ptt_sampler=None
) → None
```

Save parameters and chains.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
