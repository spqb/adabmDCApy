<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.checkpoint`
Training checkpoints and the versioned human-readable training log. 

**Global Variables**
---------------
- **DEFAULT_CHECKPOINT_INTERVAL**
- **LOG_FORMAT_VERSION**
- **HISTORY_KEYS**
- **LOG_COLUMNS**


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Checkpoint`
Save model state and write a version-2 training log. 

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    file_paths: 'dict[str, str]',
    tokens: 'str',
    metadata: 'Mapping[str, Any]',
    use_wandb: 'bool' = False,
    config: 'TrainingConfig | None' = None
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `begin_stage`

```python
begin_stage(stage: 'str', metadata: 'dict[str, Any]') → None
```

Record a phase boundary and its progress-table header. 

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check`

```python
check(updates: 'int') → bool
```

Return whether this update requires a persisted checkpoint. 

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `finish`

```python
finish(status: 'str', summary: 'Mapping[str, Any]') → None
```

Append exactly one terminal status section. 

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log`

```python
log(record: 'dict[str, Any]') → None
```

Write a record without lifecycle counters for direct callers. 

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_with_context`

```python
log_with_context(record: 'dict[str, Any]', counters: 'Any | None') → None
```

Write one metrics record with stage and lifecycle counters. 

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(
    params: 'dict[str, Tensor]',
    mask: 'Tensor',
    chains: 'Tensor',
    log_weights: 'Tensor'
) → None
```

Save parameters and chains. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
