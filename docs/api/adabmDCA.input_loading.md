<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/input_loading.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.input_loading`
Shared, side-effect-free loading of alignments and sequence weights.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/input_loading.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_alignment`

```python
load_alignment(
    source: 'AlignmentInput',
    config: 'AlignmentLoadConfig | None' = None
) → LoadedAlignment
```

Parse, validate, filter, and deduplicate one alignment.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/input_loading.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_sequence_weights`

```python
load_sequence_weights(
    source: 'WeightInput | None',
    loaded_alignment: 'LoadedAlignment',
    no_reweighting: 'bool',
    clustering_seqid: 'float',
    device: 'device',
    dtype: 'dtype',
    allow_negative: 'bool' = False,
    require_positive_sum: 'bool' = True
) → Tensor
```

Load or calculate weights and align them with retained sequences.

``allow_negative`` and ``require_positive_sum`` are intended for signed experimental adjustment vectors; ordinary statistical weights should keep their safe defaults.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/input_loading.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentLoadConfig`
Policy applied after parsing an alignment.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    alphabet: 'str' = 'protein',
    invalid_sequences: 'InvalidSequencePolicy' = 'error',
    remove_duplicates: 'bool' = False,
    expected_length: 'int | None' = None,
    format: 'AlignmentFormat' = 'auto',
    alignment_index: 'int | None' = None,
    normalize_dots: 'bool' = True
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/input_loading.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LoadedAlignment`
Validated alignment plus provenance of filtering transformations.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    alignment: 'Alignment',
    tokens: 'str',
    retained_indices: 'tuple[int, ]',
    dropped_indices: 'tuple[int, ]' = (),
    duplicate_indices: 'tuple[int, ]' = (),
    original_size: 'int' = 0
) → None
```






---

#### <kbd>property</kbd> encoded_sequences







---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/input_loading.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, object]
```

Return a JSON-serializable filtering and provenance report.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
