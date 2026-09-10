<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.preprocessing`
Pure and file-oriented multiple-sequence-alignment preprocessing.

**Global Variables**
---------------
- **RESULT_SCHEMA_VERSION**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `remove_insertions`

```python
remove_insertions(alignment: 'Alignment') → Alignment
```

Remove dots and lowercase insertion residues from every sequence.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `filter_gap_fraction`

```python
filter_gap_fraction(
    alignment: 'Alignment',
    max_gap_fraction: 'float' = 0.2,
    gap_token: 'str' = '-'
) → AlignmentFilterResult
```

Remove sequences whose gap fraction is greater than the threshold.

A sequence with a gap fraction exactly equal to the threshold is retained.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `preprocess_alignment`

```python
preprocess_alignment(
    input_path: 'str | Path',
    output_path: 'str | Path | None' = None,
    input_format: 'AlignmentFormat' = 'auto',
    alignment_index: 'int | None' = None,
    config: 'AlignmentProcessingConfig | None' = None,
    remove_insertions: 'bool | None' = None,
    max_gap_fraction: 'float | None' = None,
    remove_duplicates: 'bool | None' = None,
    alphabet: 'str | None' = None,
    gap_token: 'str | None' = None,
    line_width: 'int' = 0
) → AlignmentProcessingResult
```

Read, explicitly transform, optionally validate, and write an MSA.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentProcessingConfig`
Explicit transformations applied by :func:`preprocess_alignment`.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    remove_insertions: 'bool' = False,
    max_gap_fraction: 'float | None' = None,
    remove_duplicates: 'bool' = False,
    alphabet: 'str | None' = None,
    gap_token: 'str' = '-'
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentProcessingReport`
Auditable summary of an alignment-processing operation.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    input_sequences: 'int',
    output_sequences: 'int',
    input_length: 'int',
    output_length: 'int',
    removed_insertion_characters: 'int' = 0,
    normalized_gap_characters: 'int' = 0,
    removed_for_gap_fraction: 'int' = 0,
    removed_as_duplicates: 'int' = 0,
    max_gap_fraction: 'float | None' = None,
    removed_names: 'tuple[str, ]' = ()
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, object]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentFilterResult`
Filtered alignment plus masks and per-sequence gap fractions.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    alignment: 'Alignment',
    keep_mask: 'tuple[bool, ]',
    gap_fractions: 'tuple[float, ]',
    removed_names: 'tuple[str, ]',
    report: 'AlignmentProcessingReport'
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentProcessingResult`
Processed alignment, provenance mask, report, and optional output.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    alignment: 'Alignment',
    keep_mask: 'tuple[bool, ]',
    report: 'AlignmentProcessingReport',
    output_path: 'Path | None' = None
) → None
```











---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
