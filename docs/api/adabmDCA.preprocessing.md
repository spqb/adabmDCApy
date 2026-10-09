<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.preprocessing`
Pure and file-oriented multiple-sequence-alignment preprocessing.

**Global Variables**
---------------
- **RESULT_SCHEMA_VERSION**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `remove_insertions`

```python
remove_insertions(alignment: 'Alignment') → Alignment
```

Remove dots and lowercase insertion residues from every sequence.



**Args:**

 - <b>`alignment`</b>:  Profile alignment with insertion states in lowercase.



**Returns:**

 - <b>`A new `</b>: class:`Alignment` containing only the match columns.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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



**Args:**

 - <b>`alignment`</b>:  Alignment to filter.
 - <b>`max_gap_fraction`</b>:  Largest allowed fraction of gaps, in ``[0, 1]``.
 - <b>`gap_token`</b>:  Gap symbol.



**Returns:**

 - <b>`An `</b>: class:`AlignmentFilterResult`.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L248"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    unknown_tokens: 'str | None' = None,
    line_width: 'int' = 0
) → AlignmentProcessingResult
```

Clean a raw alignment for DCA: read, transform, filter and optionally write it.

Steps, each optional: remove insertions (or normalize dots to ``"-"``), check tokens against an alphabet, drop gappy sequences, drop duplicates. Keyword options override the corresponding :class:`AlignmentProcessingConfig` fields.



**Args:**

 - <b>`input_path`</b>:  FASTA or Stockholm file.
 - <b>`output_path`</b>:  FASTA file to write, or ``None``.
 - <b>`input_format`</b>:  ``"auto"``, ``"fasta"`` or ``"stockholm"``.
 - <b>`alignment_index`</b>:  Alignment to read from a multi-alignment Stockholm file.
 - <b>`config`</b>:  Base settings; see :class:`AlignmentProcessingConfig`.
 - <b>`remove_insertions`</b>:  Overrides ``config.remove_insertions``.
 - <b>`max_gap_fraction`</b>:  Overrides ``config.max_gap_fraction``.
 - <b>`remove_duplicates`</b>:  Overrides ``config.remove_duplicates``.
 - <b>`alphabet`</b>:  Overrides ``config.alphabet``.
 - <b>`gap_token`</b>:  Overrides ``config.gap_token``.
 - <b>`unknown_tokens`</b>:  Overrides ``config.unknown_tokens``.
 - <b>`line_width`</b>:  Largest sequence-line length of the output; ``0`` means no wrapping.



**Returns:**

 - <b>`An `</b>: class:`AlignmentProcessingResult`.



**Raises:**

 - <b>`InputValidationError`</b>:  If an option is invalid, or unknown tokens are  found with ``unknown_tokens="error"``.



**Example:**
 ``` result = preprocess_alignment("RF00379.sto", output_path="clean.fasta",```
    ...                               remove_insertions=True, max_gap_fraction=0.2)
    >>> result.report.output_sequences



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentProcessingConfig`
Transformations applied by :func:`preprocess_alignment`, in this order.



**Attributes:**

 - <b>`remove_insertions`</b>:  Delete lowercase residues and dots (insertion states  of profile alignments); otherwise dots become ``"-"``.
 - <b>`max_gap_fraction`</b>:  Remove sequences with a larger fraction of gaps, or ``None``.
 - <b>`remove_duplicates`</b>:  Keep only the first copy of identical sequences.
 - <b>`alphabet`</b>:  Check tokens against this alphabet (``"auto"`` detects it), or  ``None`` to skip the check.
 - <b>`gap_token`</b>:  Gap symbol counted by ``max_gap_fraction``.
 - <b>`unknown_tokens`</b>:  With ``alphabet``: ``"gap"`` replaces unknown tokens with  ``"-"``, ``"remove"`` drops their sequences, ``"error"`` raises.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    remove_insertions: 'bool' = False,
    max_gap_fraction: 'float | None' = None,
    remove_duplicates: 'bool' = False,
    alphabet: 'str | None' = None,
    gap_token: 'str' = '-',
    unknown_tokens: 'str' = 'gap'
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentProcessingReport`
What :func:`preprocess_alignment` changed.



**Attributes:**

 - <b>`input_sequences`</b>:  Sequences read.
 - <b>`output_sequences`</b>:  Sequences kept.
 - <b>`input_length`</b>:  Alignment length before processing.
 - <b>`output_length`</b>:  Alignment length after processing.
 - <b>`removed_insertion_characters`</b>:  Insertion residues deleted.
 - <b>`normalized_gap_characters`</b>:  Dots turned into ``"-"``.
 - <b>`removed_for_gap_fraction`</b>:  Sequences dropped for too many gaps.
 - <b>`removed_as_duplicates`</b>:  Duplicate sequences dropped.
 - <b>`replaced_unknown_characters`</b>:  Unknown tokens replaced with ``"-"``.
 - <b>`removed_for_unknown_tokens`</b>:  Sequences dropped for unknown tokens.
 - <b>`max_gap_fraction`</b>:  Gap threshold used, if any.
 - <b>`removed_names`</b>:  Names of all dropped sequences.

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
    replaced_unknown_characters: 'int' = 0,
    removed_for_unknown_tokens: 'int' = 0,
    max_gap_fraction: 'float | None' = None,
    removed_names: 'tuple[str, ]' = ()
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, object]
```

Return the report as a dictionary.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```

Write the report as a versioned JSON document.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`indent`</b>:  JSON indentation.



**Returns:**
 The written path.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentFilterResult`
Outcome of :func:`filter_gap_fraction`.



**Attributes:**

 - <b>`alignment`</b>:  Kept sequences.
 - <b>`keep_mask`</b>:  For each input sequence, whether it was kept.
 - <b>`gap_fractions`</b>:  Gap fraction of each input sequence.
 - <b>`removed_names`</b>:  Names of the removed sequences.
 - <b>`report`</b>:  Summary of the filtering.

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/preprocessing.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentProcessingResult`
Outcome of :func:`preprocess_alignment`.



**Attributes:**

 - <b>`alignment`</b>:  Processed alignment.
 - <b>`keep_mask`</b>:  For each input sequence, whether it was kept.
 - <b>`report`</b>:  What was changed, see :class:`AlignmentProcessingReport`.
 - <b>`output_path`</b>:  Written FASTA file, if ``output_path`` was given.

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
