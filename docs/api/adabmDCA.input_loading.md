<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/input_loading.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.input_loading`
Shared, side-effect-free loading of alignments and sequence weights.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/input_loading.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_alignment`

```python
load_alignment(
    source: 'AlignmentInput',
    config: 'AlignmentLoadConfig | None' = None,
    alphabet: 'str | None' = None,
    invalid_sequences: 'InvalidSequencePolicy | None' = None,
    remove_duplicates: 'bool | None' = None,
    expected_length: 'int | None' = None,
    format: 'AlignmentFormat | None' = None,
    alignment_index: 'int | None' = None,
    normalize_dots: 'bool | None' = None
) → Alignment
```

Parse, validate, filter, and deduplicate an alignment.

The alphabet is detected from standard DNA, RNA, or protein tokens by default; pass ``alphabet`` for ambiguous or custom alignments. Direct keyword options override the corresponding fields in ``config``. The returned :class:`Alignment` contains its selected encoding tokens and filtering provenance, so callers do not need a separate wrapper type.



**Args:**

 - <b>`source`</b>:  Path, :class:`Alignment` or sequences.
 - <b>`config`</b>:  Loading policy; see :class:`AlignmentLoadConfig`.
 - <b>`alphabet`</b>:  Overrides ``config.alphabet``.
 - <b>`invalid_sequences`</b>:  Overrides ``config.invalid_sequences``.
 - <b>`remove_duplicates`</b>:  Overrides ``config.remove_duplicates``.
 - <b>`expected_length`</b>:  Overrides ``config.expected_length``.
 - <b>`format`</b>:  Overrides ``config.format``.
 - <b>`alignment_index`</b>:  Overrides ``config.alignment_index``.
 - <b>`normalize_dots`</b>:  Overrides ``config.normalize_dots``.



**Returns:**

 - <b>`The filtered `</b>: class:`Alignment`, with ``tokens``, ``encoded_sequences`` and the indices of the retained sequences.



**Raises:**

 - <b>`AlignmentLoadError`</b>:  If the file cannot be read.
 - <b>`InputValidationError`</b>:  If sequences are invalid under ``"error"``, the  length is wrong, or no sequence remains.



**Example:**
 ``` alignment = load_alignment("family.fasta", invalid_sequences="drop")```
    >>> alignment.tokens, alignment.num_sequences



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/input_loading.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_sequence_weights`

```python
load_sequence_weights(
    source: 'WeightInput | None',
    loaded_alignment: 'Alignment',
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



**Args:**

 - <b>`source`</b>:  Weights file (one number per line), sequence, array or tensor, or  ``None`` to compute them. Their count may match the original or the  retained sequences.
 - <b>`loaded_alignment`</b>:  Alignment returned by :func:`load_alignment`.
 - <b>`no_reweighting`</b>:  Give every sequence weight 1 (``source`` is ignored).
 - <b>`clustering_seqid`</b>:  Computed weights: each sequence gets  ``1 / (number of sequences more than this identical to it)``,  itself included.
 - <b>`device`</b>:  Device of the returned tensor.
 - <b>`dtype`</b>:  Precision of the returned tensor.
 - <b>`allow_negative`</b>:  Accept negative weights.
 - <b>`require_positive_sum`</b>:  Reject weights whose sum is not positive.



**Returns:**
 One weight per retained sequence.



**Raises:**

 - <b>`WeightLoadError`</b>:  If the weights cannot be read, have the wrong count, or  are not finite, negative, or sum to zero when disallowed.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/input_loading.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentLoadConfig`
How :func:`load_alignment` reads, validates and filters an alignment.



**Attributes:**

 - <b>`alphabet`</b>:  ``"auto"`` (detect DNA, RNA or protein), a built-in name, or a  custom token string.
 - <b>`invalid_sequences`</b>:  ``"error"`` to reject, or ``"drop"`` to remove,  sequences with tokens outside the alphabet.
 - <b>`remove_duplicates`</b>:  Keep only the first copy of identical sequences.
 - <b>`expected_length`</b>:  Required alignment length, or ``None``.
 - <b>`format`</b>:  ``"auto"``, ``"fasta"`` or ``"stockholm"``.
 - <b>`alignment_index`</b>:  Alignment to read from a multi-alignment Stockholm file.
 - <b>`normalize_dots`</b>:  Turn ``"."`` gaps into ``"-"``.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    alphabet: 'str' = 'auto',
    invalid_sequences: 'InvalidSequencePolicy' = 'error',
    remove_duplicates: 'bool' = False,
    expected_length: 'int | None' = None,
    format: 'AlignmentFormat' = 'auto',
    alignment_index: 'int | None' = None,
    normalize_dots: 'bool' = True
) → None
```











---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
