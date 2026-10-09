<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.alignment`
Alignment containers and FASTA/Stockholm input-output helpers.

**Global Variables**
---------------
- **TYPE_CHECKING**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `detect_alignment_format`

```python
detect_alignment_format(path: 'str | Path') → Literal['fasta', 'stockholm']
```

Detect whether a file is FASTA or Stockholm from its first non-empty line.



**Args:**

 - <b>`path`</b>:  Alignment file, optionally gzip-compressed.



**Returns:**
 ``"fasta"`` or ``"stockholm"``.



**Raises:**

 - <b>`AlignmentFormatError`</b>:  If the format cannot be recognized.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_alignment`

```python
read_alignment(
    path: 'str | Path',
    format: 'AlignmentFormat' = 'auto',
    alignment_index: 'int | None' = None
) → Alignment
```

Read one aligned FASTA or Stockholm alignment, without filtering.

For validation, alphabet detection and filtering use :func:`load_alignment`.



**Args:**

 - <b>`path`</b>:  Alignment file, optionally gzip-compressed.
 - <b>`format`</b>:  ``"auto"`` (detected), ``"fasta"`` or ``"stockholm"``.
 - <b>`alignment_index`</b>:  Which alignment of a multi-alignment Stockholm file to  read (0-based); required when there is more than one.



**Returns:**

 - <b>`The `</b>: class:`Alignment`.



**Raises:**

 - <b>`AlignmentLoadError`</b>:  If the file cannot be read.
 - <b>`AlignmentFormatError`</b>:  If the content is not a valid alignment.
 - <b>`AlignmentLengthError`</b>:  If the sequences have different lengths.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L421"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `write_alignment`

```python
write_alignment(
    alignment: 'Alignment',
    path: 'str | Path',
    format: "Literal['fasta']" = 'fasta',
    line_width: 'int' = 0
) → Path
```

Write an alignment to a FASTA file.



**Args:**

 - <b>`alignment`</b>:  Alignment to write.
 - <b>`path`</b>:  Output file.
 - <b>`format`</b>:  Output format; only ``"fasta"`` is supported.
 - <b>`line_width`</b>:  Largest sequence-line length; ``0`` writes each sequence on one line.



**Returns:**
 The written path.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L458"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `normalize_gap_symbols`

```python
normalize_gap_symbols(
    alignment: 'Alignment',
    source_gap: 'str' = '.',
    target_gap: 'str' = '-'
) → Alignment
```

Replace one alignment gap symbol with another.

Stockholm permits dots as gap symbols, while adabmDCA and aligned FASTA files conventionally use hyphens. This transformation does not remove alignment columns.



**Args:**

 - <b>`alignment`</b>:  Alignment to transform.
 - <b>`source_gap`</b>:  Gap symbol to replace.
 - <b>`target_gap`</b>:  Replacement gap symbol.



**Returns:**

 - <b>`A new `</b>: class:`Alignment`.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L492"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convert_alignment`

```python
convert_alignment(
    input_path: 'str | Path',
    output_path: 'str | Path',
    input_format: 'AlignmentFormat' = 'auto',
    output_format: "Literal['fasta']" = 'fasta',
    alignment_index: 'int | None' = None,
    line_width: 'int' = 0
) → AlignmentConversionResult
```

Convert a FASTA or Stockholm alignment to canonical aligned FASTA.

Dots accepted as gaps by Stockholm are written as hyphens, the gap token used throughout adabmDCA. Lowercase residues are preserved; callers that want to delete insertion residues should use :func:`preprocess_alignment`.



**Args:**

 - <b>`input_path`</b>:  FASTA or Stockholm file.
 - <b>`output_path`</b>:  FASTA file to write.
 - <b>`input_format`</b>:  ``"auto"``, ``"fasta"`` or ``"stockholm"``.
 - <b>`output_format`</b>:  Only ``"fasta"`` is supported.
 - <b>`alignment_index`</b>:  Alignment to convert in a multi-alignment Stockholm file.
 - <b>`line_width`</b>:  Largest sequence-line length; ``0`` means no wrapping.



**Returns:**

 - <b>`An `</b>: class:`AlignmentConversionResult`.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convert_stockholm_to_fasta`

```python
convert_stockholm_to_fasta(
    input_path: 'str | Path',
    output_path: 'str | Path',
    alignment_index: 'int | None' = None,
    line_width: 'int' = 0
) → AlignmentConversionResult
```

Convert one Stockholm alignment to FASTA; see :func:`convert_alignment`.



**Args:**

 - <b>`input_path`</b>:  Stockholm file.
 - <b>`output_path`</b>:  FASTA file to write.
 - <b>`alignment_index`</b>:  Alignment to convert when the file contains several.
 - <b>`line_width`</b>:  Largest sequence-line length; ``0`` means no wrapping.



**Returns:**

 - <b>`An `</b>: class:`AlignmentConversionResult`.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Alignment`
Aligned sequences with optional alphabet and filtering provenance.

Names and sequences retain their input order. The alignment must contain at least one nonempty sequence, and every sequence must have the same length. Constructing an alignment checks these structural constraints; :func:`adabmDCA.input_loading.load_alignment` additionally resolves the alphabet, validates symbols, and optionally filters rows.



**Args:**

 - <b>`names`</b>:  Sequence names, one per row.
 - <b>`sequences`</b>:  Aligned sequence strings in the same order as ``names``.
 - <b>`source`</b>:  Path of the source file, if known.
 - <b>`tokens`</b>:  Ordered alphabet used for encoding. ``None`` until resolved;  index ``i`` in an encoding corresponds to ``tokens[i]``.
 - <b>`retained_indices`</b>:  Positions of the current rows in the original  alignment. Defaults to ``0, 1, ..., M - 1``.
 - <b>`dropped_indices`</b>:  Original positions removed for invalid symbols.
 - <b>`duplicate_indices`</b>:  Original positions removed as duplicates.
 - <b>`original_size`</b>:  Number of rows before filtering. Defaults to ``M``.



**Note:**

> The provenance fields are populated by ``load_alignment``. Plain ``Alignment`` construction does not check sequences against ``tokens``.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    names: 'tuple[str, ]',
    sequences: 'tuple[str, ]',
    source: 'Path | None' = None,
    tokens: 'str | None' = None,
    retained_indices: 'tuple[int, ]' = (),
    dropped_indices: 'tuple[int, ]' = (),
    duplicate_indices: 'tuple[int, ]' = (),
    original_size: 'int' = 0
) → None
```






---

#### <kbd>property</kbd> encoded_sequences

Return integer token indices as a NumPy array of shape ``(M, L)``.

Each element is the position of its residue in ``self.tokens``.



**Raises:**

 - <b>`InputValidationError`</b>:  If no alphabet has been resolved. Pass the  alignment through ``load_alignment`` first.

---

#### <kbd>property</kbd> num_sequences

Number of sequences (``M``) currently in the alignment.

---

#### <kbd>property</kbd> sequence_length

Number of aligned positions (``L``) in each sequence.



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dataframe`

```python
to_dataframe() → DataFrame
```

Return a pandas DataFrame with ``name`` and ``sequence`` columns.

The DataFrame has one row per sequence and preserves alignment order.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, object]
```

Return a JSON-compatible loading report without sequence data.

The report includes the source, original and retained row counts, filtering indices, aligned length, and tokens. For an alignment that has not been loaded, filtering indices describe its unchanged rows.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_onehot`

```python
to_onehot(flatten: 'bool' = False) → Tensor
```

Encode the alignment as a CPU ``torch.float32`` one-hot tensor.



**Args:**

 - <b>`flatten`</b>:  If ``False``, return shape ``(M, L, q)``. If ``True``,  combine the position and token dimensions into shape  ``(M, L * q)``, useful for PCA.



**Returns:**
 A tensor whose state features follow the order of ``self.tokens`` at each aligned position.



**Raises:**

 - <b>`InputValidationError`</b>:  If ``tokens`` is unavailable. Load the  alignment with ``load_alignment`` first.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `write_fasta`

```python
write_fasta(path: 'str | Path', line_width: 'int' = 0) → Path
```

Write names and sequences to an aligned FASTA file.



**Args:**

 - <b>`path`</b>:  Output file path.
 - <b>`line_width`</b>:  Maximum sequence line length. ``0`` writes each  sequence on one line.



**Returns:**

 - <b>`The output path as a `</b>: class:`pathlib.Path`.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentConversionResult`
Outcome of :func:`convert_alignment`.



**Attributes:**

 - <b>`alignment`</b>:  The converted alignment.
 - <b>`input_format`</b>:  Detected or given input format.
 - <b>`output_format`</b>:  Written format (``"fasta"``).
 - <b>`output_path`</b>:  Written file.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    alignment: 'Alignment',
    input_format: 'str',
    output_format: 'str',
    output_path: 'Path'
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, object]
```

Return the formats, output path and alignment size as a dictionary.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
