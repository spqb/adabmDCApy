<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.alignment`
Alignment containers and FASTA/Stockholm input-output helpers.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `detect_alignment_format`

```python
detect_alignment_format(path: 'str | Path') → Literal['fasta', 'stockholm']
```

Detect FASTA or Stockholm from the first meaningful input line.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_alignment`

```python
read_alignment(
    path: 'str | Path',
    format: 'AlignmentFormat' = 'auto',
    alignment_index: 'int | None' = None
) → Alignment
```

Read one aligned FASTA or Stockholm alignment.

If a Stockholm file contains multiple alignments, callers must explicitly select one with ``alignment_index``.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L261"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `write_alignment`

```python
write_alignment(
    alignment: 'Alignment',
    path: 'str | Path',
    format: "Literal['fasta']" = 'fasta',
    line_width: 'int' = 0
) → Path
```

Write an alignment. FASTA is currently the supported output format.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L288"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convert_stockholm_to_fasta`

```python
convert_stockholm_to_fasta(
    input_path: 'str | Path',
    output_path: 'str | Path',
    alignment_index: 'int | None' = None,
    line_width: 'int' = 0
) → AlignmentConversionResult
```

Convert one Stockholm alignment to FASTA.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Alignment`
An immutable multiple-sequence alignment.

Names and sequences retain their input order. Every sequence must have the same aligned length.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    names: 'tuple[str, ]',
    sequences: 'tuple[str, ]',
    source: 'Path | None' = None
) → None
```






---

#### <kbd>property</kbd> num_sequences





---

#### <kbd>property</kbd> sequence_length







---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dataframe`

```python
to_dataframe()
```

Return names and aligned sequences as a pandas DataFrame.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `write_fasta`

```python
write_fasta(path: 'str | Path', line_width: 'int' = 0) → Path
```

Write the alignment to FASTA and return the output path.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentConversionResult`
Result of reading and converting an alignment file.

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alignment.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, object]
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
