<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/mutations.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.mutations`
High-level mutation-scanning operations.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/mutations.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `scan_mutations`

```python
scan_mutations(
    wild_type: 'str',
    model: 'DCAModel | str | Path',
    name: 'str' = 'wild_type',
    alphabet: 'str | None' = None,
    device: 'str' = 'auto',
    dtype: 'str' = 'float32',
    include_gap: 'bool' = True
) → MutationScanResult
```

Score every single-site substitution of an aligned wild-type sequence.

For each position and each alternative token, the mutant's energy is compared with the wild type's. With the convention ``E = -sum h - sum J``, a negative ``delta_energy`` means the model favours the mutant.



**Args:**

 - <b>`wild_type`</b>:  Aligned sequence of length ``L`` using the model's tokens.
 - <b>`model`</b>:  A :class:`DCAModel`, or a path to a parameter file or PTT archive.
 - <b>`name`</b>:  Label stored in the result and used in exported files.
 - <b>`alphabet`</b>:  Alphabet of a text parameter file: ``"protein"``, ``"rna"``,  ``"dna"`` or an ordered custom token string. ``None`` reads it from a  PTT archive and assumes ``"protein"`` for text files. Ignored when
 - <b>```model`` is already a `</b>: class:`DCAModel`.
 - <b>`device`</b>:  ``"auto"`` (CUDA when available, else CPU), ``"cpu"``,  ``"cuda"`` or ``"mps"``. Ignored for an in-memory model.
 - <b>`dtype`</b>:  ``"float32"`` or ``"float64"`` for loaded parameters. Ignored for an  in-memory model.
 - <b>`include_gap`</b>:  Whether substitutions to the gap token ``"-"`` are scored.



**Returns:**

 - <b>`A `</b>: class:`MutationScanResult` with the wild-type energy and one
 - <b>`:class`</b>: `MutationRecord` per mutant, ``L * (q - 1)`` in total (fewer when gaps are excluded).



**Raises:**

 - <b>`InputValidationError`</b>:  If ``wild_type`` has the wrong length or unknown  tokens, or fewer than two tokens are eligible.



**Example:**
 ``` scan = scan_mutations("MKT...", model="params.dat.gz")```
    >>> scan.to_dataframe().sort_values("delta_energy").head()





---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
