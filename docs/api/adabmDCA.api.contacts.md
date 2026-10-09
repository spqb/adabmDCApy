<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/contacts.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.contacts`
High-level contact-prediction operations.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/contacts.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `predict_contacts`

```python
predict_contacts(
    model: 'DCAModel | str | Path | None' = None,
    fasta_path: 'AlignmentInput | None' = None,
    alphabet: 'str | None' = None,
    pseudocount: 'float' = 0.5,
    device: 'str' = 'auto',
    dtype: 'str' = 'float32'
) → ContactMapResult
```

Predict residue contacts from a DCA model or directly from an alignment.

With ``model``, scores are the average-product-corrected (APC) Frobenius norms of the zero-sum-gauge couplings, excluding the gap state. With ``fasta_path``, a mean-field DCA model is inferred from the reweighted alignment and scored the same way. Larger scores mean likelier contacts.



**Args:**

 - <b>`model`</b>:  A :class:`DCAModel`, or a path to a parameter file or PTT archive.
 - <b>`fasta_path`</b>:  Aligned sequences (path, :class:`Alignment` or sequence  strings) for mean-field prediction. Provide exactly one of ``model``  and ``fasta_path``.
 - <b>`alphabet`</b>:  ``"protein"``, ``"rna"``, ``"dna"`` or an ordered custom token  string, for the alignment or a text parameter file. ``None`` reads it  from a PTT archive and otherwise means ``"protein"``. Ignored when
 - <b>```model`` is already a `</b>: class:`DCAModel`. Must contain ``"-"``.
 - <b>`pseudocount`</b>:  Mean-field only: pseudocount in ``[0, 1]`` that regularizes the  covariance matrix before inversion.
 - <b>`device`</b>:  ``"auto"`` (CUDA when available, else CPU), ``"cpu"``,  ``"cuda"`` or ``"mps"``. Ignored for an in-memory model.
 - <b>`dtype`</b>:  ``"float32"`` or ``"float64"`` for loaded parameters. Ignored for an  in-memory model.



**Returns:**

 - <b>`A `</b>: class:`ContactMapResult` whose ``scores`` is a symmetric ``(L, L)`` array with a zero diagonal.



**Raises:**

 - <b>`InputValidationError`</b>:  If both or neither of ``model`` and ``fasta_path``  are given, the alphabet lacks the ``"-"`` gap token, or the  pseudocount is outside ``[0, 1]``.



**Example:**
 ``` result = predict_contacts(model="params.dat.gz", alphabet="rna")```
    >>> top = result.to_long_dataframe().nlargest(20, "score")  # 0-based positions



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/contacts.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_contact_map`

```python
compute_contact_map(
    model: 'DCAModel | str | Path | None' = None,
    fasta_path: 'AlignmentInput | None' = None,
    alphabet: 'str | None' = None,
    pseudocount: 'float' = 0.5,
    device: 'str' = 'auto',
    dtype: 'str' = 'float32'
) → ndarray
```

Return only the ``(L, L)`` contact-score matrix of :func:`predict_contacts`.

Takes the same arguments as :func:`predict_contacts`; see it for details.



**Returns:**
  A symmetric NumPy array of APC-corrected scores with a zero diagonal.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
