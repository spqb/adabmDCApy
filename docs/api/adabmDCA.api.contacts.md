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
    alphabet: 'str' = 'protein',
    pseudocount: 'float' = 0.5,
    device: 'str' = 'auto',
    dtype: 'str' = 'float32'
) → ContactMapResult
```

Compute APC-corrected contact scores from a model or an alignment.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/contacts.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_contact_map`

```python
compute_contact_map(
    model: 'DCAModel | str | Path | None' = None,
    fasta_path: 'AlignmentInput | None' = None,
    alphabet: 'str' = 'protein',
    pseudocount: 'float' = 0.5,
    device: 'str' = 'auto',
    dtype: 'str' = 'float32'
) → ndarray
```

Notebook-friendly shortcut returning only the contact-score matrix.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
