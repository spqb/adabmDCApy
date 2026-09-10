<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/scoring.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.scoring`
High-level sequence-energy operations.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/scoring.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `score_sequences`

```python
score_sequences(
    sequences: 'str | Iterable[str] | None' = None,
    model: 'DCAModel | str | Path',
    fasta_path: 'AlignmentInput | None' = None,
    alphabet: 'str' = 'protein',
    device: 'str' = 'auto',
    dtype: 'str' = 'float32',
    remove_duplicates: 'bool' = False
) → EnergyResult
```

Compute DCA energies and return a structured result.

Provide exactly one of ``sequences`` or ``fasta_path``. When ``model`` is a :class:`DCAModel`, its alphabet and runtime configuration are reused.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/scoring.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_energies`

```python
compute_energies(
    sequences: 'str | Iterable[str] | None' = None,
    model: 'DCAModel | str | Path',
    fasta_path: 'AlignmentInput | None' = None,
    alphabet: 'str' = 'protein',
    device: 'str' = 'auto',
    dtype: 'str' = 'float32',
    remove_duplicates: 'bool' = False
) → ndarray
```

Notebook-friendly shortcut returning only the energy vector.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
