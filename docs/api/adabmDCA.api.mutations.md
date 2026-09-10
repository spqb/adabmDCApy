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
    alphabet: 'str' = 'protein',
    device: 'str' = 'auto',
    dtype: 'str' = 'float32',
    include_gap: 'bool' = True
) → MutationScanResult
```

Score all single-residue substitutions of ``wild_type``.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
