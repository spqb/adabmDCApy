<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/splitting.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.splitting`
High-level API for profile-model alignment splitting.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/splitting.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `split_alignment`

```python
split_alignment(
    alignment: 'AlignmentInput',
    t1: 'float' = 0.5,
    t2: 'float' = 0.5,
    t3: 'float' = 1.0,
    max_train: 'int | None' = None,
    max_test: 'int | None' = None,
    attempts: 'int' = 1,
    alphabet: 'str' = 'protein',
    seed: 'int' = 0,
    device: 'str' = 'auto'
) → ProfileSplitResult
```

Split an alignment and retain the best Cobalt partition.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
