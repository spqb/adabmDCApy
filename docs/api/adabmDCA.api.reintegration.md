<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/reintegration.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.reintegration`
High-level orchestration for experimental sequence reintegration.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/reintegration.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `reintegrate_model`

```python
reintegrate_model(
    natural_alignment: 'AlignmentInput',
    experimental_alignment: 'AlignmentInput',
    adjustments: 'WeightInput',
    config: 'TrainingConfig | None' = None,
    natural_weights: 'WeightInput | None' = None,
    lambda_value: 'float | None' = None,
    output_dir: 'str | Path | None' = None,
    label: 'str | None' = None,
    initial_params_path: 'str | Path | None' = None,
    initial_chains_path: 'str | Path | None' = None,
    progress=None,
    is_cancelled=None
) → ReintegrationResult
```

Merge natural and experimentally adjusted sequences, then train directly.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
