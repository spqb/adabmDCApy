<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/entropy.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.entropy`
High-level thermodynamic-integration entropy estimation.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/entropy.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `estimate_entropy`

```python
estimate_entropy(
    model: 'str | Path',
    natural_alignment: 'AlignmentInput',
    target_alignment: 'AlignmentInput',
    initial_chains_path: 'str | Path | None' = None,
    n_chains: 'int' = 10000,
    n_sweeps: 'int' = 100,
    n_steps: 'int' = 100,
    theta_max: 'float' = 5.0,
    theta_sweeps: 'int' = 100,
    zero_sweeps: 'int' = 100,
    target_fraction: 'float' = 0.1,
    max_theta_iterations: 'int' = 10000,
    sampler: 'str' = 'metropolis',
    alphabet: 'str' = 'protein',
    seed: 'int' = 0,
    device: 'str' = 'auto',
    dtype: 'str' = 'float32',
    output_dir: 'str | Path | None' = None,
    label: 'str' = 'entropy',
    progress: 'EntropyProgress | None' = None,
    is_cancelled: 'Callable[[], bool] | None' = None
) → ThermodynamicIntegrationResult
```

Estimate model entropy with bounded, observable thermodynamic integration.

Use the first valid sequence in ``target_alignment`` in input order. If multiple valid sequences are provided, emit a warning and ignore the rest.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
