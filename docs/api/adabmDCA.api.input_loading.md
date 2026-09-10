<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/input_loading.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.input_loading`
Aggregate input loading for high-level workflows.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/input_loading.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_training_inputs`

```python
load_training_inputs(
    training: 'AlignmentInput',
    config: 'TrainingConfig',
    device: 'device',
    dtype: 'dtype',
    validation: 'AlignmentInput | None' = None,
    weights: 'WeightInput | None' = None,
    initial_params_path: 'str | Path | None' = None,
    initial_chains_path: 'str | Path | None' = None,
    allow_signed_weights: 'bool' = False
) → TrainingInputs
```

Load all training resources and validate their shared dimensions.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/input_loading.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingInputs`
Fully loaded and cross-validated inputs for model training.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    training: 'DatasetDCA',
    training_alignment: 'LoadedAlignment',
    validation: 'DatasetDCA | None' = None,
    validation_alignment: 'LoadedAlignment | None' = None,
    initial_params: 'dict[str, Tensor] | None' = None,
    initial_chains: 'Tensor | None' = None,
    initial_log_weights: 'Tensor | None' = None
) → None
```











---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
