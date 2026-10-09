<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/input_loading.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.input_loading`
Aggregate input loading for high-level workflows.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/input_loading.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Load the training inputs of :func:`train_model` and check they fit together.

Sequences with unknown tokens are dropped and duplicates removed. The validation alignment, initial parameters and initial chains must have the training alignment's length and alphabet.



**Args:**

 - <b>`training`</b>:  Training alignment (path, :class:`Alignment` or sequences).
 - <b>`config`</b>:  Training settings; provides alphabet and reweighting options.
 - <b>`device`</b>:  Device of the loaded tensors.
 - <b>`dtype`</b>:  Precision of the loaded tensors.
 - <b>`validation`</b>:  Optional validation alignment.
 - <b>`weights`</b>:  Optional training-sequence weights (path, array or tensor).
 - <b>`initial_params_path`</b>:  Optional parameter file to start from.
 - <b>`initial_chains_path`</b>:  Optional FASTA of starting chains.
 - <b>`allow_signed_weights`</b>:  Accept negative weights (experimental reintegration).



**Returns:**

 - <b>`The loaded `</b>: class:`TrainingInputs`.



**Raises:**

 - <b>`ModelCompatibilityError`</b>:  If validation data, parameters or chains do not  match the training alignment.
 - <b>`InputLoadError`</b>:  If a file cannot be read.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/input_loading.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingInputs`
Loaded and cross-checked inputs of one training run.



**Attributes:**

 - <b>`training`</b>:  Weighted training dataset.
 - <b>`training_alignment`</b>:  Training alignment after filtering.
 - <b>`validation`</b>:  Weighted validation dataset, if any.
 - <b>`validation_alignment`</b>:  Validation alignment after filtering, if any.
 - <b>`initial_params`</b>:  Parameters to start from, if given.
 - <b>`initial_chains`</b>:  One-hot chains to start from, if given.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    training: 'DatasetDCA',
    training_alignment: 'Alignment',
    validation: 'DatasetDCA | None' = None,
    validation_alignment: 'Alignment | None' = None,
    initial_params: 'dict[str, Tensor] | None' = None,
    initial_chains: 'Tensor | None' = None
) → None
```











---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
