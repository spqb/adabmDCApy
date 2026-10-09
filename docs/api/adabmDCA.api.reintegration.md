<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/reintegration.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.reintegration`
High-level orchestration for experimental sequence reintegration.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/reintegration.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    progress: 'Callable[[TrainingProgress], None] | None' = None,
    is_cancelled: 'Callable[[], bool] | None' = None
) → ReintegrationResult
```

Train a model on natural sequences plus experimentally scored ones.

Experimental sequences enter the training statistics with signed weights proportional to their ``adjustments``: positive values pull the model towards a sequence, negative values push it away. The experimental weights are scaled by ``lambda_value * Meff / n_experimental``, so ``lambda_value`` sets the weight of the experimental data relative to the natural alignment. Training uses PCD (PTT does not support signed weights).



**Args:**

 - <b>`natural_alignment`</b>:  Natural sequences (path, :class:`Alignment` or sequences).
 - <b>`experimental_alignment`</b>:  Tested sequences, aligned like the natural ones.
 - <b>`adjustments`</b>:  One signed score per experimental sequence (path, array or  sequence of numbers), e.g. +1 for functional and -1 for non-functional.
 - <b>`config`</b>:  Training settings; defaults to ``TrainingConfig()``.  A pseudocount of 1e-6 (0.1 for edgeDCA) is used unless set.
 - <b>`natural_weights`</b>:  Optional weights of the natural sequences; computed  by sequence-identity reweighting otherwise.
 - <b>`lambda_value`</b>:  Relative weight of the experimental data; defaults to  ``1 / max(|adjustments|)``.
 - <b>`output_dir`</b>:  If given, training files and the combined alignment are written there.
 - <b>`label`</b>:  File-name prefix; the written label is ``<label>-lambda_<value>``.
 - <b>`initial_params_path`</b>:  Optional parameter file to start from.
 - <b>`initial_chains_path`</b>:  Optional FASTA of starting chains.
 - <b>`progress`</b>:  Optional callback receiving a :class:`TrainingProgress` per update.
 - <b>`is_cancelled`</b>:  Optional callable; returning ``True`` stops training.



**Returns:**

 - <b>`A `</b>: class:`ReintegrationResult` with the training result, the combined alignment and its weights.



**Raises:**

 - <b>`InputValidationError`</b>:  If all adjustments are zero, ``lambda_value`` is  not positive, or the alignments are incompatible.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
