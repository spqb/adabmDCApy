<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_config.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.training_config`
Canonical configuration values and validation for DCA training.

**Global Variables**
---------------
- **DEFAULT_PTT_MAX_EPOCHS**
- **DEFAULT_PTT_N_CHAINS**
- **DEFAULT_PTT_N_SWEEPS**
- **DEFAULT_PTT_SAMPLER**
- **MODEL_TYPES**
- **SAMPLERS**
- **DEFAULT_MODEL_TYPE**
- **DEFAULT_ALPHABET**
- **DEFAULT_LEARNING_RATE**
- **DEFAULT_N_SWEEPS**
- **DEFAULT_SAMPLER**
- **DEFAULT_N_CHAINS**
- **DEFAULT_TARGET_PEARSON**
- **DEFAULT_MAX_EPOCHS**
- **DEFAULT_L2_REGULARIZATION**
- **DEFAULT_SEED**
- **DEFAULT_CLUSTERING_SEQID**
- **DEFAULT_ACTIVATION_STEPS**
- **DEFAULT_ACTIVATION_FRACTION**
- **DEFAULT_TARGET_DENSITY**
- **DEFAULT_DECIMATION_RATE**
- **DEFAULT_DEVICE**
- **DEFAULT_DTYPE**
- **DEFAULT_CHECKPOINT_INTERVAL**
- **SPARSE_CHECKPOINT_INTERVAL**
- **DEFAULT_INNER_GRADIENT_STEPS**
- **EDGE_DEFAULT_PSEUDOCOUNT**
- **EDGE_EMPIRICAL_PSEUDOCOUNT**


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_config.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConfigurationError`
Raised when a training configuration is internally inconsistent.





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_config.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingConfig`
All settings of one training run, validated on construction.

Pass it as ``config=`` to :func:`train_model`; it then takes precedence over the function's individual keyword arguments. Settings that do not apply to the chosen ``model_type`` are ignored.



**Attributes:**

 - <b>`model_type`</b>:  ``"bmDCA"`` (fully connected), ``"eaDCA"`` (coupling  activation), ``"edDCA"`` (decimation) or ``"edgeDCA"`` (edge activation).
 - <b>`alphabet`</b>:  ``"protein"``, ``"rna"``, ``"dna"`` or an ordered custom token string.
 - <b>`learning_rate`</b>:  Step size of the parameter updates (ignored by edgeDCA).
 - <b>`n_sweeps`</b>:  Monte Carlo sweeps per update.
 - <b>`sampler`</b>:  ``"metropolized_gibbs"``, ``"gibbs"`` or ``"metropolis"``.
 - <b>`n_chains`</b>:  Number of Markov chains.
 - <b>`target_pearson`</b>:  Training stops when the Pearson correlation between data and  model connected correlations reaches this value.
 - <b>`max_epochs`</b>:  Step budget: gradient steps for bmDCA and PTT edgeDCA, graph  steps for the other sparse models. Superseded by the two limits below.
 - <b>`max_gradient_steps`</b>:  Optional limit on parameter updates.
 - <b>`max_structure_steps`</b>:  Optional limit on graph activations or decimations.
 - <b>`pseudocount`</b>:  Pseudocount of the training statistics; ``None`` means ``1/Meff``  (0.1 for edgeDCA).
 - <b>`l2_regularization`</b>:  Strength of the L2 penalty on the couplings.
 - <b>`seed`</b>:  Random seed.
 - <b>`clustering_seqid`</b>:  Sequence identity above which sequences share weight.
 - <b>`no_reweighting`</b>:  Give every training sequence the same weight.
 - <b>`activation_steps`</b>:  eaDCA: gradient updates between graph activations.
 - <b>`activation_fraction`</b>:  eaDCA: fraction of the inactive couplings activated  per graph update (the upper limit for PTT adaptive activation).
 - <b>`target_density`</b>:  edDCA: coupling density at which decimation stops.
 - <b>`decimation_rate`</b>:  edDCA: fraction of active couplings removed per decimation.
 - <b>`device`</b>:  ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.
 - <b>`dtype`</b>:  ``"float32"``, ``"float64"`` or ``"bfloat16"`` (CUDA/MPS, PCD only).
 - <b>`use_wandb`</b>:  Log to Weights & Biases.
 - <b>`checkpoint_interval`</b>:  Updates between saved checkpoints; ``None`` uses the default.
 - <b>`inner_gradient_steps`</b>:  edDCA: largest number of gradient updates to  re-converge the model after each decimation.
 - <b>`edge_empirical_pseudocount`</b>:  edgeDCA: small pseudocount of the target  statistics, which avoids zero frequencies.
 - <b>`ptt`</b>:  :class:`PTTConfig` to train with PTT, or ``None`` for PCD.



**Raises:**

 - <b>`ConfigurationError`</b>:  If a setting is invalid or unsupported for the model.



**Example:**
 ``` config = TrainingConfig(model_type="eaDCA", alphabet="rna",```
    ...                         ptt=PTTConfig(activation="adaptive"))
    >>> train_model("family.fasta", config=config, output_dir="model")


<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model_type: 'str' = 'bmDCA',
    alphabet: 'str' = 'protein',
    learning_rate: 'float' = 0.01,
    n_sweeps: 'int' = 10,
    sampler: 'str' = 'metropolized_gibbs',
    n_chains: 'int' = 2000,
    target_pearson: 'float' = 0.95,
    max_epochs: 'int' = 50000,
    max_gradient_steps: 'int | None' = None,
    max_structure_steps: 'int | None' = None,
    pseudocount: 'float | None' = None,
    l2_regularization: 'float' = 0.0,
    seed: 'int' = 0,
    clustering_seqid: 'float' = 0.8,
    no_reweighting: 'bool' = False,
    activation_steps: 'int' = 10,
    activation_fraction: 'float' = 0.001,
    target_density: 'float' = 0.02,
    decimation_rate: 'float' = 0.01,
    device: 'str' = 'auto',
    dtype: 'str' = 'float32',
    use_wandb: 'bool' = False,
    checkpoint_interval: 'int | None' = 500,
    inner_gradient_steps: 'int' = 10000,
    edge_empirical_pseudocount: 'float' = 1e-06,
    ptt: 'PTTConfig | None' = None
) → None
```






---

#### <kbd>property</kbd> empirical_pseudocount

Pseudocount used for target statistics before regularization.

---

#### <kbd>property</kbd> limits

Resolve the legacy epoch limit into explicit model-aware limits.

---

#### <kbd>property</kbd> resolved_checkpoint_interval

Checkpoint interval in updates, with the default filled in.



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_config.py#L281"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `as_dict`

```python
as_dict() → dict[str, Any]
```

Return a serializable snapshot suitable for run metadata.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_config.py#L260"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `resolve_pseudocount`

```python
resolve_pseudocount(effective_size: 'float') → float
```

Return the pseudocount to use for a training set of ``effective_size`` (``Meff``).



**Args:**

 - <b>`effective_size`</b>:  Sum of the training sequence weights.



**Returns:**
 ``pseudocount`` if set; otherwise 0.1 for edgeDCA and ``1/Meff`` for the other models.



**Raises:**

 - <b>`ConfigurationError`</b>:  If ``effective_size`` is not positive.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
