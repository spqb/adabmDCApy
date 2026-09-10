<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_config.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.training_config`
Canonical configuration values and validation for DCA training.

**Global Variables**
---------------
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
- **EDGE_LOGZ_CHAIN_FRACTION**
- **SLOPE_TOLERANCE**


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_config.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConfigurationError`
Raised when a training configuration is internally inconsistent.





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_config.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingConfig`
Validated, transport-neutral configuration for one training run.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model_type: 'str' = 'bmDCA',
    alphabet: 'str' = 'protein',
    learning_rate: 'float' = 0.01,
    n_sweeps: 'int' = 10,
    sampler: 'str' = 'metropolis',
    n_chains: 'int' = 10000,
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
    checkpoint_interval: 'int | None' = 100,
    inner_gradient_steps: 'int' = 10000,
    slope_tolerance: 'float' = 0.1,
    edge_empirical_pseudocount: 'float' = 1e-06,
    edge_logz_chain_fraction: 'float' = 0.2
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







---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_config.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `as_dict`

```python
as_dict() → dict[str, Any]
```

Return a serializable snapshot suitable for run metadata.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training_config.py#L183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `resolve_pseudocount`

```python
resolve_pseudocount(effective_size: 'float') → float
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
