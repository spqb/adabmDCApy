<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/training.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.training`
High-level DCA training workflow.

**Global Variables**
---------------
- **DEFAULT_ACTIVATION_FRACTION**
- **DEFAULT_ACTIVATION_STEPS**
- **DEFAULT_ALPHABET**
- **DEFAULT_CHECKPOINT_INTERVAL**
- **DEFAULT_CLUSTERING_SEQID**
- **DEFAULT_DECIMATION_RATE**
- **DEFAULT_DEVICE**
- **DEFAULT_DTYPE**
- **DEFAULT_L2_REGULARIZATION**
- **DEFAULT_LEARNING_RATE**
- **DEFAULT_MAX_EPOCHS**
- **DEFAULT_MODEL_TYPE**
- **DEFAULT_N_CHAINS**
- **DEFAULT_N_SWEEPS**
- **DEFAULT_SAMPLER**
- **DEFAULT_SEED**
- **DEFAULT_TARGET_DENSITY**
- **DEFAULT_TARGET_PEARSON**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/training.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_model`

```python
train_model(
    data_path: 'AlignmentInput',
    config: 'TrainingConfig | None' = None,
    model_type: 'str' = 'bmDCA',
    validation_path: 'AlignmentInput | None' = None,
    weights_path: 'WeightInput | None' = None,
    output_dir: 'str | Path | None' = None,
    label: 'str | None' = None,
    initial_params_path: 'str | Path | None' = None,
    initial_chains_path: 'str | Path | None' = None,
    alphabet: 'str' = 'protein',
    learning_rate: 'float' = 0.01,
    n_sweeps: 'int' = 10,
    sampler: 'str' = 'metropolis',
    n_chains: 'int' = 10000,
    target_pearson: 'float' = 0.95,
    max_epochs: 'int' = 50000,
    max_gradient_steps: 'int | None' = None,
    max_structure_steps: 'int | None' = None,
    checkpoint_interval: 'int | None' = 100,
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
    allow_signed_weights: 'bool' = False,
    progress: 'ProgressCallback | None' = None,
    on_initialized: 'InitializationCallback | None' = None,
    is_cancelled: 'CancellationHook | None' = None
) → TrainingResult
```

Train a DCA model from a FASTA alignment.

Persistence is optional. Set ``output_dir`` to retain the historical parameter, chain, weight, and log artifacts; omit it for an in-memory notebook workflow. ``checkpoint_interval`` controls periodic parameter/chain saves (default 100 steps). The final state is also saved when training finishes. ``on_initialized`` is called once after input filtering and sequence weighting, before numerical training begins. Its event contains resolved dimensions, effective sample sizes, runtime, and optimization settings.

``dtype='bfloat16'`` uses BF16 coupling copies for CUDA/Triton sampling, with float32 master parameters, statistics, chains and saved models. Requires an NVIDIA Ampere or newer GPU. Sampling uses rounded couplings, so the training trajectory can differ from float32.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
