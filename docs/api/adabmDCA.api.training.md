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
- **DEFAULT_MODEL_TYPE**
- **DEFAULT_SEED**
- **DEFAULT_TARGET_DENSITY**
- **DEFAULT_TARGET_PEARSON**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/training.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_model`

```python
train_model(
    data_path: 'AlignmentInput',
    config: 'TrainingConfig | None' = None,
    model_type: 'str' = 'bmDCA',
    ptt: 'PTTConfig | None' = None,
    ptt_resume: 'str | Path | None' = None,
    validation_path: 'AlignmentInput | None' = None,
    weights_path: 'WeightInput | None' = None,
    output_dir: 'str | Path | None' = None,
    label: 'str | None' = None,
    initial_params_path: 'str | Path | None' = None,
    initial_chains_path: 'str | Path | None' = None,
    alphabet: 'str' = 'protein',
    learning_rate: 'float' = 0.01,
    n_sweeps: 'int' = 10,
    sampler: 'str' = 'metropolized_gibbs',
    n_chains: 'int' = 2000,
    target_pearson: 'float' = 0.95,
    max_epochs: 'int' = 50000,
    max_gradient_steps: 'int | None' = None,
    max_structure_steps: 'int | None' = None,
    checkpoint_interval: 'int | None' = 500,
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
    stage_progress: 'StageProgressCallback | None' = None,
    on_initialized: 'InitializationCallback | None' = None,
    is_cancelled: 'CancellationHook | None' = None
) → TrainingResult
```

Train a DCA model from an alignment and return its in-memory result.

``data_path`` may be an :class:`Alignment` or a FASTA, gzip-compressed FASTA, or Stockholm path. Invalid sequences are removed and duplicate sequences are collapsed. Supplied weights may match either the original rows or the retained rows. The resolved filtering decisions are available in ``result.input_report``.

Pass ``config=TrainingConfig(...)`` to specify the training policy as one object. When supplied, ``config`` is authoritative for training settings; the individual settings below do not override it. Input paths, output location, callbacks, and cancellation are still taken from this call.



**Args:**

 - <b>`data_path`</b>:  Training alignment or path to an alignment file.
 - <b>`config`</b>:  Validated training settings. If omitted, the individual  training options below are used to construct a ``TrainingConfig``.
 - <b>`model_type`</b>:  ``"bmDCA"``, ``"eaDCA"``, ``"edDCA"``, or ``"edgeDCA"``.
 - <b>`ptt`</b>:  Optional PTT settings for bmDCA, eaDCA or edgeDCA training. May also  be supplied through ``config.ptt``.
 - <b>`ptt_resume`</b>:  Full HDF5 PTT archive from which to resume a compatible  run. Requires PTT settings.
 - <b>`validation_path`</b>:  Optional alignment used for validation metrics.
 - <b>`weights_path`</b>:  Optional path, sequence, NumPy array, or tensor of  training-sequence weights.
 - <b>`output_dir`</b>:  Directory for parameter, chain, weight, and log files.  If omitted, the result stays in memory and no files are written.
 - <b>`label`</b>:  Stem used for output files when ``output_dir`` is set.
 - <b>`initial_params_path`</b>:  Optional model-parameter file for a warm start.
 - <b>`initial_chains_path`</b>:  Optional Markov-chain file for a warm start.
 - <b>`alphabet`</b>:  Standard alphabet name or explicit ordered token string.
 - <b>`learning_rate`</b>:  Parameter-update step size.
 - <b>`n_sweeps`</b>:  Sampler sweeps per training update.
 - <b>`sampler`</b>:  ``"metropolis"``, ``"gibbs"`` or ``"metropolized_gibbs"``.
 - <b>`n_chains`</b>:  Number of model chains when none are loaded.
 - <b>`target_pearson`</b>:  Target correlation for the two-point statistics.
 - <b>`max_epochs`</b>:  Legacy training-step limit; use the explicit limits below  when distinguishing gradient and structure steps.
 - <b>`max_gradient_steps`</b>:  Optional limit on parameter updates.
 - <b>`max_structure_steps`</b>:  Optional limit on graph updates.
 - <b>`checkpoint_interval`</b>:  Steps between periodic parameter and chain saves.  The final state is saved when ``output_dir`` is set.
 - <b>`pseudocount`</b>:  Empirical-frequency pseudocount, or ``None`` to derive it  from the effective sequence count.
 - <b>`l2_regularization`</b>:  Strength of the L2 penalty on model parameters.
 - <b>`seed`</b>:  Random seed for initialization and sampling.
 - <b>`clustering_seqid`</b>:  Sequence-identity threshold used for reweighting.
 - <b>`no_reweighting`</b>:  Use equal weights instead of sequence-identity weights.
 - <b>`activation_steps`</b>:  Gradient steps per activation update in eaDCA.
 - <b>`activation_fraction`</b>:  Fraction of candidate couplings activated per  structure update in eaDCA.
 - <b>`target_density`</b>:  Target coupling density for edDCA.
 - <b>`decimation_rate`</b>:  Fraction of couplings removed per edDCA update.
 - <b>`device`</b>:  Runtime device; ``"auto"`` selects an available accelerator.
 - <b>`dtype`</b>:  Training precision, ``"float32"``, ``"float64"``, or  ``"bfloat16"``.
 - <b>`use_wandb`</b>:  Enable Weights & Biases logging when output is configured.
 - <b>`allow_signed_weights`</b>:  Permit signed input weights for experimental  reintegration workflows.
 - <b>`progress`</b>:  Optional callback taking exactly one
 - <b>`:class`</b>: `TrainingProgress` event and returning ``None``. It runs synchronously after each recorded metrics update. The event has ``epoch`` (the reported step), ``stage`` (for example, ``"optimization"`` or ``"activation"``), cumulative ``gradient_steps``, ``structure_steps``, and ``sweeps`` counters, and a ``metrics`` dictionary. Numeric metrics are Python floats; keys such as ``"Pearson"`` and ``"Time"`` can vary by stage. For PTT, ``partition_estimate`` may contain normalization diagnostics; otherwise it is ``None``. Callback exceptions propagate and end the training run.
 - <b>`stage_progress`</b>:  Optional callback for transient PTT stage updates.
 - <b>`It receives a `</b>: class:`StageProgress` with ``stage``, ``kind``, completed and total work, and stage-specific details. These events do not add metric rows or alter the ``progress`` callback.
 - <b>`on_initialized`</b>:  Optional callback taking one
 - <b>`:class`</b>: `TrainingInitialization` event. It runs once after input loading and weighting, before numerical training begins.
 - <b>`is_cancelled`</b>:  Optional zero-argument function returning ``True`` to  request cancellation or ``False`` to continue.



**Returns:**

 - <b>`A `</b>: class:`TrainingResult` containing the trained model, metric history, chains, stop reason, input report, and any saved artifacts.



**Raises:**

 - <b>`InputValidationError`</b>:  If the inputs or requested settings are invalid.
 - <b>`OperationCancelledError`</b>:  If ``is_cancelled`` requests cancellation.



**Example:**
 A progress function receives an event object, not separate epoch and metric arguments. Use ``dict.get`` because metrics may differ between stages:
```

         from adabmDCA import train_model          from adabmDCA.api import TrainingProgress

         def show_progress(event: TrainingProgress) -> None:              pearson = event.metrics.get("Pearson")              if pearson is not None:                  print(                      f"{event.stage}: epoch={event.epoch}, "                      f"gradient_steps={event.gradient_steps}, "                      f"Pearson={pearson:.3f}"                  )

         result = train_model("alignment.fasta", progress=show_progress)

```


**Notes:**

> ``dtype="bfloat16"`` uses BF16 coupling copies for CUDA/Triton or MPS sampling and float32 master parameters, statistics, chains, and saved models; it requires NVIDIA Ampere or newer with Triton, or enabled MPS kernels on macOS 14 or newer. PTT requires bmDCA, eaDCA or edgeDCA with float32 or float64 and uses PTT bridges for normalization. For PTT resume, ``max_epochs`` is the total committed-step limit, including steps already saved: gradient steps for bmDCA and edgeDCA, graph-activation (structure) steps for eaDCA.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
