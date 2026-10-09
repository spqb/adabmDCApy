# Python workflows

Everything the command line does is available from Python, through the same functions: the commands are thin wrappers around them, so results, files and errors are identical. In a notebook, load alignments and models once and pass the objects around.

```python
from adabmDCA import (load_alignment, load_model, train_model, sample_sequences,
                      PTTConfig, TrainingConfig)
```

!!! warning "Defaults that differ from the command line"
    - `train_model` and `sample_sequences` use **PCD / plain MCMC unless asked for PTT** (`ptt=PTTConfig()` for training, `ptt=True` for sampling).
    - The validation stop is off unless `PTTConfig(validation_stop=True)`.
    - Workflow functions default to `alphabet="protein"`; only `load_alignment` and `load_model` on a PTT archive detect it. Pass `alphabet=` for RNA, DNA or custom data.

## Alignments

```python
from adabmDCA import load_alignment

alignment = load_alignment("family.fasta")                 # detects DNA, RNA or protein
alignment = load_alignment("family.sto", alphabet="rna",
                           invalid_sequences="drop", remove_duplicates=True)
print(alignment)                                           # sequences, length, tokens
alignment.names, alignment.sequences                       # tuples of strings
one_hot = alignment.to_onehot()                            # torch tensor (M, L, q)
```

An `Alignment` is immutable and records which input rows were retained, invalid or duplicated, so supplied weights can be matched to the right sequences. Every function that takes an alignment accepts a path or an `Alignment`. [`preprocess`](preprocess.md#in-python) shows the cleaning functions (`remove_insertions`, `filter_gap_fraction`, `preprocess_alignment`), and `write_alignment` or `alignment.write_fasta(path)` write it back.

## Models

```python
from adabmDCA import load_model

model = load_model("model/params.dat.gz", alphabet="rna", device="auto")
model = load_model("model/ptt.h5")                         # final model of an archive; alphabet read from it
print(model)                                               # DCAModel(L=..., q=..., tokens=..., device=...)
```

A `DCAModel` keeps the parameters on the chosen device and offers:

```python
energies = model.compute_energies(["ACGU...", "ACGA..."])  # NumPy array
scores   = model.score_sequences(sequences, local_lambda=1.4)   # result object with a DataFrame
contacts = model.compute_contact_map()                     # L × L APC scores
scan     = model.scan_mutations(wild_type, name="wt")      # single-mutant scan
samples  = model.sample(1000, n_sweeps=1000, seed=42)       # tuple of strings, fixed number of sweeps
```

`model.sample` is a quick fixed-length MCMC run without any mixing check; use `sample_sequences` for anything you will analyse.

## Train

```python
from adabmDCA import PTTConfig, train_model

result = train_model(
    "train.fasta", validation_path="validation.fasta", alphabet="rna",
    ptt=PTTConfig(validation_stop=True),
    output_dir="model", label="rf00379", device="cuda", seed=0,
)
result.stop_reason, result.converged                       # 'validation_plateau', True
result.gradient_steps, result.sweeps
result.partition_estimate                                  # log Z of the final model and its provenance
history = result.history_dataframe()                       # same columns as history.csv
model = result.model                                       # a DCAModel
```

All training settings can be gathered in an immutable, validated `TrainingConfig`, convenient for reproducible scripts:

```python
from adabmDCA import TrainingConfig, PTTConfig

config = TrainingConfig(model_type="eaDCA", n_chains=2000, n_sweeps=10,
                        max_gradient_steps=20000, checkpoint_interval=200,
                        ptt=PTTConfig(validation_stop=True, activation="adaptive"))
result = train_model("train.fasta", config=config, validation_path="validation.fasta",
                     output_dir="model_ea")
```

When `config=` is given it is authoritative for all training values; paths, outputs and callbacks remain arguments of `train_model`. `PTTConfig` holds every `--ptt-*` option under the same name without the prefix (see the [reference](../api/adabmDCA.ptt.config.md)).

**Resume** a PTT run with a larger budget:

```python
from dataclasses import replace

resumed = train_model("train.fasta", validation_path="validation.fasta",
                      config=replace(result.config, max_gradient_steps=20000),
                      ptt_resume="model/rf00379_ptt.h5", output_dir="model")
```

A PCD run continues from `initial_params_path=` and `initial_chains_path=`.

**Watch progress.** Four optional callbacks follow a run:

- `on_initialized=` receives the resolved setup (dimensions, `M_eff`, pseudocount, device) before training starts;
- `progress=` receives one event per update, with the update number (`epoch`) and the values of that row of the history (`metrics`);
- `stage_progress=` receives the PTT phases between updates (mixing checks, ladder equilibration, reservoir collection, recovery);
- `is_cancelled=` is polled regularly; when it returns `True`, training stops and raises `OperationCancelledError`.

For example, to print a line every 100 updates, keep the validation curve, and stop after an hour:

```python
import time
from adabmDCA import OperationCancelledError

curve, start = [], time.monotonic()

def on_update(event):
    m = event.metrics
    curve.append((event.epoch, m.get("LL_val")))
    if event.epoch % 100 == 0:
        print(f"update {event.epoch:5d}  Pearson {m['Pearson']:.3f}  "
              f"val LL/site {m.get('LL_val', float('nan')):.4f}  rungs {m.get('ptt_replicas', 0):.0f}")

def on_stage(event):
    if event.kind == "start" and event.stage != "ptt_optimization":
        print(f"  ... {event.stage}")                       # e.g. ptt_mixing, ptt_equilibration

try:
    result = train_model(
        "train.fasta", validation_path="validation.fasta", alphabet="rna",
        ptt=PTTConfig(validation_stop=True), output_dir="model",
        progress=on_update, stage_progress=on_stage,
        is_cancelled=lambda: time.monotonic() - start > 3600,
    )
except OperationCancelledError:
    print("stopped after one hour; the files written so far are in model/")
```

`metrics` holds the same quantities as `history.csv`, under the names `Pearson`, `Pearson_val`, `Slope`, `LL_train`, `LL_val`, `Entropy`, `logZ`, `ptt_replicas`, `ptt_acceptance`, `bias_learning_rate`, `coupling_learning_rate`, `ptt_predicted_kl`, `ptt_lag_drift` and `ptt_lag_drift_tail`. Quantities not yet available, such as the lag in the first updates, are absent.

## Sample

```python
from adabmDCA import sample_sequences

reference = load_alignment("train.fasta")
result = sample_sequences(
    model="model/ptt.h5", ptt=True,
    reference_fasta=reference, test_fasta="validation.fasta",
    n_sequences=2000, collect_diagnostics=True, seed=0,
)
result.sequences, result.energies
df = result.to_dataframe()                                 # sequence, energy, cde_sum
result.local_lambda_fit                                    # slope, intercept, R² of energy vs CDE
result.save_bundle("samples")                              # FASTA, CSV, JSON, logs
result.save_diagnostic_plots("samples")                    # the --plot figures
```

For plain MCMC, pass a `DCAModel` or a parameter file and leave `ptt=False`; `n_sweeps` then bounds the mixing-time measurement and `mixing_multiplier` (default 2) sets the generation length in mixing times.

To run several operations on one archive without reloading it, build the sampler once. Operations fork it and leave it untouched:

```python
from adabmDCA import PTTSampler, estimate_ptt_entropy

backend = PTTSampler.from_archive("model/ptt.h5", device="cuda", seed=7)
samples = sample_sequences(model=backend, ptt=True, reference_fasta=reference, n_sequences=2000)
entropy = estimate_ptt_entropy(model=backend)
health  = backend.ladder_health()                          # per-link overlap and mobility
```

### Steered sampling

`sample_sequences` can bias sampling toward sequences with a property you care about, then correct the bias. You provide a potential \(V(x, s)\); sampling targets \(p_s(x) \propto e^{-E(x) - V(x,s)}\), so low \(V\) is favoured. The potential receives a batch of sequences and the strength \(s\), returns one number per sequence, and must be 0 when \(s = 0\).

```python
import numpy as np

def gc_potential(sequences, strength):
    gc = np.array([sum(c in "GC" for c in s) / max(1, len(s.replace("-", ""))) for s in sequences])
    return -strength * gc                                  # favour GC-rich sequences

steered = sample_sequences(
    model="model/ptt.h5", ptt=True, alphabet="rna", n_sequences=1000,
    steering_potential=gc_potential, steering_strength=20.0,
)
steered.steering                                           # strength, acceptance, effective sample size, log Z ratio
w = np.exp(steered.log_importance_weights - steered.log_importance_weights.max())
gc = np.array([sum(c in "GC" for c in s) / len(s) for s in steered.sequences])
model_gc_mean = np.average(gc, weights=w)                  # estimate for the unsteered model
```

By default the potential receives aligned strings, which suits external tools (folding, structure predictors). `steering_input="onehot"` passes a tensor of shape `(n, L, q)`, faster for potentials written in PyTorch.

Each move proposes a block of Gibbs updates of the Potts model and accepts it with probability \(\min(1, e^{-\Delta V})\), so the potential is called once per chain and move; the block length adapts during warmup (`steering_proposal_steps`). With PTT, steered rungs of increasing strength are added above the trained model, keeping their swap acceptance above `ptt_steering_acceptance`, and the result reports \(\log Z_s - \log Z_0\).

The importance weights correct averages back to the original model only while the steered and original distributions overlap. Check `steered.steering["effective_sample_size"]`: when it falls to a few sequences (a warning is then added), the corrected averages rest on those few sequences. Weaken the steering or collect more samples. See [the mathematics](../quantities/derived.md#steering-and-importance-weights).

## Other workflows

```python
from adabmDCA import (predict_contacts, scan_mutations, score_sequences,
                      estimate_entropy, split_alignment, reintegrate_model)
```

Each has a section on its command page: [contacts](contacts.md#in-python), [energies](energies.md#in-python), [mutations](dms.md#in-python), [entropy](entropy.md#in-python), [split](split-data.md#in-python), [reintegration](reintegrate.md#in-python).

## Results and files

Every workflow returns a result object that owns its serialization:

```python
scores.to_dataframe()
scores.to_csv("scores.csv"); scores.to_json("scores.json"); scores.to_fasta("scores.fasta")
scores.save("scores.json")                                 # format from the suffix
paths = result.save_bundle("outputs", label="run1")       # all files, returns {kind: path}
```

JSON files share an envelope `{"schema_version", "result_type", "data"}`; NumPy arrays, tensors, paths and non-finite numbers are converted to strict JSON. Bundles create their folders and write atomically.

## Errors

All expected failures raise subclasses of `AdabmDCAError` (`InputValidationError`, `ModelLoadError`, `ConvergenceError`, `OperationCancelledError`, …):

```python
from adabmDCA import AdabmDCAError

try:
    model.compute_energies(["INVALID"])
except AdabmDCAError as error:
    print(error.to_dict())     # code, exit_code, message, details
```

The command line prints the same message and exits with code 2 for invalid input, 1 for loading, computation or convergence failures, and 130 for cancellation. Programming errors are not wrapped, so they keep their traceback.

## Low-level functions

The tensor-level building blocks remain available for custom algorithms: `compute_energy(one_hot, params)`, `get_freq_single_point`, `get_freq_two_points`, `get_correlation_two_points`, `compute_weights`, `get_sampler`, `init_chains`, `load_params`, `save_params`. Their signatures are in the [API reference](../api/index.md).
