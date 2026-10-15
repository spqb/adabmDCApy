# adabmDCA Python API

Everything the CLI does is a function of `adabmDCA`; the commands are thin wrappers, so files and errors are identical. Load alignments and models once and pass the objects around.

```python
from adabmDCA import (load_alignment, load_model, inspect_model, train_model, sample_sequences,
                      PTTConfig, TrainingConfig, PTTSampler, estimate_ptt_entropy, estimate_entropy,
                      predict_contacts, scan_mutations, score_sequences, split_alignment,
                      preprocess_alignment, reintegrate_model, AdabmDCAError, OperationCancelledError)
```

## Defaults that differ from the CLI

- `train_model` trains with **PCD unless `ptt=PTTConfig(...)`** is passed.
- The validation stop is off unless `PTTConfig(validation_stop=True)`.
- `sample_sequences` uses **plain MCMC unless `ptt=True`**, and `n_sequences` is required.
- Workflow functions default to `alphabet="protein"`. Only `load_alignment` and `load_model` on a `ptt.h5` detect it. Pass `alphabet="rna"`, `"dna"` or a token string otherwise.

## Alignments and models

```python
aln = load_alignment("family.fasta")                         # detects DNA/RNA/protein
aln = load_alignment("family.sto", alphabet="rna", invalid_sequences="drop", remove_duplicates=True)
aln.names, aln.sequences, aln.to_onehot()                    # tuples of str; torch (M, L, q)
aln.write_fasta("clean.fasta")

result = preprocess_alignment("family.sto", output_path="family.fasta", remove_insertions=True,
                              max_gap_fraction=0.2, remove_duplicates=True, alphabet="auto",
                              unknown_tokens="gap")
result.report.to_dict(); result.keep_mask

split = split_alignment(aln, method="clustering", identity=0.8, train_fraction=0.8, alphabet="rna", seed=0)
split.training, split.test; split.save_bundle("family")     # family.train/test.fasta, family.split.json

meta = inspect_model("model/params.dat.gz", alphabet="rna")  # L, q, tokens, device, precision; no compute
model = load_model("model/params.dat.gz", alphabet="rna", device="auto")
model = load_model("model/ptt.h5")                           # final model; alphabet read from the archive
```

`DCAModel` methods:

```python
model.compute_energies(["ACGU...", ...])                     # NumPy array
model.score_sequences(seqs, local_lambda=1.4).to_dataframe() # energy, cde_sum, local_free_energy
model.compute_contact_map()                                  # L × L APC scores (NumPy)
model.scan_mutations(wild_type, name="wt").to_dataframe()    # single-mutant scan
model.sample(1000, n_sweeps=1000, seed=42)                   # quick fixed-length MCMC, NO mixing check
```

Use `sample_sequences`, not `model.sample`, for anything you will analyse.

Function forms take `model=` a `DCAModel` or a path: `score_sequences(seqs, model=..., fasta_path=None)`, `scan_mutations(wt, model=...)`, `predict_contacts(model=...)` or `predict_contacts(fasta_path=...)` (mean-field).

## Train

```python
result = train_model(
    "train.fasta", validation_path="validation.fasta", alphabet="rna",
    ptt=PTTConfig(validation_stop=True),
    output_dir="model", label="run1", device="auto", seed=0,
)
result.stop_reason, result.converged, result.gradient_steps, result.sweeps
result.partition_estimate          # log Z of the final model and its provenance
result.history_dataframe()         # per-update metrics: Pearson, Pearson_val, LL_train, LL_val, logZ, ...
result.model                       # DCAModel
```

Keyword names (CLI → Python): `--lr` → `learning_rate`, `--nsweeps` → `n_sweeps`, `--nchains` → `n_chains`, `--target` → `target_pearson`, `--nepochs` → `max_epochs`, `-m` → `model_type`, `--l2_reg` → `l2_regularization`, `--gsteps` → `activation_steps`, `--factivate` → `activation_fraction`, `--density` → `target_density`, `--drate` → `decimation_rate`, `-w` → `weights_path`, `-p`/`-c` → `initial_params_path`/`initial_chains_path`, `--ptt-resume` → `ptt_resume`. `PTTConfig` holds every `--ptt-*` option under the same name without the prefix and with underscores (e.g. `PTTConfig(validation_stop=True, activation="adaptive", trust_radius=0.01)`).

A `TrainingConfig` gathers all training values; when passed as `config=` it is authoritative:

```python
config = TrainingConfig(model_type="eaDCA", n_chains=2048, n_sweeps=10, max_gradient_steps=20000,
                        ptt=PTTConfig(validation_stop=True, activation="adaptive"))
result = train_model("train.fasta", config=config, validation_path="validation.fasta", output_dir="model_ea")

from dataclasses import replace                                # resume PTT with a larger budget
train_model("train.fasta", validation_path="validation.fasta",
            config=replace(result.config, max_gradient_steps=40000),
            ptt_resume="model_ea/ptt.h5", output_dir="model_ea")
```

### Progress and cancellation

- `on_initialized=` gets the resolved setup (dimensions, M_eff, pseudocount, device).
- `progress=` gets one event per update: `event.epoch` and `event.metrics` with keys `Pearson`, `Pearson_val`, `Slope`, `LL_train`, `LL_val`, `Entropy`, `logZ`, `ptt_replicas`, `ptt_acceptance`, `bias_learning_rate`, `coupling_learning_rate`, `ptt_predicted_kl`, `ptt_lag_drift`, `ptt_lag_drift_tail` (absent until available).
- `stage_progress=` gets PTT phases between updates (`event.stage`, `event.kind`), e.g. `ptt_mixing`, `ptt_equilibration`.
- `is_cancelled=` is polled; returning `True` stops training and raises `OperationCancelledError`. Files written so far stay in `output_dir`.

```python
import time
start = time.monotonic()
try:
    train_model("train.fasta", validation_path="validation.fasta", alphabet="rna",
                ptt=PTTConfig(validation_stop=True), output_dir="model",
                progress=lambda e: e.epoch % 100 == 0 and print(e.epoch, e.metrics.get("Pearson")),
                is_cancelled=lambda: time.monotonic() - start > 3600)
except OperationCancelledError:
    pass
```

## Sample

```python
ref = load_alignment("train.fasta")
res = sample_sequences(model="model/ptt.h5", ptt=True, reference_fasta=ref, test_fasta="validation.fasta",
                       n_sequences=2048, collect_diagnostics=True, ptt_return_full_ladder=False, seed=0)
res.sequences, res.energies, res.to_dataframe()             # sequence, energy, cde_sum
res.local_lambda_fit                                        # slope, intercept, R² of energy vs CDE
res.ptt_diagnostics["converged"], res.ptt_diagnostics["final_pearson"]
res.save_bundle("samples")                                  # FASTA, CSV, JSON, logs
res.save_diagnostic_plots("samples")                        # the --plot figures
res.to_ladder_hdf5("samples/ladder_samples.h5")             # needs ptt_return_full_ladder=True
```

Plain MCMC: pass a `DCAModel` or parameter file and leave `ptt=False`; `n_sweeps` bounds the mixing-time measurement and `mixing_multiplier` (2) sets the generation length.

Reuse one archive for several operations without reloading:

```python
backend = PTTSampler.from_archive("model/ptt.h5", device="cuda", seed=7)
samples = sample_sequences(model=backend, ptt=True, reference_fasta=ref, n_sequences=2048)
entropy = estimate_ptt_entropy(model=backend)
health = backend.ladder_health()
```

### Steered sampling (Python only)

Bias sampling toward a property with a potential V(x, s) (low V favoured, V = 0 at s = 0), then reweight back to the model:

```python
import numpy as np
def gc_potential(sequences, strength):
    gc = np.array([sum(c in "GC" for c in s) / max(1, len(s.replace("-", ""))) for s in sequences])
    return -strength * gc

st = sample_sequences(model="model/ptt.h5", ptt=True, alphabet="rna", n_sequences=1000,
                      steering_potential=gc_potential, steering_strength=20.0)
st.steering["effective_sample_size"]          # if only a few, corrected averages are unreliable
w = np.exp(st.log_importance_weights - st.log_importance_weights.max())
```

`steering_input="onehot"` passes an `(n, L, q)` tensor instead of strings.

## Entropy and reintegration

```python
estimate_ptt_entropy(model="model/ptt.h5", n_sweeps=10).entropy      # also .log_z
estimate_entropy(model="model_pcd/params.dat.gz", natural_alignment="train.fasta",
                 target_alignment="target.fasta", n_steps=100, alphabet="rna")
reintegrate_model(load_alignment("natural.fasta"), load_alignment("tested.fasta"),
                  np.loadtxt("adjustments.txt"), output_dir="reintegrated")   # .lambda_value, .training
```

## Results and errors

Every result object serializes itself: `to_dataframe()`, `to_csv()`, `to_json()`, `to_fasta()`, `save(path)` (format from suffix), `save_bundle(dir, label=...)` → `{kind: path}`. JSON files share the envelope `{"schema_version", "result_type", "data"}`.

Expected failures raise subclasses of `AdabmDCAError` (`InputValidationError`, `ModelLoadError`, `ConvergenceError`, `OperationCancelledError`, …); `error.to_dict()` gives `code`, `exit_code`, `message`, `details`.

Low-level building blocks (`compute_energy`, `get_freq_single_point`, `get_freq_two_points`, `compute_weights`, `get_sampler`, `init_chains`, `load_params`, `save_params`) are documented in the API reference: https://spqb.github.io/adabmDCApy/api/.
