# adabmDCA output files

With `-l name`, every file below is prefixed `name_`. JSON summaries share the envelope `{"schema_version", "result_type", "data"}`: read `data`.

## Training folder (`train`, `reintegrate`)

| File | Content |
| --- | --- |
| `params.dat.gz` | The model, rewritten at every checkpoint |
| `chains.fasta` | Current Markov chains (endpoint chains with PTT) |
| `weights.dat` | One weight per retained training sequence |
| `history.csv` | One row per update |
| `events.jsonl` | One JSON object per event |
| `adabmDCA.log` | Human-readable log: configuration, data summary, progress table with events, `[END]` section |
| `training.json` | Final summary, written when the run ends |
| `ptt.h5` | PTT only: full archive (ladder, chains, optimizer, random state) for sampling, entropy and resuming |

### `params.dat.gz`

Gzip text, one line per non-zero parameter, zero-based positions, states written as alphabet symbols:

```text
J 0 1 - A -0.022051653      # coupling J_ij(a, b): J i j a b value
h 0 A -1.191999             # field h_i(a):        h i a value
```

Every command reads it compressed or not; `zcat params.dat.gz` gives plain text.

### `training.json` → `data`

Keys: `stop_reason`, `converged`, `gradient_steps`, `structure_steps`, `sweeps`, `final_metrics`, `partition_estimate` (`log_z`, `method`, `status`), `warnings`, `config` (resolved settings, `ptt` nested), `input_report` (`original_sequences`, `retained_sequences`, dropped and duplicate indices, `sequence_length`, `tokens`), `num_sequences`, `effective_sequences`, `pseudocount`, `artifacts` (paths). `final_metrics` holds `Pearson`, `Pearson_val`, `Slope`, `LL_train`, `LL_val` (per site), `Entropy`, `logZ`, `ptt_acceptance`, `ptt_replicas`, learning rates and lag values.

| `stop_reason` | Meaning |
| --- | --- |
| `validation_plateau` | Validation log-likelihood stopped improving (PTT with `-v`). The desired end |
| `target_pearson` | Pearson target reached |
| `target_density` | edDCA reached its density |
| `graph_converged` | eaDCA with adaptive activation found nothing left to add |
| `max_gradient_steps`, `max_structure_steps`, `max_epochs` | Budget ran out: usable, **not converged** (`converged: false`) |
| `cancelled` | Interrupted |

A PTT run can also end with an error after repeated failed mixing checks; the message names the cause and the setting to change, and `ptt.h5` keeps the last good state. While a run is going, `training.json` does not exist yet: use `history.csv` and `events.jsonl`.

### `history.csv`

Columns: `step, sweeps, time_s, pearson, slope, pearson_val, slope_val, ll_train, ll_val, entropy, logz, replicas, models, acceptance_min, lr_bias, lr_coupling, kl, lag_drift, lag_drift_tail` (PTT; PCD has a subset). `ll_*` are log-likelihoods per site; `entropy` and `logz` per sequence. `time_s` excludes PTT mixing and reservoir phases. Empty cells mean not yet available.

```python
import pandas as pd
h = pd.read_csv("model/history.csv")
h[["step", "pearson", "pearson_val", "ll_val", "replicas", "acceptance_min"]].tail()
```

### `events.jsonl`

One object per line with `event`, `step`, `utc`, `sweeps` and event-specific fields. Main events: `checkpoint_saved`, `ptt_equilibration`, `ptt_optimization`, `ptt_mixing` (`status`: `converged` or `budget_exceeded`, `immobile`), `ptt_snapshot_*`, `ptt_reservoir_refreshed`, `ptt_lag_pause`, `ptt_recovery`, `ptt_learning_rate`. The same events appear in `adabmDCA.log` on lines starting with `»` (`grep '»' model/adabmDCA.log`).

## Sampling folder (`sample`)

| File | Content |
| --- | --- |
| `samples.fasta` | Generated sequences |
| `samples.csv` | `name, sequence, energy, cde_sum` (+ PRIVET columns with `-v`) |
| `sampling.json` | Settings, mixing outcome, every diagnostic as numbers |
| `ladder_samples.h5` | With `--ptt-return-full-ladder` |
| `logs/` | `mix.log`, `sampling.log`, `data_vs_samples.log`, `distances.log`; PTT: `ptt.log`, `ptt_renewal.log`, `ptt_ladder_health.log`, `ptt_replicas.log` |
| `*.png` | With `--plot`; see [cli.md](cli.md#sample) |

`sampling.json` → `data` keys: `sequences`, `energies`, `cde_sum`, `num_sweeps`, `sampler`, `seed`, `model`, `local_lambda_fit` (slope λ, intercept, R² of energy vs CDE), `data_comparison`, `distance_comparison`, `mixing_history`, `sampling_history`, `warnings`, and for PTT `ptt_diagnostics` with `converged`, `final_pearson`, `final_slope`, `mixing`, `ladder_health`, `models`, `timings_seconds`.

```python
import json
d = json.load(open("samples/sampling.json"))["data"]
d["ptt_diagnostics"]["converged"], d["ptt_diagnostics"]["final_pearson"], d["warnings"]
```

`logs/ptt.log` (CSV): one row per ladder model, profile first: `replica, model_id, log_z, log_z_forward, log_z_estimator, mean_energy, entropy, log_likelihood, log_likelihood_per_residue, steering_strength, sample_round, n_chains, training_step, flag` (`flag` is `ptt` for the profile and `final` for the endpoint).

### `ladder_samples.h5`

```python
import h5py
with h5py.File("samples/ladder_samples.h5") as f:
    tokens = f.attrs["tokens"]               # string, index -> symbol
    seqs = f["sequences"][:]                 # (n_models, n_sequences, L) token indices; [0] profile, [-1] endpoint
    energies = f["energies"][:]              # (n_models, n_sequences), each under its own model
    steps = f["models/training_step"][:]     # plus log_z, entropy, log_likelihood, flag, ... per model
endpoint = ["".join(tokens[i] for i in row) for row in seqs[-1]]
```

## Analysis outputs

| Command | Files | Columns / keys |
| --- | --- | --- |
| `energies` | `<stem>_energies.{fasta,csv,json}` | `name, sequence, energy` (+ `cde_sum, local_free_energy` with `--local-lambda`) |
| `dms` | `<wt name>_DMS.{fasta,csv,json}` | `mutation, position, position_1based, wild_type, mutant, delta_energy, sequence` |
| `contacts` | `contact_map.{csv,txt,npy,json,png}` | `position_i, position_j, score` (all pairs, zero-based) |
| `entropy` (PTT) | `entropy.json` | `entropy, log_z, mean_energy, free_energy, method, status, model_id` |
| `entropy` (TI) | `entropy.{log,csv,json}` | integration path; final entropy, free energy, `theta_max` |
| `split-data` | `<prefix>.{train,test}.fasta`, `<prefix>.split.json` | |
| `preprocess` | output FASTA, optional `--report` JSON | |

Top contacts:

```python
import pandas as pd
c = pd.read_csv("contacts/contact_map.csv")
c = c[(c.position_j - c.position_i) > 4].sort_values("score", ascending=False)
top = c.head(L)   # L = alignment length
```
