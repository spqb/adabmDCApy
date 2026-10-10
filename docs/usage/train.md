# Train a model · `train`

```bash
adabmDCA train -d train.fasta -v validation.fasta -o model
```

This trains a fully connected Potts model (`bmDCA`) with Parallel Trajectory Tempering, the default strategy: 2,000 chains, 10 local Metropolized Gibbs sweeps per exchange round, learning rate 0.01, and the adaptive optimizer. Because a validation set is given, training stops when the validation log-likelihood per site stops improving. Without `-v`, it stops when the Pearson correlation between the connected correlations of the data and of the model reaches 0.95.

Sequences are reweighted by default so that clusters of close homologues count less: each sequence gets a weight inversely proportional to the number of sequences within 80% identity of it.

## Common variants

```bash
# Fast first pass with persistent contrastive divergence
adabmDCA train --strategy pcd -d train.fasta -o model_pcd

# A coarser model, quickly (Pearson rises fast up to ~0.9, then slowly)
adabmDCA train --strategy pcd -d train.fasta -o model_pcd --target 0.9

# A family with narrow subfamilies, after a run failed with trapped configurations
adabmDCA train -d train.fasta -v validation.fasta -o model --nsweeps 100

# Sparse models
adabmDCA train -m eaDCA   -d train.fasta -v validation.fasta -o model_ea
adabmDCA train -m edgeDCA -d train.fasta -v validation.fasta -o model_edge
adabmDCA train --strategy pcd -m edDCA -d train.fasta -o model_ed --density 0.02

# Equal weights for all sequences, for instance when the alignment is not redundant
adabmDCA train -d train.fasta -v validation.fasta -o model --no_reweighting

# Plot the training curves when training ends
adabmDCA train -d train.fasta -v validation.fasta -o model --plot-training-logs
```

[Choosing settings](choosing.md) explains when each variant is worth it.

## Main options

| Option | Default | Meaning |
| --- | --- | --- |
| `-d`, `--data` | required | Training alignment (FASTA, gzip FASTA or Stockholm) |
| `-v`, `--validation` | none | Validation alignment. Enables validation metrics and, with PTT, the validation stop (`--val` is an older alias) |
| `-o`, `--output` | `DCA_model` | Output folder |
| `-l`, `--label` | none | Prefix of the output files, to keep several runs in one folder |
| `-m`, `--model` | `bmDCA` | `bmDCA`, `eaDCA`, `edDCA` (PCD only) or `edgeDCA` |
| `--strategy` | `ptt` | `ptt` or `pcd` |
| `--alphabet` | `auto` | `protein`, `rna`, `dna`, or custom tokens |
| `--nchains` | 2000 | Number of Markov chains (per replica with PTT) |
| `--nsweeps` | 10 | Local sweeps per update (PCD) or per exchange round and replica (PTT) |
| `--sampler` | `metropolized_gibbs` | Local move: `metropolized_gibbs`, `gibbs` or `metropolis` |
| `--lr` | 0.01 | Learning rate; an upper bound for the adaptive PTT optimizer. Ignored by `edgeDCA` |
| `--target` | 0.95 | Pearson target on the connected correlations |
| `--pseudocount` | 1/M_eff (0.1 for edgeDCA) | Regularization of the data frequencies |
| `--l2_reg` | 0 | L2 penalty on the couplings |
| `-w`, `--weights` | computed | File with one weight per sequence |
| `--clustering_seqid` | 0.8 | Identity threshold of the automatic weights |
| `--no_reweighting` | off | Give every sequence weight 1 |
| `--max-gradient-steps` | none | Maximum number of parameter updates |
| `--max-structure-steps` | none | Maximum number of graph activations or decimations (sparse models) |
| `--nepochs` | 50000 | Older budget: gradient steps for bmDCA and PTT edgeDCA, graph steps for eaDCA and edDCA |
| `--checkpoint-interval` | 500 | Updates between saved checkpoints |
| `--device`, `--dtype` | `auto`, `float32` | See [Installation](../installation.md#choose-a-device) |
| `--seed` | 0 | Random seed |
| `--diagnostic` | off | Print the full progress table and every event in the terminal (PTT) |
| `--no-progress` | off | No terminal progress display; the files are still written |
| `--wandb` | off | Also log metrics to Weights & Biases |

Sparse models have their own options: `--gsteps` (10, parameter updates per graph change) and `--factivate` (0.001, fraction of the inactive couplings activated each time) for eaDCA; `--density` (0.02, final fraction of couplings) and `--drate` (0.01, fraction removed per step) for edDCA. For edgeDCA the pseudocount, not `--lr`, sets the size of each update: values closer to 1 give smaller steps, and some datasets need values up to 0.95.

PTT has about thirty further options, all prefixed `--ptt-`. The defaults rarely need changing; the ones that do are discussed in [Diagnose a run](../algorithms/diagnostics.md#hyperparameters-that-matter-for-diagnosis), and the full list is in `adabmDCA train --help` and the [PTT configuration reference](../api/adabmDCA.ptt.config.md).

## What you see while it runs

The command first prints the resolved configuration and the input summary: retained sequences `M`, length `L`, alphabet size `q` and effective number of sequences `M_eff`. A single status line then shows the number of updates, the Pearson correlation, and the PTT phase underway (optimizing, mixing check, reservoir collection, recovery…). Ladder changes, pauses and recoveries also get a permanent line starting with `»`. With `--diagnostic`, the terminal shows the full progress table of the log.

## Outputs

| File | Content |
| --- | --- |
| `params.dat.gz` | The model: fields and couplings in text format, gzip-compressed. Rewritten at every checkpoint |
| `chains.fasta` | The current Markov chains (the endpoint chains with PTT) |
| `weights.dat` | The weight of each training sequence |
| `history.csv` | One row per update: Pearson, slope, likelihoods, ladder and optimizer state… |
| `events.jsonl` | One JSON object per event: checkpoints, mixing checks, ladder changes, pauses, recoveries, the stop |
| `adabmDCA.log` | Human-readable log: configuration, data summary, progress table with events, end status |
| `training.json` | Final summary: stop reason, counters, final metrics, configuration, input report |
| `ptt.h5` | PTT only: the full archive (ladder, chains, optimizer, restart state) |
| `training_plots/` | With `--plot-training-logs`: one figure per recorded quantity |

With `-l name`, every file is prefixed with `name_`. [The training pipeline](../algorithms/training.md#outputs-and-logs) lists every column of `history.csv` and every event.

A PTT log looks like this (RF00379):

```text
   step   sweeps      time  pearson      val    ll/L  ll_val  reps   acc     lr_h     lr_J   lag  tail
»   189  snapshot replaced, swap acceptance 0.24
»   189  mixing check converged: renewal warmup 36 rounds, stationary renewal 47 rounds, trapped endpoint fraction 0.000
»   189  reservoir refreshed: 20000 samples, 2 active replicas, warmup 35 rounds, spacing 70 rounds, min batch fresh 0.999
   ...
   2475     153k   0:02:52   0.9518   0.8062  -0.583  -0.711     3  0.46     0.01     0.01  0.01  0.01
»  2475  validation plateau: median validation log-likelihood per site of the last 100 updates -3.2e-06 against the 100 before

[END]
status: completed
stop_reason: validation_plateau
```

`grep '»' adabmDCA.log` lists the events of a run.

## How a run ends

`training.json` (`data.stop_reason`) and the `[END]` section of the log say why training stopped:

| Stop reason | Meaning |
| --- | --- |
| `validation_plateau` | The validation log-likelihood stopped improving (PTT with `-v`). The usual, desired end |
| `target_pearson` | The Pearson target was reached |
| `target_density` | edDCA reached its target density |
| `graph_converged` | eaDCA with adaptive activation found no significant coupling left to add |
| `max_gradient_steps`, `max_structure_steps`, `max_epochs` | A budget ran out before the target. The model is usable but not converged |
| `cancelled` | Interrupted |

With PTT, training can also end with an error after repeated failed mixing checks. The message names the most likely cause and the setting to change; the state before the failure is kept in `ptt.h5`. See [Diagnose a run](../algorithms/diagnostics.md).

!!! warning "A finished run is not yet a validated model"
    Training metrics are computed from the chains used for training. If those chains fell behind the model, the metrics can look good while the model is wrong. Before relying on a model, sample it independently with [`sample`](sample.md) and go through the [checklist for a finished model](../algorithms/diagnostics.md#checking-a-finished-model). On healthy runs, the Pearson of independent samples is 0.003–0.007 below the training value at equal sample size; a larger drop deserves attention.

## Resume a run

**PTT.** Repeat the original command with the archive and a larger update budget:

```bash
adabmDCA train -d train.fasta -v validation.fasta -o model \
    --ptt-resume model/ptt.h5 --max-gradient-steps 20000
```

The archive restores the ladder, the chains, the optimizer and the random state, so a resumed run continues exactly. Only the update budget, the checkpoint interval and the logging options may change. `history.csv` is rewritten from the archive, and the log and events are appended. Do not resume a run that failed because configurations were trapped: restart it from scratch with more sweeps instead.

**PCD.** Start from the saved parameters and chains:

```bash
adabmDCA train --strategy pcd -d train.fasta -o model_pcd \
    -p model_pcd/params.dat.gz -c model_pcd/chains.fasta
```

**edDCA** uses the same two options to decimate an existing bmDCA model; without them, it first trains the dense model itself.

## Precision and speed

`--dtype bfloat16` stores the couplings used inside the sampler in BF16, keeping parameters, chains and statistics in FP32. It needs an Ampere-or-newer NVIDIA GPU with Triton, or enabled MPS kernels on macOS 14 or newer. PTT does not support BF16. Performance depends on the model and device: CUDA tests gave at most about 12% speed-up; dense MPS protein sampling can benefit more (see [MPS benchmarks](../algorithms/benchmarks.md#mps-vs-cpu)). Saved models stay FP32. Pass `--dtype bfloat16` again when resuming.

## In Python

```python
from adabmDCA import PTTConfig, load_alignment, train_model

train = load_alignment("train.fasta")
validation = load_alignment("validation.fasta")
result = train_model(
    train, validation_path=validation, alphabet="rna",
    ptt=PTTConfig(validation_stop=True), output_dir="model",
)
print(result.stop_reason, result.converged, result.gradient_steps)
history = result.history_dataframe()
```

The Python defaults differ from the command line in two ways: training uses **PCD unless a `PTTConfig` is passed** as `ptt=`, and the validation stop is off unless requested with `PTTConfig(validation_stop=True)`. The other defaults are the same. More in [Python workflows](python.md#train).
