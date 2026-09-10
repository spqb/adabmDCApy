# <span id="bmdca">Training DCA models</span>

The Python, Julia, and C++ implementations use the same general command shape.
This guide documents the Python implementation; run
`adabmDCA train --help` for the exact options in an installed release.

## Training output and logs

After the input alignment has been validated and weighted, the command prints
the resolved configuration together with the number of retained sequences
(`M`), sequence length (`L`), alphabet size (`q`), and effective number of
sequences (`M_eff`). Validation-alignment statistics are included when a
validation input is supplied.

When an output directory is configured, training writes a version-2 text log.
Its `RUN`, `TRAINING DATA`, `VALIDATION DATA`, `RUNTIME`, and `OPTIMIZATION`
sections capture the resolved setup. Each training stage has a progress table,
and the final `END` section records whether the run completed, failed, was
cancelled, or was interrupted. `Chain_ESS_frac` in the progress table is the
normalized effective sample size of the model chains; it is distinct from the
alignment-level `M_eff`.

The log can be plotted with:

```bash
adabmDCA plot-training-log path/to/model.log
```

To see the complete list of training options:

```bash
$ adabmDCA train -h
```

The standard command to start training a DCA model is:

```bash
$ adabmDCA train -m <model> -d <fasta_file> -o <output_folder> -l <label>
```

## Arguments

- **`<model>`** ∈ `{bmDCA, eaDCA, edDCA, edgeDCA}`  
  Selects the training routine.  
  By default, the fully connected `bmDCA` algorithm is used. `eaDCA` and `edgeDCA` build sparse models by activating couplings during training, while `edDCA` sparsifies a model by decimation. `edDCA` can either decimate a pre-trained `bmDCA` model, or first train a `bmDCA` model and then decimate it.
- **`<fasta_file>`** – Path to the FASTA file containing the training MSA.
- **`<output_folder>`** – Folder where results will be stored (created if missing).
- **`<label>`** – Optional tag for output files.

---

## Training Behavior

Training stops when the **Pearson correlation** between model and empirical connected correlations reaches the target value (default: `0.95`).

Training reports gradient and graph-structure work separately. A gradient step
updates model parameters; a structure step activates or decimates graph
elements. For backward compatibility, `--nepochs` limits gradient steps for
`bmDCA` and structure steps for `eaDCA`, `edDCA`, and `edgeDCA`. Use the
unambiguous limits when controlling nested sparse-model training:

```bash
--max-gradient-steps 5000 --max-structure-steps 100
```

The final summary reports the stop reason and both counters, so reaching a
budget is distinguishable from reaching the Pearson or density target.

Python callers can use `TrainingConfig` as the canonical source of defaults
and validation. It also makes advanced runtime values explicit, including
checkpoint cadence, edDCA's inner gradient budget, convergence slope
tolerance, and edgeDCA's empirical-frequency and log-partition estimation
settings. The CLI and high-level API obtain their defaults from this same
configuration module.

Training inputs may be FASTA, compressed FASTA, Stockholm, or an in-memory
`Alignment`. Invalid sequences are dropped and duplicate sequences are
removed using an explicit retained-index mapping; supplied weights may refer
to either the original or retained alignment. The returned
`TrainingResult.input_report` records every filtering decision.

- Early training is fast (e.g., Pearson ≈ 0.9 after ~100 iterations).  
- Approaching higher values takes significantly longer (power‑law decay).

For a quick coarse model, set:

```
--target 0.9
```

---

## Optional bfloat16 sampling during training

The Python implementation supports mixed-precision training on NVIDIA Ampere
or newer CUDA GPUs with Triton installed:

```bash
adabmDCA train -m bmDCA -d alignment.fasta -o model --device cuda --dtype bfloat16
```

In Python, use `train_model(..., device="cuda", dtype="bfloat16")` or
`TrainingConfig(device="cuda", dtype="bfloat16")`. Both Gibbs and Metropolis
support this mode for all four training algorithms. The default remains
`float32`; `float64` is also unchanged.

`bfloat16` is a mixed-precision training mode: fresh BF16 coupling copies are
used inside sampling, while biases, master parameters, chains, frequency
estimates, parameter updates and AIS/energy calculations remain FP32. The
kernels convert loaded couplings to FP32 before arithmetic, and uniform random
numbers remain FP32. Gibbs combines quantization and transposition in one
copy. Nothing is cached across parameter updates.

Saved models remain FP32 and work with the existing sampling, scoring and
resume workflows. `result.config.dtype` records `bfloat16`, while
`result.model.metadata.dtype` reports the actual master dtype, `float32`.
Use `--dtype bfloat16` again when resuming to retain mixed-precision sampling.
The `sample` command also accepts `--dtype bfloat16` for fixed-model sampling;
its parameters, chains, diagnostics, and reported energies remain FP32.

Rounding couplings slightly changes the sampled model and can change the
training trajectory. This mode is optional; compare convergence and final
statistics for your data. It reduces sampling coupling bandwidth, but does
not halve total training memory or accelerate FP32 statistics and energy
calculations. Overall speedups depend on the workload. CPU and pre-Ampere
GPUs reject this mode with an explicit error.

---

## Output Files

During training, adabmDCA maintains three output files:

- **`<label>_params.dat`** – Non‑zero model parameters  
  - Lines starting with `J` → couplings  
  - Lines with `h` → biases

- **`<label>_chains.fasta`** – State of the Markov chains

- **`<label>.log`** – Versioned training log updated throughout training

Parameters and chains are saved every **100 training steps** by default for
all models. Set a positive interval with `--checkpoint-interval`, for example:

```bash
adabmDCA train -d alignment.fasta -o model --checkpoint-interval 200
```

In Python, use `train_model(..., checkpoint_interval=200)` or
`TrainingConfig(checkpoint_interval=200)`. Checkpoints follow the training
stage's step counter (gradient updates during optimization, graph updates
during activation/decimation). Final states and explicit phase-boundary
snapshots are saved regardless of the interval. Metrics are still logged
every step.

---

## Restoring Interrupted Training

Resume training using:

```bash
$ adabmDCA train [...] -p <file_params> -c <file_chains>
```

---

## Importance Weights

Provide custom weights with:

```bash
--weights <path>
```

Otherwise, weights are computed automatically and stored as:

```
<label>_weights.dat
```

Options:

- `--clustering_seqid <value>` – default: 0.8  
- `--no_reweighting` – use uniform weights  

---

## Choosing the Alphabet

The command-line default is **`auto`**. It detects standard DNA, RNA, or
protein data after normalizing alignment gaps. DNA wins ambiguous nucleotide
matches, so an alignment containing only `A`, `C`, and `G` is classified as
DNA.

Specify an alphabet when detection is ambiguous or the tokens are custom:

- RNA → `--alphabet rna`
- DNA → `--alphabet dna`
- Custom →  
  ```
  --alphabet ABCD-
  ```

---
# <span id="eadca">eaDCA</span>

Enable with:

```
--model eaDCA
```

Key hyperparameters:

- `--factivate` – fraction of inactive couplings activated (default: 0.001)  
- `--gsteps` – parameter updates per graph update (default: 10)

Recommended: reduce sweeps to **5**.

---

# <span id="eddca">edDCA (Decimated DCA)</span>

Run decimation:

```bash
$ adabmDCA train -m edDCA -d <fasta_file> -p <params> -c <chains>
```

Two workflows:

1. Use pre‑trained bmDCA (`params` + `chains`)
2. Train bmDCA automatically, then decimate

Key hyperparameters:

- `--gsteps` – default: 10  
- `--drate` – pruning fraction (default: 0.01)  
- `--density` – target graph density (default: 0.02)  
- `--target` – Pearson threshold (default: 0.95)

---

# <span id="edgedca">edgeDCA (Edge Activation DCA)</span>

Enable with:

```
--model edgeDCA
```

`edgeDCA` trains a sparse model by starting from an empty coupling graph and progressively activating whole residue-residue edges. At each graph update, the algorithm compares empirical and model two-site statistics, selects the inactive edge with the largest KL discrepancy, activates all couplings for that pair of sites, and initializes them from the empirical/model frequency ratio. The Markov chains are then resampled and the process repeats until the target Pearson correlation is reached or the maximum number of epochs is exceeded.

This differs from `eaDCA`, which activates individual coupling entries. `edgeDCA` activates complete site pairs, so it is useful when you want a sparse interaction graph at the residue-pair level.

Important defaults:

- `--pseudocount`: if not set, defaults to `0.1` for `edgeDCA`
- `--lr`: ignored by `edgeDCA`
- `--target`: Pearson threshold on two-site statistics, default `0.95`
- `--nsweeps`: Monte Carlo sweeps between edge activations, default `10`

For `edgeDCA`, the pseudocount acts as an effective learning rate for edge activation: the closer the pseudocount is to `1`, the smaller the effective learning rate is. Some datasets can require much stronger regularization, with `--pseudocount` values up to `0.95`. The explicit learning-rate parameter `--lr` is not used by `edgeDCA`.

Example:

```bash
$ adabmDCA train -m edgeDCA -d <fasta_file> -o <output_folder>
```

---

# Choosing Hyperparameters

Defaults work well for clean and moderately diverse MSAs. For more difficult datasets, consider tuning:

---

### Learning Rate

- Default: **0.01**  
- If chains mix poorly, try:  
  ```
  --lr 0.005
  ```
- For `edgeDCA`, `--lr` is ignored; tune `--pseudocount` instead.

### Number of Markov Chains

- Default: **10,000**  
- Using fewer chains reduces the memory required to train the model, but it may also lead to a longer algorithm convergence time.  
- Change with:  
  ```
  --nchains <value>
  ```

### Number of Monte Carlo Steps

- Controlled by `--nsweeps`  
- Default: **10**  
- Recommended range: **10–50**. Higher values drastically increase the training time and, empirically, do not help much the model convergence.

### Regularization (Pseudocount)

Controlled by `--pseudocount`.

Default for `bmDCA`, `eaDCA`, and `edDCA`:
```
α = 1 / M_eff
```

Default for `edgeDCA`:
```
α = 0.1
```

For `bmDCA`, `eaDCA`, and `edDCA`, increasing α (e.g. α = 0.001 or 0.01) may help when the training struggle converging or the mixing time of the model is very high, but it also makes the model less expressive. For `edgeDCA`, α controls the effective learning rate of edge activation: values closer to `1` make updates smaller, and values up to `0.95` can be useful on some datasets. `edgeDCA` still points towards the original statistics, without pseudocount, so the pseudocount does not interfere with the expressivity of the model.
