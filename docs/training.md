# <span id="bmdca">Training DCA models 🚀</span>

All versions of **adabmDCA** — Python, Julia, and C++ — expose the same command-line interface through the `adabmDCA` command.

To see the complete list of training options:

```bash
$ adabmDCA train -h
```

The standard command to start training a DCA model is:

```bash
$ adabmDCA train -m <model> -d <fasta_file> -o <output_folder> -l <label>
```

## Arguments 🧩

- **`<model>`** ∈ `{bmDCA, eaDCA, edDCA}`  
  Selects the training routine.  
  By default, the fully connected `bmDCA` algorithm is used. `edDCA` can follow two different routines: either it decimates a pre-trained `bmDCA` model, or it first trains a `bmDCA` model and then decimates it.
- **`<fasta_file>`** – Path to the FASTA file containing the training MSA.
- **`<output_folder>`** – Folder where results will be stored (created if missing).
- **`<label>`** – Optional tag for output files.

---

## Training Behavior ⚙️

Training stops when the **Pearson correlation** between model and empirical connected correlations reaches the target value (default: `0.95`).

- Early training is fast (e.g., Pearson ≈ 0.9 after ~100 iterations).  
- Approaching higher values takes significantly longer (power‑law decay).

For a quick coarse model, set:

```
--target 0.9
```

---

## Output Files 📁

During training, adabmDCA maintains three output files:

- **`<label>_params.dat`** – Non‑zero model parameters  
  - Lines starting with `J` → couplings  
  - Lines with `h` → biases

- **`<label>_chains.fasta`** – State of the Markov chains

- **`<label>_adabmDCA.log`** – Log file updated throughout training

**Update intervals:**
- `bmDCA`: every 50 updates  
- `eaDCA`, `edDCA`: every 10 updates  

---

## Restoring Interrupted Training 🔄

Resume training using:

```bash
$ adabmDCA train [...] -p <file_params> -c <file_chains>
```

---

## Importance Weights 🏋️‍♂️

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

## Choosing the Alphabet 🔠

Default alphabet: **protein**.

Specify alternatives:

- RNA → `--alphabet rna`
- DNA → `--alphabet dna`
- Custom →  
  ```
  --alphabet ABCD-
  ```

---
# <span id="eadca">eaDCA 🌱</span>

Enable with:

```
--model eaDCA
```

Key hyperparameters:

- `--factivate` – fraction of inactive couplings activated (default: 0.001)  
- `--gsteps` – parameter updates per graph update (default: 10)

Recommended: reduce sweeps to **5**.

---

# <span id="eddca">edDCA ✂️ (Decimated DCA)</span>

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

# Choosing Hyperparameters 🎚️

Defaults work well for clean and moderately diverse MSAs. For more difficult datasets, consider tuning:

---

### Learning Rate

- Default: **0.01**  
- If chains mix poorly, try:  
  ```
  --lr 0.005
  ```

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

Default:
```
α = 1 / M_eff
```

Increasing α (e.g. α = 0.001 or 0.01) may help when the training struggle converging or the mixing time of the model is very high, but it also makes the model less expressive.