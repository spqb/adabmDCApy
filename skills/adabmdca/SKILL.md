---
name: adabmdca
description: Run Direct Coupling Analysis with the adabmDCA Python package (CLI or API). Use when asked to train a Potts/Boltzmann machine model (bmDCA, eaDCA, edDCA) on a protein, RNA or DNA multiple sequence alignment, generate sequences from it (PTT or MCMC), score sequences, scan single mutations, predict contacts, estimate entropy or log Z, or retrain with experimental feedback.
---

# adabmDCA

adabmDCA learns a Potts model from a multiple sequence alignment (MSA) and uses it to generate, score and analyse sequences. Every task is a subcommand of one CLI, `adabmDCA`, and every subcommand is a thin wrapper around a function of the Python API, so both give identical files and errors.

Prefer the CLI from a shell: it has the right defaults (PTT, validation stop, alphabet detection) and prints a summary you can read. Use the Python API when chaining many operations on loaded objects, steering sampling, or following a run with callbacks.

`adabmDCA <command> --help` is the authoritative list of options of the installed version. Check it before using any flag not shown here.

## Install and check

```bash
python -m pip install 'adabmDCA[cpu]'   # or: uv tool install 'adabmDCA[cpu]'
adabmDCA --help
python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"
```

- The `cpu` extra installs Numba for compiled multithreaded CPU samplers (7 to 36 times faster per sweep). Install it on any machine without an NVIDIA GPU.
- `--device auto` (the default everywhere) picks CUDA, then Apple `mps`, then CPU. NVIDIA GPUs use Triton kernels; Apple GPUs use Metal kernels with PyTorch 2.7+.
- From a clone of the repository: `uv sync --locked` and prefix commands with `uv run`.

## The standard workflow

```text
family.sto ─preprocess─► family.fasta ─split-data─► train + test
                                                       │
                                              train (PTT, validation stop)
                                                       │
                                   model/params.dat.gz + model/ptt.h5 + logs
                                                       │
            sample --plot (validate) ─► energies · dms · contacts · entropy
```

```bash
# 1. Clean the alignment (Stockholm or FASTA, insertions, unknown symbols, gappy rows)
adabmDCA preprocess family.sto -o family.fasta --remove-insertions --remove-duplicates --report family.preprocess.json

# 2. Homology-aware split: writes family.train.fasta, family.test.fasta, family.split.json
adabmDCA split-data family family.fasta

# 3. Train bmDCA with PTT; with -v it stops at the best validation log-likelihood
adabmDCA train -d family.train.fasta -v family.test.fasta -o model

# 4. Validate the model by sampling it, with diagnostic plots
adabmDCA sample -p model/ptt.h5 -d family.train.fasta -v family.test.fasta --plot -o samples

# 5. Use it
adabmDCA energies -d sequences.fasta -p model/params.dat.gz -o energies
adabmDCA dms      -d wildtype.fasta  -p model/params.dat.gz -o dms        # first sequence = wild type
adabmDCA contacts -p model/params.dat.gz -o contacts
adabmDCA entropy  -p model/ptt.h5 -o entropy
```

Skip steps 1–2 when the user already has a clean aligned FASTA and a validation set. Skip what the user did not ask for, but do not skip step 4 for a model whose samples or scores will be relied on.

## Rules that prevent the common mistakes

1. **Training and sampling are long jobs.** PTT training takes minutes for small RNA families and hours for protein families of 200–300 positions, even on a GPU. Run them in the background with output redirected to a file (`> train.out 2>&1`), and follow `model/history.csv`, `model/events.jsonl` and `grep '»' model/adabmDCA.log` instead of the terminal. Add `--no-progress` to `train` so the redirected output stays readable. For a quick check of a pipeline, cap the budget with `--max-gradient-steps 100`, and say the result is not converged.
2. **Never overwrite a previous run.** Use a new `-o` folder (or `-l label`) per run. Only `--ptt-resume` reuses a folder, deliberately.
3. **A finished training run is not a validated model.** Always read `training.json` → `data.stop_reason` and `data.converged`, then sample the model with `--plot` and a held-out set, and compare the sampled Pearson with the last training Pearson. See [references/diagnostics.md](references/diagnostics.md).
4. **Prefer PTT, use PCD to explore.** `--strategy pcd` is 9 to 41 times faster but nothing checks that its chains stay at equilibrium; on clustered protein families PCD models could not be sampled correctly. Train the model the user will rely on with PTT (the CLI default).
5. **Python defaults differ from the CLI.** `train_model` uses PCD unless `ptt=PTTConfig(...)` is passed; the validation stop needs `PTTConfig(validation_stop=True)`; `sample_sequences` uses plain MCMC unless `ptt=True`; workflow functions default to `alphabet="protein"` (only `load_alignment` and `load_model` on `ptt.h5` detect it).
6. **Alphabets.** Detection tries DNA, then RNA, then protein. An RNA alignment without any `U` is read as DNA: pass `--alphabet rna`. For custom symbols pass the ordered token string (e.g. `--alphabet ABCD-`) and use the same string in every command that reads that model.
7. **Positions are zero-based alignment columns** in every output (contacts, DMS labels such as `G27A`, parameter files). The DMS CSV also has `position_1based`. Alignment columns are not residue numbers of the full-length protein; map them before comparing with structures or assays.
8. **Sample with PTT whenever a `ptt.h5` archive exists.** Look for `ptt.h5` (or `<label>_ptt.h5`) in the training folder before sampling. If it is there, pass it with `-p` and keep the default `--strategy ptt` (in Python: `sample_sequences(model="model/ptt.h5", ptt=True, ...)`), even if the user only mentions `params.dat.gz`. Use plain MCMC (`--strategy pcd`) only when there is no archive (PCD-trained or imported models), or as an extra cross-check, and then confirm it reached its mixing time.
9. **Pick the right model file.** `params.dat.gz` for energies, DMS, contacts and plain MCMC; `ptt.h5` for PTT sampling, PTT entropy and resuming PTT training (it also works wherever `params.dat.gz` does). Resuming PCD needs `params.dat.gz` **and** `chains.fasta`.
10. **Interpretation.** Lower energy means more probable under that model; energies of different models are not comparable. DMS `delta_energy` < 0 means the model prefers the mutant. Contact scores rank pairs; they are not probabilities (take the top L or L/2, excluding |i−j| ≤ 4 for proteins).
11. **Errors are structured.** The CLI prints `Error [<code>]: <message>` plus details and exits with 2 for invalid input, 1 for loading, computation or convergence failures, 130 for cancellation. Report the message and the setting it names; do not retry blindly.

## Choosing settings

| Decision | Default | Change when |
| --- | --- | --- |
| Strategy | `ptt` | `--strategy pcd` for a fast first look or small families without clusters |
| Model | `bmDCA` | `-m eaDCA` (PTT or PCD) or `-m edDCA` / `-m edgeDCA` (PCD only) when a sparse graph is itself the goal |
| Validation set | none | Always provide `-v` if possible: it enables the validation stop and the memorization checks |
| `--nsweeps` | 10 | 100 when PTT reports trapped configurations or stalls in a mixing check; restart from scratch, do not resume |
| `--lr` | 0.01 | Do not raise it with PTT |
| `--sampler` | `metropolized_gibbs` | `gibbs` is equally good; avoid `metropolis` on proteins |
| `--target` | 0.95 | Only matters without `-v` (Pearson stop), or with `--no-ptt-validation-stop` |
| Reweighting | 80% identity | `--clustering_seqid`, `-w weights.dat`, or `--no_reweighting` for non-redundant data |

## Reference files

Read the one that matches the task:

- [references/cli.md](references/cli.md): every command with its main options, variants and outputs (train, resume, sample incl. `--ptt-return-full-ladder`, energies, dms, contacts, entropy, reintegrate, preprocess, split-data, plot-training-log).
- [references/python-api.md](references/python-api.md): the same workflows from Python, progress callbacks and cancellation, steered sampling, result objects.
- [references/outputs.md](references/outputs.md): what each output file contains and how to read it (`params.dat.gz`, `history.csv` columns, `events.jsonl`, `training.json`, `sampling.json`, `ladder_samples.h5`).
- [references/diagnostics.md](references/diagnostics.md): how to tell whether a run is healthy, a triage script, and what to change when it is not.

The full documentation is at https://spqb.github.io/adabmDCApy/ and, in a clone, under `docs/`.

## MCP server

For clients without a shell (Claude Desktop, claude.ai chat), and for structured management of long training jobs (start, poll progress, cancel), the separate agentDCA MCP server (https://github.com/rossetl/agentDCA) exposes adabmDCA as tools. With a shell, this skill and the CLI are enough.

## Citation

When adabmDCA results go into a report, suggest citing Rosset et al., *adabmDCA 2.0*, Methods Mol. Biol. 2979, 83–104 (2026), and Béreux et al., arXiv:2607.27077 (2026) when PTT was used.
