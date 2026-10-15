# adabmDCA command line

`adabmDCA <command> [options]`. Commands: `preprocess`, `split-data`, `train`, `sample`, `energies`, `dms`, `contacts`, `entropy`, `reintegrate`, `plot-training-log`. Every command accepts `--alphabet` (`auto`, `protein`, `rna`, `dna` or an ordered token string), `--device` (`auto`, `cuda`, `cuda:1`, `mps`, `cpu`) and `--dtype` (`float32` default; `float64`; `bfloat16` only for sampling couplings on Ampere+ CUDA with Triton or MPS on macOS 14+, PTT on CUDA only). Input alignments can be FASTA, gzip FASTA or Stockholm. With `-l name`, output files are prefixed `name_`.

## preprocess

```bash
adabmDCA preprocess family.sto -o family.fasta --remove-insertions --remove-duplicates --report family.json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `input` (positional) | required | FASTA or Stockholm |
| `-o` | required | Output FASTA |
| `--input-format` | `auto` | `fasta` or `stockholm` |
| `--alignment-index` | | Which alignment of a multi-alignment Stockholm file |
| `--remove-insertions` | off | Drop dots and lowercase insertion residues (A2M/A3M style) |
| `--max-gap-fraction` | 0.2 | Drop sequences with more gaps; `1.0` disables |
| `--remove-duplicates` | off | Keep the first copy of each sequence |
| `--unknown-tokens` | `gap` | `gap` (replace with `-`), `remove` the sequence, or `error` |
| `--gap-token` | `-` | Character counted as a gap |
| `--report` | none | JSON report: counts before/after, dropped names |

`train` and the other commands always drop sequences with invalid symbols and exact duplicates themselves; `preprocess` is for anything more.

## split-data

```bash
adabmDCA split-data family family.fasta                 # clustering at 80% identity, 80% train
adabmDCA split-data family family.fasta --method cobalt -t1 0.5 -t2 0.5
```

Writes `<prefix>.train.fasta`, `<prefix>.test.fasta`, `<prefix>.split.json`. Clustering options: `--identity` (0.8), `--train-fraction` (0.8). Cobalt options (Petti & Eddy 2022): `-t1` max train–test identity (0.5), `-t2` max test–test identity (0.5), `-t3` max train–train identity (1.0), `--bestof`, `--maxtrain`, `--maxtest`. `--seed` (0).

## train

```bash
adabmDCA train -d train.fasta -v validation.fasta -o model --no-progress > train.out 2>&1 &
```

Default: bmDCA, PTT, 2048 chains, 10 Metropolized Gibbs sweeps per exchange round, lr 0.01, adaptive optimizer. With `-v` training stops at the validation plateau; without it at Pearson `--target` 0.95.

| Option | Default | Meaning |
| --- | --- | --- |
| `-d`, `--data` | required | Training alignment |
| `-v`, `--validation` | none | Validation alignment (`--val` older alias) |
| `-o`, `--output` | `DCA_model` | Output folder |
| `-l`, `--label` | none | Prefix of output files |
| `-m`, `--model` | `bmDCA` | `bmDCA`, `eaDCA` (PTT or PCD), `edDCA`, `edgeDCA` (PCD only) |
| `--strategy` | `ptt` | `ptt` or `pcd` |
| `--nchains` | 2048 | Chains (per replica with PTT) |
| `--nsweeps` | 10 | Sweeps per update (PCD) or per exchange round (PTT) |
| `--sampler` | `metropolized_gibbs` | `gibbs`, `metropolis` |
| `--lr` | 0.01 | Learning rate (upper bound for the adaptive PTT optimizer) |
| `--target` | 0.95 | Pearson target on connected correlations |
| `--max-gradient-steps` | none | Budget of parameter updates |
| `--max-structure-steps` | none | Budget of graph steps (sparse models) |
| `--nepochs` | 50000 | Older budget (gradient steps for bmDCA, graph steps for sparse) |
| `--checkpoint-interval` | 500 | Updates between checkpoints |
| `--pseudocount` | 1/M_eff (0.1 edgeDCA) | Regularization of data frequencies |
| `--l2_reg` | 0 | L2 penalty |
| `-w`, `--weights` / `--clustering_seqid` / `--no_reweighting` | computed at 0.8 | Sequence weights |
| `--seed` | 0 | Random seed |
| `--plot-training-logs` | off | Write `<label>_plots/` (or `training_plots/`) at the end |
| `--diagnostic` | off | Full PTT progress table and every event in the terminal |
| `--no-progress` | off | No terminal progress bar (files are still written) |
| `--wandb` | off | Log to Weights & Biases |

Sparse models: eaDCA `--gsteps` (10), `--factivate` (0.001), `--ptt-activation {fixed,adaptive}`; edDCA `--density` (0.02), `--drate` (0.01). For edgeDCA the pseudocount, not `--lr`, sets the step size.

PTT has about thirty `--ptt-*` options; the defaults rarely need changing. The ones that matter:

| Option | Default | Meaning |
| --- | --- | --- |
| `--ptt-validation-stop` / `--no-ptt-validation-stop` | on with `-v` | Stop at the validation plateau |
| `--ptt-validation-window` | 200 | Updates per compared window |
| `--ptt-validation-min-gain` | 0 | Minimal gain between windows |
| `--ptt-target-acceptance` | 0.25 | Swap acceptance that triggers a snapshot insertion |
| `--ptt-optimizer` | `adaptive` | `sgd` uses the fixed `--lr` |
| `--ptt-trust-radius` | 0.01 | Max KL per update (adaptive optimizer) |
| `--ptt-max-recoveries` | 3 | Learning-rate halvings after mixing failures before stopping |
| `--ptt-mixing-method` | `renewal` | or `autocorrelation` |
| `--ptt-resume` | | Resume from a `ptt.h5` archive |

Common variants:

```bash
adabmDCA train --strategy pcd -d train.fasta -o model_pcd                         # fast first look
adabmDCA train -d train.fasta -v validation.fasta -o model --nsweeps 100          # after trapped configurations
adabmDCA train -m eaDCA -d train.fasta -v validation.fasta -o model_ea            # sparse, PTT
adabmDCA train --strategy pcd -m edDCA -d train.fasta -o model_ed --density 0.02  # decimated, PCD only
```

**Resume PTT** (same command plus the archive and a larger budget). Only the budget, stopping criteria, checkpoint interval and logging options may change; `--no-ptt-validation-stop` continues past a plateau. Do not resume a run that failed with trapped configurations; restart with more sweeps.

```bash
adabmDCA train -d train.fasta -v validation.fasta -o model --ptt-resume model/ptt.h5 --max-gradient-steps 20000
```

**Resume PCD** (also how edDCA decimates an existing bmDCA model):

```bash
adabmDCA train --strategy pcd -d train.fasta -o model_pcd -p model_pcd/params.dat.gz -c model_pcd/chains.fasta
```

Outputs (in `-o`): `params.dat.gz`, `chains.fasta`, `weights.dat`, `history.csv`, `events.jsonl`, `adabmDCA.log`, `training.json`, and with PTT `ptt.h5`. See [outputs.md](outputs.md).

## sample

```bash
# Default whenever the training folder has a ptt.h5: PTT through the archived ladder; -d is required
adabmDCA sample -p model/ptt.h5 -d train.fasta -v validation.fasta --plot -o samples

# Plain MCMC: only when there is no ptt.h5 (PCD-trained or imported models), or as a cross-check
adabmDCA sample --strategy pcd -p model/params.dat.gz -d train.fasta --plot -o samples_mcmc
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-p` | required | `ptt.h5` (PTT) or a parameter file / `ptt.h5` (MCMC) |
| `-o` | required | Output folder |
| `--ngen` | training chains per model (PTT), 2048 (MCMC) | Number of sequences |
| `-d`, `--data` | required for PTT | Reference alignment, normally the training set |
| `-v`, `--validation` | none | Held-out set: enables the memorization checks (`distances.png`, PRIVET) |
| `--plot` | off | Diagnostic figures; requires `-d` |
| `--ptt-return-full-ladder` | off | Also write `ladder_samples.h5`: `--ngen` samples of every ladder model, profile first, endpoint last |
| `--ptt-stationary` | off | Renew the ladder a second time (about twice the cost) |
| `--ptt-max-rounds` | 20000 | Budget of exchange rounds |
| `--ptt-local-sweeps` | 10 | Local sweeps per round |
| `--ptt-local-kernel` | `metropolized_gibbs` | `archived` keeps the training kernel |
| `--beta` | 1.0 | Inverse temperature (MCMC); > 1 concentrates on low energies |
| `--nmix` | 2 | Mixing times to run (MCMC) |
| `--max_nsweeps` | 5000 | Sweep budget (MCMC). Hitting it means samples are not at equilibrium |
| `--nmeasure` | 10000 | Reference sequences used for mixing, PCA and distances |
| `--seed` | 0 | |

Sample as many sequences as training chains when comparing Pearson with training (the PTT default does this).

Outputs: `samples.fasta`, `samples.csv` (`name, sequence, energy, cde_sum`, plus PRIVET columns with `-v`), `sampling.json`, `logs/` (`ptt.log` has log Z, mean energy, entropy and data log-likelihood of every ladder model), and with `--plot`: `cij_scatter.png`, `pca_1_2.png`, `pca_3_4.png`, `data_vs_samples.png`, `distances.png`, `energy_vs_cde.png`, plus `ptt_renewal.png`, `ptt_ladder_health.png` (PTT) or `autocorrelation.png`, `pearson_sampling.png` (MCMC).

## energies

```bash
adabmDCA energies -d sequences.fasta -p model/params.dat.gz -o energies [--local-lambda 1.4]
```

Writes `<input stem>_energies.fasta` (energy in the header), `.csv` (`name, sequence, energy`; with `--local-lambda` also `cde_sum` and `local_free_energy` = E − λ·CDE) and `.json`. Sequences must be aligned to the model (same length L). A sensible λ is the one fitted by `sample` (`sampling.json` → `local_lambda_fit`).

## dms

```bash
adabmDCA dms -d wildtype.fasta -p model/params.dat.gz -o dms
```

Scans every single substitution (gap included) of the **first** sequence of `-d`. Writes `<name>_DMS.fasta`, `.csv`, `.json`, where `<name>` is the wild type's header with non-alphanumeric characters removed: list the folder to find it. CSV columns: `mutation` (e.g. `G27A`, zero-based), `position`, `position_1based`, `wild_type`, `mutant`, `delta_energy` (= E(mutant) − E(wt); negative = preferred), `sequence`.

## contacts

```bash
adabmDCA contacts -p model/params.dat.gz -o contacts      # from a trained model
adabmDCA contacts -d family.fasta -o contacts_mf          # mean-field, no training needed
```

Writes `contact_map.csv` (`position_i, position_j, score` for every pair, diagonal included with score 0), `contact_map.txt` (headerless, older format), `contact_map.npy` (L×L symmetric matrix), `contact_map.json`, `contact_map.png`. Scores are APC-corrected Frobenius norms in the zero-sum gauge, gaps excluded. Sparse models give exactly 0 to inactive pairs. To rank: drop `position_i >= position_j`, drop |i−j| ≤ 4 for proteins, sort by score.

## entropy

```bash
adabmDCA entropy -p model/ptt.h5 -o entropy                                              # PTT bridge, seconds to minutes
adabmDCA entropy --strategy pcd -p model/params.dat.gz -d train.fasta -t target.fasta -o entropy_ti   # thermodynamic integration
```

PTT writes `entropy.json` with `entropy`, `log_z`, `mean_energy`, `free_energy`. Thermodynamic integration (any model, needs chains that mix) writes `entropy.log`, `entropy.csv`, `entropy.json`; options `--nchains` (8192), `--nsteps` (100), `--nsweeps` (100), `--theta_max` (5). Values are in nats per sequence; training logs report per-site log-likelihoods.

## reintegrate

```bash
adabmDCA reintegrate -d natural.fasta --reint tested.fasta --adj adjustments.txt -o reintegrated [--lambda_ 1] [--strategy ptt]
```

`--adj` has one number per tested sequence, in order: `1` passed, `-1` failed, values in between grade it. `--lambda_` defaults to 1/max|adj|. **Defaults to PCD** (unlike `train`); add `--strategy ptt` for bmDCA/eaDCA. Accepts all `train` options. Outputs: the training outputs, prefixed `<label>-lambda_<λ>_` (label defaults to `reintegrated`, λ to two decimals), plus the combined alignment `..._msa.fasta`, its signed weights `..._weights.dat` and a JSON summary. Check that adjustments line up with the tested sequences in the combined files.

## plot-training-log

```bash
adabmDCA plot-training-log model/history.csv [-o plots]
```

Redraws the training curves of a finished or running job (default output `<label>_plots` next to the history).
