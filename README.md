<p align="center">
  <img src="docs/figures/logo-adabmDCA.png" alt="adabmDCA logo" width="160">
</p>

# adabmDCA 2.1 — Direct Coupling Analysis in Python

`adabmDCA` learns a Potts model (a Boltzmann machine over sequences) from a multiple sequence alignment of proteins, RNA or DNA, and uses it to:

- **generate** new sequences with the statistics of the family;
- **score** sequences and single mutations by their statistical energy;
- **predict contacts** between alignment positions from the learned couplings;
- **estimate the entropy** and the normalization of the model;
- **retrain** the model with experimental feedback on tested sequences.

adabmDCA 2.1 is a major update of the Python implementation. It introduces equilibrium training and sampling with Parallel Trajectory Tempering (below), and rewrites the Monte Carlo samplers: optimized **Triton kernels on NVIDIA GPUs** and multithreaded **Numba kernels on CPUs** are 24–74× and 7–36× faster per sweep than the previous PyTorch implementation, while producing the same Markov chains. It also runs on Apple GPUs (Metal) using optimized mps kernels, and offers both a command-line interface and a Python API. The Python package has its own version number.

## Equilibrium training with Parallel Trajectory Tempering

A Potts model is only as good as the Monte Carlo chains used to train and sample it. When a family contains well-separated subfamilies, ordinary chains stay inside one of them for a very long time. The training gradient is then estimated from unrepresentative samples, and generated sequences miss whole clusters, while the usual metrics still look fine.

By default, adabmDCA trains and samples with **Parallel Trajectory Tempering (PTT)**. PTT keeps a ladder of models saved along the training trajectory, from an independent-site profile that can be sampled exactly up to the current model. Configurations swap along this ladder, so fresh, independent ones keep reaching the top, and built-in checks verify that the ladder keeps mixing ([Béreux et al., 2026](https://arxiv.org/abs/2607.27077)). Because the bottom of the ladder has an exactly known normalization, PTT also gives a reliable estimate of the model's partition function. This enables online tracking of the training and validation log-likelihood, and an estimate of the model's entropy. The faster persistent contrastive divergence (`--strategy pcd`) remains available for families without strong cluster structure.

bmDCA training starts by default from a tractable **anchor** model instead of the profile (`--anchor`, default `tree-low-rank`). The anchor is fitted to the data before training, sampled exactly and has a known normalization, like the profile, but it already captures part of the correlations. With PTT it sits at the bottom of the ladder, which then needs fewer models; with PCD it gives the initial parameters and chains. It builds on the exactly sampled low-rank models of [Béreux et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4fe1859112230a032c7143a9adc3be78-Abstract-Conference.html).

On three large protein families, plain Monte Carlo sampling of trained models did not reach equilibrium within 10,000 sweeps, while PTT samples reproduced every cluster of the data.

## Features

- Dense (`bmDCA`) and sparse (`eaDCA`, `edDCA`, `edgeDCA`) Potts models.
- PTT or PCD training, with a held-out validation set and automatic stopping at the best validation likelihood.
- Sampling with convergence checks and diagnostic plots: connected correlations, PCA projections, cluster balance, distances to training and held-out sequences.
- Contact maps, sequence energies, single-mutant scans, entropy and log Z estimates.
- Retraining with experimental feedback on tested sequences.
- Alignment preprocessing (FASTA and Stockholm) and homology-aware train/test splits.
- Automatic detection of protein, RNA and DNA alphabets; custom alphabets on request.
- Fast sampling kernels: Triton on NVIDIA GPUs, multithreaded Numba on CPUs.

## Installation

```bash
uv tool install adabmDCA            # or: python -m pip install adabmDCA
adabmDCA --help
```

Without a GPU, add the `cpu` extra for compiled multithreaded samplers, 7–36× faster per sweep:

```bash
uv tool install 'adabmDCA[cpu]'     # or: python -m pip install 'adabmDCA[cpu]'
```

To use it as a library in a Python project, run `uv add adabmDCA`. To work from source:

```bash
git clone https://github.com/spqb/adabmDCApy.git
cd adabmDCApy
uv sync --locked
uv run adabmDCA --help
```

## Quick start

The repository includes an RNA family to try the commands, RF00379, already split into training and validation sets (`example_data/RF00379/splits`):

```bash
D=example_data/RF00379/splits

# Train a fully connected model with PTT; stop at the best validation likelihood (a few minutes on a GPU)
adabmDCA train -d $D/RF00379_train.fasta -v $D/RF00379_validation.fasta -o model

# Generate sequences, check that sampling equilibrated, and plot the diagnostics
adabmDCA sample -p model/ptt.h5 -d $D/RF00379_train.fasta -v $D/RF00379_validation.fasta --plot-sampling-logs -o samples

# Use the model
adabmDCA contacts -p model/params.dat.gz -o contacts
adabmDCA energies -d $D/RF00379_validation.fasta -p model/params.dat.gz -o energies
adabmDCA dms      -d $D/RF00379_validation.fasta -p model/params.dat.gz -o mutations   # first sequence as wild type
adabmDCA entropy  -p model/ptt.h5 -o entropy
```

For your own alignment, `adabmDCA preprocess` cleans FASTA or Stockholm files, and `adabmDCA split-data family family.fasta` creates a training/validation split that keeps close homologues on the same side.

A finished training run is not yet a validated model. Read the plots written by `sample --plot-sampling-logs`: the generated sequences should reproduce the training Pearson and cover every cluster of the data. `adabmDCA <command> --help` lists every option of a command.

## Python API

The same workflows run from Python. In Python, PTT has to be requested explicitly:

```python
from adabmDCA import PTTConfig, load_alignment, load_model, sample_sequences, train_model

train = load_alignment("example_data/RF00379/splits/RF00379_train.fasta")
validation = load_alignment("example_data/RF00379/splits/RF00379_validation.fasta")

result = train_model(train, validation_path=validation, alphabet="rna",
                     ptt=PTTConfig(validation_stop=True), output_dir="model")
print(result.stop_reason, result.history_dataframe().tail())

samples = sample_sequences(model="model/ptt.h5", ptt=True, reference_fasta=train,
                           n_sequences=2048, collect_diagnostics=True)
samples.save_bundle("samples")
samples.save_diagnostic_plots("samples")

model = load_model("model/params.dat.gz", alphabet="rna")
energies = model.compute_energies(list(validation.sequences))
contacts = model.compute_contact_map()
```

### Steered sampling

Use `SteeringConfig` to specify the support and the reduced-energy bias. This
example uses a 193-site RNA model and an aligned reference string `start`.

```python
from adabmDCA import HammingBias, HammingWindow, SteeringConfig, sample_sequences

steering = SteeringConfig(
    allowed_tokens="ACGU",
    windows=(HammingWindow(start, 5, 10),),
    distance_biases=(HammingBias(start, 0.1),),
    entropy=True,
)
result = sample_sequences(
    model="model/ptt.h5", ptt=True, steering=steering,
    beta=1.25, n_sequences=1024, ptt_max_rounds=1000,
)
print(result.steering["status"], result.steering["entropy"])
```

The reduced energy is `beta * H + sum(weight * cost)` within the allowed
support. Linear Hamming costs are folded into the fields, and inclusive
windows are enforced during proposals. Add deterministic
`WeightedPotential(callback, weight)` terms to `SteeringConfig.potentials`
for other costs. Each callback takes a batch of aligned sequences and
returns one finite unweighted value per sequence. It can also take
categorical indices with `input="categorical"`. The sampler reuses these
values across weights. `beta > 1` cools the Potts model with the specified
bias weights fixed.

Steering is available through the Python API using `sample_sequences` with
`steering=SteeringConfig(...)`. Steered PTT also accepts a standalone Potts
file. With `ptt=True`
and `beta > 1`, the API builds a temperature ladder even without another bias.

The steering summary records rank-normalized split R-hat and ESS on
independent group-mean traces. PTT also tracks refresh from the exact bottom
of the ladder. A PTT archive trained from an anchor other than the profile uses its exact anchor
at the bottom for unrestricted or alphabet-only support. Joint Hamming
windows use an exact constrained profile at the bottom and keep the restricted
training anchor on the ladder. PTT entropy uses adjacent Bennett ratios and
reports its Monte Carlo uncertainty. If warmup exhausts the budget, output
draws and entropy are marked provisional.

Use `result.steering["history"]` to plot a structured steering run. The
`warmup` and `production` records have absolute round numbers, elapsed time,
endpoint observables and group means, acceptance rates, and cumulative
callback evaluations. Production records correspond to retained batches and
average acceptance over each batch interval. Endpoint means use the full
chain population, including when the last batch retains fewer chains. PTT
also records rung energies, available callback means, and the old population
count on each rung. Rung arrays follow `history["rungs"]` from bottom to top.
Callback columns follow `history["potential_indices"]`, which indexes the
original configuration and skips zero-weight callbacks. Distance columns
follow `history["windows"]`.

The `checks` records give R-hat and ESS at each stopping check. The `pilot`
records show the ladder and swap acceptance before each insertion. When
entropy is enabled, `history["entropy"]` gives running log Z and entropy after
1, 2, 4, … retained batches and at the final batch. These histories are
included when the result is saved as JSON.

```python
import matplotlib.pyplot as plt

warmup = result.steering["history"]["warmup"]
plt.plot([row["round"] for row in warmup],
         [row["observables"]["reduced_energy"] for row in warmup])
plt.xlabel("Round")
plt.ylabel("Mean reduced energy")
```

## Use with AI agents

[`skills/adabmdca`](skills/) is an Agent Skill that teaches coding agents such as Claude Code or Codex to run adabmDCA and check its results, and `skills/adabmdca-steering` and `skills/adabmdca-local-free-energy` add steered generation and local free energies. Copy them to `~/.claude/skills/` (or your agent's skills folder); see [skills/README.md](skills/README.md).

## Documentation

The [documentation](https://spqb.github.io/adabmDCApy/) explains:

- how to use every command and which settings to choose;
- how training and sampling work, and how to read their plots;
- how to diagnose a run that stalls or a model that does not reproduce the data;
- benchmarks on six protein and RNA families;
- the mathematical definition of every reported quantity.

An interactive [Colab tutorial](https://colab.research.google.com/drive/1uMY1mIlurutquw87FcfX8Rmqfzsyk74Z?usp=sharing) covers training, sampling and analysis.

adabmDCA is also available in [Julia](https://github.com/spqb/adabmDCA.jl) (multi-core CPU) and [C++](https://github.com/spqb/adabmDCAc) (single-core CPU), with the same general command shape. PTT and the diagnostics described here are specific to the Python package.

## Development

```bash
uv sync --locked                                       # environment with test tools and Numba
uv run pytest -q                                       # test suite
uv run ruff check .                                    # lint
uv sync --locked --group docs
uv run --group docs mkdocs serve                       # documentation, live preview
```

## Citation

If you use adabmDCA in your research, please cite:

- The original adabmDCA:
  A. P. Muntoni, A. Pagnani, M. Weigt and F. Zamponi, *adabmDCA: adaptive Boltzmann machine learning for biological sequences*, BMC Bioinformatics **22**, 528 (2021). [doi:10.1186/s12859-021-04441-9](https://doi.org/10.1186/s12859-021-04441-9)

- The reference paper of the package:
  L. Rosset, R. Netti, A. P. Muntoni, M. Weigt and F. Zamponi, *adabmDCA 2.0 — a flexible but easy-to-use package for Direct Coupling Analysis*, Methods in Molecular Biology **2979**, 83–104 (2026). [doi:10.1007/978-1-0716-4828-5_6](https://doi.org/10.1007/978-1-0716-4828-5_6); preprint [arXiv:2501.18456](https://arxiv.org/abs/2501.18456)

- Parallel Trajectory Tempering, when you train or sample with PTT:
  N. Béreux, A. Decelle, C. Furtlehner and B. Seoane, *Equilibrium training of energy-based models with Parallel Trajectory Tempering*, arXiv:2607.27077 (2026). [arXiv:2607.27077](https://arxiv.org/abs/2607.27077)

- The anchor, when you train bmDCA with its default `--anchor` or another low-rank anchor:
  N. Béreux, A. Decelle, C. Furtlehner, L. Rosset and B. Seoane, *Fast training and sampling of Restricted Boltzmann Machines*, ICLR 2025. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4fe1859112230a032c7143a9adc3be78-Abstract-Conference.html)

## License

adabmDCA is distributed under the [Apache License 2.0](LICENSE).
