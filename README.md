# adabmDCA 2.0 — Direct Coupling Analysis in Python

`adabmDCA 2.0` trains and analyzes Potts models for Direct Coupling Analysis
(DCA). This repository contains the Python implementation, with support for
CUDA, Apple Metal, and CPU execution. The “2.0” name refers to the method and
paper; Python package releases use their own semantic version numbers.

The package provides command-line tools and a high-level Python API for model
training, sequence generation, contact prediction, mutational-effect scoring,
entropy estimation, experimental-data reintegration, and alignment processing.

## Highlights

- Dense `bmDCA` and sparse `eaDCA`, `edDCA`, and `edgeDCA` training.
- Automatic detection of standard protein, DNA, and RNA alphabets; explicit
  custom alphabets remain available for nonstandard data.
- Metropolis and Gibbs sampling with mixing-time estimation.
- Sampling diagnostics for autocorrelation, Pearson correlation, connected
  correlations, and PCA projections with marginal distributions.
- FASTA and Stockholm input, auditable preprocessing, sequence reweighting,
  validation alignments, reproducible seeds, and resumable checkpoints.
- Structured result objects with JSON, CSV, FASTA, NumPy, and plot serializers.
- Optional BF16 sampling kernels on supported NVIDIA GPUs.

See the [documentation](https://spqb.github.io/adabmDCApy/) for the complete
guides, or open the
[Colab tutorial notebook](https://colab.research.google.com/drive/1uMY1mIlurutquw87FcfX8Rmqfzsyk74Z?usp=sharing) for an interactive
training and analysis workflow.

## Installation

Install the released package with [`uv`](https://docs.astral.sh/uv/):

```bash
uv tool install adabmDCA
adabmDCA --help
```

For use as a Python dependency:

```bash
uv add adabmDCA
```

The equivalent pip installation is also supported:

```bash
python -m pip install adabmDCA
```

To work from source:

```bash
git clone https://github.com/spqb/adabmDCApy.git
cd adabmDCApy
uv sync --locked
uv run adabmDCA --help
```

See the [installation guide](https://spqb.github.io/adabmDCApy/installation/)
for development,
documentation, Julia, and C++ setup.

## Command-line quick start

Train a dense Potts model. The sequence alphabet is detected automatically
when it matches a standard protein, DNA, or RNA alphabet:

```bash
adabmDCA train \
    --data alignment.fasta \
    --output model \
    --model bmDCA
```

Generate sequences and save convergence and PCA diagnostics:

```bash
adabmDCA sample \
    --path_params model/params.dat \
    --data alignment.fasta \
    --output samples \
    --ngen 1000 \
    --plot
```

Run `adabmDCA --help` to list all workflows and
`adabmDCA <command> --help` for command-specific options. The
[quick-reference page](https://spqb.github.io/adabmDCApy/quicklist/) contains
more examples.

## Python API

The same workflows are available through reusable Python objects:

```python
from adabmDCA import load_model, sample_sequences

model = load_model("model/params.dat", alphabet="protein", device="auto")

energies = model.compute_energies(["ACDEFGHIK", "ACDEYGHIK"])
contacts = model.compute_contact_map()
mutations = model.scan_mutations("ACDEFGHIK").to_dataframe()

result = sample_sequences(
    model=model,
    n_sequences=100,
    n_sweeps=1000,
    reference_fasta="alignment.fasta",
    collect_diagnostics=True,
    seed=42,
)
result.save_bundle("samples", label="family")
result.save_diagnostic_plots("samples", label="family")
```

The [high-level API guide](https://spqb.github.io/adabmDCApy/high_level_api/)
covers training, progress
callbacks, input validation, serialization, and structured errors. Low-level
tensor functions remain available for numerical workflows.

## Documentation and development

Build the documentation locally with:

```bash
uv sync --locked --group docs
uv run --group docs mkdocs serve
```

Run the CPU test suite and lint checks with:

```bash
uv run pytest -q
uv run ruff check .
```

Release verification, including the GPU gate, is described in
[the release-verification guide](https://spqb.github.io/adabmDCApy/release_checks/).

## Citation

If you use `adabmDCA` in your research, please cite:

> Rosset, L., Netti, R., Muntoni, A. P., Weigt, M., & Zamponi, F. (2025).
> *adabmDCA 2.0: A flexible but easy-to-use package for Direct Coupling
> Analysis.* [Preprint](https://doi.org/10.1101/2025.01.31.635874).

The original implementation is described in
[Muntoni et al. (2021)](https://doi.org/10.1186/s12859-021-04441-9).

## License

`adabmDCA` is distributed under the
[Apache License 2.0](https://github.com/spqb/adabmDCApy/blob/main/LICENSE).
