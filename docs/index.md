# adabmDCA 2.0 documentation

`adabmDCA 2.0` trains and analyzes Potts models for Direct Coupling Analysis
on protein, DNA, RNA, and custom-alphabet sequence alignments. The Python
package provides a command-line interface, a reusable high-level API, and
lower-level PyTorch operations. “2.0” is the project and paper name; the Python
package has an independent semantic release number.

!!! info "Where to start"

    - Follow the [installation guide](installation.md), then use the
      [quick reference](quicklist.md) for common commands.
    - Read [input data and preprocessing](preprocessing.md) before training a
      new family.
    - Use the [high-level Python API](high_level_api.md) in notebooks and
      applications.
    - Open the [Colab tutorial notebook](https://colab.research.google.com/drive/1uMY1mIlurutquw87FcfX8Rmqfzsyk74Z?usp=sharing)
      for an interactive training, sampling, and analysis workflow.

The methods are described in
[Rosset et al. (2025)](https://doi.org/10.1101/2025.01.31.635874) and the
original implementation in
[Muntoni et al. (2021)](https://doi.org/10.1186/s12859-021-04441-9).

## Main workflows

### Train Potts models

Choose among four training strategies:

- [`bmDCA`](training.md#bmdca): a fully connected Boltzmann machine.
- [`eaDCA`](training.md#eadca): a sparse model built by activating individual
  couplings.
- [`edDCA`](training.md#eddca): a sparse model obtained by
  decimating couplings.
- [`edgeDCA`](training.md#edgedca): a sparse model built by
  activating complete residue-pair edges.

The training guide covers stopping budgets, checkpoints, structured logs,
precision choices, and resuming interrupted runs.

### Analyze and generate sequences

The [applications guide](applications.md) covers:

- sequence generation with mixing and PCA diagnostics;
- model-energy scoring and single-mutant scans;
- residue-residue contact prediction;
- experimental-feedback reintegration; and
- profile-aware train/test splitting.

The Python implementation also provides
[thermodynamic integration](thermodynamic_integration.md) for estimating the
log-partition function and entropy.

### Prepare alignments

The CLI detects standard protein, DNA, and RNA alphabets automatically.
Nonstandard data requires an explicit custom alphabet. FASTA and Stockholm
inputs can be converted and cleaned with the
[alignment preprocessing workflow](alignment_preprocessing.md).

## Implementations

The project has three language-specific implementations:

- [Python](https://github.com/spqb/adabmDCApy): CUDA, Apple Metal, and CPU.
- [Julia](https://github.com/spqb/adabmDCA.jl): multi-core CPU.
- [C++](https://github.com/spqb/adabmDCAc): single-core CPU.

This site documents the Python implementation. Command availability and exact
options can differ between implementations; use `adabmDCA <command> --help`
for the installed program's authoritative command-line reference.
