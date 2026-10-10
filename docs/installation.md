# Installation

adabmDCA needs Python 3.10 or newer and runs on NVIDIA GPUs (CUDA), Apple GPUs (Metal) and CPUs.

## Install the command-line tool

```bash
uv tool install adabmDCA
# or
python -m pip install adabmDCA
```

Check the installation with `adabmDCA --help`. To use the package from Python inside an existing project, add it as a dependency instead (`uv add adabmDCA`).

### Faster CPU sampling

Without a GPU, install the `cpu` extra:

```bash
uv tool install 'adabmDCA[cpu]'
# or
python -m pip install 'adabmDCA[cpu]'
```

It adds [Numba](https://numba.pydata.org/) and enables compiled, multithreaded sampling kernels, which the package then uses automatically on CPU. They need no compiler, give the same result for a given seed whatever the number of threads, and are 7–36× faster per sweep than the plain PyTorch samplers (see [Benchmarks](algorithms/benchmarks.md#sampling-kernels)). They use as many threads as PyTorch (`torch.set_num_threads`).

## Choose a device

Every command accepts `--device`:

| Value | Uses |
| --- | --- |
| `auto` (default) | CUDA if available, then Apple Metal (`mps`), then CPU |
| `cuda`, `cuda:1`, … | A specific NVIDIA GPU; sampling uses Triton kernels |
| `mps` | An Apple GPU; sampling uses optimized Apple Metal kernels with PyTorch 2.7+ |
| `cpu` | The CPU; Numba kernels when the `cpu` extra is installed |

In the NVIDIA benchmarks, an RTX A5000 is 5–20× faster per sampling sweep, and 12–16× faster for a whole bmDCA training, than a 16-thread CPU. It is recommended for protein families of a few hundred positions. CPU runs reach models of the same quality and are perfectly usable for small RNA families: RF00379 (136 positions) trains in about 5 minutes with PCD and 35 minutes with PTT.

On a MacBook Air with an Apple M4, MPS is **3.9–8.3 times faster
than its CPU** for Metropolized Gibbs sampling on the tested RNA and protein
models (2,000 chains; CPU with 4 Numba threads). This uses different hardware
from the Threadripper/RTX A5000 workstation in the NVIDIA comparison above.
See [MPS vs CPU](algorithms/benchmarks.md#mps-vs-cpu) for sampling and PCD
training results.

The default precision is `float32`; `--dtype float64` is available. `--dtype bfloat16` stores the couplings used inside the sampler in BF16 while keeping all other state in FP32. On CUDA it requires an Ampere-or-newer NVIDIA GPU with Triton and gave at most about 12% speed-up in our tests. It is also available on MPS as described below. PTT does not support BF16.

### Apple Silicon

Use `--device mps` to train or sample on an Apple GPU. Optimized kernels are
used automatically with PyTorch 2.7 or newer; no extra package is needed.
Both PCD and PTT support float32 on MPS. MPS does not support float64.

On macOS 14 or newer, `--device mps --dtype bfloat16` is also available for
PCD training and ordinary sampling. It keeps master parameters and training
statistics in float32. Performance gains depend on the model; PTT does not
support BF16.

```bash
adabmDCA train --device mps -d train.fasta -v validation.fasta -o model
adabmDCA sample --device mps -p model/params.dat.gz -d train.fasta -o samples
```

See [MPS vs CPU](algorithms/benchmarks.md#mps-vs-cpu) for the laptop comparison.

## Environment variables

These change how the sampling kernels are selected. They never change the sampled distribution.

| Variable | Effect |
| --- | --- |
| `ADABMDCA_NUMBA=0` | Use the PyTorch samplers on CPU even when Numba is installed |
| `ADABMDCA_NUMBA_SWAPS=0` | Disable fused movement on fixed seven-model CPU PTT sampling ladders, retaining Numba local and exchange kernels |
| `ADABMDCA_MPS=0` | Use the original PyTorch samplers on MPS instead of fused Metal kernels |
| `ADABMDCA_NUMBA_PIN=1` | Pin each Numba thread to its own core. Can help on idle multi-chiplet CPUs (AMD Zen), can hurt when other work shares the cores: measure before relying on it |
| `ADABMDCA_SPARSE=0` | Always use the dense kernels, even for sparse coupling graphs |

## Install from source

```bash
git clone https://github.com/spqb/adabmDCApy.git
cd adabmDCApy
uv sync --locked
uv run adabmDCA --help
```

The default `dev` dependency group includes the test tools and Numba. For an editable installation into another environment, use `uv pip install -e .`.

### Build this documentation

```bash
uv sync --locked --group docs
uv run --group docs mkdocs serve     # live preview
uv run --group docs mkdocs build     # writes site/
```

Next: [prepare an alignment](usage/preprocess.md) and [train a model](usage/train.md).
