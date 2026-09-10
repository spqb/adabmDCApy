# Installation Guide

`adabmDCA` is available in three language-specific implementations:

- **Python** – optimized for GPU execution  
- **Julia** – designed for multi-core CPU usage  
- **C++** – lightweight and single-core CPU compatible

Follow the instructions below based on your preferred environment.

---

## Python

### Option 1: Install from PyPI with uv (recommended)

```bash
uv add adabmDCA
```

This adds the latest stable release to the current Python project. To install
only the command-line application in an isolated environment, use:

```bash
uv tool install adabmDCA
adabmDCA --help
```

The equivalent pip command remains supported:

```bash
python -m pip install adabmDCA
```

### Option 2: Install from GitHub with uv

Clone the repository and synchronize its locked environment:

```bash
git clone https://github.com/spqb/adabmDCApy.git
cd adabmDCApy
uv sync --locked
uv run adabmDCA --help
```

The default `dev` dependency group includes the test and lint tools. To install
the documentation dependencies too, run:

```bash
uv sync --locked --group docs
uv run --group docs mkdocs serve
```

For an editable installation without project synchronization:

```bash
uv venv
uv pip install -e .
```

GitHub repository: [adabmDCApy](https://github.com/spqb/adabmDCApy)

The Python package runs on CUDA, Apple Metal, and CPU. CUDA is recommended for
large training and sampling workloads. `--device auto` selects CUDA when
available, then Apple Metal, and finally CPU. BF16 sampling requires an NVIDIA
Ampere-or-newer GPU and Triton; the default FP32 mode has no such requirement.

The [Colab tutorial notebook](https://colab.research.google.com/drive/1uMY1mIlurutquw87FcfX8Rmqfzsyk74Z?usp=sharing)
can run in Colab or from a local checkout.

---

## Julia (Multi-core CPU)

Make sure you’ve installed [Julia](https://julialang.org/downloads/). Then choose one of the following:

### Option 1: Automatic Setup via Shell

```bash
# Download main scripts
wget -O adabmDCA.sh https://raw.githubusercontent.com/spqb/adabmDCA.jl/refs/heads/main/adabmDCA.sh
wget -O execute.jl https://raw.githubusercontent.com/spqb/adabmDCA.jl/refs/heads/main/execute.jl
chmod +x adabmDCA.sh

# Install dependencies and the package
julia --eval 'using Pkg; Pkg.add("ArgParse"); Pkg.add(PackageSpec(url="https://github.com/spqb/adabmDCA.jl"))'
```

### Option 2: Manual Setup via Julia REPL

1. Launch Julia and run:
```bash
using Pkg
Pkg.add(url="https://github.com/spqb/adabmDCA.jl")
Pkg.add("ArgParse")
```

2. Download execution scripts:
```bash
wget https://raw.githubusercontent.com/spqb/adabmDCA.jl/main/adabmDCA.sh
wget https://raw.githubusercontent.com/spqb/adabmDCA.jl/main/execute.jl
chmod +x adabmDCA.sh
```
        

GitHub repo: [adabmDCA.jl](https://github.com/spqb/adabmDCA.jl.git)

---

## C++ (Single-core CPU)

A minimal setup with no external dependencies beyond `make`.

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/spqb/adabmDCAc.git
cd adabmDCAc/src
make
```

2. Return to the root folder and make the main script executable:
```bash
chmod +x adabmDCA.sh
```

3. Verify installation and available options:
```bash
./adabmDCA --help
```

GitHub repo: [adabmDCAc](https://github.com/spqb/adabmDCAc.git)

---

!!! tip
    The implementations share the same general command shape, although some
    runtime and workflow options are implementation-specific. Check the
    installed program's `--help` output when switching languages.
