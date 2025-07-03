# Installation Guide

`adabmDCA` is available in three language-specific implementations:

- **Python** – optimized for GPU execution  
- **Julia** – designed for multi-core CPU usage  
- **C++** – lightweight and single-core CPU compatible

Follow the instructions below based on your preferred environment.

---

## 🔷 Python (GPU-oriented)

### 🔹 Option 1: Install from PyPI (Recommended)

```{bash}
pip install adabmDCA
```

Fastest way to get started. This installs the latest stable release.

### 🔹 Option 2: Install from GitHub

Clone the repository and install the package locally:

```{bash}
git clone https://github.com/spqb/adabmDCApy.git
cd adabmDCApy
pip install .
```

📦 GitHub repo: [adabmDCApy](https://github.com/spqb/adabmDCApy.git)

!!! info
    This version of the code assumes the user to be provided with a GPU. If this is not the case, we provide a [Colab notebook](https://colab.research.google.com/drive/1l5e1W8pk4cB92JAlBElLzpkEk6Hdjk7B?usp=sharing) that can be used with GPU hardware acceleration provided by Google.

---

## 🟣 Julia (Multi-core CPU)

Make sure you’ve installed [Julia](https://julialang.org/downloads/). Then choose one of the following:

### 🔹 Option 1: Automatic Setup via Shell

```{bash}
# Download main scripts
wget -O adabmDCA.sh https://raw.githubusercontent.com/spqb/adabmDCA.jl/refs/heads/main/adabmDCA.sh
wget -O execute.jl https://raw.githubusercontent.com/spqb/adabmDCA.jl/refs/heads/main/execute.jl
chmod +x adabmDCA.sh

# Install dependencies and the package
julia --eval 'using Pkg; Pkg.add("ArgParse"); Pkg.add(PackageSpec(url="https://github.com/spqb/adabmDCA.jl"))'
```

### 🔹 Option 2: Manual Setup via Julia REPL

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
        

📦 GitHub repo: [adabmDCA.jl](https://github.com/spqb/adabmDCA.jl.git)

---

## 🟦 C++ (Single-core CPU)

A minimal setup with no external dependencies beyond `make`.

### 🔹 Installation Steps

1. Clone the repository:
```{bash}
git clone https://github.com/spqb/adabmDCAc.git
cd adabmDCAc/src
make
```

2. Return to the root folder and make the main script executable:
```{bash}
chmod +x adabmDCA.sh
```

3. Verify installation and available options:
```{bash}
./adabmDCA --help
```

📦 GitHub repo: [adabmDCAc](https://github.com/spqb/adabmDCAc.git)

---

!!! tip
    All implementations share a consistent command-line interface. You can switch between them based on your hardware and performance needs without learning new syntax.
