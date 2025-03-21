# Installation

`adabmDCA` is available in three languages: C++ (single-core CPU), Julia (multi-core CPU) and Python (GPU-oriented). Follow the instructions for installing the desired implementation.

## Python implementation

### Option 1: Install from PyPI
Open a terminal and run
```bash
pip install adabmDCA
```

### Option 2: Install from the GitHub repository
Clone the repository locally and then install the requirements and the package. In a terminal, run:

```bash
git clone https://github.com/spqb/adabmDCApy.git
cd adabmDCApy
pip install .
```

The main repository of the implementation can be found at [adabmDCApy](https://github.com/spqb/adabmDCApy.git).

## Julia implementation

After installing [Julia](https://julialang.org/downloads/) on your system, you can install the package in one of the following ways:

### Option 1: Using bash command

Open a terminal in the desired folder, and run the following commands:

```bash
# Download scripts from Github
wget -O adabmDCA.sh https://raw.githubusercontent.com/spqb/adabmDCA.jl/refs/heads/main/adabmDCA.sh
wget -O execute.jl https://raw.githubusercontent.com/spqb/adabmDCA.jl/refs/heads/main/execute.jl
chmod +x adabmDCA.sh

# Install ArgParse and adabmDCA.jl from the GitHub repo
julia --eval 'using Pkg; Pkg.add("ArgParse"); Pkg.add(PackageSpec(url="https://github.com/spqb/adabmDCA.jl"))'
```
This will install all necessary dependencies and set up the package.

### Option 2: Manual Installation via Julia

1.  Open Julia and install the package by running:
    ```Julia
    using Pkg
    Pkg.add(url="https://github.com/spqb/adabmDCA.jl")
    Pkg.add("ArgParse")
    ```

2.  Download the files `adabmDCA.sh` and `execute.jl` into the same folder
    ```bash
    wget https://github.com/spqb/adabmDCA.jl/blob/main/install.sh
    wget https://github.com/spqb/adabmDCA.jl/blob/main/execute.jl
    ```

3.  Make the script executable by opening a terminal in the folder and running:
    ```bash
    chmod +x adabmDCA.sh
    ```
This will set up the package for use.

The main repository of the implementation can be found at [adabmDCA.jl](https://github.com/spqb/adabmDCA.jl.git).

## C++ implementation

1.    Clone the repository
      ```bash
      git clone git@github.com:spqb/adabmDCAc.git
      ```
2.    In the __src__ folder run
      ```bash
      make
      ```
3.    It will generate the executable file __adabmDCA__. In the main folder run also `chmod +x adabmDCA.sh` to use the main script file. See
      ```bash
      ./adabmDCA --help
      ```
      for a complete list of features.

The main repository of the implementation can be found at [adabmDCAc](https://github.com/spqb/adabmDCAc.git).