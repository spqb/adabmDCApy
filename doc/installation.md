## Installation

`adabmDCA` is available in three languages: C++ (single-core CPU), Julia (multi-core CPU) and Python (GPU-oriented). Follow the instructions for installing the desired implementation.

### Python implementation
Open a terminal, clone the github repository locally, install the dependencies and the package:

```{bash}
git clone git@github.com:spqb/adabmDCApy.git
cd adabmDCApy
pip install -r requirements.txt
pip install -e .
```

### Julia implementation
After installing [Julia](https://julialang.org/downloads/) on your system, you can install the package in one of the following ways:

#### Option 1: Using bash command
Open a terminal in the desired folder, and run the following commands:

```{bash}
# Download scripts from Github
wget -O adabmDCA.sh https://raw.githubusercontent.com/spqb/adabmDCA.jl/refs/heads/main/adabmDCA.sh
wget -O execute.jl https://raw.githubusercontent.com/spqb/adabmDCA.jl/refs/heads/main/execute.jl
chmod +x adabmDCA.sh

# Install ArgParse and adabmDCA.jl from the GitHub repo
julia --eval 'using Pkg; Pkg.add("ArgParse"); Pkg.add(PackageSpec(url="https://github.com/spqb/adabmDCA.jl"))'
```
This will install all necessary dependencies and set up the package.

#### Option 2: Manual Installation via Julia

1.  Open Julia and install the package by running:
    ```{Julia}
    using Pkg
    Pkg.add(url="https://github.com/spqb/adabmDCA.jl")
    Pkg.add("ArgParse")
    ```
2.  Download the files `adabmDCA.sh` and `execute.jl` into the same folder.

3.  Make the script executable by opening a terminal in the folder and running:
    ```{bash}
    chmod +x adabmDCA.sh
    ```
This will set up the package for use.

### C++ implementation
TODO