# Welcome to the `adabmDCA 2.0` Documentation

**`adabmDCA`** is a versatile library for Direct Coupling Analysis (DCA), enabling the training, sampling, and application of Boltzmann Machines (Potts models) on biological sequence data.

!!! info "Instructons"
    This documentation is meant for providing a user-friendly description of the `adabmDCA` package main features. It is supported by:
    
    - The main article [[Rosset et al., 2025](https://doi.org/10.1101/2025.01.31.635874)], with detailed explanations of the main features. The present documentation is a shorter version of the paper, but it includes additional features
    - The [Colab notebook](https://colab.research.google.com/drive/1l5e1W8pk4cB92JAlBElLzpkEk6Hdjk7B?usp=sharing) providing a tutorial of the APIs for training, sampling and analyzing a `bmDCA` model (Python only)

This tutorial introduces the **new and enhanced version** of `adabmDCA` [[Muntoni at al., 2021](https://doi.org/10.1186/s12859-021-04441-9)]. The software is available in three language-specific implementations:

- [**C++**](https://github.com/spqb/adabmDCAc.git) – optimized for single-core CPUs  
- [**Julia**](https://github.com/spqb/adabmDCA.jl.git) – ideal for multi-core CPU setups  
- [**Python**](https://github.com/spqb/adabmDCApy.git) – GPU-accelerated and feature-rich

All versions share a unified terminal-based interface, allowing users to choose based on their hardware and performance needs.

## Core Capabilities

### 🧠 Model Training
Choose from three training strategies to fit your model complexity and goals:

- [**`bmDCA`**](training.md#bmdca): Fully-connected Boltzmann Machine [[Figliuzzi et al., 2018](https://doi.org/10.1093/molbev/msv211)]
- [**`eaDCA`**](training.md#eadca): Sparse model with progressively added couplings [[Calvanese et al., 2024](https://doi.org/10.1093/nar/gkae289)]
- [**`edDCA`**](training.md#eddca): Prunes an existing `bmDCA` model down to a sparse network [[Barrat-Charlaix et al., 2021](https://doi.org/10.1103/PhysRevE.104.024407)]

### ⚙️ Applications of Pretrained Models
Once trained, models can be used to:

- [Generate new sequences](applications.md#sampling)
- [Predict structural contacts](applications.md#contact-prediction) [[Ekeberg et al., 2013](https://doi.org/10.1103/PhysRevE.87.012707)].
- [Score sequence datasets](applications.md#scoring) based on model energy
- [Build mutational libraries](applications.md#DMS) with DCA-based scoring

### 🚀 Advanced Features in Python (`adabmDCApy`)
The Python version includes exclusive features:

- [Experimental feedback reintegration](applications.md#reintegration) for refined models [[Calvanese et al., 2025](https://doi.org/10.48550/arXiv.2504.01593)]
- Thermodynamic integration to estimate model entropy
- [`Profmark`](applications.md#profmark): GPU-accelerated dataset splitting with phylogenetic and sampling bias control, based on the `cobalt` algorithm [[Petti et al., 2022](https://doi.org/10.1371/journal.pcbi.1009492)]

## Get Started
Ready to run? Skip ahead to the [Quicklist](quicklist.md#quicklist) for command-line usage examples.
