# adabmDCA 2.0 - Direct Coupling Analysis in Python

## ⚡ Overview

**adabmDCA 2.0** is a flexible yet easy-to-use implementation of Direct Coupling Analysis (DCA) based on Boltzmann machine learning. This package provides tools for analyzing residue-residue contacts, predicting mutational effects, scoring sequence libraries, and generating artificial sequences, applicable to both protein and RNA families. The package is designed for flexibility and performance, supporting multiple programming languages (C++, Julia, Python) and architectures (single-core/multi-core CPUs and GPUs).  
This repository contains the Python GPU version of adabmDCA, maintained by **Lorenzo Rosset**.

> [!NOTE]
>   - 📖 Check out our [Documentation](https://spqb.github.io/adabmDCApy/) website if you want to dive into the package's main features
>   - ❓ Read the reference paper [Rosset et al., 2025](https://doi.org/10.1101/2025.01.31.635874) and its previous version [Muntoni et al., 2021](https://doi.org/10.1186/s12859-021-04441-9) for a detailed description of the proposed methods
>   - 🌐 Explore the [Colab notebook](https://colab.research.google.com/drive/1l5e1W8pk4cB92JAlBElLzpkEk6Hdjk7B?usp=sharing) with a tutorial on some of the package APIs
    

The project's main repository can be found at [adabmDCA 2.0](https://github.com/spqb/adabmDCA.git).

**Authors:**  
- **Lorenzo Rosset** (Ecole Normale Supérieure ENS, Sorbonne Université)
- **Roberto Netti** (Sorbonne Université)
- **Anna Paola Muntoni** (Politecnico di Torino)
- **Martin Weigt** (Sorbonne Université)
- **Francesco Zamponi** (Sapienza Università di Roma)
  
**Maintainer:** Lorenzo Rosset

## 🚀 Features

- **Direct Coupling Analysis (DCA)** based on Boltzmann machine learning.
- Support for **dense** and **sparse** generative DCA models.
- Available on multiple architectures: single-core and multi-core CPUs, GPUs.
- Ready-to-use for **residue-residue contact prediction**, **mutational-effect prediction**, and **sequence design**.
- Compatible with protein and RNA family analysis.

## ⬇️ Installation

### Option 1: Install from PyPI
Open a terminal and run
```bash
python -m pip install adabmDCA
```

### Option 2: Install from the GitHub repository
Clone the repository locally and then install the requirements and the package. In a terminal, run:

```bash
git clone git@github.com:spqb/adabmDCApy.git
cd adabmDCApy
pip install .
```

## 🕶️ Usage

After installation, all the main routines can be launched through the command-line interface using the command `adabmDCA`.

To get started with adabmDCA in Python, please refer to the [Documentation](https://spqb.github.io/adabmDCApy/) or the [Colab notebook](https://colab.research.google.com/drive/1l5e1W8pk4cB92JAlBElLzpkEk6Hdjk7B?usp=sharing).

## License

This package is open-sourced under the MIT License.

## Citation

If you use this package in your research, please cite:

> Rosset, L., Netti, R., Muntoni, A.P., Weigt, M., & Zamponi, F. (2024). adabmDCA 2.0: A flexible but easy-to-use package for Direct Coupling Analysis.

## Acknowledgments

This work was developed in collaboration with Sorbonne Université, Sapienza Università di Roma, and Politecnico di Torino.


