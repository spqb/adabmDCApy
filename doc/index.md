% adabmDCA documentation master file, created by
% sphinx-quickstart on Mon Oct  7 16:05:18 2024.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# adabmDCA documentation

This tutorial presents a new version of `adabmDCA` {cite:p}`muntoni_adabmdca_2021`. The package comes in three different languages: C++ (single-core CPU), Julia (multi-core CPU), and Python (GPU-oriented). They share the same front-end interface from the terminal allowing the user to install and use one of the three equivalent versions based on hardware or software constraints.

We provide three different training routines:
- **bmDCA** |:whale2:| : Trains a fully-connected DCA model {cite:p}`figliuzzi_how_2018`;
- **eaDCA** |:dolphin:| : Trains a DCA model on a sparse coupling network by progressively adding couplings during the training {cite:p}`calvanese_towards_2024`;
- **edDCA** |:fish:| : Starts from a trained bmDCA model and iteratively removes the less informative couplings until the target sparsity is reached {cite:p}`barrat-charlaix_sparse_2021`.

Additionally, we provide several routines for sampling and analyzing the generated sequences once a DCA model is trained, for constructing and evaluating - according to a DCA model - a single mutant library from a given wild type, and finally, for computing the pairwise contact scores, in terms of average-product corrected Frobenius norms of the DCA couplings {cite:p}`ekeberg_improved_2013`.

```{toctree}
:caption: 'Contents:'
:maxdepth: 4

installation.md
preprocessing.md
implementation.md
quicklist.md
adabmDCApy.md
bibliography.md
```
