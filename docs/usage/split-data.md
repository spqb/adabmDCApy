# Split an alignment · `split-data`

Create a training set and a held-out test set in which close homologues are kept on the same side:

```bash
adabmDCA split-data family family.fasta
```

This writes `family.train.fasta`, `family.test.fasta` and `family.split.json`. Pass the test file to `train -v` and to `sample -v`.

## Why not a random split?

Protein and RNA families contain clusters of near-identical sequences. With a random split, most held-out sequences have a close relative in the training set, so the held-out likelihood measures memorization as much as generalization, and an overfitted model looks good. Assigning whole clusters to one side makes the test set a fairer stand-in for new members of the family.

The price is that the held-out sequences are, by construction, somewhat farther from the training set than a random member of the family would be. Keep this in mind when comparing distances in the [sampling diagnostics](../algorithms/diagnostics.md#distances-and-overfitting). In the benchmarks, generated sequences were never markedly closer to the training set than held-out ones. The two cases where they were marginally closer had no generated sequence identical to a training sequence: LBD (median 0.38–0.42 against 0.43) and RF00379 trained without reweighting (0.221 against 0.228). A small shift of this kind is not, by itself, evidence of copying; a pile of distances near zero is.

## Methods

**`clustering` (default).** Sequences are grouped into clusters at `--identity` minimum sequence identity (default 0.8), and whole clusters are assigned to training or test until the training set holds about `--train-fraction` of the sequences (default 0.8). No sequence is discarded; because clusters are indivisible, the actual fraction can differ from the target. At least two clusters are needed. When the `mmseqs` executable is on the `PATH` and the alignment has at least 100 sequences, clustering uses MMseqs2 `easy-cluster`; otherwise, or if MMseqs2 fails, identities are computed position by position on the aligned sequences (gaps included) on the selected `--device`. The two backends define identity differently (MMseqs2 aligns the sequences again), so the clusters can differ slightly.

**`cobalt`.** Follows [Petti and Eddy (2022)](https://doi.org/10.1371/journal.pcbi.1009492) and enforces hard limits: no training–test pair above `-t1` identity (default 0.5), no test–test pair above `-t2` (0.5), no training–training pair above `-t3` (1.0, no limit). Sequences that cannot be placed are discarded. `--bestof N` tries N random orders and keeps the split with the largest product of set sizes; `--maxtrain` and `--maxtest` cap the set sizes.

```bash
adabmDCA split-data --identity 0.9 --train-fraction 0.7 family family.fasta
adabmDCA split-data --method cobalt -t1 0.5 -t2 0.5 --bestof 10 family family.fasta
```

!!! warning "Check that the test set still looks like the family"
    Strict COBALT thresholds can push whole subfamilies into one of the two sets. The test set then covers regions of sequence space that the training set does not contain at all. For a generative model trained on the training set, such a test set is an *out-of-distribution* sample: validation likelihood and Pearson stay low however good the model is, the validation stop loses its meaning, and the [distance checks](../algorithms/diagnostics.md#distances-and-overfitting) compare the samples with the wrong yardstick.

    Before training, project both sets on the principal components of the training set and compare them. The test sequences should fall within the clusters of the training sequences. If they form clusters of their own, loosen `-t1` (for example from 0.5 to 0.7) or use the default `clustering` method.

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    from adabmDCA import load_alignment

    train = load_alignment("family.train.fasta").to_onehot(flatten=True).numpy()
    test = load_alignment("family.test.fasta").to_onehot(flatten=True).numpy()
    mean = train.mean(axis=0)
    _, _, components = np.linalg.svd(train - mean, full_matrices=False)
    project = lambda x: (x - mean) @ components[:2].T

    for data, label in ((train, "training"), (test, "test")):
        pcs = project(data)
        plt.scatter(pcs[:, 0], pcs[:, 1], s=4, alpha=0.4, label=label)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(); plt.savefig("split_pca.png")
    ```

`--seed` makes the split reproducible.

## Outputs

| File | Content |
| --- | --- |
| `<prefix>.train.fasta` | Training sequences |
| `<prefix>.test.fasta` | Held-out sequences |
| `<prefix>.split.json` | Method, parameters and resulting set sizes |

## In Python

```python
from adabmDCA import load_alignment, split_alignment, train_model, PTTConfig

alignment = load_alignment("family.fasta", alphabet="rna")
split = split_alignment(alignment, method="clustering", identity=0.8,
                        train_fraction=0.8, alphabet="rna", seed=7)
print(split.training.num_sequences, split.test.num_sequences)
split.save_bundle("family")                    # writes the three files

result = train_model(split.training, validation_path=split.test, alphabet="rna",
                     ptt=PTTConfig(validation_stop=True), output_dir="model")
```

Unlike the command line, the Python workflow functions (all but `load_alignment`) do not detect the alphabet: they default to `"protein"`, so pass `alphabet=` for nucleotide or custom data. `attempts`, `max_train` and `max_test` are the Python names of `--bestof`, `--maxtrain` and `--maxtest`.
