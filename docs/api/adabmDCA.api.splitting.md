<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/splitting.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.splitting`
High-level API for homology-aware alignment splitting.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/splitting.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `split_alignment`

```python
split_alignment(
    alignment: 'AlignmentInput',
    method: 'str' = 'clustering',
    identity: 'float' = 0.8,
    train_fraction: 'float' = 0.8,
    t1: 'float' = 0.5,
    t2: 'float' = 0.5,
    t3: 'float' = 1.0,
    max_train: 'int | None' = None,
    max_test: 'int | None' = None,
    attempts: 'int' = 1,
    alphabet: 'str' = 'protein',
    seed: 'int' = 0,
    device: 'str' = 'auto'
) → ProfileSplitResult
```

Split an alignment into training and test sets that are not too similar.

``method="clustering"`` clusters sequences at ``identity`` (MMseqs2 when installed and there are at least 100 sequences, otherwise a built-in greedy clustering) and assigns whole clusters to the training set until about ``train_fraction`` of the sequences are in it. ``method="cobalt"`` runs the Cobalt algorithm (Petti and Eddy, 2022), whose thresholds ``t1``-``t3`` bound the identities between and within the two sets.



**Args:**

 - <b>`alignment`</b>:  Alignment to split (path, :class:`Alignment` or sequences).
 - <b>`method`</b>:  ``"clustering"`` or ``"cobalt"``.
 - <b>`identity`</b>:  Clustering: sequence identity, in ``(0, 1]``, that joins two  sequences into one cluster.
 - <b>`train_fraction`</b>:  Clustering: target fraction of sequences in the training set.
 - <b>`t1`</b>:  Cobalt: no test sequence is more than ``t1`` identical to a training sequence.
 - <b>`t2`</b>:  Cobalt: no two test sequences are more than ``t2`` identical.
 - <b>`t3`</b>:  Cobalt: no two training sequences are more than ``t3`` identical.
 - <b>`max_train`</b>:  Cobalt: largest training set, or ``None`` for no limit.
 - <b>`max_test`</b>:  Cobalt: largest test set, or ``None`` for no limit.
 - <b>`attempts`</b>:  Cobalt: random attempts; the one maximizing  ``len(training) * len(test)`` is kept.
 - <b>`alphabet`</b>:  ``"protein"``, ``"rna"``, ``"dna"`` or a custom token string.
 - <b>`seed`</b>:  Random seed.
 - <b>`device`</b>:  ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"`` for identity computations.



**Returns:**

 - <b>`A `</b>: class:`ProfileSplitResult` with the two alignments.



**Raises:**

 - <b>`InputValidationError`</b>:  If a setting is out of range or the alignment has  invalid sequences.
 - <b>`ConvergenceError`</b>:  If Cobalt finds no split with both sets non-empty.



**Example:**
 ``` split = split_alignment("family.fasta", alphabet="rna", identity=0.8)```
    >>> split.save_bundle("splits/family")





---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
