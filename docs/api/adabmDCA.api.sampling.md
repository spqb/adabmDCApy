<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/sampling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.sampling`
High-level sequence-generation operations.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/sampling.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sample_sequences`

```python
sample_sequences(
    model: 'DCAModel | str | Path',
    n_sequences: 'int',
    n_sweeps: 'int' = 1000,
    sampler: 'str' = 'metropolis',
    beta: 'float' = 1.0,
    seed: 'int' = 0,
    reference_fasta: 'AlignmentInput | None' = None,
    weights_path: 'WeightInput | None' = None,
    n_measure: 'int' = 10000,
    mixing_multiplier: 'int' = 2,
    pseudocount: 'float | None' = None,
    clustering_seqid: 'float' = 0.8,
    no_reweighting: 'bool' = False,
    alphabet: 'str' = 'protein',
    device: 'str' = 'auto',
    dtype: 'str' = 'float32',
    collect_diagnostics: 'bool' = False,
    progress: 'ProgressCallback | None' = None,
    is_cancelled: 'CancellationHook | None' = None
) → SamplingResult
```

Generate sequences from a DCA model.

When ``reference_fasta`` is provided, ``n_sweeps`` is the maximum number of sweeps used to estimate mixing and generation runs for ``mixing_multiplier`` times the estimated mixing time. Otherwise, generation runs for exactly ``n_sweeps``.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/sampling.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `generate_sequences`

```python
generate_sequences(**kwargs) → tuple[str, ]
```

Notebook-friendly shortcut returning only generated sequences.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
