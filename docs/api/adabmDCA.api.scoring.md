<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/scoring.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.scoring`
High-level sequence-energy operations.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/scoring.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `summed_cde`

```python
summed_cde(
    data: 'Tensor',
    params: 'dict[str, Tensor]',
    batch_size: 'int' = 64
) → ndarray
```

Return summed conditional entropy for each one-hot sequence, in nats.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/scoring.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fit_local_lambda`

```python
fit_local_lambda(
    energies: 'ndarray',
    cde_sum: 'ndarray'
) → dict[str, float | int]
```

Fit E = intercept + lambda * summed CDE by ordinary least squares.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/scoring.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `score_sequences`

```python
score_sequences(
    sequences: 'str | Iterable[str] | None' = None,
    model: 'DCAModel | str | Path',
    fasta_path: 'AlignmentInput | None' = None,
    alphabet: 'str | None' = None,
    device: 'str' = 'auto',
    dtype: 'str' = 'float32',
    remove_duplicates: 'bool' = False,
    local_lambda: 'float | None' = 1.0
) → EnergyResult
```

Compute DCA energies of aligned sequences, with their local free energies.

Energies follow ``E(s) = -sum_i h_i(s_i) - sum_{i<j} J_ij(s_i, s_j)``: lower is more probable under the model. Unless ``local_lambda`` is ``None``, each sequence also gets its summed context-dependent entropy (CDE) and the local free energy ``E - local_lambda * CDE``.



**Args:**

 - <b>`sequences`</b>:  One aligned sequence or an iterable of them, each of length  ``L`` over the model's tokens. Provide exactly one of ``sequences``  and ``fasta_path``.
 - <b>`model`</b>:  A :class:`DCAModel`, or a path to a parameter file or PTT archive.
 - <b>`fasta_path`</b>:  Alignment to score (path, :class:`Alignment` or sequences);  sequence names are kept in the result.
 - <b>`alphabet`</b>:  Alphabet of a text parameter file: ``"protein"``, ``"rna"``,  ``"dna"`` or an ordered custom token string. ``None`` reads it from a  PTT archive and assumes ``"protein"`` for text files. Ignored when
 - <b>```model`` is already a `</b>: class:`DCAModel`.
 - <b>`device`</b>:  ``"auto"`` (CUDA when available, else CPU), ``"cpu"``,  ``"cuda"`` or ``"mps"``. Ignored for an in-memory model.
 - <b>`dtype`</b>:  ``"float32"`` or ``"float64"`` for loaded parameters. Ignored for an  in-memory model.
 - <b>`remove_duplicates`</b>:  With ``fasta_path``, score each distinct sequence once.
 - <b>`local_lambda`</b>:  Weight of the summed CDE in the local free energy, or  ``None`` to skip the CDE computation.



**Returns:**

 - <b>`An `</b>: class:`EnergyResult` with one energy per sequence, in input order.



**Raises:**

 - <b>`InputValidationError`</b>:  If both or neither of ``sequences`` and  ``fasta_path`` are given, a sequence is incompatible with the model,  or ``local_lambda`` is not finite.



**Example:**
 ``` result = score_sequences(fasta_path="designs.fasta", model="params.dat.gz")```
    >>> result.to_dataframe().sort_values("energy").head()



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/scoring.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_energies`

```python
compute_energies(
    sequences: 'str | Iterable[str] | None' = None,
    model: 'DCAModel | str | Path',
    fasta_path: 'AlignmentInput | None' = None,
    alphabet: 'str | None' = None,
    device: 'str' = 'auto',
    dtype: 'str' = 'float32',
    remove_duplicates: 'bool' = False
) → ndarray
```

Return only the DCA energies of :func:`score_sequences`, without CDE.

Takes the same arguments as :func:`score_sequences` except ``local_lambda``.



**Returns:**
  A one-dimensional NumPy array with one energy per sequence, also for a  single sequence string.



**Example:**
 ``` compute_energies(["ACGU...", "ACGA..."], model=model)```
     array([-212.4, -208.9])





---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
