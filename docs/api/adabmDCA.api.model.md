<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.model`
Notebook-friendly DCA model object.

**Global Variables**
---------------
- **TYPE_CHECKING**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L248"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_model`

```python
load_model(
    path: 'str | Path',
    alphabet: 'str | None' = None,
    device: 'str' = 'auto',
    dtype: 'str' = 'float32'
) → DCAModel
```

Load DCA parameters from a text parameter file or a PTT archive.

The file type is detected from its content. For a PTT archive (``.h5``), the final model of the training run is loaded together with its alphabet.



**Args:**

 - <b>`path`</b>:  Parameter file written by ``adabmDCA train`` (``params.dat``,  optionally gzipped) or a PTT archive (``ptt.h5``).
 - <b>`alphabet`</b>:  ``"protein"``, ``"rna"``, ``"dna"`` or an ordered custom token  string. ``None`` reads it from a PTT archive and assumes  ``"protein"`` for text files. An explicit value must match the archive.
 - <b>`device`</b>:  ``"auto"`` (CUDA when available, else CPU), ``"cpu"``, ``"cuda"``  or ``"mps"``.
 - <b>`dtype`</b>:  ``"float32"`` or ``"float64"``. PTT archives keep their precision.



**Returns:**

 - <b>`A `</b>: class:`DCAModel` on the requested device.



**Raises:**

 - <b>`ModelLoadError`</b>:  If the file is missing or cannot be parsed.
 - <b>`InputValidationError`</b>:  If the alphabet is invalid or conflicts with the file.



**Example:**
 ``` model = load_model("output/params.dat.gz", alphabet="rna")```
    >>> model
    DCAModel(L=136, q=5, tokens='-ACGU', ...)



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L317"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `inspect_model`

```python
inspect_model(
    model: 'DCAModel | str | Path',
    alphabet: 'str | None' = None,
    device: 'str' = 'auto',
    dtype: 'str' = 'float32'
) → ModelMetadata
```

Describe a model without using it: length, alphabet, device and precision.



**Args:**

 - <b>`model`</b>:  A :class:`DCAModel`, or a path to a parameter file or PTT archive.
 - <b>`alphabet`</b>:  Alphabet of a text parameter file: ``"protein"``, ``"rna"``,  ``"dna"`` or an ordered custom token string. ``None`` reads it from a  PTT archive and assumes ``"protein"`` for text files. Ignored when
 - <b>```model`` is already a `</b>: class:`DCAModel`.
 - <b>`device`</b>:  ``"auto"`` (CUDA when available, else CPU), ``"cpu"``,  ``"cuda"`` or ``"mps"``. Ignored for an in-memory model.
 - <b>`dtype`</b>:  ``"float32"`` or ``"float64"`` for loaded parameters. Ignored for an  in-memory model.



**Returns:**

 - <b>`The model's `</b>: class:`ModelMetadata`.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DCAModel`
DCA parameters and convenience methods for scoring and sampling.

Use :func:`load_model` for a saved model. Construct ``DCAModel`` directly when parameter tensors are already in memory. The bias tensor determines the model length ``L``, number of states ``q``, device, and dtype. The ordered ``tokens`` string maps state indices to residue symbols.



**Attributes:**

 - <b>`params`</b>:  Parameter tensors, including ``bias`` with shape ``(L, q)``  and ``coupling_matrix`` with shape ``(L, q, L, q)``.
 - <b>`alphabet`</b>:  Standard alphabet name or custom alphabet passed at creation.
 - <b>`tokens`</b>:  Resolved, ordered token string of length ``q``.
 - <b>`source`</b>:  Source path as a string, or ``None`` for an in-memory model.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    params: 'Mapping[str, Tensor]',
    alphabet: 'str' = 'protein',
    source: 'str | Path | None' = None
) → None
```

Validate and retain a set of DCA parameter tensors.



**Args:**

 - <b>`params`</b>:  Mapping containing finite ``bias`` and  ``coupling_matrix`` tensors with compatible shapes.
 - <b>`alphabet`</b>:  ``"protein"``, ``"dna"``, ``"rna"``, or an ordered  custom token string whose length equals ``q``.
 - <b>`source`</b>:  Optional path to the file from which the parameters came.



**Raises:**

 - <b>`InputValidationError`</b>:  If required tensors are missing, nonfinite,  incompatible in shape, or inconsistent with ``alphabet``.


---

#### <kbd>property</kbd> metadata

Portable model description derived from the current bias tensor.

Includes length, alphabet, tokens, source, package version, device, and dtype. ``num_states`` is available on the returned metadata.



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_contact_map`

```python
compute_contact_map() → ndarray
```

Return the model's APC-corrected contact scores.



**Returns:**
  A NumPy array of shape ``(L, L)``. Contact prediction requires  the ``"-"`` gap token in the model alphabet.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_energies`

```python
compute_energies(sequences: 'str | Iterable[str]') → ndarray
```

Compute model energies for one or more aligned sequences.



**Args:**

 - <b>`sequences`</b>:  One sequence string or an iterable of strings, each of  length ``L`` and containing only the model's tokens.



**Returns:**
 A one-dimensional NumPy array with one energy per input sequence, including when a single string is supplied.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_contacts`

```python
predict_contacts() → ContactMapResult
```

Return contact scores together with method and model metadata.



**Returns:**

 - <b>`A `</b>: class:`ContactMapResult` whose score matrix has shape ``(L, L)``. The model alphabet must include ``"-"``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L198"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sample`

```python
sample(
    n_sequences: 'int',
    n_sweeps: 'int' = 1000,
    sampler: 'str' = 'metropolized_gibbs',
    beta: 'float' = 1.0,
    seed: 'int' = 0
) → tuple[str, ]
```

Generate sequences and return only their decoded strings.



**Args:**

 - <b>`n_sequences`</b>:  Number of sequences to generate.
 - <b>`n_sweeps`</b>:  Sampling sweeps applied to the initial chains.
 - <b>`sampler`</b>:  ``"metropolis"``, ``"gibbs"`` or ``"metropolized_gibbs"``.
 - <b>`beta`</b>:  Positive inverse temperature used for sampling.
 - <b>`seed`</b>:  Random seed for reproducible initialization and sampling.



**Returns:**
 A tuple of ``n_sequences`` strings, each of length ``L``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L230"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sample_sequences`

```python
sample_sequences(n_sequences: 'int', **kwargs) → SamplingResult
```

Generate sequences with energies and optional diagnostics.



**Args:**

 - <b>`n_sequences`</b>:  Number of sequences to generate.
 - <b>`**kwargs`</b>:  Additional options accepted by
 - <b>`:func`</b>: `adabmDCA.api.sampling.sample_sequences`, such as ``n_sweeps``, ``sampler``, ``seed``, or ``reference_fasta``.



**Returns:**

 - <b>`A `</b>: class:`SamplingResult` containing generated sequences, energies, model metadata, and any requested diagnostics.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `scan_mutations`

```python
scan_mutations(wild_type: 'str', name: 'str' = 'wild_type') → MutationScanResult
```

Score every single-token substitution of a wild-type sequence.



**Args:**

 - <b>`wild_type`</b>:  Aligned sequence of length ``L`` using model tokens.
 - <b>`name`</b>:  Label attached to the resulting mutation scan.



**Returns:**

 - <b>`A `</b>: class:`MutationScanResult` with wild-type energy and each mutant's energy difference relative to the wild type. Gap substitutions are included when ``"-"`` is a model token.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `score_sequences`

```python
score_sequences(
    sequences: 'str | Iterable[str]',
    local_lambda: 'float' = 1.0
) → EnergyResult
```

Compute energy and CDE-based local free energy with sequence metadata.



**Args:**

 - <b>`sequences`</b>:  One aligned sequence or an iterable of aligned  sequences compatible with this model.
 - <b>`local_lambda`</b>:  Coefficient of summed CDE; defaults to 1.



**Returns:**

 - <b>`An `</b>: class:`EnergyResult` containing the sequences, energy and local-free-energy vectors, and model metadata.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
