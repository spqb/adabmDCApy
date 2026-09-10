<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.model`
Notebook-friendly DCA model object.

**Global Variables**
---------------
- **TYPE_CHECKING**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L150"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_model`

```python
load_model(
    path: 'str | Path',
    alphabet: 'str' = 'protein',
    device: 'str' = 'auto',
    dtype: 'str' = 'float32'
) → DCAModel
```

Load DCA parameters into a reusable :class:`DCAModel`.

``device='auto'`` selects CUDA when available and otherwise uses CPU. Explicit ``'cpu'``, ``'cuda'``, and ``'mps'`` values remain supported.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `inspect_model`

```python
inspect_model(
    model: 'DCAModel | str | Path',
    alphabet: 'str' = 'protein',
    device: 'str' = 'auto',
    dtype: 'str' = 'float32'
) → ModelMetadata
```

Return portable metadata for an in-memory or saved model.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DCAModel`
Loaded DCA parameters with convenient analysis methods.

Prefer :func:`load_model` when loading a model saved by adabmDCA. Direct construction is useful for advanced users and tests that already have the parameter tensors in memory.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    params: 'Mapping[str, Tensor]',
    alphabet: 'str' = 'protein',
    source: 'str | Path | None' = None
) → None
```






---

#### <kbd>property</kbd> metadata







---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_contact_map`

```python
compute_contact_map()
```

Return the model's APC-corrected contact-score matrix.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_energies`

```python
compute_energies(sequences: 'str | Iterable[str]')
```

Return a NumPy vector with one energy per sequence.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_contacts`

```python
predict_contacts() → ContactMapResult
```

Return contact scores plus method and model metadata.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sample`

```python
sample(
    n_sequences: 'int',
    n_sweeps: 'int' = 1000,
    sampler: 'str' = 'metropolis',
    beta: 'float' = 1.0,
    seed: 'int' = 0
) → tuple[str, ]
```

Generate sequences and return them as ordinary strings.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sample_sequences`

```python
sample_sequences(n_sequences: 'int', **kwargs) → SamplingResult
```

Generate sequences and return structured diagnostics.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `scan_mutations`

```python
scan_mutations(wild_type: 'str', name: 'str' = 'wild_type') → MutationScanResult
```

Score every single-residue mutant of ``wild_type``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/model.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `score_sequences`

```python
score_sequences(sequences: 'str | Iterable[str]') → EnergyResult
```

Return energies plus sequence and model metadata.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
