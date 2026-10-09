<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.dataset`




**Global Variables**
---------------
- **CPU_DEVICE**


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DatasetDCA`
Encoded alignment with sequence weights, as a PyTorch dataset.

Build one with :meth:`from_alignment`. Items are ``(sequence, weight)`` pairs of encoded sequences.



**Attributes:**

 - <b>`names`</b>:  Sequence names.
 - <b>`data`</b>:  Encoded sequences, shape ``(M, L)``, integer states.
 - <b>`weights`</b>:  Weight of each sequence.
 - <b>`tokens`</b>:  Ordered alphabet.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args: Any, **kwargs: Any) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_alignment`

```python
from_alignment(
    alignment: str | Path | Alignment,
    weights: str | Path | Sequence[float] | ndarray | Tensor | None = None,
    load_config: AlignmentLoadConfig | None = None,
    clustering_th: float = 0.8,
    no_reweighting: bool = False,
    device: device = device(type='cpu'),
    dtype: dtype = torch.float32,
    allow_signed_weights: bool = False
) → DatasetDCA
```

Load an alignment and its weights into a dataset.

An ``Alignment`` already returned by :func:`load_alignment` is reused when ``load_config`` is omitted, preserving its filtering provenance. Pass ``load_config`` to apply a new loading policy.



**Args:**

 - <b>`alignment`</b>:  Path, :class:`Alignment` or sequences.
 - <b>`weights`</b>:  Sequence weights (file, array or tensor), or ``None`` to compute them.
 - <b>`load_config`</b>:  Loading policy; see :class:`AlignmentLoadConfig`.
 - <b>`clustering_th`</b>:  Identity threshold of the computed weights.
 - <b>`no_reweighting`</b>:  Give every sequence weight 1.
 - <b>`device`</b>:  Device of the tensors.
 - <b>`dtype`</b>:  Precision of the weights and one-hot encodings.
 - <b>`allow_signed_weights`</b>:  Accept negative weights (experimental reintegration).



**Returns:**

 - <b>`A `</b>: class:`DatasetDCA`.



**Example:**
 ``` dataset = DatasetDCA.from_alignment("family.fasta", load_config=AlignmentLoadConfig(alphabet="rna"))```
    >>> fi, fij = dataset.get_frequencies(pseudocount=1 / dataset.get_effective_size())


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_effective_size`

```python
get_effective_size() → float
```

Returns the effective size (Meff) of the dataset.



**Returns:**

 - <b>`float`</b>:  Sum of the sequence weights.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_frequencies`

```python
get_frequencies(
    pseudocount: float = 0.0,
    batch_size: int = 10000
) → tuple[Tensor, Tensor]
```

Computes the single-site and two-site frequencies of the dataset. When there are too many sequences, computing the frequencies directly from the one-hot encoding can be memory-intensive. Therefore, we compute the frequencies using batched operations.



**Args:**

 - <b>`pseudocount`</b> (float, optional):  Pseudocount to be added to the frequencies. Defaults to 0.0.
 - <b>`batch_size`</b> (int, optional):  Batch size to use when computing the frequencies. Defaults to 10000.



**Returns:**

 - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  Single-site frequencies fi of shape (L, q) and two-site frequencies fij of shape (L, q, L, q).

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_num_residues`

```python
get_num_residues() → int
```

Returns the number of residues (L) in the multi-sequence alignment.



**Returns:**

 - <b>`int`</b>:  Length of the MSA.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_num_states`

```python
get_num_states() → int
```

Returns the number of states (q) in the alphabet.



**Returns:**

 - <b>`int`</b>:  Number of states.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `shuffle`

```python
shuffle() → None
```

Shuffles the dataset.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L152"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_one_hot`

```python
to_one_hot() → Tensor
```

Converts the dataset to one-hot encoding.



**Returns:**

 - <b>`torch.Tensor`</b>:  One-hot encoded dataset of shape (M, L, q).




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
