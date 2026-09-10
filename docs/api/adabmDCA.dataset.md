<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.dataset`




**Global Variables**
---------------
- **CPU_DEVICE**


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DatasetDCA`
Dataset class for handling multi-sequence alignments data.



**Args:**

 - <b>`path_data`</b> (str):  Path to multi sequence alignment in fasta format.
 - <b>`path_weights`</b> (Optional[str], optional):  Path to the file containing the importance weights of the sequences. If None, the weights are computed automatically.
 - <b>`alphabet`</b> (str, optional):  Selects the type of encoding of the sequences. Default choices are ("protein", "rna", "dna"). Defaults to "protein".
 - <b>`clustering_th`</b> (float, optional):  Sequence identity threshold for clustering. Defaults to 0.8.
 - <b>`no_reweighting`</b> (bool, optional):  If True, the weights are not computed. Defaults to False.
 - <b>`remove_duplicates`</b> (bool, optional):  If True, removes duplicate sequences from the dataset. Defaults to False.
 - <b>`filter_sequences`</b> (bool, optional):  If True, removes sequences containing tokens not in the alphabet. Defaults to False.
 - <b>`message`</b> (bool, optional):  Print the import message. Defaults to True.
 - <b>`device`</b> (torch.device, optional):  Device to be used. Defaults to "cpu".
 - <b>`dtype`</b> (torch.dtype, optional):  Data type of the dataset. Defaults to torch.float32.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    path_data: str | Path | Alignment,
    path_weights: str | Path | Sequence[float] | ndarray | Tensor | None = None,
    alphabet: str = 'protein',
    clustering_th: float = 0.8,
    no_reweighting: bool = False,
    remove_duplicates: bool = False,
    filter_sequences: bool = False,
    message: bool = True,
    device: device = device(type='cpu'),
    dtype: dtype = torch.float32
)
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Load and materialize an alignment without constructor-side policy.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_loaded_alignment`

```python
from_loaded_alignment(
    loaded: LoadedAlignment,
    weights: str | Path | Sequence[float] | ndarray | Tensor | None = None,
    clustering_th: float = 0.8,
    no_reweighting: bool = False,
    device: device = device(type='cpu'),
    dtype: dtype = torch.float32,
    allow_signed_weights: bool = False
) → DatasetDCA
```

Materialize an in-memory dataset from a validated alignment.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_path`

```python
from_path(path: str | Path, **kwargs) → DatasetDCA
```

Compatibility factory mirroring :meth:`from_alignment`.

.. deprecated:: 0.7.8  Use :meth:`from_alignment` instead.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_effective_size`

```python
get_effective_size() → float
```

Returns the effective size (Meff) of the dataset.



**Returns:**

 - <b>`float`</b>:  Sum of the sequence weights.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_num_residues`

```python
get_num_residues() → int
```

Returns the number of residues (L) in the multi-sequence alignment.



**Returns:**

 - <b>`int`</b>:  Length of the MSA.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_num_states`

```python
get_num_states() → int
```

Returns the number of states (q) in the alphabet.



**Returns:**

 - <b>`int`</b>:  Number of states.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `shuffle`

```python
shuffle() → None
```

Shuffles the dataset.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dataset.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_one_hot`

```python
to_one_hot() → Tensor
```

Converts the dataset to one-hot encoding.



**Returns:**

 - <b>`torch.Tensor`</b>:  One-hot encoded dataset of shape (M, L, q).




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
