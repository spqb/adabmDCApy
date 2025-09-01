<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/dataset.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `dataset`






---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/dataset.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DatasetDCA`
Dataset class for handling multi-sequence alignments data. 

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/dataset.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    path_data: str | Path,
    path_weights: str | Path | None = None,
    alphabet: str = 'protein',
    clustering_th: float = 0.8,
    no_reweighting: bool = False,
    device: device = device(type='cpu'),
    dtype: dtype = torch.float32,
    message: bool = True
)
```

Initialize the dataset. 



**Args:**
 
 - <b>`path_data`</b> (str | Path):  Path to multi sequence alignment in fasta format. 
 - <b>`path_weights`</b> (str | Path | None, optional):  Path to the file containing the importance weights of the sequences. If None, the weights are computed automatically. 
 - <b>`alphabet`</b> (str, optional):  Selects the type of encoding of the sequences. Default choices are ("protein", "rna", "dna"). Defaults to "protein". 
 - <b>`clustering_th`</b> (float, optional):  Sequence identity threshold for clustering. Defaults to 0.8. 
 - <b>`no_reweighting`</b> (bool, optional):  If True, the weights are not computed. Defaults to False. 
 - <b>`device`</b> (torch.device, optional):  Device to be used. Defaults to "cpu". 
 - <b>`dtype`</b> (torch.dtype, optional):  Data type of the dataset. Defaults to torch.float32. 
 - <b>`message`</b> (bool, optional):  Print the import message. Defaults to True. 




---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/dataset.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_effective_size`

```python
get_effective_size() → int
```

Returns the effective size (Meff) of the dataset. 



**Returns:**
 
 - <b>`int`</b>:  Effective size of the dataset. 

---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/dataset.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_num_residues`

```python
get_num_residues() → int
```

Returns the number of residues (L) in the multi-sequence alignment. 



**Returns:**
 
 - <b>`int`</b>:  Length of the MSA. 

---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/dataset.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_num_states`

```python
get_num_states() → int
```

Returns the number of states (q) in the alphabet. 



**Returns:**
 
 - <b>`int`</b>:  Number of states. 

---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/dataset.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `shuffle`

```python
shuffle() → None
```

Shuffles the dataset.  






---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
