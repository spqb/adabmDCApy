<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.utils`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/utils.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `init_parameters`

```python
init_parameters(fi: Tensor) → Dict[str, Tensor]
```

Initialize the parameters of the DCA model. The bias terms are initialized from the single-point frequencies 'fi', while the coupling matrix is initialized to zero.



**Args:**

 - <b>`fi`</b> (torch.Tensor):  Single-point frequencies of the data.



**Returns:**
 Dict[str, torch.Tensor]:
 - <b>`"bias" (torch.Tensor)`</b>:  Bias terms.
 - <b>`"coupling_matrix" (torch.Tensor)`</b>:  Coupling matrix.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/utils.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `init_chains`

```python
init_chains(
    num_chains: int,
    L: int,
    q: int,
    device: device,
    dtype: dtype = torch.float32,
    fi: Optional[Tensor] = None
) → Tensor
```

Initialize the Markov chains of the DCA model. If 'fi' is provided, the chains are sampled from the profile model, otherwise they are sampled uniformly at random.



**Args:**

 - <b>`num_chains`</b> (int):  Number of parallel chains.
 - <b>`L`</b> (int):  Length of the MSA.
 - <b>`q`</b> (int):  Number of values that each residue can assume.
 - <b>`device`</b> (torch.device):  Device where to store the chains.
 - <b>`dtype`</b> (torch.dtype, optional):  Data type of the chains. Defaults to torch.float32.
 - <b>`fi`</b> (Optional[torch.Tensor], optional):  Single-point frequencies. Defaults to None.



**Returns:**

 - <b>`torch.Tensor`</b>:  Initialized Markov chains in one-hot encoding format, shape (num_chains, L, q).


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/utils.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_mask_save`

```python
get_mask_save(L: int, q: int, device: device) → Tensor
```

Returns the mask to save the upper-triangular part of the coupling matrix.



**Args:**

 - <b>`L`</b> (int):  Length of the MSA.
 - <b>`q`</b> (int):  Number of values that each residue can assume.
 - <b>`device`</b> (torch.device):  Device where to store the mask.



**Returns:**

 - <b>`torch.Tensor`</b>:  Mask.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/utils.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `systematic_resampling`

```python
systematic_resampling(chains: Tensor, weights: Tensor) → Tensor
```

Performs the systematic resampling of the chains according to their relative weight.



**Args:**

 - <b>`chains`</b> (torch.Tensor):  Chains.
 - <b>`weights`</b> (torch.Tensor):  Weights of the chains.



**Returns:**

 - <b>`torch.Tensor`</b>:  Resampled chains.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/utils.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `resample_sequences`

```python
resample_sequences(data: Tensor, weights: Tensor, nextract: int) → Tensor
```

Extracts nextract sequences from data with replacement according to the weights.



**Args:**

 - <b>`data`</b> (torch.Tensor):  Data array.
 - <b>`weights`</b> (torch.Tensor):  Weights of the sequences.
 - <b>`nextract`</b> (int):  Number of sequences to be extracted.



**Returns:**

 - <b>`torch.Tensor`</b>:  Extracted sequences.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/utils.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_device`

```python
get_device(device: str, message: bool = True) → device
```

Returns the device where to store the tensors.



**Args:**

 - <b>`device`</b> (str):  Device to use. ``auto`` prefers CUDA, then MPS, then CPU.  Explicit values include ``cpu``, ``cuda`` and ``mps``.
 - <b>`message`</b> (bool, optional):  Print the device. Defaults to True.



**Returns:**

 - <b>`torch.device`</b>:  Device.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/utils.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_dtype`

```python
get_dtype(dtype: str) → dtype
```

Returns the data type of the tensors.



**Args:**

 - <b>`dtype`</b> (str):  Data type. Possible values are 'float32' and 'float64'.



**Returns:**

 - <b>`torch.dtype`</b>:  Data type.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/utils.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parse_log_file`

```python
parse_log_file(log_path: str) → Tuple[Dict[str, str], Dict[str, ndarray]]
```

Parse a DCA training log file.



**Args:**

 - <b>`log_path`</b> (str):  Path to the log file.



**Returns:**

 - <b>`Tuple[Dict[str, str], Dict[str, np.ndarray]]`</b>:  Dictionary containing metadata and training data.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
