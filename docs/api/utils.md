<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `utils`





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/utils.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `init_parameters`

```python
init_parameters(fi: Tensor) → Dict[str, Tensor]
```

Initialize the parameters of the DCA model. 



**Args:**
 
 - <b>`fi`</b> (torch.Tensor):  Single-point frequencies of the data. 



**Returns:**
 
 - <b>`Dict[str, torch.Tensor]`</b>:  Parameters of the model. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/utils.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `init_chains`

```python
init_chains(
    num_chains: int,
    L: int,
    q: int,
    device: device,
    dtype: dtype = torch.float32,
    fi: Tensor | None = None
) → Tensor
```

Initialize the chains of the DCA model. If 'fi' is provided, the chains are sampled from the profile model, otherwise they are sampled uniformly at random. 



**Args:**
 
 - <b>`num_chains`</b> (int):  Number of parallel chains. 
 - <b>`L`</b> (int):  Length of the MSA. 
 - <b>`q`</b> (int):  Number of values that each residue can assume. 
 - <b>`device`</b> (torch.device):  Device where to store the chains. 
 - <b>`dtype`</b> (torch.dtype, optional):  Data type of the chains. Defaults to torch.float32. 
 - <b>`fi`</b> (torch.Tensor | None, optional):  Single-point frequencies. Defaults to None. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Initialized parallel chains in one-hot encoding format. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/utils.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/utils.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/utils.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_device`

```python
get_device(device: str, message: bool = True) → device
```

Returns the device where to store the tensors. 



**Args:**
 
 - <b>`device`</b> (str):  Device to be used. 
 - <b>`message`</b> (bool, optional):  Print the device. Defaults to True. 



**Returns:**
 
 - <b>`torch.device`</b>:  Device. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/utils.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_dtype`

```python
get_dtype(dtype: str) → dtype
```

Returns the data type of the tensors. 



**Args:**
 
 - <b>`dtype`</b> (str):  Data type. 



**Returns:**
 
 - <b>`torch.dtype`</b>:  Data type. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
