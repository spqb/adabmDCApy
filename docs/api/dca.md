<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/dca.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `dca`





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/dca.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_seqid`

```python
get_seqid(
    s1: Tensor,
    s2: Tensor | None = None,
    average: bool = False
) → Union[Tensor, Tuple[Tensor, Tensor]]
```

When average is True: 
- If s2 is provided, computes the mean and the standard deviation of the mean sequence identity between two sets of one-hot encoded sequences. 
- If s2 is a single sequence (L, q), it computes the mean and the standard deviation of the mean sequence identity between the dataset s1 and s2. 
- If s2 is none, computes the mean and the standard deviation of the mean of the sequence identity between s1 and a permutation of s1. 

When average is False it returns the array of sequence identities. 



**Args:**
 
 - <b>`s1`</b> (torch.Tensor):  Sequence dataset 1. 
 - <b>`s2`</b> (torch.Tensor | None):  Sequence dataset 2. Defaults to None. 
 - <b>`average`</b> (bool):  Whether to return the average and standard deviation of the sequence identity or the array of sequence identities. 



**Returns:**
 
 - <b>`torch.Tensor | Tuple[torch.Tensor, torch.Tensor]`</b>:  List of sequence identities or mean sequence identity and standard deviation of the mean. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/dca.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_zerosum_gauge`

```python
set_zerosum_gauge(params: Dict[str, Tensor]) → Dict[str, Tensor]
```

Sets the zero-sum gauge on the coupling matrix. 



**Args:**
 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 



**Returns:**
 
 - <b>`Dict[str, torch.Tensor]`</b>:  Parameters with fixed gauge. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/dca.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_contact_map`

```python
get_contact_map(params: Dict[str, Tensor], tokens: str) → ndarray
```

Computes the contact map from the model coupling matrix. 



**Args:**
 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Model parameters. 
 - <b>`tokens`</b> (str):  Alphabet. 



**Returns:**
 
 - <b>`np.ndarray`</b>:  Contact map. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/dca.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_mf_contact_map`

```python
get_mf_contact_map(
    data: Tensor,
    tokens: str,
    weights: Tensor | None = None
) → ndarray
```

Computes the contact map from the model coupling matrix. 



**Args:**
 
 - <b>`data`</b> (torch.Tensor):  Input one-hot data tensor. 
 - <b>`tokens`</b> (str):  Alphabet. 
 - <b>`weights`</b> (torch.Tensor | None):  Weights for the data points. Defaults to None. 



**Returns:**
 
 - <b>`np.ndarray`</b>:  Contact map. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
