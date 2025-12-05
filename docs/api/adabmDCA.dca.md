<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dca.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.dca`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dca.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_seqid`

```python
get_seqid(s1: Tensor, s2: Optional[Tensor] = None) → Tensor
```

Returns a tensor containing the sequence identities between two sets of one-hot encoded sequences. 
- If s2 is provided, computes the sequence identity between the corresponding sequences in s1 and s2. 
- If s2 is a single sequence (L, q), it computes the sequence identities between the dataset s1 and s2. 
- If s2 is none, computes the sequence identity between s1 and a permutation of s1. 



**Args:**
 
 - <b>`s1`</b> (torch.Tensor):  One-hot encoded sequence dataset 1 of shape (batch_size, L, q) or (L, q). 
 - <b>`s2`</b> (Optional[torch.Tensor]):  One-hot encoded sequence dataset 2 of shape (batch_size, L, q) or (L, q). Defaults to None. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Tensor of sequence identities. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dca.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_seqid_stats`

```python
get_seqid_stats(s1: Tensor, s2: Optional[Tensor] = None) → Tuple[Tensor, Tensor]
```


- If s2 is provided, computes the mean and the standard deviation of the mean sequence identity between two sets of one-hot encoded sequences. 
- If s2 is a single sequence (L, q), it computes the mean and the standard deviation of the mean sequence identity between the dataset s1 and s2. 
- If s2 is none, computes the mean and the standard deviation of the mean of the sequence identity between s1 and a permutation of s1. 



**Args:**
 
 - <b>`s1`</b> (torch.Tensor):  One-hot encoded sequence dataset 1 of shape (batch_size, L, q) or (L, q). 
 - <b>`s2`</b> (Optional[torch.Tensor]):  One-hot encoded sequence dataset 2 of shape (batch_size, L, q) or (L, q). Defaults to None. 



**Returns:**
 Tuple[torch.Tensor, torch.Tensor]:  (torch.Tensor) Mean sequence identity  (torch.Tensor) Standard deviation of the mean sequence identity. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dca.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_zerosum_gauge`

```python
set_zerosum_gauge(params: Dict[str, Tensor]) → Dict[str, Tensor]
```

Sets the zero-sum gauge on the coupling matrix. 



**Args:**
 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 



**Returns:**
 
 - <b>`Dict[str, torch.Tensor]`</b>:  New dictionary with modified coupling matrix. 
 - <b>`"bias"`</b>:  torch.Tensor of shape (L, q) 
 - <b>`"coupling_matrix"`</b>:  torch.Tensor of shape (L, q, L, q) 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dca.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_contact_map`

```python
get_contact_map(params: Dict[str, Tensor], tokens: str) → ndarray
```

Computes the contact map from the model coupling matrix. 



**Args:**
 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Model parameters. Should contain: 
        - "coupling_matrix": torch.Tensor of shape (L, q, L, q) 
        - "bias": torch.Tensor of shape (L, q) 
 - <b>`tokens`</b> (str):  Alphabet to be used. 



**Returns:**
 
 - <b>`np.ndarray`</b>:  Contact map. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/dca.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_mf_contact_map`

```python
get_mf_contact_map(
    data: Tensor,
    tokens: str,
    weights: Optional[Tensor] = None
) → ndarray
```

Computes the contact map using mean-field approximation from the data. 



**Args:**
 
 - <b>`data`</b> (torch.Tensor):  Input one-hot data tensor. 
 - <b>`tokens`</b> (str):  Alphabet to be used. 
 - <b>`weights`</b> (Optional[torch.Tensor]):  Weights for the data points. Defaults to None. 



**Returns:**
 
 - <b>`np.ndarray`</b>:  Contact map. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
