<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/stats.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `stats`





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/stats.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_freq_single_point`

```python
get_freq_single_point(
    data: Tensor,
    weights: Tensor | None = None,
    pseudo_count: float = 0.0
) → Tensor
```

Computes the single point frequencies of the input MSA. 

**Args:**
 
 - <b>`data`</b> (torch.Tensor):  One-hot encoded data array. 
 - <b>`weights`</b> (torch.Tensor | None, optional):  Weights of the sequences. 
 - <b>`pseudo_count`</b> (float, optional):  Pseudo count to be added to the frequencies. Defaults to 0.0. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the input data is not a 3D tensor. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Single point frequencies. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/stats.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_freq_two_points`

```python
get_freq_two_points(
    data: Tensor,
    weights: Tensor | None = None,
    pseudo_count: float = 0.0
) → Tensor
```

Computes the 2-points statistics of the input MSA. 



**Args:**
 
 - <b>`data`</b> (torch.Tensor):  One-hot encoded data array. 
 - <b>`weights`</b> (torch.Tensor | None, optional):  Array of weights to assign to the sequences of shape. 
 - <b>`pseudo_count`</b> (float, optional):  Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.0. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the input data is not a 3D tensor. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Matrix of two-point frequencies of shape (L, q, L, q). 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/stats.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `generate_unique_triplets`

```python
generate_unique_triplets(
    L: int,
    ntriplets: int,
    device: device = device(type='cpu')
) → Tensor
```

Generates a set of unique triplets of positions. Used to compute the 3-points statistics. 



**Args:**
 
 - <b>`L`</b> (int):  Length of the sequences. 
 - <b>`ntriplets`</b> (int):  Number of triplets to be generated. 
 - <b>`device`</b> (torch.device, optional):  Device to perform computations on. Defaults to "cpu". 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Tensor of shape (ntriplets, 3) containing the indices of the triplets. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/stats.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_freq_three_points`

```python
get_freq_three_points(
    nat: Tensor,
    gen: Tensor,
    ntriplets: int,
    weights: Tensor | None = None,
    device: device = device(type='cpu')
) → Tuple[Tensor, Tensor]
```

Computes the 3-body connected correlation statistics of the input MSAs. 



**Args:**
 
 - <b>`nat`</b> (torch.Tensor):  Input MSA representing natural data in one-hot encoding. 
 - <b>`gen`</b> (torch.Tensor):  Input MSA representing generated data in one-hot encoding. 
 - <b>`ntriplets`</b> (int):  Number of triplets to test. 
 - <b>`weights`</b> (torch.Tensor | None, optional):  Importance weights for the natural sequences. Defaults to None. 
 - <b>`device`</b> (torch.device, optional):  Device to perform computations on. Defaults to "cpu". 



**Returns:**
 
 - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  Natural and generated 3-points connected correlation for ntriplets randomly extracted triplets. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/stats.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_covariance_matrix`

```python
get_covariance_matrix(
    data: Tensor,
    weights: Tensor | None = None,
    pseudo_count: float = 0.0
) → Tensor
```

Computes the weighted covariance matrix of the input multi sequence alignment. 



**Args:**
 
 - <b>`data`</b> (torch.Tensor):  Input MSA in one-hot variables. 
 - <b>`weights`</b> (torch.Tensor | None, optional):  Importance weights of the sequences. 
 - <b>`pseudo_count`</b> (float, optional):  Pseudo count. Defaults to 0.0. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Covariance matrix. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/stats.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `extract_Cij_from_freq`

```python
extract_Cij_from_freq(
    fij: Tensor,
    pij: Tensor,
    fi: Tensor,
    pi: Tensor,
    mask: Tensor | None = None
) → Tuple[Tensor, Tensor]
```

Extracts the lower triangular part of the covariance matrices of the data and chains starting from the frequencies. 



**Args:**
 
 - <b>`fij`</b> (torch.Tensor):  Two-point frequencies of the data. 
 - <b>`pij`</b> (torch.Tensor):  Two-point frequencies of the chains. 
 - <b>`fi`</b> (torch.Tensor):  Single-point frequencies of the data. 
 - <b>`pi`</b> (torch.Tensor):  Single-point frequencies of the chains. 
 - <b>`mask`</b> (torch.Tensor | None, optional):  Mask for comparing just a subset of the couplings. Defaults to None. 



**Returns:**
 
 - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  Extracted two-point frequencies of the data and chains. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/stats.py#L270"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `extract_Cij_from_seqs`

```python
extract_Cij_from_seqs(
    data: Tensor,
    chains: Tensor,
    weights: Tensor | None = None,
    pseudo_count: float = 0.0,
    mask: Tensor | None = None
) → Tuple[Tensor, Tensor]
```

Extracts the lower triangular part of the covariance matrices of the data and chains starting from the sequences. 



**Args:**
 
 - <b>`data`</b> (torch.Tensor):  Data sequences. 
 - <b>`chains`</b> (torch.Tensor):  Chain sequences. 
 - <b>`weights`</b> (torch.Tensor | None, optional):  Weights of the sequences. Defaults to None. 
 - <b>`pseudo_count`</b> (float, optional):  Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.0. 
 - <b>`mask`</b> (torch.Tensor | None, optional):  Mask for comparing just a subset of the couplings. Defaults to None. 



**Returns:**
 
 - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  Two-point frequencies of the data and chains. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/stats.py#L297"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_correlation_two_points`

```python
get_correlation_two_points(
    fij: Tensor,
    pij: Tensor,
    fi: Tensor,
    pi: Tensor,
    mask: Tensor | None = None
) → Tuple[float, float]
```

Computes the Pearson coefficient and the slope between the two-point frequencies of data and chains. 



**Args:**
 
 - <b>`fij`</b> (torch.Tensor):  Two-point frequencies of the data. 
 - <b>`pij`</b> (torch.Tensor):  Two-point frequencies of the chains. 
 - <b>`fi`</b> (torch.Tensor):  Single-point frequencies of the data. 
 - <b>`pi`</b> (torch.Tensor):  Single-point frequencies of the chains. 
 - <b>`mask`</b> (torch.Tensor | None, optional):  Mask to select the couplings to use for the correlation coefficient. Defaults to None.  



**Returns:**
 
 - <b>`Tuple[float, float]`</b>:  Pearson correlation coefficient of the two-sites statistics and slope of the interpolating line. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
