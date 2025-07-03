<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/resampling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `resampling`





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/resampling.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_seqID`

```python
compute_seqID(a1: Tensor, single_seq: Tensor)
```

Computes the Hamming distance  between a set of one-hot encoded sequences and a single one-hot encoded sequence. 



**Args:**
 
 - <b>`a1`</b> (torch.Tensor):  Sequence dataset, shape (N, L, C), where N is the number of sequences,  L is the length, and C is the number of categories (one-hot size). 
 - <b>`single_seq`</b> (torch.Tensor):  Single one-hot encoded sequence, shape (L, C). 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Hamming distances for each sequence in the dataset. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/resampling.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_mixing_time`

```python
compute_mixing_time(
    sampler: Callable,
    data: Tensor,
    params: Dict[str, Tensor],
    n_max_sweeps: int,
    beta: float
) → Dict[str, list]
```

Computes the mixing time using the t and t/2 method. The sampling will halt when the mixing time is reached or the limit of `n_max_sweeps` sweeps is reached. 



**Args:**
 
 - <b>`sampler`</b> (Callable):  Sampling function. 
 - <b>`data`</b> (torch.Tensor):  Initial data. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters for the sampling. 
 - <b>`n_max_sweeps`</b> (int):  Maximum number of sweeps. 
 - <b>`beta`</b> (float):  Inverse temperature for the sampling. 



**Returns:**
 
 - <b>`Dict[str, list]`</b>:  Results of the mixing time analysis. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
