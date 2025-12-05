<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/resampling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.resampling`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/resampling.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_mixing_time`

```python
compute_mixing_time(
    sampler: Callable[, Tensor],
    data: Tensor,
    params: Dict[str, Tensor],
    n_max_sweeps: int,
    beta: float
) → Dict[str, List[Union[float, int]]]
```

Computes the mixing time using the t and t/2 method. The sampling will halt when the mixing time is reached or the limit of `n_max_sweeps` sweeps is reached. 



**Args:**
 
 - <b>`sampler`</b> (Callable):  Sampling function. 
 - <b>`data`</b> (torch.Tensor):  Initial data. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters for the sampling. 
        - "bias": Tensor of shape (L, q) - local biases. 
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix. 
 - <b>`n_max_sweeps`</b> (int):  Maximum number of sweeps. 
 - <b>`beta`</b> (float):  Inverse temperature for the sampling. 



**Returns:**
 
 - <b>`Dict[str, List[Union[float, int]]]`</b>:  Results of the mixing time analysis. 
        - "seqid_t": List of average sequence identities at time t. 
        - "std_seqid_t": List of standard deviations of sequence identities at time t. 
        - "seqid_t_t_half": List of average sequence identities between t and t/2. 
        - "std_seqid_t_t_half": List of standard deviations of sequence identities between t and t/2. 
        - "t_half": List of t/2 values (integers). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
