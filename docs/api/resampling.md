<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/resampling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `resampling`





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/resampling.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
