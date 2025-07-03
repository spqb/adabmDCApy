<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/sampling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sampling`





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/sampling.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gibbs_sampling`

```python
gibbs_sampling(
    chains: Tensor,
    params: Dict[str, Tensor],
    nsweeps: int,
    beta: float = 1.0
) → Tensor
```

Gibbs sampling. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  Initial chains. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`nsweeps`</b> (int):  Number of sweeps. 
 - <b>`beta`</b> (float, optional):  Inverse temperature. Defaults to 1.0. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/sampling.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolis`

```python
metropolis(
    chains: Tensor,
    params: Dict[str, Tensor],
    nsweeps: int,
    beta: float = 1.0
) → Tensor
```

Metropolis sampling. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  One-hot encoded sequences. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`nsweeps`</b> (int):  Number of sweeps to be performed. 
 - <b>`beta`</b> (float, optional):  Inverse temperature. Defaults to 1.0. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/sampling.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sampler`

```python
get_sampler(sampling_method: str) → Callable
```

Returns the sampling function corresponding to the chosen method. 



**Args:**
 
 - <b>`sampling_method`</b> (str):  String indicating the sampling method. Choose between 'metropolis' and 'gibbs'. 



**Raises:**
 
 - <b>`KeyError`</b>:  Unknown sampling method. 



**Returns:**
 
 - <b>`Callable`</b>:  Sampling function. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
