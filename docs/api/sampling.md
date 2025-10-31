<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sampling`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gibbs_step_uniform_sites`

```python
gibbs_step_uniform_sites(
    chains: Tensor,
    params: Dict[str, Tensor],
    beta: float = 1.0
) → Tensor
```

Performs a single mutation using the Gibbs sampler. In this version, the mutation is attempted at the same sites for all chains. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  One-hot encoded sequences of shape (batch_size, L, q). 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`beta`</b> (float):  Inverse temperature. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gibbs_step_independent_sites`

```python
gibbs_step_independent_sites(
    chains: Tensor,
    params: Dict[str, Tensor],
    beta: float = 1.0
) → Tensor
```

Performs a single mutation using the Gibbs sampler. This version selects different random sites for each chain. It is less efficient than the 'gibbs_step_uniform_sites' function, but it is more suitable for mutating staring from the same wild-type sequence since mutations are independent across chains. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  One-hot encoded sequences of shape (batch_size, L, q). 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`beta`</b> (float):  Inverse temperature. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
 
 - <b>`chains`</b> (torch.Tensor):  Initial one-hot encoded chains of size (batch_size, L, q). 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`nsweeps`</b> (int):  Number of sweeps, where one sweep corresponds to attempting L mutations. 
 - <b>`beta`</b> (float, optional):  Inverse temperature. Defaults to 1.0. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolis_step_uniform_sites`

```python
metropolis_step_uniform_sites(
    chains: Tensor,
    params: Dict[str, Tensor],
    beta: float = 1.0
) → Tensor
```

Performs a single mutation using the Metropolis sampler. In this version, the mutation is attempted at the same sites for all chains. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  One-hot encoded sequences of shape (batch_size, L, q). 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`beta`</b> (float, optional):  Inverse temperature. Defaults to 1.0. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolis_step_independent_sites`

```python
metropolis_step_independent_sites(
    chains: Tensor,
    params: Dict[str, Tensor],
    beta: float = 1.0
) → Tensor
```

Performs a single mutation using the Metropolis sampler. This version selects different random sites for each chain. It is less efficient than the 'metropolis_step_uniform_sites' function, but it is more suitable for mutating staring from the same wild-type sequence since mutations are independent across chains. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  One-hot encoded sequences of shape (batch_size, L, q). 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`beta`</b> (float, optional):  Inverse temperature. Defaults to 1.0. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolis_sampling`

```python
metropolis_sampling(
    chains: Tensor,
    params: Dict[str, Tensor],
    nsweeps: int,
    beta: float = 1.0
) → Tensor
```

Metropolis sampling. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  One-hot encoded sequences of shape (batch_size, L, q). 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`nsweeps`</b> (int):  Number of sweeps to be performed, where one sweep corresponds to attempting L mutations. 
 - <b>`beta`</b> (float, optional):  Inverse temperature. Defaults to 1.0. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
