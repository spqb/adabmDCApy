<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.sampling`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sampling_profile`

```python
sampling_profile(params: dict[str, Tensor], nsamples: int, beta: float) → Tensor
```

Samples from the profile model defined by the local biases only. 



**Args:**
 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
        - "bias": Tensor of shape (L, q) - local biases. 
 - <b>`nsamples`</b> (int):  Number of samples to generate. 
 - <b>`beta`</b> (float):  Inverse temperature. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Sampled one-hot encoded sequences of shape (nsamples, L, q). 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gibbs_step_uniform_sites`

```python
gibbs_step_uniform_sites(
    chains: Tensor,
    params: dict[str, Tensor],
    beta: float = 1.0
) → Tensor
```

Performs a single mutation using the Gibbs sampler. In this version, the mutation is attempted at the same sites for all chains. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  One-hot encoded sequences of shape (batch_size, L, q). 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
        - "bias": Tensor of shape (L, q) - local biases. 
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix. 
 - <b>`beta`</b> (float, optional):  Inverse temperature. Defaults to 1.0. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gibbs_step_independent_sites`

```python
gibbs_step_independent_sites(
    chains: Tensor,
    params: dict[str, Tensor],
    beta: float = 1.0
) → Tensor
```

Performs a single mutation using the Gibbs sampler. This version selects different random sites for each chain. It is less efficient than the 'gibbs_step_uniform_sites' function, but it is more suitable for mutating starting from the same wild-type sequence since mutations are independent across chains. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  One-hot encoded sequences of shape (batch_size, L, q). 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
        - "bias": Tensor of shape (L, q) - local biases. 
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix. 
 - <b>`beta`</b> (float, optional):  Inverse temperature. Defaults to 1.0. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gibbs_sampling`

```python
gibbs_sampling(
    chains: Tensor,
    params: dict[str, Tensor],
    nsweeps: int,
    beta: float = 1.0
) → Tensor
```

Gibbs sampling. Attempts L * nsweeps mutations to each sequence in 'chains'. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  Initial one-hot encoded samples of size (batch_size, L, q). 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
        - "bias": Tensor of shape (L, q) - local biases. 
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix. 
 - <b>`nsweeps`</b> (int):  Number of sweeps, where one sweep corresponds to attempting L mutations. 
 - <b>`beta`</b> (float, optional):  Inverse temperature. Defaults to 1.0. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolis_step_uniform_sites`

```python
metropolis_step_uniform_sites(
    chains: Tensor,
    params: dict[str, Tensor],
    beta: float = 1.0
) → Tensor
```

Performs a single mutation using the Metropolis sampler. In this version, the mutation is attempted at the same sites for all chains. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  One-hot encoded sequences of shape (batch_size, L, q). 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
        - "bias": Tensor of shape (L, q) - local biases. 
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix. 
 - <b>`beta`</b> (float, optional):  Inverse temperature. Defaults to 1.0. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolis_step_independent_sites`

```python
metropolis_step_independent_sites(
    chains: Tensor,
    params: dict[str, Tensor],
    beta: float = 1.0
) → Tensor
```

Performs a single mutation using the Metropolis sampler. This version selects different random sites for each chain. It is less efficient than the 'metropolis_step_uniform_sites' function, but it is more suitable for mutating starting from the same wild-type sequence since mutations are independent across chains. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  One-hot encoded sequences of shape (batch_size, L, q). 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
        - "bias": Tensor of shape (L, q) - local biases. 
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix. 
 - <b>`beta`</b> (float, optional):  Inverse temperature. Defaults to 1.0. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolis_sampling`

```python
metropolis_sampling(
    chains: Tensor,
    params: dict[str, Tensor],
    nsweeps: int,
    beta: float = 1.0
) → Tensor
```

Metropolis sampling. Attempts L * nsweeps mutations to each sequence in 'chains'. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  One-hot encoded sequences of shape (batch_size, L, q). 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
        - "bias": Tensor of shape (L, q) - local biases. 
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix. 
 - <b>`nsweeps`</b> (int):  Number of sweeps to be performed, where one sweep corresponds to attempting L mutations. 
 - <b>`beta`</b> (float, optional):  Inverse temperature. Defaults to 1.0. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated chains. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L255"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `prepare_sampler`

```python
prepare_sampler(sampling_method: str, device: device) → Callable
```

Select a fused CUDA sampler, or the scripted sampler without Triton. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L289"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `prepare_fixed_model_sampler`

```python
prepare_fixed_model_sampler(
    sampling_method: str,
    device: device,
    dtype: str,
    params: dict[str, Tensor]
) → tuple[Callable, dict[str, Tensor]]
```

Prepare a sampler and parameters for a model that will not be updated. 

BF16 mode keeps biases, chains, statistics, and energy calculations in float32. Only the fixed coupling matrix is rounded to BF16, once, before sampling. This is the inference counterpart of :func:`prepare_training_sampler`, where the coupling copy must instead be refreshed after every parameter update. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling.py#L314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `prepare_training_sampler`

```python
prepare_training_sampler(
    sampling_method: str,
    device: device,
    dtype: str = 'float32'
) → Callable
```

Prepare sampling for training with optional BF16 coupling storage. 

In bfloat16 mode, parameters, chains and statistics outside the sampler stay float32. A fresh BF16 coupling copy is made after each parameter update; biases and all field/acceptance arithmetic remain float32. The rounded coupling matrix approximates the master model, so trajectories can change. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
