<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/statmech.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `statmech`





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/statmech.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_energy`

```python
compute_energy(X: Tensor, params: Dict[str, Tensor]) → Tensor
```

Compute the DCA energy of the sequences in X. 



**Args:**
 
 - <b>`X`</b> (torch.Tensor):  Sequences in one-hot encoding format. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  DCA Energy of the sequences. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/statmech.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_log_likelihood`

```python
compute_log_likelihood(
    fi: Tensor,
    fij: Tensor,
    params: Dict[str, Tensor],
    logZ: float
) → float
```

Compute the log-likelihood of the model. 



**Args:**
 
 - <b>`fi`</b> (torch.Tensor):  Single-site frequencies of the data. 
 - <b>`fij`</b> (torch.Tensor):  Two-site frequencies of the data. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`logZ`</b> (float):  Log-partition function of the model. 



**Returns:**
 
 - <b>`float`</b>:  Log-likelihood of the model. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/statmech.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `enumerate_states`

```python
enumerate_states(L: int, q: int, device: device = device(type='cpu')) → Tensor
```

Enumerate all possible states of a system of L sites and q states. 



**Args:**
 
 - <b>`L`</b> (int):  Number of sites. 
 - <b>`q`</b> (int):  Number of states. 
 - <b>`device`</b> (torch.device, optional):  Device to store the states. Defaults to "cpu". 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  All possible states. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/statmech.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_logZ_exact`

```python
compute_logZ_exact(all_states: Tensor, params: Dict[str, Tensor]) → float
```

Compute the log-partition function of the model. 



**Args:**
 
 - <b>`all_states`</b> (torch.Tensor):  All possible states of the system. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 



**Returns:**
 
 - <b>`float`</b>:  Log-partition function of the model. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/statmech.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_entropy`

```python
compute_entropy(chains: Tensor, params: Dict[str, Tensor], logZ: float) → float
```

Compute the entropy of the DCA model. 



**Args:**
 
 - <b>`chains`</b> (torch.Tensor):  Chains that are supposed to be an equilibrium realization of the model. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`logZ`</b> (float):  Log-partition function of the model. 



**Returns:**
 
 - <b>`float`</b>:  Entropy of the model. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/statmech.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `iterate_tap`

```python
iterate_tap(
    mag: Tensor,
    params: Dict[str, Tensor],
    max_iter: int = 500,
    epsilon: float = 0.0001
)
```

Iterates the TAP equations until convergence. 



**Args:**
 
 - <b>`mag`</b> (torch.Tensor):  Initial magnetizations. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`max_iter`</b> (int, optional):  Maximum number of iterations. Defaults to 2000. 
 - <b>`epsilon`</b> (float, optional):  Convergence threshold. Defaults to 1e-6. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Fixed point magnetizations of the TAP equations. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
