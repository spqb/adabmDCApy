<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/statmech.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.statmech`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/statmech.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `couplings_are_symmetric`

```python
couplings_are_symmetric(couplings: Tensor) → bool
```

Whether ``J[i, a, j, b] == J[j, b, i, a]`` exactly, as for every trained Potts model.

The answer is cached per tensor and recomputed after in-place changes, so kernels can ask on every call. Kernels use it to sum pair terms once.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/statmech.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_energy`

```python
compute_energy(x: Tensor, params: Dict[str, Tensor]) → Tensor
```

Compute the DCA energy for a batch of sequences.



**Args:**

 - <b>`x`</b> (torch.Tensor):  Tensor of shape (batch_size, L, q) - batch of one-hot encoded sequences.
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model.
        - "bias": Tensor of shape (L, q) - local biases.
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.





**Returns:**

 - <b>`torch.Tensor`</b>:  Tensor of shape (batch_size,) - DCA energy for each sequence in the batch.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/statmech.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_cde`

```python
get_cde(x: Tensor, params: dict[str, Tensor]) → Tensor
```

Compute per-site context-dependent entropy of one-hot sequences.

For each site ``i``, hold all other residues fixed, evaluate the model probability of every state at ``i``, and return the Shannon entropy of that conditional distribution in nats. The conditional probabilities follow the same energy convention as :func:`compute_energy`, including asymmetric couplings and nonzero same-site terms.



**Args:**

 - <b>`x`</b>:  One-hot sequence of shape ``(L, q)`` or batch of shape  ``(N, L, q)``. It is moved to the model's device and dtype.
 - <b>`params`</b>:  Model parameters with ``bias`` of shape ``(L, q)`` and  ``coupling_matrix`` of shape ``(L, q, L, q)``.



**Returns:**
 Tensor of shape ``(L,)`` for one sequence or ``(N, L)`` for a batch.



**Raises:**

 - <b>`ValueError`</b>:  If the parameter or sequence dimensions are incompatible,  or ``x`` is not one-hot encoded.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/statmech.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_log_likelihood`

```python
compute_log_likelihood(
    fi: Tensor,
    fij: Tensor,
    params: Dict[str, Tensor],
    logZ: float
) → float
```

Compute the log-likelihood per residue of the model.



**Args:**

 - <b>`fi`</b> (torch.Tensor):  Single-site frequencies of the data.
 - <b>`fij`</b> (torch.Tensor):  Two-site frequencies of the data.
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model.
 - <b>`logZ`</b> (float):  Log-partition function of the model.



**Returns:**

 - <b>`float`</b>:  Log-likelihood per residue of the model.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/statmech.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/statmech.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/statmech.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/statmech.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exchange_log_acceptance`

```python
exchange_log_acceptance(prev_params, curr_params, prev_chains, curr_chains)
```

Deterministic log Metropolis acceptance for Hamiltonian exchange.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/statmech.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `iterate_tap`

```python
iterate_tap(
    mag: Tensor,
    params: Dict[str, Tensor],
    max_iter: int = 500,
    epsilon: float = 0.0001
) → Tensor
```

Iterates the TAP equations until convergence.



**Args:**

 - <b>`mag`</b> (torch.Tensor):  Initial magnetizations.
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model.
 - <b>`max_iter`</b> (int, optional):  Maximum number of iterations. Defaults to 500.
 - <b>`epsilon`</b> (float, optional):  Convergence threshold. Defaults to 1e-4.



**Returns:**

 - <b>`torch.Tensor`</b>:  Fixed point magnetizations of the TAP equations.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
