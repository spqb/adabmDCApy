<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.graph`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_density`

```python
compute_density(mask: Tensor) → float
```

Computes the density of active couplings in the coupling matrix.



**Args:**

 - <b>`mask`</b> (torch.Tensor):  Mask.



**Returns:**

 - <b>`float`</b>:  Density.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_Dkl_element_activation`

```python
compute_Dkl_element_activation(fij: Tensor, pij: Tensor) → Tensor
```

Computes the Kullback-Leibler divergence matrix of all the possible couplings.



**Args:**

 - <b>`fij`</b> (torch.Tensor):  Two-point frequences of the dataset.
 - <b>`pij`</b> (torch.Tensor):  Two-point marginals of the model.



**Returns:**

 - <b>`torch.Tensor`</b>:  Kullback-Leibler divergence matrix.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `select_inactive_elements`

```python
select_inactive_elements(
    Dkl: Tensor,
    mask: Tensor,
    fraction: float
) → Tuple[Tensor, int, int]
```

Activates the inactive off-diagonal coupling entries with the largest Dkl.

Entries are counted once per symmetric pair (i < j). Only inactive entries are eligible: active ones are already refitted at every gradient update. At least one entry is activated while inactive entries remain, and never more than remain.



**Args:**

 - <b>`Dkl`</b> (torch.Tensor):  Element-wise Kullback-Leibler divergence matrix.
 - <b>`mask`</b> (torch.Tensor):  Symmetric boolean mask of the active couplings.
 - <b>`fraction`</b> (float):  Fraction of the inactive unique entries to activate.



**Returns:**

 - <b>`Tuple[torch.Tensor, int, int]`</b>:  Updated symmetric mask, number of entries requested (``int(fraction * inactive)``) and number actually activated.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rank_inactive_elements`

```python
rank_inactive_elements(Dkl: Tensor, mask: Tensor) → Tensor
```

Flat indices of the inactive unique (i < j) coupling entries, by decreasing Dkl.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `activate_elements`

```python
activate_elements(mask: Tensor, indices: Tensor) → Tensor
```

Activates the unique entries at the flat ``indices`` together with their transposes.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `activate_graph_elements`

```python
activate_graph_elements(
    mask: Tensor,
    fij: Tensor,
    pij: Tensor,
    fraction: float
) → Tensor
```

Updates the interaction graph by activating a fraction of the inactive couplings.



**Args:**

 - <b>`mask`</b> (torch.Tensor):  Mask.
 - <b>`fij`</b> (torch.Tensor):  Two-point frequencies of the dataset.
 - <b>`pij`</b> (torch.Tensor):  Two-point marginals of the model.
 - <b>`fraction`</b> (float):  Fraction of the inactive unique coupling entries to activate.



**Returns:**

 - <b>`torch.Tensor`</b>:  Updated mask.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_Dkl_edge_activation`

```python
compute_Dkl_edge_activation(fij: Tensor, pij: Tensor) → Tensor
```

Computes the Kullback-Leibler divergence matrix of all the possible edges.



**Args:**

 - <b>`fij`</b> (torch.Tensor):  Two-point frequences of the dataset.
 - <b>`pij`</b> (torch.Tensor):  Two-point marginals of the model.



**Returns:**

 - <b>`torch.Tensor`</b>:  Kullback-Leibler divergence matrix.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_sym_Dkl`

```python
compute_sym_Dkl(params: Dict[str, Tensor], pij: Tensor) → Tensor
```

Computes the symmetric Kullback-Leibler divergence matrix between the initial distribution and the same  distribution once removing one coupling J_ij(a, b).



**Args:**

 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model.
 - <b>`pij`</b> (torch.Tensor):  Two-point marginal probability distribution.



**Returns:**

 - <b>`torch.Tensor`</b>:  Kullback-Leibler divergence matrix.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_Dkl_decimation`

```python
compute_Dkl_decimation(params: Dict[str, Tensor], pij: Tensor) → Tensor
```

Computes the Kullback-Leibler divergence matrix between the initial distribution and the same  distribution once removing one coupling J_ij(a, b).



**Args:**

 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model.
 - <b>`pij`</b> (torch.Tensor):  Two-point marginal probability distribution.



**Returns:**

 - <b>`torch.Tensor`</b>:  Kullback-Leibler divergence matrix.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `update_mask_decimation`

```python
update_mask_decimation(mask: Tensor, Dkl: Tensor, drate: float) → Tensor
```

Updates the mask by removing the n_remove couplings with the smallest Dkl.



**Args:**

 - <b>`mask`</b> (torch.Tensor):  Mask.
 - <b>`Dkl`</b> (torch.Tensor):  Kullback-Leibler divergence matrix.
 - <b>`drate`</b> (float):  Percentage of active couplings to be pruned at each decimation step.



**Returns:**

 - <b>`torch.Tensor`</b>:  Updated mask.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L211"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `decimate_graph`

```python
decimate_graph(
    pij: Tensor,
    params: Dict[str, Tensor],
    mask: Tensor,
    drate: float
) → Tuple[Dict[str, Tensor], Tensor]
```

Performs one decimation step and updates the parameters and mask.



**Args:**

 - <b>`pij`</b> (torch.Tensor):  Two-point marginal probability distribution.
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model.
 - <b>`mask`</b> (torch.Tensor):  Mask.
 - <b>`drate`</b> (float):  Percentage of active couplings to be pruned at each decimation step.



**Returns:**

 - <b>`Tuple[Dict[str, torch.Tensor], torch.Tensor]`</b>:  Updated parameters and mask.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
