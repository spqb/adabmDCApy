<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.graph`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `update_mask_element_activation`

```python
update_mask_element_activation(
    Dkl: Tensor,
    mask: Tensor,
    nactivate: int
) → Tensor
```

Updates the mask by activating the nactivate couplings with the largest Dkl. 



**Args:**
 
 - <b>`Dkl`</b> (torch.Tensor):  Kullback-Leibler divergence matrix. 
 - <b>`mask`</b> (torch.Tensor):  Mask. 
 - <b>`nactivate`</b> (int):  Number of couplings to be activated at each graph update. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated mask. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `activate_graph_elements`

```python
activate_graph_elements(
    mask: Tensor,
    fij: Tensor,
    pij: Tensor,
    nactivate: int
) → Tensor
```

Updates the interaction graph by activating a maximum of nactivate couplings. 



**Args:**
 
 - <b>`mask`</b> (torch.Tensor):  Mask. 
 - <b>`fij`</b> (torch.Tensor):  Two-point frequencies of the dataset. 
 - <b>`pij`</b> (torch.Tensor):  Two-point marginals of the model. 
 - <b>`nactivate`</b> (int):  Number of couplings to activate. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated mask. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/graph.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
