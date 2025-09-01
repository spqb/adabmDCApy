<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/graph.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `graph`





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/graph.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `update_mask_activation`

```python
update_mask_activation(Dkl: Tensor, mask: Tensor, nactivate: int) → Tensor
```

Updates the mask by removing the nactivate couplings with the smallest Dkl. 



**Args:**
 
 - <b>`Dkl`</b> (torch.Tensor):  Kullback-Leibler divergence matrix. 
 - <b>`mask`</b> (torch.Tensor):  Mask. 
 - <b>`nactivate`</b> (int):  Number of couplings to be activated at each graph update. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Updated mask. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/graph.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/graph.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
