<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/adabmDCA/training.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `training`





---

<a href="https://github.com/spqb/adabmDCApy/adabmDCA/training.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `update_params`

```python
update_params(
    fi: Tensor,
    fij: Tensor,
    pi: Tensor,
    pij: Tensor,
    params: Dict[str, Tensor],
    mask: Tensor,
    lr: float
) → Dict[str, Tensor]
```

Updates the parameters of the model. 



**Args:**
 
 - <b>`fi`</b> (torch.Tensor):  Single-point frequencies of the data. 
 - <b>`fij`</b> (torch.Tensor):  Two-points frequencies of the data. 
 - <b>`pi`</b> (torch.Tensor):  Single-point marginals of the model. 
 - <b>`pij`</b> (torch.Tensor):  Two-points marginals of the model. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`mask`</b> (torch.Tensor):  Mask of the interaction graph. 
 - <b>`lr`</b> (float):  Learning rate. 



**Returns:**
 
 - <b>`Dict[str, torch.Tensor]`</b>:  Updated parameters. 


---

<a href="https://github.com/spqb/adabmDCApy/adabmDCA/training.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_graph`

```python
train_graph(
    sampler: Callable,
    chains: Tensor,
    mask: Tensor,
    fi: Tensor,
    fij: Tensor,
    params: Dict[str, Tensor],
    nsweeps: int,
    lr: float,
    max_epochs: int,
    target_pearson: float,
    fi_test: Tensor | None = None,
    fij_test: Tensor | None = None,
    checkpoint: Checkpoint | None = None,
    check_slope: bool = False,
    log_weights: Tensor | None = None,
    progress_bar: bool = True
) → Tuple[Tensor, Dict[str, Tensor], Tensor, Dict[str, List[float]]]
```

Trains the model on a given graph until the target Pearson correlation is reached or the maximum number of epochs is exceeded. 



**Args:**
 
 - <b>`sampler`</b> (Callable):  Sampling function. 
 - <b>`chains`</b> (torch.Tensor):  Markov chains simulated with the model. 
 - <b>`mask`</b> (torch.Tensor):  Mask encoding the sparse graph. 
 - <b>`fi`</b> (torch.Tensor):  Single-point frequencies of the data. 
 - <b>`fij`</b> (torch.Tensor):  Two-point frequencies of the data. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`nsweeps`</b> (int):  Number of Gibbs steps for each gradient estimation. 
 - <b>`lr`</b> (float):  Learning rate. 
 - <b>`max_epochs`</b> (int):  Maximum number of gradient updates to be done. 
 - <b>`target_pearson`</b> (float):  Target Pearson coefficient. 
 - <b>`fi_test`</b> (torch.Tensor | None, optional):  Single-point frequencies of the test data. Defaults to None. 
 - <b>`fij_test`</b> (torch.Tensor | None, optional):  Two-point frequencies of the test data. Defaults to None. 
 - <b>`checkpoint`</b> (Checkpoint | None, optional):  Checkpoint class to be used for saving the model. Defaults to None. 
 - <b>`check_slope`</b> (bool, optional):  Whether to take into account the slope for the convergence criterion or not. Defaults to False. 
 - <b>`log_weights`</b> (torch.Tensor, optional):  Log-weights used for the online computation of the log-likelihood. Defaults to None. 
 - <b>`progress_bar`</b> (bool, optional):  Whether to display a progress bar or not. Defaults to True. 



**Returns:**
 
 - <b>`Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]`</b>:  Updated chains and parameters, log-weights for the log-likelihood computation. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
