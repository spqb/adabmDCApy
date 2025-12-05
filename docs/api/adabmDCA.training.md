<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.training`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_graph`

```python
train_graph(
    sampler: Callable,
    chains: Tensor,
    mask: Tensor,
    fi_target: Tensor,
    fij_target: Tensor,
    params: Dict[str, Tensor],
    nsweeps: int,
    lr: float,
    max_epochs: int,
    target_pearson: float,
    fi_test: Optional[Tensor] = None,
    fij_test: Optional[Tensor] = None,
    checkpoint: Optional[Checkpoint] = None,
    check_slope: bool = False,
    log_weights: Optional[Tensor] = None,
    progress_bar: bool = True,
    *args,
    **kwargs
) → Tuple[Tensor, Dict[str, Tensor], Tensor, Dict[str, List[float]]]
```

Trains the model on a given graph until the target Pearson correlation is reached or the maximum number of epochs is exceeded. 



**Args:**
 
 - <b>`sampler`</b> (Callable):  Sampling function. 
 - <b>`chains`</b> (torch.Tensor):  Markov chains simulated with the model. 
 - <b>`mask`</b> (torch.Tensor):  Mask encoding the sparse graph. 
 - <b>`fi_target`</b> (torch.Tensor):  Single-point frequencies of the data. 
 - <b>`fij_target`</b> (torch.Tensor):  Two-point frequencies of the data. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`nsweeps`</b> (int):  Number of Gibbs steps for each gradient estimation. 
 - <b>`lr`</b> (float):  Learning rate. 
 - <b>`max_epochs`</b> (int):  Maximum number of gradient updates to be done. 
 - <b>`target_pearson`</b> (float):  Target Pearson coefficient. 
 - <b>`fi_test`</b> (Optional[torch.Tensor], optional):  Single-point frequencies of the test data. Defaults to None. 
 - <b>`fij_test`</b> (Optional[torch.Tensor], optional):  Two-point frequencies of the test data. Defaults to None. 
 - <b>`checkpoint`</b> (Optional[Checkpoint], optional):  Checkpoint class to be used for saving the model. Defaults to None. 
 - <b>`check_slope`</b> (bool, optional):  Whether to take into account the slope for the convergence criterion or not. Defaults to False. 
 - <b>`log_weights`</b> (Optional[torch.Tensor], optional):  Log-weights used for the online computation of the log-likelihood. Defaults to None. 
 - <b>`progress_bar`</b> (bool, optional):  Whether to display a progress bar or not. Defaults to True. 



**Returns:**
 
 - <b>`Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]`</b>:  Updated chains and parameters, log-weights for the log-likelihood computation. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training.py#L251"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_eaDCA`

```python
train_eaDCA(
    sampler: Callable,
    fi_target: Tensor,
    fij_target: Tensor,
    params: Dict[str, Tensor],
    mask: Tensor,
    chains: Tensor,
    log_weights: Tensor,
    target_pearson: float,
    nsweeps: int,
    max_epochs: int,
    pseudo_count: float,
    lr: float,
    factivate: float,
    gsteps: int,
    fi_test: Optional[Tensor] = None,
    fij_test: Optional[Tensor] = None,
    checkpoint: Optional[Checkpoint] = None,
    *args,
    **kwargs
) → Tuple[Tensor, Dict[str, Tensor], Tensor, Dict[str, List[float]]]
```

Fits an eaDCA model on the training data and saves the results in a file. 



**Args:**
 
 - <b>`sampler`</b> (Callable):  Sampling function to be used. 
 - <b>`fi_target`</b> (torch.Tensor):  Single-point frequencies of the data. 
 - <b>`fij_target`</b> (torch.Tensor):  Two-point frequencies of the data. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Initialization of the model's parameters. 
 - <b>`mask`</b> (torch.Tensor):  Initialization of the coupling matrix's mask. 
 - <b>`chains`</b> (torch.Tensor):  Initialization of the Markov chains. 
 - <b>`log_weights`</b> (torch.Tensor):  Log-weights of the chains. Used to estimate the log-likelihood. 
 - <b>`target_pearson`</b> (float):  Pearson correlation coefficient on the two-points statistics to be reached. 
 - <b>`nsweeps`</b> (int):  Number of Monte Carlo steps to update the state of the model. 
 - <b>`max_epochs`</b> (int):  Maximum number of epochs to be performed. 
 - <b>`pseudo_count`</b> (float):  Pseudo count for the single and two points statistics. Acts as a regularization. 
 - <b>`lr`</b> (float):  Learning rate. 
 - <b>`factivate`</b> (float):  Fraction of inactive couplings to activate at each step. 
 - <b>`gsteps`</b> (int):  Number of gradient updates to be performed on a given graph. 
 - <b>`fi_test`</b> (Optional[torch.Tensor], optional):  Single-point frequencies of the test data. Defaults to None. 
 - <b>`fij_test`</b> (Optional[torch.Tensor], optional):  Two-point frequencies of the test data. Defaults to None. 
 - <b>`checkpoint`</b> (Optional[Checkpoint], optional):  Checkpoint class to be used to save the model. Defaults to None. 



**Returns:**
 
 - <b>`Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]`</b>:  Updated chains and parameters, log-weights for the log-likelihood computation, and training history. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training.py#L488"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_edDCA`

```python
train_edDCA(
    sampler: Callable,
    chains: Tensor,
    log_weights: Tensor,
    fi_target: Tensor,
    fij_target: Tensor,
    params: Dict[str, Tensor],
    mask: Tensor,
    lr: float,
    nsweeps: int,
    target_pearson: float,
    target_density: float,
    drate: float,
    checkpoint: Checkpoint,
    fi_test: Optional[Tensor] = None,
    fij_test: Optional[Tensor] = None,
    *args,
    **kwargs
) → Tuple[Tensor, Dict[str, Tensor], Tensor, Dict[str, List[float]]]
```

Fits an edDCA model on the training data and saves the results in a file. 



**Args:**
 
 - <b>`sampler`</b> (Callable):  Sampling function to be used. 
 - <b>`chains`</b> (torch.Tensor):  Initialization of the Markov chains. 
 - <b>`log_weights`</b> (torch.Tensor):  Log-weights of the chains. Used to estimate the log-likelihood. 
 - <b>`fi_target`</b> (torch.Tensor):  Single-point frequencies of the data. 
 - <b>`fij_target`</b> (torch.Tensor):  Two-point frequencies of the data. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Initialization of the model's parameters. 
 - <b>`mask`</b> (torch.Tensor):  Initialization of the coupling matrix's mask. 
 - <b>`lr`</b> (float):  Learning rate. 
 - <b>`nsweeps`</b> (int):  Number of Monte Carlo steps to update the state of the model. 
 - <b>`target_pearson`</b> (float):  Pearson correlation coefficient on the two-points statistics to be reached. 
 - <b>`target_density`</b> (float):  Target density of the coupling matrix. 
 - <b>`drate`</b> (float):  Percentage of active couplings to be pruned at each decimation step. 
 - <b>`checkpoint`</b> (Checkpoint):  Checkpoint class to be used to save the model. 
 - <b>`fi_test`</b> (Optional[torch.Tensor], optional):  Single-point frequencies of the test data. Defaults to None. 
 - <b>`fij_test`</b> (Optional[torch.Tensor], optional):  Two-point frequencies of the test data. Defaults to None. 



**Returns:**
 
 - <b>`Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]`</b>:  Updated chains and parameters, log-weights for the log-likelihood computation, and training history. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
