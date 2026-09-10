<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.training`




**Global Variables**
---------------
- **DEFAULT_INNER_GRADIENT_STEPS**
- **EDGE_EMPIRICAL_PSEUDOCOUNT**
- **EDGE_LOGZ_CHAIN_FRACTION**
- **SLOPE_TOLERANCE**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_gradient`

```python
compute_gradient(
    fi: Tensor,
    fij: Tensor,
    pi: Tensor,
    pij: Tensor
) → dict[str, Tensor]
```

Computes the gradient of the log-likelihood of the model using PyTorch.



**Args:**

 - <b>`fi`</b> (torch.Tensor):  Single-point frequencies of the data.
 - <b>`fij`</b> (torch.Tensor):  Target two-points frequencies.
 - <b>`pi`</b> (torch.Tensor):  Single-point marginals of the model.
 - <b>`pij`</b> (torch.Tensor):  Two-points marginals of the model.



**Returns:**

 - <b>`Dict[str, torch.Tensor]`</b>:  Gradient.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `update_params`

```python
update_params(
    fi: Tensor,
    fij: Tensor,
    pi: Tensor,
    pij: Tensor,
    params: dict[str, Tensor],
    mask: Tensor,
    lr: float,
    l2_reg: float = 0.0
) → dict[str, Tensor]
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
 - <b>`l2_reg`</b> (float, optional):  L2 regularization coefficient. Defaults to 0.0.



**Returns:**

 - <b>`Dict[str, torch.Tensor]`</b>:  Updated parameters.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `update_params_edge_activation`

```python
update_params_edge_activation(
    fij: Tensor,
    pij: Tensor,
    params: dict[str, Tensor],
    mask: Tensor
) → tuple[tuple[int, int], Tensor, dict[str, Tensor]]
```

Updates the mask and the coupling parameters using the edge-activation algorithm.



**Args:**

 - <b>`fij`</b> (torch.Tensor):  Two-point frequences of the dataset.
 - <b>`pij`</b> (torch.Tensor):  Two-point marginals of the model.
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model.
 - <b>`mask`</b> (torch.Tensor):  Mask.



**Returns:**

 - <b>`torch.Tensor`</b>:  Indices of the activated edge.
 - <b>`torch.Tensor`</b>:  Updated mask.
 - <b>`Dict[str, torch.Tensor]`</b>:  Updated parameters.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_graph`

```python
train_graph(
    sampler: Callable[, Tensor],
    chains: Tensor,
    mask: Tensor,
    fi_target: Tensor,
    fij_target: Tensor,
    params: dict[str, Tensor],
    nsweeps: int,
    lr: float,
    max_epochs: int,
    target_pearson: float,
    fi_val: Tensor | None = None,
    fij_val: Tensor | None = None,
    check_slope: bool = False,
    log_weights: Tensor | None = None,
    l2_reg: float = 0.0,
    controller: TrainingController | None = None,
    slope_tolerance: float = 0.1
) → tuple[Tensor, dict[str, Tensor], Tensor, dict[str, list[Any]]]
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
 - <b>`fi_val`</b> (Optional[torch.Tensor], optional):  Single-point frequencies of the validation data. Defaults to None.
 - <b>`fij_val`</b> (Optional[torch.Tensor], optional):  Two-point frequencies of the validation data. Defaults to None.
 - <b>`check_slope`</b> (bool, optional):  Whether to take into account the slope for the convergence criterion or not. Defaults to False.
 - <b>`log_weights`</b> (Optional[torch.Tensor], optional):  Log-weights used for the online computation of the log-likelihood. Defaults to None.
 - <b>`l2_reg`</b> (float, optional):  L2 regularization coefficient. Defaults to 0.0.



**Returns:**

 - <b>`Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]`</b>:  Updated chains and parameters, log-weights for the log-likelihood computation.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training.py#L305"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_eaDCA`

```python
train_eaDCA(
    sampler: Callable[, Tensor],
    fi_target: Tensor,
    fij_target: Tensor,
    params: dict[str, Tensor],
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
    fi_val: Tensor | None = None,
    fij_val: Tensor | None = None,
    l2_reg: float = 0.0,
    controller: TrainingController | None = None,
    max_gradient_steps: int | None = None
) → tuple[Tensor, dict[str, Tensor], Tensor, dict[str, list[Any]]]
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
 - <b>`fi_val`</b> (Optional[torch.Tensor], optional):  Single-point frequencies of the validation data. Defaults to None.
 - <b>`fij_val`</b> (Optional[torch.Tensor], optional):  Two-point frequencies of the validation data. Defaults to None.
 - <b>`l2_reg`</b> (float, optional):  L2 regularization coefficient. Defaults to 0.0.



**Returns:**

 - <b>`Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]`</b>:  Updated chains and parameters, log-weights for the log-likelihood computation, and training history.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training.py#L497"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_edDCA`

```python
train_edDCA(
    sampler: Callable[, Tensor],
    chains: Tensor,
    log_weights: Tensor,
    fi_target: Tensor,
    fij_target: Tensor,
    params: dict[str, Tensor],
    mask: Tensor,
    lr: float,
    nsweeps: int,
    target_pearson: float,
    target_density: float,
    drate: float,
    fi_val: Tensor | None = None,
    fij_val: Tensor | None = None,
    l2_reg: float = 0.0,
    max_epochs: int = 10000,
    controller: TrainingController | None = None,
    max_gradient_steps: int | None = None,
    inner_gradient_steps: int = 10000
) → tuple[Tensor, dict[str, Tensor], Tensor, dict[str, list[Any]]]
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
 - <b>`fi_val`</b> (Optional[torch.Tensor], optional):  Single-point frequencies of the validation data. Defaults to None.
 - <b>`fij_val`</b> (Optional[torch.Tensor], optional):  Two-point frequencies of the validation data. Defaults to None.
 - <b>`l2_reg`</b> (float, optional):  L2 regularization coefficient. Defaults to 0.0.



**Returns:**

 - <b>`Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]`</b>:  Updated chains and parameters, log-weights for the log-likelihood computation, and training history.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/training.py#L738"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_edgeDCA`

```python
train_edgeDCA(
    sampler: Callable[, Tensor],
    fi_target: Tensor,
    fij_target: Tensor,
    fi_pseudocounted: Tensor,
    fij_pseudocounted: Tensor,
    params: dict[str, Tensor],
    mask: Tensor,
    chains: Tensor,
    target_pearson: float,
    nsweeps: int,
    max_epochs: int,
    pseudo_count: float,
    fi_val: Tensor | None = None,
    fij_val: Tensor | None = None,
    controller: TrainingController | None = None,
    empirical_pseudocount: float = 1e-06,
    logz_chain_fraction: float = 0.2
) → tuple[Tensor, dict[str, Tensor], Tensor, dict[str, list[Any]]]
```

Fits an edge activation DCA model (edgeDCA) on the training data and saves the results in a file.



**Args:**

 - <b>`sampler`</b> (Callable):  Sampling function to be used.
 - <b>`fi_target`</b> (torch.Tensor):  Single-point frequencies of the data.
 - <b>`fij_target`</b> (torch.Tensor):  Two-point frequencies of the data.
 - <b>`fi_pseudocounted`</b> (torch.Tensor):  Pseudocounted single-point frequencies.
 - <b>`fij_pseudocounted`</b> (torch.Tensor):  Pseudocounted two-point frequencies.
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Initialization of the model's parameters.
 - <b>`mask`</b> (torch.Tensor):  Initialization of the coupling matrix's mask.
 - <b>`chains`</b> (torch.Tensor):  Initialization of the Markov chains.
 - <b>`target_pearson`</b> (float):  Pearson correlation coefficient on the two-points statistics to be reached.
 - <b>`nsweeps`</b> (int):  Number of Monte Carlo steps to update the state of the model.
 - <b>`max_epochs`</b> (int):  Maximum number of epochs to be performed.
 - <b>`pseudo_count`</b> (float):  Pseudo count for the single and two points statistics. Acts as a regularization.
 - <b>`fi_val`</b> (Optional[torch.Tensor], optional):  Single-point frequencies of the validation data. Defaults to None.
 - <b>`fij_val`</b> (Optional[torch.Tensor], optional):  Two-point frequencies of the validation data. Defaults to None.



**Returns:**

 - <b>`Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]`</b>:  Updated chains and parameters, log-weights for the log-likelihood computation, and training history.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
