<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/adabmDCA/models/eaDCA.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `models.eaDCA`





---

<a href="https://github.com/spqb/adabmDCApy/adabmDCA/models/eaDCA.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fit`

```python
fit(
    sampler: Callable,
    fi_target: Tensor,
    fij_target: Tensor,
    params: dict,
    mask: Tensor,
    chains: Tensor,
    log_weights: Tensor,
    target_pearson: float,
    nsweeps: int,
    nepochs: int,
    pseudo_count: float,
    lr: float,
    factivate: float,
    gsteps: int,
    fi_test: Tensor | None = None,
    fij_test: Tensor | None = None,
    checkpoint: Checkpoint | None = None,
    *args,
    **kwargs
) → None
```

Fits an eaDCA model on the training data and saves the results in a file. 



**Args:**
 
 - <b>`sampler`</b> (Callable):  Sampling function to be used. 
 - <b>`fi_target`</b> (torch.Tensor):  Single-point frequencies of the data. 
 - <b>`fij_target`</b> (torch.Tensor):  Two-point frequencies of the data. 
 - <b>`params`</b> (dict):  Initialization of the model's parameters. 
 - <b>`mask`</b> (torch.Tensor):  Initialization of the coupling matrix's mask. 
 - <b>`chains`</b> (torch.Tensor):  Initialization of the Markov chains. 
 - <b>`log_weights`</b> (torch.Tensor):  Log-weights of the chains. Used to estimate the log-likelihood. 
 - <b>`target_pearson`</b> (float):  Pearson correlation coefficient on the two-points statistics to be reached. 
 - <b>`nsweeps`</b> (int):  Number of Monte Carlo steps to update the state of the model. 
 - <b>`nepochs`</b> (int):  Maximum number of epochs to be performed. Defaults to 50000. 
 - <b>`pseudo_count`</b> (float):  Pseudo count for the single and two points statistics. Acts as a regularization. 
 - <b>`lr`</b> (float):  Learning rate. 
 - <b>`factivate`</b> (float):  Fraction of inactive couplings to activate at each step. 
 - <b>`gsteps`</b> (int):  Number of gradient updates to be performed on a given graph. 
 - <b>`fi_test`</b> (torch.Tensor | None):  Single-point frequencies of the test data. Defaults to None. 
 - <b>`fij_test`</b> (torch.Tensor | None):  Two-point frequencies of the test data. Defaults to None. 
 - <b>`checkpoint`</b> (Checkpoint | None):  Checkpoint class to be used to save the model. Defaults to None. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
