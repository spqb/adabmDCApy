<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/adabmDCA/models/edDCA.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `models.edDCA`




**Global Variables**
---------------
- **MAX_EPOCHS**

---

<a href="https://github.com/spqb/adabmDCApy/adabmDCA/models/edDCA.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fit`

```python
fit(
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
    fi_test: Tensor | None = None,
    fij_test: Tensor | None = None,
    *args,
    **kwargs
)
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
 - <b>`fi_test`</b> (torch.Tensor | None, optional):  Single-point frequencies of the test data. Defaults to None. 
 - <b>`fij_test`</b> (torch.Tensor | None, optional):  Two-point frequencies of the test data. Defaults to None. 






---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
