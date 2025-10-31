<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `checkpoint`






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Checkpoint`
Helper class to save the model's parameters and chains at regular intervals during training and to log the progress of the training. 

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    file_paths: dict,
    tokens: str,
    args: dict,
    params: Optional[Dict[str, Tensor]] = None,
    chains: Tensor | None = None,
    use_wandb: bool = False
)
```

Initializes the Checkpoint class. 



**Args:**
 
 - <b>`file_paths`</b> (dict):  Dictionary containing the paths of the files to be saved. 
 - <b>`tokens`</b> (str):  Alphabet to be used for encoding the sequences. 
 - <b>`args`</b> (dict):  Dictionary containing the arguments of the training. 
 - <b>`params`</b> (Dict[str, torch.Tensor] | None, optional):  Parameters of the model. Defaults to None. 
 - <b>`chains`</b> (torch.Tensor | None, optional):  Chains. Defaults to None. 
 - <b>`use_wandb`</b> (bool, optional):  Whether to use Weights & Biases for logging. Defaults to False. 




---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check`

```python
check(updates: int) → bool
```

Checks if a checkpoint has been reached. 



**Args:**
 
 - <b>`updates`</b> (int):  Number of gradient updates performed. 



**Returns:**
 
 - <b>`bool`</b>:  Whether a checkpoint has been reached. 

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log`

```python
log(record: Dict[str, Any]) → None
```

Adds a key-value pair to the log dictionary 



**Args:**
 
 - <b>`record`</b> (Dict[str, Any]):  Key-value pairs to be added to the log dictionary. 

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/checkpoint.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(
    params: Dict[str, Tensor],
    mask: Tensor,
    chains: Tensor,
    log_weights: Tensor
) → None
```

Saves the chains and the parameters of the model. 



**Args:**
 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`mask`</b> (torch.Tensor):  Mask of the model's coupling matrix representing the interaction graph 
 - <b>`chains`</b> (torch.Tensor):  Chains. 
 - <b>`log_weights`</b> (torch.Tensor):  Log of the chain weights. Used for AIS. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
