<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `checkpoint`





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_checkpoint`

```python
get_checkpoint(chpt: str) → Checkpoint
```






---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Checkpoint`
Helper class to save the model's parameters and chains at regular intervals during training and to log the progress of the training. 

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check`

```python
check(updates: int, *args, **kwargs) → bool
```

Checks if a checkpoint has been reached. 



**Args:**
 
 - <b>`updates`</b> (int):  Number of gradient updates performed. 



**Returns:**
 
 - <b>`bool`</b>:  Whether a checkpoint has been reached. 

---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log`

```python
log(record: Dict[str, Any]) → None
```

Adds a key-value pair to the log dictionary 



**Args:**
 
 - <b>`record`</b> (Dict[str, Any]):  Key-value pairs to be added to the log dictionary. 

---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LinearCheckpoint`




<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    file_paths: dict,
    tokens: str,
    args: dict,
    params: Optional[Dict[str, Tensor]] = None,
    chains: Tensor | None = None,
    checkpt_interval: int = 50,
    use_wandb: bool = False,
    **kwargs
)
```








---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check`

```python
check(updates: int, *args, **kwargs) → bool
```

Checks if a checkpoint has been reached. 



**Args:**
 
 - <b>`updates`</b> (int):  Number of gradient updates performed. 



**Returns:**
 
 - <b>`bool`</b>:  Whether a checkpoint has been reached. 

---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log`

```python
log(record: Dict[str, Any]) → None
```

Adds a key-value pair to the log dictionary 



**Args:**
 
 - <b>`record`</b> (Dict[str, Any]):  Key-value pairs to be added to the log dictionary. 

---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AcceptanceCheckpoint`




<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    file_paths: Dict,
    tokens: str,
    args: Dict,
    params: Dict[str, Tensor],
    chains: Tensor | None = None,
    target_acc_rate: float = 0.5,
    use_wandb: bool = False,
    **kwargs
)
```








---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L244"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check`

```python
check(
    updates: int,
    curr_params: Dict[str, Tensor],
    curr_chains: Tensor,
    *args,
    **kwargs
) → bool
```

Checks if a checkpoint has been reached by computing the acceptance rate of swapping the  configurations of the present model and the last saved model. 



**Args:**
 
 - <b>`updates`</b> (int):  Number of gradient updates performed. 
 - <b>`curr_params`</b> (Dict[str, torch.Tensor]):  Current parameters of the model. 
 - <b>`curr_chains`</b> (torch.Tensor):  Current chains of the model. 



**Returns:**
 
 - <b>`bool`</b>:  Whether a checkpoint has been reached. 

---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log`

```python
log(record: Dict[str, Any]) → None
```

Adds a key-value pair to the log dictionary 



**Args:**
 
 - <b>`record`</b> (Dict[str, Any]):  Key-value pairs to be added to the log dictionary. 

---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/checkpoint.py#L272"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(
    params: Dict[str, Tensor],
    mask: Tensor,
    chains: Tensor,
    log_weights: Tensor,
    *args,
    **kwargs
) → None
```

Saves the chains and the parameters of the model and appends the current parameters to the file containing the parameters history. 



**Args:**
 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`mask`</b> (torch.Tensor):  Mask of the model's coupling matrix representing the interaction graph. 
 - <b>`chains`</b> (torch.Tensor):  Chains. 
 - <b>`log_weights`</b> (torch.Tensor):  Log of the chain weights. Used for AIS. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
