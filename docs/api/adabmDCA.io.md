<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.io`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_chains`

```python
load_chains(
    fname: str,
    tokens: str,
    load_weights: bool = False,
    device: device = device(type='cpu'),
    dtype: dtype = torch.float32
) → Tuple[Tensor, ]
```

Loads the sequences from a fasta file and returns the one-hot encoded version. If the sequences are weighted, the log-weights are also returned. If the sequences are not weighted, the log-weights are set to 0. 



**Args:**
 
 - <b>`fname`</b> (str):  Path to the file containing the sequences. 
 - <b>`tokens`</b> (str):  "protein", "dna", "rna" or another string with the alphabet to be used. 
 - <b>`load_weights`</b> (bool, optional):  If True, the log-weights are loaded and returned. Defaults to False. 
 - <b>`device`</b> (torch.device, optional):  Device where to store the sequences. Defaults to "cpu". 
 - <b>`dtype`</b> (torch.dtype, optional):  Data type of the sequences. Defaults to torch.float32 

Return: 
 - <b>`Tuple[torch.Tensor, ...]`</b>:  One-hot encoded sequences and log-weights if load_weights is True. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_chains`

```python
save_chains(
    fname: str,
    chains: Union[list, ndarray, Tensor],
    tokens: str,
    log_weights: Optional[Tensor, ndarray] = None
) → None
```

Saves the chains in a fasta file. 



**Args:**
 
 - <b>`fname`</b> (str):  Path to the file where to save the chains. 
 - <b>`chains`</b> (Union[list, np.ndarray, torch.Tensor]):  Iterable with sequences in string, categorical or one-hot encoded format. 
 - <b>`tokens`</b> (str):  "protein", "dna", "rna" or another string with the alphabet to be used. 
 - <b>`log_weights`</b> (Union[torch.Tensor, np.ndarray, None], optional):  Log-weights of the chains. Defaults to None. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_params`

```python
load_params(
    fname: str,
    tokens: str,
    device: device,
    dtype: dtype = torch.float32
) → Dict[str, Tensor]
```

Import the parameters of the model from a text file. 



**Args:**
 
 - <b>`fname`</b> (str):  Path of the file that stores the parameters. 
 - <b>`tokens`</b> (str):  "protein", "dna", "rna" or another string with a compatible alphabet to be used. 
 - <b>`device`</b> (torch.device):  Device where to store the parameters. 
 - <b>`dtype`</b> (torch.dtype):  Data type of the parameters. Defaults to torch.float32. 



**Returns:**
 
 - <b>`Dict[str, torch.Tensor]`</b>:  Parameters of the model. 
        - "bias": Tensor of shape (L, q) - local biases. 
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_params_old`

```python
load_params_old(
    fname: str,
    tokens: str,
    device: device,
    dtype: dtype = torch.float32
) → Dict[str, Tensor]
```

Import the parameters of the model from a file. 



**Args:**
 
 - <b>`fname`</b> (str):  Path of the file that stores the parameters. 
 - <b>`tokens`</b> (str):  "protein", "dna", "rna" or another string with a compatible alphabet to be used. 
 - <b>`device`</b> (torch.device):  Device where to store the parameters. 
 - <b>`dtype`</b> (torch.dtype):  Data type of the parameters. Defaults to torch.float32. 



**Returns:**
 
 - <b>`Dict[str, torch.Tensor]`</b>:  Parameters of the model. 
        - "bias": Tensor of shape (L, q) - local biases. 
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L260"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_params`

```python
save_params(
    fname: str,
    params: Dict[str, Tensor],
    tokens: str,
    mask: Optional[Tensor] = None
) → None
```

Saves the parameters of the model in a file. 



**Args:**
 
 - <b>`fname`</b> (str):  Path to the file where to save the parameters. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
        - "bias": Tensor of shape (L, q) - local biases. 
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix. 
 - <b>`tokens`</b> (str):  "protein", "dna", "rna" or another string with a compatible alphabet to be used. 
 - <b>`mask`</b> (Optional[torch.Tensor]):  Tensor of shape (L, q, L, q) - Mask of the coupling matrix that determines which are the non-zero entries.  If None, the lower-triangular part of the coupling matrix is masked. Defaults to None. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L317"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_params_oldformat`

```python
load_params_oldformat(
    fname: str,
    device: device,
    dtype: dtype = torch.float32
) → Dict[str, Tensor]
```

Import the parameters of the model from a file. Assumes the old DCA format. 



**Args:**
 
 - <b>`fname`</b> (str):  Path of the file that stores the parameters. 
 - <b>`device`</b> (torch.device):  Device where to store the parameters. 
 - <b>`dtype`</b> (torch.dtype):  Data type of the parameters. Defaults to torch.float32. 



**Returns:**
 
 - <b>`Dict[str, torch.Tensor]`</b>:  Parameters of the model. 
        - "bias": Tensor of shape (L, q) - local biases. 
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L364"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_params_oldformat`

```python
save_params_oldformat(
    fname: str,
    params: Dict[str, Tensor],
    mask: Optional[Tensor] = None
) → None
```

Saves the parameters of the model in a file. Assumes the old DCA format. 



**Args:**
 
 - <b>`fname`</b> (str):  Path to the file where to save the parameters. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
        - "bias": Tensor of shape (L, q) - local biases. 
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix. 
 - <b>`mask`</b> (Optional[torch.Tensor]):  Tensor of shape (L, q, L, q) - Mask of the coupling matrix that determines which are the non-zero entries.  If None, the lower-triangular part of the coupling matrix is masked. Defaults to None. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
