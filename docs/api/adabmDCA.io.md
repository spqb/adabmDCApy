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
    device: device = device(type='cpu'),
    dtype: dtype = torch.float32
) → Tuple[Tensor, ]
```

Load chain sequences from FASTA and return their one-hot encoding.



**Args:**

 - <b>`fname`</b> (str):  Path to the file containing the sequences.
 - <b>`tokens`</b> (str):  "protein", "dna", "rna" or another string with the alphabet to be used.
 - <b>`device`</b> (torch.device, optional):  Device where to store the sequences. Defaults to "cpu".
 - <b>`dtype`</b> (torch.dtype, optional):  Data type of the sequences. Defaults to torch.float32

Return:
 - <b>`Tuple[torch.Tensor, ...]`</b>:  One-hot encoded sequences.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_chains`

```python
save_chains(
    fname: str,
    chains: Union[list, ndarray, Tensor],
    tokens: str
) → None
```

Saves the chains in a fasta file.



**Args:**

 - <b>`fname`</b> (str):  Path to the file where to save the chains.
 - <b>`chains`</b> (Union[list, np.ndarray, torch.Tensor]):  Iterable with sequences in string, categorical or one-hot encoded format.
 - <b>`tokens`</b> (str):  "protein", "dna", "rna" or another string with the alphabet to be used.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_gzip`

```python
is_gzip(fname: Union[str, Path]) → bool
```

Whether a file is gzip-compressed, judged by its content rather than its name.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `open_params`

```python
open_params(fname: Union[str, Path], mode: str = 'r') → IO[str]
```

Open a parameter file as text; gzip is detected on reading and chosen by a ``.gz`` name on writing.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_params`

```python
load_params(
    fname: str,
    tokens: str,
    device: device,
    dtype: dtype = torch.float32
) → Dict[str, Tensor]
```

Import parameters from the established ``J``/``h`` text format.

Gzip-compressed files (such as ``params.dat.gz``) are read transparently, whatever their name. The file is parsed in two streaming passes so memory use is bounded by the final tensors and a small coupling chunk. Files containing one or both coupling triangles are supported.



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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L238"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_params`

```python
save_params(
    fname: str,
    params: Dict[str, Tensor],
    tokens: str,
    mask: Optional[Tensor] = None
) → None
```

Save parameters in the established ``J``/``h`` text format.

Couplings are streamed in bounded chunks using the canonical ``i < j`` triangle. A supplied symmetric mask is collapsed onto that triangle. A file name ending in ``.gz`` is written gzip-compressed.



**Args:**

 - <b>`fname`</b> (str):  Path to the file where to save the parameters.
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model.
        - "bias": Tensor of shape (L, q) - local biases.
        - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
 - <b>`tokens`</b> (str):  "protein", "dna", "rna" or another string with a compatible alphabet to be used.
 - <b>`mask`</b> (Optional[torch.Tensor]):  Tensor of shape (L, q, L, q) - Mask of the coupling matrix that determines which are the non-zero entries.  If None, the lower-triangular part of the coupling matrix is masked. Defaults to None.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L401"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L448"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
