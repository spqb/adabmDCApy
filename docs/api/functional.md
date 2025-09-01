<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/functional.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `functional`





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/functional.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `one_hot`

```python
one_hot(x: Tensor, num_classes: int = -1, dtype: dtype = torch.float32)
```

A fast one-hot encoding function faster than the PyTorch one working with torch.int32 and returning a float Tensor. Works only for 2D tensors. 



**Args:**
 
 - <b>`x`</b> (torch.Tensor):  Input tensor to be one-hot encoded. 
 - <b>`num_classes`</b> (int, optional):  Number of classes. If -1, the number of classes is inferred from the input tensor. Defaults to -1. 
 - <b>`dtype`</b> (torch.dtype, optional):  Data type of the output tensor. Defaults to torch.float32. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  One-hot encoded tensor. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
