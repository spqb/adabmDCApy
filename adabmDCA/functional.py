# Description: This file contains the custom functions used in the package.

import torch


@torch.jit.script
def one_hot(x: torch.Tensor, num_classes: int = -1, dtype: torch.dtype = torch.float32):
    """A fast one-hot encoding function faster than the PyTorch one working with torch.int32 and returning a float Tensor.
    Works only for 2D tensors."""
    
    if x.dim() != 2:
        raise ValueError("Input tensor x must be 2D")
    
    if num_classes < 0:
        num_classes = x.max() + 1
    res = torch.zeros(x.shape[0], x.shape[1], num_classes, device=x.device, dtype=dtype)
    tmp = torch.meshgrid(
        torch.arange(x.shape[0], device=x.device),
        torch.arange(x.shape[1], device=x.device),
        indexing="ij",
    )
    index = (tmp[0], tmp[1], x)
    values = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=dtype)
    res.index_put_(index, values)
    
    return res