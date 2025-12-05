import torch


@torch.jit.script
def _one_hot(x: torch.Tensor, num_classes: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
   
    if x.dim() not in (1, 2):
        raise ValueError("Input tensor x must be 1D or 2D")
    
    if num_classes < 0:
        num_classes = int(x.max() + 1)
    
    # Handle 1D case (single sequence)
    if x.dim() == 1:
        res = torch.zeros(x.shape[0], num_classes, device=x.device, dtype=dtype)
        index = (torch.arange(x.shape[0], device=x.device), x)
        values = torch.ones(x.shape[0], device=x.device, dtype=dtype)
        res.index_put_(index, values)
        return res
    
    # Handle 2D case (batch of sequences)
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


def one_hot(x: torch.Tensor, num_classes: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A fast one-hot encoding function faster than the PyTorch one working with torch.int32 and returning a float Tensor.
    Works for both 1D (single sequence) and 2D (batch of sequences) tensors.
    
    Args:
        x (torch.Tensor): Input tensor to be one-hot encoded. Shape (L,) or (batch_size, L).
        num_classes (int, optional): Number of classes. If -1, the number of classes is inferred from the input tensor. Defaults to -1.
        dtype (torch.dtype, optional): Data type of the output tensor. Defaults to torch.float32.
        
    Returns:
        torch.Tensor: One-hot encoded tensor. Shape (L, num_classes) for 1D input or (batch_size, L, num_classes) for 2D input.
    """
    return _one_hot(x, num_classes, dtype)