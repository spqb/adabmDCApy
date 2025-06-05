import torch


@torch.jit.script
def _one_hot(x: torch.Tensor, num_classes: int = -1, dtype: torch.dtype = torch.float32):
   
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


def one_hot(x: torch.Tensor, num_classes: int = -1, dtype: torch.dtype = torch.float32):
    """A fast one-hot encoding function faster than the PyTorch one working with torch.int32 and returning a float Tensor.
    Works only for 2D tensors.
    
    Args:
        x (torch.Tensor): Input tensor to be one-hot encoded.
        num_classes (int, optional): Number of classes. If -1, the number of classes is inferred from the input tensor. Defaults to -1.
        dtype (torch.dtype, optional): Data type of the output tensor. Defaults to torch.float32.
        
    Returns:
        torch.Tensor: One-hot encoded tensor.
    """
    return _one_hot(x, num_classes, dtype)


@torch.jit.script
def multinomial_one_hot(logits: torch.Tensor) -> torch.Tensor:
    """Sample from a multinomial distribution and return one-hot encoded samples."""
    assert logits.dim() == 2, "Input logits must be a 2D tensor (N, q) where N is the number of samples and q is the number of classes."
    batch_size = logits.size(0)
    one_hot_samples = torch.zeros_like(logits)
    probs = torch.softmax(logits, dim=-1)
    cdf = torch.cumsum(probs, dim=1)
    u = torch.rand(batch_size, 1, device=logits.device, dtype=logits.dtype) * cdf[:, -1].unsqueeze(1)
    indices = torch.searchsorted(cdf, u, right=True)
    one_hot_samples.scatter_(-1, indices, 1.0)
    
    return one_hot_samples