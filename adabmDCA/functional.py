import torch


def one_hot(x: torch.Tensor, num_classes: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Encode int32 or int64 indices directly into a tensor of the requested dtype.
    
    Args:
        x (torch.Tensor): Input tensor to be one-hot encoded. Shape (L,) or (batch_size, L).
        num_classes (int, optional): Number of classes. If -1, the number of classes is inferred from the input tensor. Defaults to -1.
        dtype (torch.dtype, optional): Data type of the output tensor. Defaults to torch.float32.
        
    Returns:
        torch.Tensor: One-hot encoded tensor. Shape (L, num_classes) for 1D input or (batch_size, L, num_classes) for 2D input.
    """
    if x.dim() not in (1, 2):
        raise ValueError("Input tensor x must be 1D or 2D")
    if x.dtype not in (torch.int32, torch.int64):
        raise TypeError("Class indices must have dtype int32 or int64.")
    if num_classes < -1:
        raise ValueError("num_classes must be nonnegative or -1 to infer it.")
    if num_classes == -1:
        if x.numel() == 0:
            raise ValueError("Cannot infer num_classes from an empty tensor.")
        num_classes = int(x.max()) + 1
    if ((x < 0) | (x >= num_classes)).any():
        raise ValueError("Class indices must be between 0 and num_classes - 1.")
    result = torch.zeros((*x.shape, num_classes), device=x.device, dtype=dtype)
    return result.scatter_(-1, x.long().unsqueeze(-1), 1)
