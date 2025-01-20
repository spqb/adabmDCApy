from typing import Dict

import torch

from adabmDCA.functional import one_hot
    
    
def set_zerosum_gauge(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Sets the zero-sum gauge on the coupling matrix.
    
    Args:
        params (Dict[str, torch.Tensor]): Parameters of the model.
        
    Returns:
        Dict[str, torch.Tensor]: Parameters with fixed gauge.
    """
    coupling_matrix = params["coupling_matrix"]
    coupling_matrix -= coupling_matrix.mean(dim=1, keepdim=True) + \
                       coupling_matrix.mean(dim=3, keepdim=True) - \
                       coupling_matrix.mean(dim=(1, 3), keepdim=True)
    
    params["coupling_matrix"] = coupling_matrix
    
    return params


def init_parameters(fi: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Initialize the parameters of the DCA model.

    Args:
        fi (torch.Tensor): Single-point frequencies of the data.

    Returns:
        Dict[str, torch.Tensor]: Parameters of the model.
    """
    L, q = fi.shape
    params = {}
    params["bias"] = torch.log(fi)
    params["coupling_matrix"] = torch.zeros((L, q, L, q), device=fi.device, dtype=fi.dtype)
    
    return params


def init_chains(
    num_chains: int,
    L: int,
    q: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    fi: torch.Tensor | None = None,
) -> torch.Tensor:
    """Initialize the chains of the DCA model. If 'fi' is provided, the chains are sampled from the
    profile model, otherwise they are sampled uniformly at random.

    Args:
        num_chains (int): Number of parallel chains.
        L (int): Length of the MSA.
        q (int): Number of values that each residue can assume.
        device (torch.device): Device where to store the chains.
        dtype (torch.dtype, optional): Data type of the chains. Defaults to torch.float32.
        fi (torch.Tensor | None, optional): Single-point frequencies. Defaults to None.

    Returns:
        torch.Tensor: Initialized parallel chains in one-hot encoding format.
    """
    if fi is None:
        chains = torch.randint(low=0, high=q, size=(num_chains, L), device=device)
    else:
        chains = torch.multinomial(fi, num_samples=num_chains, replacement=True).to(device=device).T
    
    return one_hot(chains, num_classes=q).to(dtype)


def get_mask_save(L: int, q: int, device: torch.device) -> torch.Tensor:
    """Returns the mask to save the upper-triangular part of the coupling matrix.
    
    Args:
        L (int): Length of the MSA.
        q (int): Number of values that each residue can assume.
        device (torch.device): Device where to store the mask.
        
    Returns:
        torch.Tensor: Mask.
    """
    mask_save = torch.ones((L, q, L, q), dtype=torch.bool, device=device)
    idx1_rm, idx2_rm = torch.tril_indices(L, L, offset=0)
    mask_save[idx1_rm, :, idx2_rm, :] = 0
    
    return mask_save


@torch.jit.script
def systematic_resampling(
    chains: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Performs the systematic resampling of the chains according to their relative weight.

    Args:
        chains (torch.Tensor): Chains.
        weights (torch.Tensor): Weights of the chains.

    Returns:
        torch.Tensor: Resampled chains.
    """
    num_chains = len(chains)
    device = chains.device
    # Normalize the weights
    weights = weights.view(-1) / weights.sum()
    weights_span = torch.cumsum(weights.double(), dim=0).float()
    rand_unif = torch.rand(size=(1,), device=device)
    arrow_span = (torch.arange(num_chains, device=device) + rand_unif) / num_chains
    mask = (weights_span.reshape(num_chains, 1) >= arrow_span).sum(1)
    counts = torch.diff(mask, prepend=torch.tensor([0], device=device))
    chains = torch.repeat_interleave(chains, counts, dim=0)

    return chains


def resample_sequences(
    data: torch.Tensor,
    weights: torch.Tensor,
    nextract: int,
) -> torch.Tensor:
    """Extracts nextract sequences from data with replacement according to the weights.
    
    Args:
        data (torch.Tensor): Data array.
        weights (torch.Tensor): Weights of the sequences.
        nextract (int): Number of sequences to be extracted.

    Returns:
        torch.Tensor: Extracted sequences.
    """
    weights = weights.view(-1)
    indices = torch.multinomial(weights, nextract, replacement=True)
    
    return data[indices]


def get_device(device: str, message: bool = True) -> torch.device:
    """Returns the device where to store the tensors.
    
    Args:
        device (str): Device to be used.
        message (bool, optional): Print the device. Defaults to True.
        
    Returns:
        torch.device: Device.
    """
    if "cuda" in device and torch.cuda.is_available():
        device = torch.device(device)
        if message:
            print(f"Running on {torch.cuda.get_device_name(device)}")
        return device
    else:
        if message:
            print("Running on CPU")
        return torch.device("cpu")
    
    
def get_dtype(dtype: str) -> torch.dtype:
    """Returns the data type of the tensors.
    
    Args:
        dtype (str): Data type.
        
    Returns:
        torch.dtype: Data type.
    """
    if dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise ValueError(f"Data type {dtype} not supported.")