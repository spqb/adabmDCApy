from typing import Dict

import torch

from adabmDCA.functional import one_hot


def compute_energy(
    X: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute the DCA energy of the sequences in X.
    
    Args:
        X (torch.Tensor): Sequences in one-hot encoding format.
        params (Dict[str, torch.Tensor]): Parameters of the model.
    
    Returns:
        torch.Tensor: DCA Energy of the sequences.
    """
    
    if X.dim() != 3:
        raise ValueError("Input tensor X must be 3-dimensional of size (_, L, q)")
    
    @torch.jit.script
    def compute_energy_sequence(x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        L, q = params["bias"].shape
        x_oh = x.ravel()
        bias_oh = params["bias"].ravel()
        couplings_oh = params["coupling_matrix"].view(L * q, L * q)
        energy = - x_oh @ bias_oh - 0.5 * x_oh @ (couplings_oh @ x_oh)
        
        return energy
    
    return torch.vmap(compute_energy_sequence, in_dims=(0, None))(X, params)
    
    
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
    params["coupling_matrix"] = torch.zeros((L, q, L, q), device=fi.device)
    
    return params


def init_chains(
    num_chains: int,
    L: int,
    q: int,
    device: torch.device,
    fi: torch.Tensor = None,
) -> torch.Tensor:
    """Initialize the chains of the DCA model. If 'fi' is provided, the chains are sampled from the
    profile model, otherwise they are sampled uniformly at random.

    Args:
        num_chains (int): Number of parallel chains.
        L (int): Length of the MSA.
        q (int): Number of values that each residue can assume.
        device (torch.device): Device where to store the chains.
        fi (torch.Tensor, optional): Single-point frequencies. Defaults to None.

    Returns:
        torch.Tensor: Initialized parallel chains in one-hot encoding format.
    """
    if fi is None:
        chains = torch.randint(low=0, high=q, size=(num_chains, L), device=device)
    else:
        chains = torch.multinomial(fi, num_samples=num_chains, replacement=True).T
    
    return one_hot(chains, num_classes=q).float()


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
    indices = torch.multinomial(weights.view(-1), nextract, replacement=True)
    
    return data[indices]


def get_device(device: str) -> torch.device:
    """Returns the device where to store the tensors.
    
    Args:
        device (str): Device to be used.
        
    Returns:
        torch.device: Device.
    """
    if "cuda" in device and torch.cuda.is_available():
        device = torch.device(device)
        print(f"Running on {torch.cuda.get_device_name(device)}")
        return device
    else:
        print("Running on CPU")
        return torch.device("cpu")