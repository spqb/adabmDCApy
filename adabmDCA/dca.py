import numpy as np
import torch
from typing import Tuple, Dict
from adabmDCA.stats import get_covariance_matrix


def get_seqid(
    s1: torch.Tensor,
    s2: torch.Tensor | None = None,
    average: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    When average is True:
    - If s2 is provided, computes the mean and the standard deviation of the mean sequence identity between two sets of one-hot encoded sequences.
    - If s2 is a single sequence (L, q), it computes the mean and the standard deviation of the mean sequence identity between the dataset s1 and s2.
    - If s2 is none, computes the mean and the standard deviation of the mean of the sequence identity between s1 and a permutation of s1.
    
    When average is False it returns the array of sequence identities.

    Args:
        s1 (torch.Tensor): Sequence dataset 1.
        s2 (torch.Tensor | None): Sequence dataset 2. Defaults to None.
        average (bool): Whether to return the average and standard deviation of the sequence identity or the array of sequence identities.

    Returns:
        torch.Tensor | Tuple[torch.Tensor, torch.Tensor]: List of sequence identities or mean sequence identity and standard deviation of the mean.
    """
    if len(s1.shape) == 2:
        s1 = s1.unsqueeze(0)
    if s2 is None:
        s2 = s1[torch.randperm(s1.shape[0])]
    if len(s2.shape) == 2:
        s2 = s2.unsqueeze(0)
        
    s1 = s1.view(s1.shape[0], -1)
    s2 = s2.view(s2.shape[0], -1)
    
    seqids = (s1 * s2).sum(1)
    if average is False:
        return seqids
    else:
        mean_seqid = seqids.mean()
        if len(seqids) == 1:
            std_seqid = torch.tensor(0.0, device=seqids.device)
        else:
            std_seqid = seqids.std() / np.sqrt(len(seqids))

        return mean_seqid, std_seqid


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


def get_contact_map(
    params: Dict[str, torch.Tensor],
    tokens: str,
) -> np.ndarray:
    """
    Computes the contact map from the model coupling matrix.

    Args:
        params (Dict[str, torch.Tensor]): Model parameters.
        tokens (str): Alphabet.

    Returns:
        np.ndarray: Contact map.
    """
    q = params["coupling_matrix"].shape[1]

    # Zero-sum gauge  
    params = set_zerosum_gauge(params)
    
    # Get index of the gap symbol
    gap_idx = tokens.index("-")
    
    Jij = params["coupling_matrix"]
    # Take all the entries of the coupling matrix except where the gap is involved
    mask = torch.arange(q) != gap_idx
    Jij_reduced = Jij[:, mask, :, :][:, :, :, mask]

    # Compute the Frobenius norm
    cm = torch.sqrt(torch.square(Jij_reduced).sum([1, 3]))
    # Set to zero the diagonal
    cm = cm - torch.diag(cm.diag())
    # Compute the average-product corrected Frobenius norm
    Fapc = cm - torch.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    # set to zero the diagonal
    Fapc = Fapc - torch.diag(Fapc.diag())

    return Fapc.cpu().numpy()


def get_mf_contact_map(
    data: torch.Tensor,
    tokens: str,
    weights: torch.Tensor | None = None
) -> np.ndarray:
    """
    Computes the contact map from the model coupling matrix.

    Args:
        data (torch.Tensor): Input one-hot data tensor.
        tokens (str): Alphabet.
        weights (torch.Tensor | None): Weights for the data points. Defaults to None.

    Returns:
        np.ndarray: Contact map.
    """
    device = data.device
    dtype = data.dtype
    L, q = data.shape[1], data.shape[2]
    # Get index of the gap symbol
    gap_idx = tokens.index("-")
    
    # Compute the covariance matrix
    Cij = get_covariance_matrix(data, weights=weights)
    shrink = 4.5 / torch.sqrt(torch.tensor(data.shape[0], dtype=dtype, device=device)) * torch.eye(Cij.shape[0], device=device, dtype=dtype)
    Cij += shrink
        
    # Invert the covariance matrix to get the coupling matrix
    Jij = -torch.linalg.inv(Cij)

    # partial correlation coefficient
    Jij_diag = torch.diag(Jij)
    pcc = Jij / torch.sqrt(Jij_diag[:, None] * Jij_diag[None, :])
    pcc = pcc.reshape(L, q, L, q)
    
    # Take all the entries of the coupling matrix except where the gap is involved
    mask = torch.arange(q) != gap_idx
    pcc = pcc[:, mask, :, :][:, :, :, mask]
    
    # Compute the Frobenius norm
    F = torch.sqrt(torch.square(pcc).sum([1, 3]))
    # Set to zero the diagonal
    F = F - torch.diag(F.diag())
    # Compute the average-product corrected Frobenius norm
    Fapc = F - (F.sum(1, keepdim=True) * F.sum(0, keepdim=True)) / F.sum()
    # Set to zero the diagonal
    Fapc = Fapc - torch.diag(Fapc.diag())

    return Fapc.cpu().numpy()