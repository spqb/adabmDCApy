import itertools
from typing import Tuple

import torch

def resample_sequences(
    data: torch.Tensor,
    weights: torch.Tensor,
    nextract: int,
) -> torch.Tensor:
    """Extracts nextract sequences from data with replacement according to the weights.
    Args:
    key (torch.Generator): Random key.
    data (torch.Tensor): Data array.
    weights (torch.Tensor): Weights of the sequences.
    nextract (int): Number of sequences to be extracted.

    Returns:
        torch.Tensor: Extracted sequences.
    """
    indices = torch.multinomial(weights, nextract, replacement=True)
    
    return data[indices]


@torch.jit.script
def get_freq_single_point(
    data: torch.Tensor,
    weights: torch.Tensor | None,
    pseudo_count: float = 0.,
) -> torch.Tensor:
    """Computes the single point frequencies of the input MSA.
    Args:
        data (torch.Tensor): One-hot encoded data array.
        weights (torch.Tensor): Weights of the sequences.
        pseudo_count (float): Pseudo count to be added to the frequencies.

    Returns:
        torch.Tensor: Single point frequencies.
    """
    if data.dim() != 3:
        raise ValueError(f"Expected data to be a 3D tensor, but got {data.dim()}D tensor instead")
    
    M, _, q = data.shape
    if weights is not None:
        norm_weights = weights.reshape(M, 1, 1) / weights.sum()
    else:
        norm_weights = torch.ones((M, 1, 1), device=data.device) / M

    frequencies = torch.clamp((data.float() * norm_weights).sum(dim=0), min=1e-6, max=1. - 1e-6)

    return (1. - pseudo_count) * frequencies + (pseudo_count / q)


@torch.jit.script
def get_freq_two_points(
    data: torch.Tensor,
    weights: torch.Tensor | None,
    pseudo_count: float=0.,
) -> torch.Tensor:
    """
    Computes the 2-points statistics of the input MSA.

    Args:
        data (torch.Tensor): One-hot encoded data array.
        weights (torch.Tensor, optional): Array of weights to assign to the sequences of shape.
        pseudo_count (float, optional): Pseudo count for the single and two points statistics. Acts as a regularization.

    Returns:
        torch.Tensor: Matrix of two-point frequencies of shape (L, q, L, q).
    """
    if data.dim() != 3:
        raise ValueError(f"Expected data to be a 3D tensor, but got {data.dim()}D tensor instead")
    
    M, L, q = data.shape
    data_oh = data.reshape(M, q * L)
    
    if weights is not None:
        norm_weights = weights.reshape(M, 1) / weights.sum()
    else:
        norm_weights = torch.ones((M, 1), device=data.device) / M
    
    fij = (data_oh * norm_weights).T @ data_oh
    # Apply the pseudo count
    fij = (1. - pseudo_count) * fij + (pseudo_count / q**2)
    # Diagonal terms must represent the single point frequencies
    fi = get_freq_single_point(data, weights, pseudo_count).ravel()
    # Apply the pseudo count on the single point frequencies
    fij_diag = (1. - pseudo_count) * fi + (pseudo_count / q)
    # Set the diagonal terms of fij to the single point frequencies
    fij = torch.diagonal_scatter(fij, fij_diag, dim1=0, dim2=1)
    
    return fij.reshape(L, q, L, q)


def generate_unique_triplets(
    L: int,
    ntriplets: int,
    device: str = "cpu",
) -> torch.Tensor:
    
    # Generate all possible unique triplets
    all_triplets = torch.tensor(list(itertools.combinations(range(L), 3)), device=device)
    # Shuffle the triplets to ensure randomness
    shuffled_triplets = all_triplets[torch.randperm(all_triplets.size(0))]
    # Select the first ntriplets from the shuffled triplets
    selected_triplets = shuffled_triplets[:ntriplets]
    
    return selected_triplets


@torch.jit.script
def get_C_ijk(
    triplet: torch.Tensor,
    data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the third-order correlation tensor for a given triplet of positions.

    Args:
        triplet (torch.Tensor): Tensor of shape (3,) containing the indices of the triplet.
        data (torch.Tensor): Tensor of shape (M, L, q) containing the one-hot encoded data.
        weights (torch.Tensor): Tensor of shape (M,) containing the weights of the sequences.

    Returns:
        torch.Tensor: Tensor of shape (q, q, q) containing the third-order correlations.
    """
    # Center the dataset
    data_c = data - data.mean(dim=0, keepdim=True)
    norm_weights = weights.view(-1, 1) / weights.sum()
    x = data_c[:, triplet[0], :] * norm_weights
    y = data_c[:, triplet[1], :]
    z = data_c[:, triplet[2], :]

    C = torch.einsum("mi, mj, mk -> ijk", x, y, z)

    return C


def get_freq_three_points(
    data: torch.Tensor,
    weights: torch.Tensor,
    ntriplets: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Computes the 3-body statistics of the input MSA.

    Args:
        data (torch.Tensor): Input MSA in one-hot encoding.
        weights (torch.Tensor): Importance weights for the sequences.
        ntriplets (int): Number of triplets to test.
        device (str): Device to perform computations on.

    Returns:
        torch.Tensor: 3-points connected correlation for ntriplets randomly extracted triplets.
    """
    
    if data.dim() != 3:
        raise ValueError(f"Expected data to be a 3D tensor, but got {data.dim()}D tensor instead")
    
    L = data.shape[1]
    triplets = generate_unique_triplets(L=L, ntriplets=ntriplets, device=device)
    Cijk = []
    for triplet in triplets:
        Cijk.append(get_C_ijk(triplet, data, weights).flatten())
        
    return torch.stack(Cijk)


def get_covariance_matrix(
    data: torch.Tensor,
    weights: torch.Tensor,
    pseudo_count: float = 0.
) -> torch.Tensor:
    """Computes the weighted covariance matrix of the input multi sequence alignment.

    Args:
        data (torch.Tensor): Input MSA in one-hot variables.
        weights (torch.Tensor): Importance weights of the sequences.
        pseudo_count (float, optional): Pseudo count. Defaults to 0..

    Returns:
        torch.Tensor: Covariance matrix.
    """
    
    _, L, q = data.shape
    fi = get_freq_single_point(data, weights, pseudo_count)
    fij = get_freq_two_points(data, weights, pseudo_count)
    cov_matrix = fij - torch.einsum('ij,kl->ijkl', fi, fi)
    
    return cov_matrix.reshape(L * q, L * q)


def get_slope(x, y):
    n = len(x)
    num = n * (x @ y) - y.sum() * x.sum()
    den = n * (x @ x) - torch.square(x.sum())
    return torch.abs(num / den)


def get_correlation_two_points(
    fij: torch.Tensor,
    pij: torch.Tensor,
    fi: torch.Tensor,
    pi: torch.Tensor,
    mask: torch.Tensor = None,
) -> Tuple[float, float]:
    """Computes the Pearson coefficient and the slope between the two-point frequencies of data and chains.

    Args:
        fij (torch.Tensor): Two-point frequencies of the data.
        pij (torch.Tensor): Two-point frequencies of the chains.
        fi (torch.Tensor): Single-point frequencies of the data.
        pi (torch.Tensor): Single-point frequencies of the chains.
        mask (torch.Tensor, optional): Mask to select the couplings to use for the correlation coefficient. Defaults to None. 

    Returns:
        Tuple[float, float]: Pearson correlation coefficient of the two-sites statistics and slope of the interpolating line.
    """
    
    L = fi.shape[0]
    
    # Compute the covariance matrices
    cov_data = fij - torch.einsum('ij,kl->ijkl', fi, fi)
    cov_chains = pij - torch.einsum('ij,kl->ijkl', pi, pi)
    
    # Only use a subset of couplings if a mask is provided
    if mask is not None:
        cov_data = torch.where(mask, cov_data, torch.tensor(0.0, device=cov_data.device))
        cov_chains = torch.where(mask, cov_chains, torch.tensor(0.0, device=cov_chains.device))
    
    # Extract only the entries of half the matrix and out of the diagonal blocks
    idx_row, idx_col = torch.tril_indices(L, L, offset=-1)
    fij_extract = cov_data[idx_row, :, idx_col, :].reshape(-1)
    pij_extract = cov_chains[idx_row, :, idx_col, :].reshape(-1)
    
    pearson = torch.corrcoef(torch.stack([fij_extract, pij_extract]))[0, 1].item()
    slope = get_slope(fij_extract, pij_extract).item()
    
    return pearson, slope
