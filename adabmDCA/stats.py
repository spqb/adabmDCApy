import itertools
from typing import Tuple

import torch


@torch.jit.script
def _get_freq_single_point(
    data: torch.Tensor,
    weights: torch.Tensor,
    pseudo_count: float,
) -> torch.Tensor:    
    _, _, q = data.shape
    frequencies = (data * weights).sum(dim=0)
    # Set to zero the negative frequencies. Used for the reintegration.
    torch.clamp_(frequencies, min=0.0)

    return (1. - pseudo_count) * frequencies + (pseudo_count / q)


def get_freq_single_point(
    data: torch.Tensor,
    weights: torch.Tensor | None = None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:
    """Computes the single point frequencies of the input MSA.
    Args:
        data (torch.Tensor): One-hot encoded data array.
        weights (torch.Tensor | None, optional): Weights of the sequences.
        pseudo_count (float, optional): Pseudo count to be added to the frequencies. Defaults to 0.0.
    
    Raises:
        ValueError: If the input data is not a 3D tensor.

    Returns:
        torch.Tensor: Single point frequencies.
    """
    if data.dim() != 3:
        raise ValueError(f"Expected data to be a 3D tensor, but got {data.dim()}D tensor instead")
    M = len(data)
    if weights is not None:
        norm_weights = weights.reshape(M, 1, 1) / weights.sum()
    else:
        norm_weights = torch.ones((M, 1, 1), device=data.device, dtype=data.dtype) / M
    
    return _get_freq_single_point(data, norm_weights, pseudo_count)


@torch.jit.script
def _get_freq_two_points(
    data: torch.Tensor,
    weights: torch.Tensor,
    pseudo_count: float,
) -> torch.Tensor:
    
    M, L, q = data.shape
    data_oh = data.reshape(M, q * L)
    
    fij = (data_oh * weights).T @ data_oh
    # Apply the pseudo count
    fij = (1. - pseudo_count) * fij + (pseudo_count / q**2)
    # Diagonal terms must represent the single point frequencies
    fi = get_freq_single_point(data, weights, pseudo_count).ravel()
    # Apply the pseudo count on the single point frequencies
    fij_diag = (1. - pseudo_count) * fi + (pseudo_count / q)
    # Set the diagonal terms of fij to the single point frequencies
    fij = torch.diagonal_scatter(fij, fij_diag, dim1=0, dim2=1)
    # Set to zero the negative frequencies. Used for the reintegration.
    torch.clamp_(fij, min=0.0)
    
    return fij.reshape(L, q, L, q)


def get_freq_two_points(
    data: torch.Tensor,
    weights: torch.Tensor | None = None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:
    """
    Computes the 2-points statistics of the input MSA.

    Args:
        data (torch.Tensor): One-hot encoded data array.
        weights (torch.Tensor | None, optional): Array of weights to assign to the sequences of shape.
        pseudo_count (float, optional): Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.0.
    
    Raises:
        ValueError: If the input data is not a 3D tensor.

    Returns:
        torch.Tensor: Matrix of two-point frequencies of shape (L, q, L, q).
    """
    if data.dim() != 3:
        raise ValueError(f"Expected data to be a 3D tensor, but got {data.dim()}D tensor instead")
    
    M = len(data)
    if weights is not None:
        norm_weights = weights.reshape(M, 1) / weights.sum()
    else:
        norm_weights = torch.ones((M, 1), device=data.device, dtype=data.dtype) / M
    
    return _get_freq_two_points(data, norm_weights, pseudo_count)


def generate_unique_triplets(
    L: int,
    ntriplets: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates a set of unique triplets of positions. Used to compute the 3-points statistics.
    
    Args:
        L (int): Length of the sequences.
        ntriplets (int): Number of triplets to be generated.
        device (torch.device, optional): Device to perform computations on. Defaults to "cpu".
    
    Returns:
        torch.Tensor: Tensor of shape (ntriplets, 3) containing the indices of the triplets.
    """    
    # Generate all possible unique triplets
    all_triplets = torch.tensor(list(itertools.combinations(range(L), 3)), device=device)
    # Shuffle the triplets to ensure randomness
    shuffled_triplets = all_triplets[torch.randperm(all_triplets.size(0))]
    # Select the first ntriplets from the shuffled triplets
    selected_triplets = shuffled_triplets[:ntriplets]
    
    return selected_triplets


@torch.jit.script
def _get_C_ijk(
    triplet: torch.Tensor,
    data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the third-order correlation tensor for a given triplet of positions.

    Args:
        triplet (torch.Tensor): Tensor of shape (3,) containing the indices of the triplet.
        data (torch.Tensor): Tensor of shape (M, L, q) containing the one-hot encoded data.
        weights (torch.Tensor): Tensor of shape (M, 1) containing the normalized weights of the sequences.

    Returns:
        torch.Tensor: Tensor of shape (q, q, q) containing the third-order correlations.
    """
    # Center the dataset
    data_c = data - data.mean(dim=0, keepdim=True)
    x = data_c[:, triplet[0], :] * weights
    y = data_c[:, triplet[1], :]
    z = data_c[:, triplet[2], :]

    C = torch.einsum("mi, mj, mk -> ijk", x, y, z)

    return C


def get_freq_three_points(
    data: torch.Tensor,
    ntriplets: int,
    weights: torch.Tensor | None = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Computes the 3-body statistics of the input MSA.

    Args:
        data (torch.Tensor): Input MSA in one-hot encoding.
        ntriplets (int): Number of triplets to test.
        weights (torch.Tensor | None, optional): Importance weights for the sequences. Defaults to None.
        device (torch.device, optional): Device to perform computations on. Defaults to "cpu".

    Returns:
        torch.Tensor: 3-points connected correlation for ntriplets randomly extracted triplets.
    """
    if data.dim() != 3:
        raise ValueError(f"Expected data to be a 3D tensor, but got {data.dim()}D tensor instead")
    
    M = len(data)
    if weights is not None:
        norm_weights = weights.view(-1, 1) / weights.sum()
    else:
        norm_weights = torch.ones((M, 1), device=data.device, dtype=data.dtype) / M
    
    L = data.shape[1]
    triplets = generate_unique_triplets(L=L, ntriplets=ntriplets, device=device)
    Cijk = []
    for triplet in triplets:
        Cijk.append(_get_C_ijk(triplet, data, norm_weights).flatten())
        
    return torch.stack(Cijk)


def get_covariance_matrix(
    data: torch.Tensor,
    weights: torch.Tensor | None = None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:
    """Computes the weighted covariance matrix of the input multi sequence alignment.

    Args:
        data (torch.Tensor): Input MSA in one-hot variables.
        weights (torch.Tensor | None, optional): Importance weights of the sequences.
        pseudo_count (float, optional): Pseudo count. Defaults to 0.0.

    Returns:
        torch.Tensor: Covariance matrix.
    """
    _, L, q = data.shape
    fi = get_freq_single_point(data, weights, pseudo_count)
    fij = get_freq_two_points(data, weights, pseudo_count)
    cov_matrix = fij - torch.einsum('ij,kl->ijkl', fi, fi)
    
    return cov_matrix.reshape(L * q, L * q)


def _get_slope(x, y):
    n = len(x)
    num = n * (x @ y) - y.sum() * x.sum()
    den = n * (x @ x) - torch.square(x.sum())
    return torch.abs(num / den)


def extract_Cij_from_freq(
    fij: torch.Tensor,
    pij: torch.Tensor,
    fi: torch.Tensor,
    pi: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Tuple[float, float]:
    """Extracts the lower triangular part of the covariance matrices of the data and chains starting from the frequencies.

    Args:
        fij (torch.Tensor): Two-point frequencies of the data.
        pij (torch.Tensor): Two-point frequencies of the chains.
        fi (torch.Tensor): Single-point frequencies of the data.
        pi (torch.Tensor): Single-point frequencies of the chains.
        mask (torch.Tensor | None, optional): Mask for comparing just a subset of the couplings. Defaults to None.

    Returns:
        Tuple[float, float]: Extracted two-point frequencies of the data and chains.
    """
    L = fi.shape[0]
    
    # Compute the covariance matrices
    cov_data = fij - torch.einsum('ij,kl->ijkl', fi, fi)
    cov_chains = pij - torch.einsum('ij,kl->ijkl', pi, pi)
    
    # Only use a subset of couplings if a mask is provided
    if mask is not None:
        cov_data = torch.where(mask, cov_data, torch.tensor(0.0, device=cov_data.device, dtype=cov_data.dtype))
        cov_chains = torch.where(mask, cov_chains, torch.tensor(0.0, device=cov_chains.device, dtype=cov_chains.dtype))
    
    # Extract only the entries of half the matrix and out of the diagonal blocks
    idx_row, idx_col = torch.tril_indices(L, L, offset=-1)
    fij_extract = cov_data[idx_row, :, idx_col, :].reshape(-1)
    pij_extract = cov_chains[idx_row, :, idx_col, :].reshape(-1)
    
    return fij_extract, pij_extract


def extract_Cij_from_seqs(
    data: torch.Tensor,
    chains: torch.Tensor,
    weights: torch.Tensor | None = None,
    pseudo_count: float = 0.0,
    mask: torch.Tensor | None = None,
) -> Tuple[float, float]:
    """Extracts the lower triangular part of the covariance matrices of the data and chains starting from the sequences.

    Args:
        data (torch.Tensor): Data sequences.
        chains (torch.Tensor): Chain sequences.
        weights (torch.Tensor | None, optional): Weights of the sequences. Defaults to None.
        pseudo_count (float, optional): Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.0.
        mask (torch.Tensor | None, optional): Mask for comparing just a subset of the couplings. Defaults to None.

    Returns:
        Tuple[float, float]: Two-point frequencies of the data and chains.
    """
    fi = get_freq_single_point(data, weights=weights, pseudo_count=pseudo_count)
    pi = get_freq_single_point(chains, weights=None)
    fij = get_freq_two_points(data, weights=weights, pseudo_count=pseudo_count)
    pij = get_freq_two_points(chains, weights=None)
    
    return extract_Cij_from_freq(fij, pij, fi, pi, mask)


def get_correlation_two_points(
    fij: torch.Tensor,
    pij: torch.Tensor,
    fi: torch.Tensor,
    pi: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Tuple[float, float]:
    """Computes the Pearson coefficient and the slope between the two-point frequencies of data and chains.

    Args:
        fij (torch.Tensor): Two-point frequencies of the data.
        pij (torch.Tensor): Two-point frequencies of the chains.
        fi (torch.Tensor): Single-point frequencies of the data.
        pi (torch.Tensor): Single-point frequencies of the chains.
        mask (torch.Tensor | None, optional): Mask to select the couplings to use for the correlation coefficient. Defaults to None. 

    Returns:
        Tuple[float, float]: Pearson correlation coefficient of the two-sites statistics and slope of the interpolating line.
    """
    
    fij_extract, pij_extract = extract_Cij_from_freq(fij, pij, fi, pi, mask)
    # torch.corrcoef does not support half precision
    pearson = torch.corrcoef(torch.stack([fij_extract.float(), pij_extract.float()]))[0, 1].item()
    slope = _get_slope(fij_extract.float(), pij_extract.float()).item()
    
    return pearson, slope
