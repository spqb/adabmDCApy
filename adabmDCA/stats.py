import random
from bisect import bisect_right
from math import comb, nan
from typing import Tuple, Optional
import torch


def _normalize_weights(data: torch.Tensor, weights: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError("data must have nonempty shape (N, L, q).")
    dtype = data.dtype if data.is_floating_point() else torch.get_default_dtype()
    if weights is None:
        data = data.to(dtype)
        return data, data.new_full((len(data), 1), 1 / len(data))
    if weights.numel() != len(data):
        raise ValueError("Provide one weight per sequence.")
    dtype = torch.promote_types(dtype, weights.dtype)
    weights = weights.reshape(-1, 1).to(device=data.device, dtype=dtype)
    total = weights.sum()
    if not (torch.isfinite(total) & (total != 0)):
        raise ValueError("Sequence weights must have a finite, nonzero sum.")
    return data.to(dtype), weights / total


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
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:
    """Computes the single point frequencies of the input MSA.
    Args:
        data (torch.Tensor): One-hot encoded data array.
        weights (Optional[torch.Tensor], optional): Weights of the sequences.
        pseudo_count (float, optional): Pseudo count to be added to the frequencies. Defaults to 0.0.
    
    Raises:
        ValueError: If the input data is not a 3D tensor.

    Returns:
        torch.Tensor: Single point frequencies.
    """
    data, norm_weights = _normalize_weights(data, weights)
    return _get_freq_single_point(data, norm_weights.unsqueeze(-1), pseudo_count)


def _get_freq_two_points(
    data: torch.Tensor,
    weights: torch.Tensor,
    pseudo_count: float,
) -> torch.Tensor:
    
    M, L, q = data.shape
    data_oh = data.reshape(M, q * L)
    
    fij = (data_oh * weights).T @ data_oh
    # Set to zero the negative frequencies. Used for the reintegration.
    torch.clamp_(fij, min=0.0)
    # Apply the pseudo count
    fij = (1. - pseudo_count) * fij + (pseudo_count / q**2)
    # Within a site, two different one-hot states cannot occur together.
    fi = _get_freq_single_point(data, weights.unsqueeze(-1), pseudo_count)
    blocks = fij.reshape(L, q, L, q)
    sites = torch.arange(L, device=data.device)
    blocks[sites, :, sites, :] = torch.diag_embed(fi)
    return blocks


def get_freq_two_points(
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:
    """
    Computes the 2-points statistics of the input MSA.

    Args:
        data (torch.Tensor): One-hot encoded data array.
        weights (Optional[torch.Tensor], optional): Array of weights to assign to the sequences of shape.
        pseudo_count (float, optional): Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.0.
    
    Raises:
        ValueError: If the input data is not a 3D tensor.

    Returns:
        torch.Tensor: Matrix of two-point frequencies of shape (L, q, L, q).
    """
    data, norm_weights = _normalize_weights(data, weights)
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
        torch.Tensor: Indices of up to ntriplets unique triplets, with shape (count, 3).
    """    
    if L < 0 or ntriplets < 0:
        raise ValueError("L and ntriplets must be nonnegative.")
    total = comb(L, 3)
    count = min(total, ntriplets)
    if count == 0:
        return torch.empty((0, 3), dtype=torch.long, device=device)

    # Sample lexicographic ranks and locate the first two sites by their prefix counts.
    seed = torch.randint(0, 2**63 - 1, (), device="cpu").item()
    ranks = random.Random(seed).sample(range(total), count)
    triplet_prefix = [total - comb(L - i, 3) for i in range(L - 1)]
    pair_count = comb(L, 2)
    pair_prefix = [pair_count - comb(L - i, 2) for i in range(L)]
    triplets = []
    for rank in ranks:
        i = bisect_right(triplet_prefix, rank) - 1
        pair_rank = rank - triplet_prefix[i] + pair_prefix[i + 1]
        j = bisect_right(pair_prefix, pair_rank) - 1
        k = j + 1 + pair_rank - pair_prefix[j]
        triplets.append((i, j, k))
    return torch.tensor(triplets, dtype=torch.long, device=device)


def _get_C_ijk(triplet: list[int], centered: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute the third central moment of three sites in already centered data."""
    i, j, k = triplet
    return torch.einsum("ma,mb,mc->abc", centered[:, i] * weights, centered[:, j], centered[:, k])


def get_freq_three_points(
    nat: torch.Tensor,
    gen: torch.Tensor,
    ntriplets: int = 10000,
    triplets: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the 3-body connected correlation statistics of the input MSAs.

    Args:
        nat (torch.Tensor): Input MSA representing natural data in one-hot encoding.
        gen (torch.Tensor): Input MSA representing generated data in one-hot encoding.
        ntriplets (int): Number of triplets to test. Defaults to 10000.
        triplets (Optional[torch.Tensor], optional): Predefined triplets to use for the computation. Defaults to None.
        weights (Optional[torch.Tensor], optional): Importance weights for the natural sequences. Defaults to None.
        device (torch.device, optional): Device to perform computations on. Defaults to "cpu".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Natural and generated 3-points connected correlation for ntriplets randomly extracted triplets.
    """
    nat, norm_weights = _normalize_weights(nat, weights)
    gen, uniform_weights = _normalize_weights(gen, None)
    if nat.shape[1:] != gen.shape[1:]:
        raise ValueError("The two MSAs must have the same length and state count.")
    if triplets is None:
        triplets = generate_unique_triplets(L=nat.shape[1], ntriplets=ntriplets, device=device)
    if triplets.ndim != 2 or triplets.shape[1] != 3:
        raise ValueError("triplets must have shape (count, 3).")
    if len(triplets) == 0:
        return nat.new_empty(0), gen.new_empty(0)
    if triplets.is_floating_point() or triplets.is_complex() or triplets.dtype == torch.bool:
        raise ValueError("Triplet site indices must be integers.")

    nat_mean = nat.mean(dim=0) if weights is None else (nat * norm_weights.unsqueeze(-1)).sum(dim=0)
    nat_centered = nat - nat_mean
    gen_centered = gen - gen.mean(dim=0)
    indices = triplets.tolist()
    return (
        torch.stack([_get_C_ijk(t, nat_centered, norm_weights) for t in indices]).flatten(),
        torch.stack([_get_C_ijk(t, gen_centered, uniform_weights) for t in indices]).flatten(),
    )


def get_covariance_matrix(
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
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


def extract_Cij_from_freq(
    fij: torch.Tensor,
    pij: torch.Tensor,
    fi: torch.Tensor,
    pi: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts the lower triangular part of the covariance matrices of the natural data and generated data starting from the frequencies.

    Args:
        fij (torch.Tensor): Two-point frequencies of the natural data.
        pij (torch.Tensor): Two-point frequencies of the generated data.
        fi (torch.Tensor): Single-point frequencies of the natural data.
        pi (torch.Tensor): Single-point frequencies of the generated data.
        mask (Optional[torch.Tensor], optional): Mask for comparing just a subset of the couplings. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Extracted covariance matrix entries of the natural data and generated data.
    """
    idx_row, idx_col = torch.tril_indices(fi.shape[0], fi.shape[0], offset=-1, device=fij.device)
    cov_data = fij[idx_row, :, idx_col, :] - fi[idx_row, :, None] * fi[idx_col, None, :]
    cov_chains = pij[idx_row, :, idx_col, :] - pi[idx_row, :, None] * pi[idx_col, None, :]
    if mask is not None:
        selected = torch.broadcast_to(mask, fij.shape)[idx_row, :, idx_col, :]
        return cov_data[selected], cov_chains[selected]
    return cov_data.flatten(), cov_chains.flatten()


def extract_Cij_from_seqs(
    data: torch.Tensor,
    chains: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts the lower triangular part of the covariance matrices of the natural data and generated data starting from the sequences.

    Args:
        data (torch.Tensor): Natural data sequences.
        chains (torch.Tensor): Generated data sequences.
        weights (torch.Tensor | None, optional): Weights of the sequences. Defaults to None.
        pseudo_count (float, optional): Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.0.
        mask (torch.Tensor | None, optional): Mask for comparing just a subset of the couplings. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Selected connected correlations of the natural and generated data.
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
    mask: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Compute Pearson correlation and absolute regression slope for connected correlations.

    Args:
        fij (torch.Tensor): Two-point frequencies of the natural data.
        pij (torch.Tensor): Two-point frequencies of the generated data.
        fi (torch.Tensor): Single-point frequencies of the natural data.
        pi (torch.Tensor): Single-point frequencies of the generated data.
        mask (Optional[torch.Tensor], optional): Mask to select the couplings to use for the correlation coefficient. Defaults to None. 

    Returns:
        Tuple[float, float]: Pearson correlation and absolute least-squares slope. Undefined values are NaN.
    """
    
    x, y = extract_Cij_from_freq(fij, pij, fi, pi, mask)
    if x.numel() < 2:
        return nan, nan
    dtype = torch.promote_types(torch.promote_types(x.dtype, y.dtype), torch.float32)
    x, y = x.to(dtype), y.to(dtype)
    x, y = x - x[0], y - y[0]
    x, y = x - x.mean(), y - y.mean()
    xy = x @ y
    xx = x @ x
    yy = y @ y
    pearson, slope = torch.stack([(xy / (xx.sqrt() * yy.sqrt())).clamp(-1, 1), (xy / xx).abs()]).tolist()
    return pearson, slope
