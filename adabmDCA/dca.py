import numpy as np
import torch
from typing import Tuple, Dict, Optional
from adabmDCA.stats import get_freq_single_point, get_freq_two_points


def get_seqid(
    s1: torch.Tensor,
    s2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Return the number of matching sites for each pair of one-hot encoded sequences.
    - If s2 is provided, computes the sequence identity between the corresponding sequences in s1 and s2.
    - If s2 is a single sequence (L, q), it computes the sequence identities between the dataset s1 and s2.
    - If s2 is none, computes the sequence identity between s1 and a permutation of s1.

    Args:
        s1 (torch.Tensor): One-hot encoded sequence dataset 1 of shape (batch_size, L, q) or (L, q).
        s2 (Optional[torch.Tensor]): One-hot encoded sequence dataset 2 of shape (batch_size, L, q) or (L, q). Defaults to None.

    Returns:
        torch.Tensor: Tensor of sequence identities.
    """
    if s1.ndim not in (2, 3) or (s2 is not None and s2.ndim not in (2, 3)):
        raise ValueError("Sequences must have shape (L, q) or (N, L, q).")
    if s1.ndim == 2:
        s1 = s1.unsqueeze(0)
    if s2 is None:
        # Pair shuffled neighbours. A single sequence pairs with itself.
        s1 = s1[torch.randperm(s1.shape[0], device=s1.device)]
        s2 = s1.roll(1, dims=0)
    if s2.ndim == 2:
        s2 = s2.unsqueeze(0)
    if s1.shape[1:] != s2.shape[1:]:
        raise ValueError("Sequences must have the same length and state count.")
    if len(s1) != len(s2) and 1 not in (len(s1), len(s2)):
        raise ValueError("Sequence batches must have equal sizes or one must contain a single sequence.")
    return (s1 * s2).sum((1, 2))
    

def get_seqid_stats(
    s1: torch.Tensor,
    s2: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    - If s2 is provided, computes the mean and the standard deviation of the sequence identity between two sets of one-hot encoded sequences.
    - If s2 is a single sequence (L, q), it computes the mean and the standard deviation of the sequence identity between the dataset s1 and s2.
    - If s2 is none, computes the mean and the standard deviation of the sequence identity between s1 and a permutation of s1.

    Args:
        s1 (torch.Tensor): One-hot encoded sequence dataset 1 of shape (batch_size, L, q) or (L, q).
        s2 (Optional[torch.Tensor]): One-hot encoded sequence dataset 2 of shape (batch_size, L, q) or (L, q). Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (torch.Tensor) Mean sequence identity
            (torch.Tensor) Standard deviation of the sequence identity.
    """
    seqids = get_seqid(s1, s2)
    if not seqids.is_floating_point():
        seqids = seqids.to(torch.get_default_dtype())
    if len(seqids) == 1:
        mean_seqid = seqids[0]
        std_seqid = seqids.new_zeros(())
    else:
        mean_seqid = seqids.mean()
        std_seqid = seqids.std()
    return mean_seqid, std_seqid


def _zerosum_couplings(couplings: torch.Tensor) -> torch.Tensor:
    return couplings - (
        couplings.mean(dim=1, keepdim=True)
        + couplings.mean(dim=3, keepdim=True)
        - couplings.mean(dim=(1, 3), keepdim=True)
    )


def set_zerosum_gauge(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Project coupling blocks to zero-sum form for contact scoring.

    The returned fields are copies of the original fields. A gauge
    transformation of the full probability model also requires field shifts.
    
    Args:
        params (Dict[str, torch.Tensor]): Parameters of the model.
        
    Returns:
        Dict[str, torch.Tensor]: New dictionary with modified coupling matrix.
            "bias": torch.Tensor of shape (L, q)
            "coupling_matrix": torch.Tensor of shape (L, q, L, q)
    """
    return {key: _zerosum_couplings(value) if key == "coupling_matrix" else value.clone()
            for key, value in params.items()}


def _gap_index(tokens: str, num_states: int) -> int:
    if len(tokens) != num_states:
        raise ValueError("Alphabet length must match the number of states.")
    if "-" not in tokens:
        raise ValueError(f"Gap symbol '-' not found in alphabet: {tokens}")
    return tokens.index("-")


def get_contact_map(
    params: Dict[str, torch.Tensor],
    tokens: str,
) -> np.ndarray:
    """
    Computes the contact map from the model coupling matrix.

    Args:
        params (Dict[str, torch.Tensor]): Model parameters. Should contain:
            - "coupling_matrix": torch.Tensor of shape (L, q, L, q)
            - "bias": torch.Tensor of shape (L, q)
        tokens (str): Alphabet to be used.

    Returns:
        np.ndarray: Contact map.
    """
    Jij = params["coupling_matrix"].detach()
    q = Jij.shape[1]
    gap_idx = _gap_index(tokens, q)
    if Jij.dtype in (torch.float16, torch.bfloat16):
        Jij = Jij.float()
    # The quadratic Potts energy uses only the symmetric part of the couplings.
    Jij = _zerosum_couplings((Jij + Jij.permute(2, 3, 0, 1)).mul_(0.5))
    # Take all the entries of the coupling matrix except where the gap is involved
    mask = torch.arange(q, device=Jij.device) != gap_idx
    Jij_reduced = Jij[:, mask, :, :][:, :, :, mask]

    # Compute the Frobenius norm
    cm = Jij_reduced.square().sum((1, 3)).sqrt()
    cm = ((cm + cm.T) * 0.5).fill_diagonal_(0)
    row_sums = cm.sum(1)
    total = row_sums.sum()
    if total > 0:
        scaled = row_sums / total.sqrt()
        cm = (cm - torch.outer(scaled, scaled)).fill_diagonal_(0)
    return cm.cpu().numpy()


def get_mf_contact_map(
    data: torch.Tensor,
    tokens: str,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.5,
) -> np.ndarray:
    """
    Computes the contact map using the mean-field DCA approximation from the data.

    Args:
        data (torch.Tensor): Input one-hot data tensor.
        tokens (str): Alphabet to be used.
        weights (Optional[torch.Tensor]): Weights for the data points. Defaults to None.
        pseudo_count (float): Pseudocount used to regularize the empirical frequencies. Defaults to 0.5.

    Returns:
        np.ndarray: Contact map.
    """
    if data.dim() != 3:
        raise ValueError(f"Expected data to be a 3D tensor, but got {data.dim()}D tensor instead")
    if not 0.0 <= pseudo_count <= 1.0:
        raise ValueError(f"pseudo_count must be between 0 and 1, got {pseudo_count}")

    device = data.device
    L, q = data.shape[1], data.shape[2]

    gap_idx = _gap_index(tokens, q)
    state_mask = torch.arange(q, device=device) != gap_idx
    state_idx = torch.where(state_mask)[0]
    q_reduced = len(state_idx)

    if q_reduced < 1:
        raise ValueError("At least one non-gap state is required to compute mfDCA contacts")
    if data.dtype in (torch.float16, torch.bfloat16):
        data = data.float()
    
    fi = get_freq_single_point(data=data, weights=weights, pseudo_count=pseudo_count)
    fij = get_freq_two_points(data=data, weights=weights, pseudo_count=pseudo_count)

    fi_reduced = fi[:, state_mask]
    fij_reduced = fij[:, state_mask, :, :][:, :, :, state_mask]
    covariance = fij_reduced - torch.einsum("ia,jb->iajb", fi_reduced, fi_reduced)
    covariance = covariance.reshape(L * q_reduced, L * q_reduced)

    # Mean-field DCA inverts the reduced Potts covariance matrix and uses the
    # negative inverse as the coupling matrix in the reference-state gauge.
    try:
        Jij_reduced = -torch.linalg.inv(covariance).reshape(L, q_reduced, L, q_reduced)
    except torch.linalg.LinAlgError as exc:
        raise ValueError("Mean-field covariance is singular. Use a larger positive pseudocount.") from exc

    Jij = Jij_reduced.new_zeros((L, q, L, q))
    Jij[:, state_idx[:, None], :, state_idx] = Jij_reduced.permute(1, 3, 0, 2)

    params = {
        "bias": Jij.new_zeros((L, q)),
        "coupling_matrix": Jij,
    }

    return get_contact_map(params=params, tokens=tokens)
