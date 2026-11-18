from math import isqrt
from typing import Iterable, Union

import numpy as np
import torch

from adabmDCA.alphabet import get_tokens  # noqa: F401  (re-exported: imported from this module elsewhere)

_WEIGHT_ENCODING_ELEMENTS = 1 << 23
_WEIGHT_TILE_ELEMENTS = 1 << 20


def encode_sequence(sequence: Union[str, Iterable[str]], tokens: str) -> np.ndarray:
    """Encodes a sequence or a list of sequences into a numeric format.

    Args:
        sequence (Union[str, Iterable[str]]): Input sequence or iterable of sequences of size (batch_size,).
        tokens (str): Alphabet to be used for the encoding.

    Returns:
        np.ndarray: Array of shape (L,) or (batch_size, L) with the encoded sequence or sequences.
    """
    letter_map = {token: n for n, token in enumerate(tokens)}
    
    def _encode(sequence):
        return [letter_map[token] for token in sequence]
    
    if isinstance(sequence, str):
        return np.asarray(_encode(sequence), dtype=np.int64)
    return np.asarray([_encode(row) for row in sequence], dtype=np.int64)


def decode_sequence(sequence: Union[np.ndarray, torch.Tensor, list], tokens: str) -> Union[str, np.ndarray]:
    """Takes a numeric sequence or list of seqences in input an returns the corresponding string encoding.

    Args:
        sequence (Union[np.ndarray, torch.Tensor, list]): Input sequences. Can be of shape
            - (L,): single sequence in encoded format
            - (batch_size, L): multiple sequences in encoded format
            - (batch_size, L, q) multiple one-hot encoded sequences
        tokens (str): Alphabet to be used for the encoding.

    Returns:
        Union[str, np.ndarray]: string or array of strings with the decoded input.
    """
    if isinstance(sequence, list):
        sequence = np.array(sequence)
    elif isinstance(sequence, torch.Tensor):
        sequence = sequence.detach().cpu()
        if sequence.dtype == torch.bfloat16:
            sequence = sequence.float()
        sequence = sequence.numpy()
    if not isinstance(sequence, np.ndarray):
        raise TypeError("Input sequence must be either a numpy array, a list or a torch tensor.")
    if sequence.dtype.kind not in "biuf":
        raise ValueError("Encoded sequences must contain real numeric values.")
    if sequence.ndim == 3:
        if sequence.shape[2] != len(tokens):
            raise ValueError("The last dimension of the input one-hot encoded sequence must be equal to the length of the alphabet.")
        if not np.all((sequence == 0) | (sequence == 1)) or not np.all((sequence == 1).sum(axis=2) == 1):
            raise ValueError("Each one-hot row must contain exactly one 1 and otherwise zeros.")
        sequence = np.argmax(sequence, axis=2)
    elif sequence.ndim not in (1, 2):
        raise ValueError("Input sequence must be either a 1D, 2D or a 3D (one-hot encoded) iterable.")
    if not np.all((sequence >= 0) & (sequence < len(tokens))):
        raise ValueError("Encoded states must be valid indices into the alphabet.")
    if sequence.dtype.kind == "f" and not np.all(sequence == np.floor(sequence)):
        raise ValueError("Encoded states must be integers.")
    sequence = sequence.astype(np.intp, copy=False)

    def _decode(row):
        return "".join(tokens[aa] for aa in row)

    if sequence.ndim == 1:
        return _decode(sequence)
    return np.asarray([_decode(row) for row in sequence], dtype=str)


def _count_sequence_neighbors(data: torch.Tensor, th: float) -> torch.Tensor:
    """Count neighbors in bounded tiles, using a matrix product when a dense encoding fits."""
    size, length = data.shape
    # Compare integer counts to avoid rounding an identity threshold up to 1.
    minimum = round(th * length)
    if minimum / length <= th:
        minimum += 1
    counts = torch.empty(size, dtype=torch.int64, device=data.device)
    if not data.is_floating_point():
        indices = data.long()
        low, high = torch.stack(torch.aminmax(indices)).tolist()
        classes = high - low + 1
        if size * length * classes <= _WEIGHT_ENCODING_ELEMENTS:
            encoded = torch.zeros((size, length, classes), dtype=torch.float32, device=data.device)
            encoded.scatter_(-1, (indices - low).unsqueeze(-1), 1)
            rows = encoded.flatten(1)
            chunk = max(1, _WEIGHT_TILE_ELEMENTS // size)
            # Mixed precision can round integer match counts before thresholding.
            with torch.autocast(device_type=data.device.type, enabled=False):
                for start in range(0, size, chunk):
                    block = rows[start:start + chunk]
                    counts[start:start + len(block)] = ((block @ rows.T) >= minimum).sum(dim=1)
            return counts

    chunk = max(1, isqrt(_WEIGHT_TILE_ELEMENTS // length))
    for start in range(0, size, chunk):
        block = data[start:start + chunk]
        count = counts[start:start + len(block)].zero_()
        for other in data.split(chunk):
            matches = (block[:, None, :] == other[None, :, :]).sum(dim=-1)
            count += (matches >= minimum).sum(dim=1)
    return counts


def compute_weights(
    data: Union[np.ndarray, torch.Tensor],
    th: float = 0.8,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Weight each sequence by the reciprocal of its neighbor count.

    Each sequence counts itself. Other sequences count when their identity is
    strictly greater than the threshold.

    Args:
        data (Union[np.ndarray, torch.Tensor]): Input dataset. Must be either a (batch_size, L) or a (batch_size, L, q) (one-hot encoded) array.
        th (float, optional): Sequence identity threshold for the clustering. Defaults to 0.8.
        device (torch.device, optional): Device. Defaults to "cpu".
        dtype (torch.dtype, optional): Data type. Defaults to torch.float32.

    Returns:
        torch.Tensor: Array with the weights of the sequences.
    """
    if not 0 <= th <= 1:
        raise ValueError("The sequence identity threshold must be in [0, 1].")
    data = torch.as_tensor(data, device=device)
    if data.ndim not in (2, 3):
        raise ValueError("'data' must be either a (batch_size, L) or a (batch_size, L, q) (one-hot encoded) array.")
    if data.shape[1] == 0 or (data.ndim == 3 and data.shape[2] == 0):
        raise ValueError("Sequences must have at least one site and state.")
    if data.is_complex():
        raise ValueError("Sequence encodings must contain real values.")
    if len(data) == 0 or th == 1:
        return torch.ones(len(data), device=device, dtype=dtype)
    if data.is_floating_point() and not torch.isfinite(data).all():
        raise ValueError("Sequence encodings must contain finite values.")
    if data.ndim == 3:
        data = (data.to(torch.uint8) if data.dtype == torch.bool else data).argmax(dim=-1)
    counts = _count_sequence_neighbors(data, th)
    precision = torch.float64 if dtype == torch.float64 else torch.float32
    return counts.to(precision).reciprocal().to(dtype)


def validate_alphabet(sequences: Iterable[str], tokens: str):
    """Validates that all characters in the sequences are present in the provided alphabet.
    Args:
        sequences (Iterable[str]): Iterable of sequences to be validated.
        tokens (str): Alphabet to be used for the validation.
    """
    all_char = "".join(sequences)
    tokens_data = "".join(sorted(set(all_char)))
    for c in tokens_data:
        if c not in tokens:
            raise KeyError(
                f"The chosen alphabet is incompatible with the Multi-Sequence Alignment. The unexpected token is: '{c}'"
            )
