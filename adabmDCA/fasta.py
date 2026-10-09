from typing import Iterable, Union

import numpy as np
import torch

from adabmDCA.alphabet import get_tokens  # noqa: F401  (re-exported: imported from this module elsewhere)


def encode_sequence(sequence: Union[str, Iterable[str]], tokens: str) -> np.ndarray:
    """Encodes a sequence or a list of sequences into a numeric format.

    Args:
        sequence (Union[str, Iterable[str]]): Input sequence or iterable of sequences of size (batch_size,).
        tokens (str): Alphabet to be used for the encoding.

    Returns:
        np.ndarray: Array of shape (L,) or (batch_size, L) with the encoded sequence or sequences.
    """
    letter_map = {l : n for n, l in enumerate(tokens)}
    
    def _encode(sequence):
        return [letter_map[l] for l in sequence]
    
    if isinstance(sequence, str):
        return np.array(_encode(sequence))
    elif isinstance(sequence, np.ndarray):
        sequence = list(sequence)
        return np.array(list(map(_encode, sequence)))
    elif isinstance(sequence, torch.Tensor):
        sequence = sequence.cpu().numpy()
        sequence = list(sequence)
        return np.array(list(map(_encode, sequence)))
    elif isinstance(sequence, list):
        return np.array(list(map(_encode, sequence)))
    else:        
        raise ValueError("Input sequence must be either a string or a numpy array.")


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
        sequence = sequence.cpu().numpy()
    if not isinstance(sequence, np.ndarray):
        raise TypeError("Input sequence must be either a numpy array, a list or a torch tensor.")
    sequence = sequence.astype(int)
    
    def _decode(sequence):
        return ''.join([tokens[aa] for aa in sequence])
    
    if sequence.ndim == 1:
        return _decode(sequence)
    elif sequence.ndim == 2:
        sequence = list(sequence)
        return np.array(list(map(_decode, sequence)))
    elif sequence.ndim == 3:
        if sequence.shape[2] != len(tokens):
            raise ValueError("The last dimension of the input one-hot encoded sequence must be equal to the length of the alphabet.")
        sequence = np.argmax(sequence, axis=2)
        sequence = list(sequence)
        return np.array(list(map(_decode, sequence)))
    else:
        raise ValueError("Input sequence must be either a 1D, 2D or a 3D (one-hot encoded) iterable.")


def _get_sequence_weight(s: torch.Tensor, data: torch.Tensor, L: int, th: float):
    seq_id = torch.sum(s == data, dim=1) / L
    n_clust = torch.sum(seq_id > th)
    
    return 1.0 / n_clust


def compute_weights(
    data: Union[np.ndarray, torch.Tensor],
    th: float = 0.8,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Computes the weight to be assigned to each sequence 's' in 'data' as 1 / n_clust, where 'n_clust' is the number of sequences
    that have a sequence identity with 's' > th (including 's' itself).

    Args:
        data (Union[np.ndarray, torch.Tensor]): Input dataset. Must be either a (batch_size, L) or a (batch_size, L, q) (one-hot encoded) array.
        th (float, optional): Sequence identity threshold for the clustering. Defaults to 0.8.
        device (torch.device, optional): Device. Defaults to "cpu".
        dtype (torch.dtype, optional): Data type. Defaults to torch.float32.

    Returns:
        torch.Tensor: Array with the weights of the sequences.
    """
    if len(data.shape) not in (2, 3):
        raise ValueError("'data' must be either a (batch_size, L) or a (batch_size, L, q) (one-hot encoded) array.")
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, device=device)
    if len(data.shape) == 3:
        data_encoded = data.argmax(dim=2)
    else:
        data_encoded = data
    _, L = data_encoded.shape
    weights = torch.vstack([_get_sequence_weight(s, data_encoded, L, th) for s in data_encoded]).view(-1)

    return weights.to(dtype)


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
