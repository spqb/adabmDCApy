import warnings
from typing import Iterable, Literal, Optional, Tuple, Union, overload

import numpy as np
import torch

from adabmDCA.alignment import Alignment, read_alignment, write_alignment
from adabmDCA.alphabet import get_tokens
    
    
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


@overload
def import_from_fasta(
    fasta_name: str,
    tokens: Optional[str] = None,
    filter_sequences: bool = False,
    remove_duplicates: bool = False,
    return_mask: Literal[False] = False,
) -> Tuple[np.ndarray, np.ndarray]: ...

@overload
def import_from_fasta(
    fasta_name: str,
    tokens: Optional[str] = None,
    filter_sequences: bool = False,
    remove_duplicates: bool = False,
    return_mask: Literal[True] = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def import_from_fasta(
    fasta_name: str,
    tokens: Optional[str] = None,
    filter_sequences: bool = False,
    remove_duplicates: bool = False,
    return_mask: bool = False,
):
    """Import sequences from a FASTA file using the legacy array interface.

    .. deprecated:: 0.7.8
       Use :func:`adabmDCA.read_alignment` for parsing or
       :func:`adabmDCA.load_alignment` for validated filtering and provenance.

    The following operations are performed:
    - If 'tokens' is provided, encodes the sequences in numeric format.
    - If 'filter_sequences' is True, removes the sequences whose tokens are not present in the alphabet.
    - If 'remove_duplicates' is True, removes the duplicated sequences.
    - If 'return_mask' is True, returns also the mask selecting the retained sequences from the original ones.

    Args:
        fasta_name (str | Path): Path to the fasta or compressed fasta (.fas.gz) file.
        tokens (str | None, optional): Alphabet to be used for the encoding. If provided, encodes the sequences in numeric format.
        filter_sequences (bool, optional): If True, removes the sequences whose tokens are not present in the alphabet. Defaults to False.
        remove_duplicates (bool, optional): If True, removes the duplicated sequences. Defaults to False.
        return_mask (bool, optional): If True, returns also the mask selecting the retained sequences from the original ones. Defaults to False.

    Raises:
        RuntimeError: The file is not in fasta format.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        - If 'return_mask' is False: Tuple of (headers, sequences)
        - If 'return_mask' is True: Tuple of (headers, sequences, mask)
    """
    warnings.warn(
        "import_from_fasta is deprecated; use read_alignment for parsing or "
        "load_alignment with AlignmentLoadConfig for filtering and provenance.",
        DeprecationWarning,
        stacklevel=2,
    )

    alignment = read_alignment(fasta_name, format="fasta")
    names = np.asarray(alignment.names, dtype=str)
    sequences = np.asarray(alignment.sequences, dtype=str)
    mask = np.ones(len(sequences), dtype=bool)
    
    # Filter sequences
    if filter_sequences:
        if tokens is None:
            raise ValueError("Argument 'tokens' must be provided if 'filter_sequences' is True.")
        from adabmDCA.input_loading import AlignmentLoadConfig, load_alignment

        loaded = load_alignment(
            alignment,
            config=AlignmentLoadConfig(
                alphabet=tokens,
                invalid_sequences="drop",
                remove_duplicates=remove_duplicates,
                normalize_dots=False,
            ),
        )
        for index in loaded.dropped_indices:
            print(f"Unknown token found: removing sequence {alignment.names[index]}")
        names = np.asarray(loaded.alignment.names, dtype=str)
        sequences = np.asarray(loaded.alignment.sequences, dtype=str)
        mask = np.zeros(len(alignment), dtype=bool)
        mask[list(loaded.retained_indices)] = True
        remove_duplicates = False
    
    # Remove duplicates
    if remove_duplicates:
        seen = set()
        retained = []
        for index, sequence in enumerate(sequences):
            if sequence not in seen:
                seen.add(sequence)
                retained.append(index)
        names = names[retained]
        sequences = sequences[retained]
        mask = np.zeros(len(alignment), dtype=bool)
        mask[retained] = True
        
    if (tokens is not None) and (len(sequences) > 0):
        sequences = encode_sequence(sequences, get_tokens(tokens))
        
    out = (names, sequences)
    if return_mask:
        out = out + (mask,)
    
    return out


def write_fasta(
    fname: str,
    headers: Union[Iterable[str], np.ndarray, torch.Tensor],
    sequences: Union[Iterable[str], np.ndarray, torch.Tensor],
    remove_gaps: bool = False,
    tokens: str = "protein",
) -> None:
    """Generate a FASTA file using the legacy array interface.

    .. deprecated:: 0.7.8
       Construct an :class:`adabmDCA.Alignment` and call
       :func:`adabmDCA.write_alignment`, or use a high-level result object's
       ``to_fasta``/``save_bundle`` method.

    Args:
        fname (str): Name of the output fasta file.
        headers (Union[Iterable[str], np.ndarray, torch.Tensor]): Iterable with sequences' headers.
        sequences (Union[Iterable[str], np.ndarray, torch.Tensor]): Iterable with sequences in string, categorical or one-hot encoded format.
        remove_gaps (bool, optional): If True, removes the gap from the alignment. Defaults to False.
        tokens (str): Alphabet to be used for the encoding. Defaults to 'protein'.
    """
    warnings.warn(
        "write_fasta is deprecated; use Alignment.write_fasta/write_alignment "
        "or a high-level result object's to_fasta/save_bundle method.",
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(headers, torch.Tensor):
        headers = headers.cpu().numpy()
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.cpu().numpy()
    if isinstance(headers, list):
        headers = np.array(headers)
    if isinstance(sequences, list):
        sequences = np.array(sequences)
    if not isinstance(sequences, np.ndarray):
        sequences = np.array(list(sequences))
    if not isinstance(headers, np.ndarray):
        headers = np.array(list(headers))
    sequences_arr: np.ndarray = sequences
    headers_arr: np.ndarray = headers
    
    tokens = get_tokens(tokens)
    
    # Handle the case when the sequenes are one-hot encoded
    if len(sequences_arr.shape) == 3:
        if sequences_arr.shape[2] != len(tokens):
            raise ValueError("The last dimension of the input one-hot encoded sequence must be equal to the length of the alphabet.")
        sequences_arr = np.argmax(sequences_arr, axis=2)
        seqs_decoded = decode_sequence(sequences_arr, tokens)
    else:
        # Handle the case when the sequences are in categorical or string format
        if np.issubdtype(sequences_arr.dtype, np.integer) or np.issubdtype(sequences_arr.dtype, np.floating):
            seqs_decoded = decode_sequence(sequences_arr, tokens)
        elif np.issubdtype(sequences_arr.dtype, np.str_):
            seqs_decoded = sequences_arr.copy()
        else:
            raise ValueError("Input sequences must be either in string or numeric format.")
        
    if isinstance(seqs_decoded, str):
        seqs_decoded = np.asarray([seqs_decoded])
    else:
        seqs_decoded = np.asarray(seqs_decoded, dtype=str)

    if remove_gaps:
        seqs_decoded = np.asarray([sequence.replace("-", "") for sequence in seqs_decoded], dtype=str)

    alignment = Alignment(
        names=tuple(str(header) for header in headers_arr),
        sequences=tuple(str(sequence) for sequence in seqs_decoded),
    )
    write_alignment(alignment, fname, format="fasta")
            

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
    that have a sequence identity with 's' >= th.

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
