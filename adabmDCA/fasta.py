import numpy as np
from typing import Iterable, Tuple, Union, overload, Literal, Optional
from Bio import SeqIO

import torch

# Default alphabets
TOKENS_PROTEIN = "-ACDEFGHIKLMNPQRSTVWY"
TOKENS_RNA = "-ACGU"
TOKENS_DNA = "-ACGT"


def get_tokens(alphabet: str) -> str:
    """Converts a known alphabet into the corresponding tokens, otherwise returns the custom alphabet.

    Args:
        alphabet (str): Alphabet to be used for the encoding. It can be either "protein", "rna", "dna" or a custom string of tokens.

    Returns:
        str: Tokens of the alphabet.
    """
    if not isinstance(alphabet, str):
        raise TypeError("Argument 'alphabet' must be of type str")
    if alphabet == "protein":
        return TOKENS_PROTEIN
    elif alphabet == "rna":
        return TOKENS_RNA
    elif alphabet == "dna":
        return TOKENS_DNA
    else:
        return alphabet
    
    
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
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Import sequences from a fasta file. The following operations are performed:
    - If 'tokens' is provided, encodes the sequences in numeric format.
    - If 'filter_sequences' is True, removes the sequences whose tokens are not present in the alphabet.
    - If 'remove_duplicates' is True, removes the duplicated sequences.
    - If 'return_mask' is True, returns also the mask selecting the retained sequences from the original ones.

    Args:
        fasta_name (Union[str, Path]): Path to the fasta file.
        tokens (Optional[str]): Alphabet to be used for the encoding. If provided, encodes the sequences in numeric format.
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
    # Import headers and sequences
    sequences = []
    names = []
    for record in SeqIO.parse(fasta_name, "fasta"):
        names.append(str(record.id))
        sequences.append(str(record.seq))
    
    # Filter sequences
    if filter_sequences:
        if tokens is None:
            raise ValueError("Argument 'tokens' must be provided if 'filter_sequences' is True.")
        tokens = get_tokens(tokens)
        tokens_list = [a for a in tokens]
        clean_names = []
        clean_sequences = []
        clean_mask = []
        for n, s in zip(names, sequences):
            if all(c in tokens_list for c in s):
                if n == "":
                    n = "unknown_sequence"
                clean_names.append(n)
                clean_sequences.append(s)
                clean_mask.append(True)
            else:
                print(f"Unknown token found: removing sequence {n}")
                clean_mask.append(False)
        names = np.array(clean_names)
        sequences = np.array(clean_sequences)
        mask = np.array(clean_mask)
        
    else:
        names = np.array(names)
        sequences = np.array(sequences)
        mask = np.full(len(sequences), True)
    
    # Remove duplicates
    if remove_duplicates:
        sequences, unique_ids = np.unique(sequences, return_index=True)
        # sort to preserve the original order
        order = np.argsort(unique_ids)
        sequences = sequences[order]
        names = names[unique_ids[order]]
        # set to false the mask elements corresponding to the duplicates
        original_indices_post_filtering = np.where(mask)[0]
        original_indices_of_unique_items = original_indices_post_filtering[unique_ids]
        mask_unique = np.full(len(mask), False)
        mask_unique[original_indices_of_unique_items] = True
        mask = mask & mask_unique
        
    if (tokens is not None) and (len(sequences) > 0):
        sequences = encode_sequence(sequences, tokens)
        
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
    """Generate a fasta file with the input sequences.

    Args:
        fname (str): Name of the output fasta file.
        headers (Union[Iterable[str], np.ndarray, torch.Tensor]): Iterable with sequences' headers.
        sequences (Union[Iterable[str], np.ndarray, torch.Tensor]): Iterable with sequences in string, categorical or one-hot encoded format.
        remove_gaps (bool, optional): If True, removes the gap from the alignment. Defaults to False.
        tokens (str): Alphabet to be used for the encoding. Defaults to 'protein'.
    """
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
        
    if remove_gaps:
        seqs_decoded = np.vectorize(lambda s: s.replace("-", ""))(seqs_decoded)
        
    with open(fname, 'w') as f:
        for name_seq, seq in zip(headers_arr, seqs_decoded):
            f.write('>' + name_seq + '\n')
            f.write(seq)
            f.write('\n')
            

@torch.jit.script
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
    weights = torch.vstack([_get_sequence_weight(s, data_encoded, L, th) for s in data_encoded])

    return weights.to(dtype)


def validate_alphabet(sequences: np.ndarray, tokens: str):
    """Check if the chosen alphabet is compatible with the input sequences.

    Args:
        sequences (np.ndarray): Input sequences.
        tokens (str): Alphabet to be used for the encoding.

    Raises:
        ValueError: The chosen alphabet is incompatible with the Multi-Sequence Alignment.
    """
    all_char = "".join(sequences)
    tokens_data = "".join(sorted(set(all_char)))
    sorted_tokens = "".join(sorted(tokens))
    if not sorted_tokens == tokens_data:
        raise ValueError(f"The chosen alphabet is incompatible with the Multi-Sequence Alignment. The missing tokens are: {[c for c in tokens_data if c not in sorted_tokens]}. Current alphabet: {tokens}")