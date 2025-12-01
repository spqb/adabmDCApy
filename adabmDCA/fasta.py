from pathlib import Path
import numpy as np
from typing import Tuple
from Bio import SeqIO
import gzip

import torch

# Default alphabets
TOKENS_PROTEIN = "-ACDEFGHIKLMNPQRSTVWY"
TOKENS_RNA = "-ACGU"
TOKENS_DNA = "-ACGT"


def get_tokens(alphabet: str) -> str:
    """Converts the alphabet into the corresponding tokens.

    Args:
        alphabet (str): Alphabet to be used for the encoding. It can be either "protein", "rna", "dna" or a custom string of tokens.

    Returns:
        str: Tokens of the alphabet.
    """
    assert isinstance(alphabet, str), "Argument 'alphabet' must be of type str"
    if alphabet == "protein":
        return TOKENS_PROTEIN
    elif alphabet == "rna":
        return TOKENS_RNA
    elif alphabet == "dna":
        return TOKENS_DNA
    else:
        return alphabet
    
    
def encode_sequence(sequence: str | np.ndarray | list, tokens: str) -> np.ndarray:
    """Encodes a sequence or a list of sequences into a numeric format.

    Args:
        sequence (str | np.ndarray | list): Input sequence.
        tokens (str): Alphabet to be used for the encoding.

    Returns:
        np.ndarray: Encoded sequence or sequences.
    """
    letter_map = {l : n for n, l in enumerate(tokens)}
    
    def _encode(sequece):
        return [letter_map[l] for l in sequece]
    
    if isinstance(sequence, str):
        return np.array(_encode(sequence))
    elif isinstance(sequence, np.ndarray):
        sequence = list(sequence)
        return np.array(list(map(_encode, sequence)))
    elif isinstance(sequence, list):
        return np.array(list(map(_encode, sequence)))
    else:        
        raise ValueError("Input sequence must be either a string or a numpy array.")


def decode_sequence(sequence: list | np.ndarray | torch.Tensor, tokens: str) -> str | np.ndarray:
    """Takes a numeric sequence or list of seqences in input an returns the corresponding string encoding.

    Args:
        sequence (np.ndarray): Input sequences. Can be either a 1D, 2D or a 3D (one-hot encoded) iterable.
        tokens (str): Alphabet to be used for the encoding.

    Returns:
        str | np.ndarray: string or array of strings with the decoded input.
    """
    if isinstance(sequence, list):
        sequence = np.array(sequence)
    elif isinstance(sequence, torch.Tensor):
        sequence = sequence.cpu().numpy()
    assert isinstance(sequence, np.ndarray), "Input sequence must be either a numpy array, a list or a torch tensor."
    sequence = sequence.astype(int)
    
    def _decode(sequence):
        return ''.join([tokens[aa] for aa in sequence])
    
    if sequence.ndim == 1:
        return _decode(sequence)
    elif sequence.ndim == 2:
        sequence = list(sequence)
        return np.array(list(map(_decode, sequence)))
    elif sequence.ndim == 3:
        assert sequence.shape[2] == len(tokens), "The last dimension of the input one-hot encoded sequence must be equal to the length of the alphabet."
        sequence = np.argmax(sequence, axis=2)
        sequence = list(sequence)
        return np.array(list(map(_decode, sequence)))
    else:
        raise ValueError("Input sequence must be either a 1D, 2D or a 3D (one-hot encoded) iterable.")


def import_from_fasta(
    fasta_name: str | Path,
    tokens: str | None = None,
    filter_sequences: bool = False,
    remove_duplicates: bool = False,
    return_mask: bool = False,
):
    """Import sequences from a fasta or compressed fasta (.fas.gz) file. The following operations are performed:
    - If 'tokens' is provided, encodes the sequences in numeric format.
    - If 'filter_sequences' is True, removes the sequences whose tokens are not present in the alphabet.
    - If 'remove_duplicates' is True, removes the duplicated sequences.
    - If 'return_ids' is True, returns also the indices of the original sequences.

    Args:
        fasta_name (str | Path): Path to the fasta or compressed fasta (.fas.gz) file.
        tokens (str | None, optional): Alphabet to be used for the encoding. If provided, encodes the sequences in numeric format.
        filter_sequences (bool, optional): If True, removes the sequences whose tokens are not present in the alphabet. Defaults to False.
        remove_duplicates (bool, optional): If True, removes the duplicated sequences. Defaults to False.
        return_ids (bool, optional): If True, returns also the indices of the original sequences. Defaults to False.

    Raises:
        RuntimeError: The file is not in fasta format.

    Returns:
        Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of (headers, sequences) or (headers, sequences, original_ids) if 'return_ids' is True.
    """
    # Open the file, handling both .fasta and .fas.gz formats
    if str(fasta_name).endswith(".gz"):
        with gzip.open(fasta_name, "rt") as fasta_file: 
            records = list(SeqIO.parse(fasta_file, "fasta"))
    else:
        with open(fasta_name, "r") as fasta_file:
            records = list(SeqIO.parse(fasta_file, "fasta"))

    # Import headers and sequences
    sequences = []
    names = []
    for record in records:
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
    headers: list | np.ndarray | torch.Tensor,
    sequences: list | np.ndarray | torch.Tensor,
    remove_gaps: bool = False,
    tokens: str = "protein",
):
    """Generate a fasta file with the input sequences.

    Args:
        fname (str): Name of the output fasta file.
        headers (list | np.ndarray | torch.Tensor): Iterable with sequences' headers.
        sequences (list | np.ndarray | torch.Tensor): Iterable with sequences in string, categorical or one-hot encoded format.
        remove_gaps (bool, optional): If True, removes the gap from the alignment. Defaults to False.
        tokens (str): Alphabet to be used for the encoding. Defaults to protein.
    """
    if isinstance(headers, torch.Tensor):
        headers = headers.cpu().numpy()
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.cpu().numpy()
    if isinstance(headers, list):
        headers = np.array(headers)
    if isinstance(sequences, list):
        sequences = np.array(sequences)
    tokens = get_tokens(tokens)
    
    # Handle the case when the sequenes are one-hot encoded
    if len(sequences.shape) == 3:
        assert sequences.shape[2] == len(tokens), "The last dimension of the input one-hot encoded sequence must be equal to the length of the alphabet."
        sequences = np.argmax(sequences, axis=2)
        seqs_decoded = decode_sequence(sequences, tokens)
    else:
        # Handle the case when the sequences are in categorical or string format
        if np.issubdtype(sequences.dtype, np.integer) or np.issubdtype(sequences.dtype, np.floating):
            seqs_decoded = decode_sequence(sequences, tokens)
        elif np.issubdtype(sequences.dtype, np.str_):
            seqs_decoded = sequences.copy()
        else:
            raise ValueError("Input sequences must be either in string or numeric format.")
        
    if remove_gaps:
        seqs_decoded = np.vectorize(lambda s: s.replace("-", ""))(seqs_decoded)
        
    with open(fname, 'w') as f:
        for name_seq, seq in zip(headers, seqs_decoded):
            f.write('>' + name_seq + '\n')
            f.write(seq)
            f.write('\n')
            

@torch.jit.script
def _get_sequence_weight(s: torch.Tensor, data: torch.Tensor, L: int, th: float):
    seq_id = torch.sum(s == data, dim=1) / L
    n_clust = torch.sum(seq_id > th)
    
    return 1.0 / n_clust


def compute_weights(
    data: np.ndarray | torch.Tensor,
    th: float = 0.8,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Computes the weight to be assigned to each sequence 's' in 'data' as 1 / n_clust, where 'n_clust' is the number of sequences
    that have a sequence identity with 's' >= th.

    Args:
        data (np.ndarray | torch.Tensor): Input dataset. Must be either a 2D or a 3D (one-hot encoded) array.
        th (float, optional): Sequence identity threshold for the clustering. Defaults to 0.8.
        device (torch.device, optional): Device. Defaults to "cpu".
        dtype (torch.dtype, optional): Data type. Defaults to torch.float32.

    Returns:
        torch.Tensor: Array with the weights of the sequences.
    """
    assert len(data.shape) == 2 or len(data.shape) == 3, "'data' must be either a 2D or a 3D array"
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
        KeyError: The chosen alphabet is incompatible with the Multi-Sequence Alignment.
    """
    all_char = "".join(sequences)
    tokens_data = "".join(sorted(set(all_char)))
    sorted_tokens = "".join(sorted(tokens))
    if not sorted_tokens == tokens_data:
        raise KeyError(f"The chosen alphabet is incompatible with the Multi-Sequence Alignment. The missing tokens are: {[c for c in tokens_data if c not in sorted_tokens]}. Current alphabet: {tokens}")