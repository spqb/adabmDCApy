from pathlib import Path
import numpy as np
from typing import Union, Tuple

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
    
    
def encode_sequence(sequence: Union[str, np.ndarray], tokens: str) -> np.ndarray:
    """Encodes a sequence or a list of sequences into a numeric format.

    Args:
        sequence (Union[str, Array]): Input sequence.
        tokens (str): Alphabet to be used for the encoding.

    Returns:
        list: Encoded sequence.
    """
    letter_map = {l : n for n, l in enumerate(tokens)}
    if isinstance(sequence, str):
        return np.array([letter_map[l] for l in sequence])
    elif isinstance(sequence, np.ndarray):
        return np.vectorize(encode_sequence, excluded=["tokens"], signature="(), () -> (n)")(sequence, tokens)
    elif isinstance(sequence, list):
        sequence = np.array(sequence)
        return encode_sequence(sequence, tokens)
    else:        
        raise ValueError("Input sequence must be either a string or a numpy array.")


def decode_sequence(sequence: np.ndarray, tokens: str) -> str:
    """Takes a numeric sequence or list of seqences in input an returns the corresponding string encoding.

    Args:
        sequence (np.ndarray): Input sequences. Can be either a 1D or a 2D array.
        tokens (str): Alphabet to be used for the encoding.

    Returns:
        list: Decoded input.
    """
    if sequence.ndim == 1:
        return ''.join([tokens[aa] for aa in sequence])
    elif sequence.ndim == 2:
        return np.vectorize(decode_sequence, signature="(m), () -> ()")(sequence, tokens)
    else:
        raise ValueError("Input sequence must be either a 1D or a 2D array.")


def import_from_fasta(fasta_name: Union[str, Path], tokens: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """Import data from a fasta file.

    Args:
        fasta_name (Union[str, Path]): Path to the fasta file.
        tokens (str): Alphabet to be used for the encoding. If provided, encodes the sequences in numeric format.

    Raises:
        RuntimeError: The file is not in fasta format.

    Returns:
        Tuple[np.ndarray, np.ndarray]: headers, sequences.
    """
    sequences = []
    names = []
    seq = ''
    with open(fasta_name, 'r') as f:
        first_line = f.readline()
        if not first_line.startswith('>'):
            raise RuntimeError(f"The file {fasta_name} is not in a fasta format.")
        f.seek(0)
        for line in f:
            if not line.strip():
                continue
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                header = line[1:].strip()
                names.append(header)
                seq = ''
            else:
                seq += line.strip()
    if seq:
        sequences.append(seq)
    
    if tokens is not None:
        sequences = encode_sequence(sequences, tokens)
    
    return np.array(names), np.array(sequences)
    
    
def write_fasta(
    fname: str,
    headers: np.ndarray,
    sequences: np.ndarray,
    numeric_input: bool = False,
    remove_gaps: bool = False,
    alphabet: str = "protein"
):
    """Generate a fasta file with the input sequences.

    Args:
        fname (str): Name of the output fasta file.
        headers (np.ndarray): Array of sequences' headers.
        sequences (np.ndarray): Array of sequences.
        numeric_input (bool, optional): Whether the sequences are in numeric (encoded) format or not. Defaults to False.
        remove_gaps (bool, optional): If True, removes the gap from the alignment. Defaults to False.
        tokens (str): Alphabet to be used for the encoding. Defaults to protein.
    """
    tokens = get_tokens(alphabet)

    if numeric_input:
        # Decode the sequences
        seqs_decoded = decode_sequence(sequences, tokens)
    else:
        seqs_decoded = sequences.copy()
    if remove_gaps:
        seqs_decoded = np.vectorize(lambda s: s.replace("-", ""))(seqs_decoded)
        
    with open(fname, 'w') as f:
        for name_seq, seq in zip(headers, seqs_decoded):
            f.write('>' + name_seq + '\n')
            f.write(seq)
            f.write('\n')
         
                
def import_clean_dataset(filein: str, tokens: str = "protein") -> Tuple[np.ndarray, np.ndarray]:
    """Imports data from a fasta file and removes all the sequences whose tokens are not present in a specified alphabet.

    Args:
        filein (str): Input fasta.
        tokens (str, optional): Alphabet to be used for the encoding. Defaults to "protein".
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: headers, sequences.
    """
    tokens = get_tokens(tokens)
    names, sequences = import_from_fasta(filein)
    tokens_list = [a for a in tokens]
    clean_names = []
    clean_sequences = []
    for n, s in zip(names, sequences):
        good_sequence = np.full(shape=(len(s),), fill_value=False)
        splitline = np.array([a for a in s])
        for token in tokens_list:
            good_sequence += (token == splitline)
        if np.all(good_sequence):
            if n == "":
                n = "unknown_sequence"
            clean_names.append(n)
            clean_sequences.append(s)
        else:
            print(f"Unknown token found: removing sequence {n}")
            
    return clean_names, clean_sequences


def compute_weights(
    data: np.ndarray | torch.Tensor,
    th: float = 0.8,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Computes the weight to be assigned to each sequence 's' in 'data' as 1 / n_clust, where 'n_clust' is the number of sequences
    that have a sequence identity with 's' >= th.

    Args:
        data (np.ndarray | torch.Tensor): Encoded input dataset.
        th (float, optional): Sequence identity threshold for the clustering. Defaults to 0.8.
        device (toch.device, optional): Device. Defaults to "cpu".

    Returns:
        torch.Tensor: Array with the weights of the sequences.
    """
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    else:
        data = torch.tensor(data, device=device)

    assert len(data.shape) == 2, "'data' must be a 2-dimensional array"
    _, L = data.shape

    @torch.jit.script
    def get_sequence_weight(s: torch.Tensor, data: torch.Tensor, L: int, th: float):
        seq_id = torch.sum(s == data, dim=1) / L
        n_clust = torch.sum(seq_id >= th)
        
        return 1.0 / n_clust

    weights = torch.vstack([get_sequence_weight(s, data, L, th) for s in data])
    
    return weights


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