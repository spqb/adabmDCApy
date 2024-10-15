from typing import Union, Any
from pathlib import Path
import numpy as np

from torch.utils.data import Dataset
import torch

from adabmDCA.fasta_utils import (
    get_tokens,
    import_clean_dataset,
    encode_sequence,
    compute_weights,
)


class DatasetDCA(Dataset):
    
    def __init__(
        self,
        path_data: Union[str, Path],
        path_weights: Union[str, Path] = None,
        alphabet: str = "protein",
        device: str = "cuda",
    ):
        """Initialize the dataset.

        Args:
            path_data (Union[str, Path]): Path to multi sequence alignment in fasta format.
            path_weights (Union[str, Path], optional): Path to the file containing the importance weights of the sequences. If None, the weights are computed automatically.
            alphabet (str, optional): Selects the type of encoding of the sequences. Default choices are ("protein", "rna", "dna"). Defaults to "protein".
            device (str, optional): Device to be used. Choose among ['cpu', 'cuda']. Defaults to "cuda".
        """
        path_data = Path(path_data)
        self.names = []
        self.data = []
        
        # Select the proper encoding
        self.tokens = get_tokens(alphabet)
        
        # Automatically detects if the file is in fasta format and imports the data
        with open(path_data, "r") as f:
            first_line = f.readline()
        if first_line.startswith(">"):
            names, sequences = import_clean_dataset(path_data, tokens=self.tokens)
            # Check if data is empty
            if len(sequences) == 0:
                raise ValueError(f"The input dataset is empty. Check that the alphabet is correct. Current alphabet: {alphabet}")
            self.names = np.array(names)
            self.data = encode_sequence(sequences, tokens=self.tokens)
        else:
            raise KeyError("The input dataset is not in fasta format")
        
        # Computes the weights to be assigned to the data
        if path_weights is None:
            print("Automatically computing the sequence weights...")
            self.weights = compute_weights(data=self.data, th=0.8, device=device)
            
        else:
            with open(path_weights, "r") as f:
                weights = [float(line.strip()) for line in f]
            self.weights = torch.tensor(weights, device=device)
        
        print(f"Multi-sequence alignment imported: M = {self.data.shape[0]}, L = {self.data.shape[1]}, q = {self.get_num_states()}, M_eff = {int(self.weights.sum())}.")


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx: int) -> Any:
        sample = self.data[idx]
        weight = self.weights[idx]
        return (sample, weight)
    
    
    def get_num_residues(self) -> int:
        return self.data.shape[1]
    
    
    def get_num_states(self) -> int:
        return np.max(self.data) + 1
    
    
    def get_effective_size(self) -> int:
        return int(self.weights.sum())
    
    
    def shuffle(self) -> None:
        """Shuffles the dataset.
        """
        perm = torch.randperm(len(self.data))
        self.data = self.data[perm]
        self.names = self.names[perm]
        self.weights = self.weights[perm]