from typing import Any
from pathlib import Path
import numpy as np

from torch.utils.data import Dataset
import torch

from adabmDCA.fasta import (
    get_tokens,
    import_from_fasta,
    compute_weights,
)


class DatasetDCA(Dataset):
    
    def __init__(
        self,
        path_data: str | Path,
        path_weights: str | Path | None = None,
        alphabet: str = "protein",
        clustering_th: float = 0.8,
        no_reweighting: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the dataset.

        Args:
            path_data (str | Path): Path to multi sequence alignment in fasta format.
            path_weights (str | Path | None, optional): Path to the file containing the importance weights of the sequences. If None, the weights are computed automatically.
            alphabet (str, optional): Selects the type of encoding of the sequences. Default choices are ("protein", "rna", "dna"). Defaults to "protein".
            clustering_th (float, optional): Sequence identity threshold for clustering. Defaults to 0.8.
            no_reweighting (bool, optional): If True, the weights are not computed. Defaults to False.
            device (torch.device, optional): Device to be used. Defaults to "cpu".
            dtype (torch.dtype, optional): Data type of the dataset. Defaults to torch.float32.
        """
        path_data = Path(path_data)
        self.names = None
        self.data = None
        self.device = device
        self.dtype = dtype
        
        # Select the proper encoding
        self.tokens = get_tokens(alphabet)
        
        # Automatically detects if the file is in fasta format and imports the data
        with open(path_data, "r") as f:
            first_line = f.readline()
        if first_line.startswith(">"):
            self.names, self.data = import_from_fasta(path_data, tokens=self.tokens, filter_sequences=True)
            self.data = torch.tensor(self.data, device=device, dtype=torch.int32)
            # Check if data is empty
            if len(self.data) == 0:
                raise ValueError(f"The input dataset is empty. Check that the alphabet is correct. Current alphabet: {alphabet}")
        else:
            raise KeyError("The input dataset is not in fasta format")
        
        # Computes the weights to be assigned to the data
        if no_reweighting:
            self.weights = torch.ones(len(self.data), device=device, dtype=dtype)
        elif path_weights is None:
            print("Automatically computing the sequence weights...")
            self.weights = compute_weights(data=self.data, th=clustering_th, device=device, dtype=dtype)
        else:
            with open(path_weights, "r") as f:
                weights = [float(line.strip()) for line in f]
            self.weights = torch.tensor(weights, device=device, dtype=dtype)
        
        print(f"Multi-sequence alignment imported: M = {self.data.shape[0]}, L = {self.data.shape[1]}, q = {self.get_num_states()}, M_eff = {int(self.weights.sum())}.")


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx: int) -> Any:
        sample = self.data[idx]
        weight = self.weights[idx]
        return (sample, weight)
    
    
    def get_num_residues(self) -> int:
        """Returns the number of residues (L) in the multi-sequence alignment.

        Returns:
            int: Length of the MSA.
        """
        return self.data.shape[1]
    
    
    def get_num_states(self) -> int:
        """Returns the number of states (q) in the alphabet.

        Returns:
            int: Number of states.
        """
        return torch.max(self.data).item() + 1
    
    
    def get_effective_size(self) -> int:
        """Returns the effective size (Meff) of the dataset.

        Returns:
            int: Effective size of the dataset.
        """
        return int(self.weights.sum())
    
    
    def shuffle(self) -> None:
        """Shuffles the dataset.
        """
        perm = torch.randperm(len(self.data))
        self.data = self.data[perm]
        self.names = self.names[perm]
        self.weights = self.weights[perm]