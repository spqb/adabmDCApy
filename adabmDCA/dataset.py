from typing import Any
import numpy as np

from torch.utils.data import Dataset
import torch
from torch.nn.functional import one_hot

from adabmDCA.fasta import (
    get_tokens,
    import_from_fasta,
    compute_weights,
)


class DatasetDCA(Dataset):
    """Dataset class for handling multi-sequence alignments data."""
    def __init__(
        self,
        path_data: str,
        path_weights: str | None = None,
        alphabet: str = "protein",
        clustering_th: float = 0.8,
        no_reweighting: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        remove_duplicates: bool = False,
        filter_sequences: bool = False,
        message: bool = True,
    ):
        """Initialize the dataset.

        Args:
            path_data (str): Path to multi sequence alignment in fasta format.
            path_weights (str | None, optional): Path to the file containing the importance weights of the sequences. If None, the weights are computed automatically.
            alphabet (str, optional): Selects the type of encoding of the sequences. Default choices are ("protein", "rna", "dna"). Defaults to "protein".
            clustering_th (float, optional): Sequence identity threshold for clustering. Defaults to 0.8.
            no_reweighting (bool, optional): If True, the weights are not computed. Defaults to False.
            device (torch.device, optional): Device to be used. Defaults to "cpu".
            dtype (torch.dtype, optional): Data type of the dataset. Defaults to torch.float32.
            remove_duplicates (bool, optional): If True, removes duplicate sequences from the dataset. Defaults to False.
            filter_sequences (bool, optional): If True, removes sequences containing tokens not in the alphabet. Defaults to False.
            message (bool, optional): Print the import message. Defaults to True.
        """
        self.names = np.array([], dtype=str)
        self.data = torch.tensor([], device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        
        # Select the proper encoding
        self.tokens = get_tokens(alphabet)
        
        # Automatically detects if the file is in fasta format and imports the data
        with open(path_data, "r") as f:
            first_line = f.readline()
        if first_line.startswith(">"):
            self.names, data_enc, mask = import_from_fasta(
                path_data,
                tokens=self.tokens,
                filter_sequences=filter_sequences,
                remove_duplicates=remove_duplicates,
                return_mask=True,
            )
            data_enc = torch.tensor(data_enc, dtype=torch.int64)
            self.data = one_hot(data_enc, num_classes=len(self.tokens)).to(device=device, dtype=dtype)
            # Check if data is empty
            if len(self.data) == 0:
                raise ValueError(f"The input dataset is empty. Check that the alphabet is correct. Current alphabet: {alphabet}")
        else:
            raise KeyError("The input dataset is not in fasta format")
        
        # Computes the weights to be assigned to the data
        if no_reweighting:
            self.weights = torch.ones(len(self.data), device=device, dtype=dtype)
        elif path_weights is None:
            if message:
                print("Automatically computing the sequence weights...")
            self.weights = compute_weights(data=self.data, th=clustering_th, device=device, dtype=dtype)
        else:
            with open(path_weights, "r") as f:
                weights = [float(line.strip()) for line in f]
            weights = torch.tensor(weights, device=device, dtype=dtype)
            assert len(weights) == len(mask), f"The number of weights ({len(weights)}) does not match the number of sequences in the dataset ({len(mask)})."
            self.weights = weights[mask]
            
        
        if message:
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
        return self.data.shape[2]
    
    
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