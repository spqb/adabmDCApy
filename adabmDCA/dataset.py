from typing import Optional, Tuple
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
    """Dataset class for handling multi-sequence alignments data.
    
     Args:
        path_data (str): Path to multi sequence alignment in fasta format.
        path_weights (Optional[str], optional): Path to the file containing the importance weights of the sequences. If None, the weights are computed automatically.
        alphabet (str, optional): Selects the type of encoding of the sequences. Default choices are ("protein", "rna", "dna"). Defaults to "protein".
        clustering_th (float, optional): Sequence identity threshold for clustering. Defaults to 0.8.
        no_reweighting (bool, optional): If True, the weights are not computed. Defaults to False.
        remove_duplicates (bool, optional): If True, removes duplicate sequences from the dataset. Defaults to False.
        filter_sequences (bool, optional): If True, removes sequences containing tokens not in the alphabet. Defaults to False.
        message (bool, optional): Print the import message. Defaults to True.
        device (torch.device, optional): Device to be used. Defaults to "cpu".
        dtype (torch.dtype, optional): Data type of the dataset. Defaults to torch.float32.
    """
    
    def __init__(
        self,
        path_data: str,
        path_weights: Optional[str] = None,
        alphabet: str = "protein",
        clustering_th: float = 0.8,
        no_reweighting: bool = False,
        remove_duplicates: bool = False,
        filter_sequences: bool = False,
        message: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.names = np.array([], dtype=str)
        self.data = torch.tensor([], device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        
        # Select the proper encoding
        self.tokens = get_tokens(alphabet)
        
        # Automatically detects if the file is in fasta format and imports the data
        with open(path_data, "r") as f:
            first_line = f.readline()
        if not first_line:
            raise ValueError(f"The input file is empty: {path_data}")
        if first_line.startswith(">"):
            self.names, data_enc, mask = import_from_fasta(
                path_data,
                tokens=self.tokens,
                filter_sequences=filter_sequences,
                remove_duplicates=remove_duplicates,
                return_mask=True,
            )
            self.data = torch.tensor(data_enc, dtype=torch.int64, device=device)
            if len(self.data) == 0:
                raise ValueError(f"The input dataset is empty. Check that the alphabet is correct. Current alphabet: {alphabet}")
        else:
            raise ValueError(f"The input file is not in fasta format: {path_data}")
        
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
            if len(weights) == len(self.data):
                self.weights = weights
            elif len(weights) == len(mask):
                self.weights = weights[mask]
            else:
                raise ValueError(f"The number of weights ({len(weights)}) does not match the neither number of sequences in the fasta file ({len(mask)}), nor the number of sequences in the dataset after filtering ({len(self.data)}).")
        
        if message:
            print(f"Multi-sequence alignment imported: M = {self.data.shape[0]}, L = {self.data.shape[1]}, q = {len(self.tokens)}, M_eff = {int(self.weights.sum())}.")


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return len(self.tokens)
    
    
    def get_effective_size(self) -> int:
        """Returns the effective size (Meff) of the dataset.

        Returns:
            int: Effective size of the dataset.
        """
        return int(self.weights.sum())
    
    
    def shuffle(self) -> None:
        """Shuffles the dataset.
        """
        perm = torch.randperm(len(self.data), device=self.device)
        self.data = self.data[perm]
        self.names = self.names[perm.cpu().numpy()]
        self.weights = self.weights[perm]
        
        
    def to_one_hot(self) -> torch.Tensor:
        """Converts the dataset to one-hot encoding.

        Returns:
            torch.Tensor: One-hot encoded dataset of shape (M, L, q).
        """
        q = len(self.tokens)
        return one_hot(self.data.long(), num_classes=q).to(dtype=self.dtype, device=self.device)
    
    
    def get_frequencies(self, pseudocount: float = 0.0, batch_size: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the single-site and two-site frequencies of the dataset. When there are too many sequences, computing the frequencies directly from the one-hot encoding can be memory-intensive.
        Therefore, we compute the frequencies using batched operations.
        
        Args:
            pseudocount (float, optional): Pseudocount to be added to the frequencies. Defaults to 0.0.
            batch_size (int, optional): Batch size to use when computing the frequencies. Defaults to 10000.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Single-site frequencies fi of shape (L, q) and two-site frequencies fij of shape (L, q, L, q).
        """
        L, q = self.get_num_residues(), self.get_num_states()
        # 
        fi = torch.zeros((L, q), device=self.device, dtype=self.dtype)
        fij = torch.zeros((L, q, L, q), device=self.device, dtype=self.dtype)
        num_batches = (len(self.data) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch_data = self.data[i*batch_size : (i+1)*batch_size]
            batch_weights = self.weights[i*batch_size : (i+1)*batch_size]
            batch_one_hot = one_hot(batch_data.long(), num_classes=q).to(dtype=self.dtype, device=self.device)
            fi += (batch_one_hot * batch_weights.reshape(-1, 1, 1)).sum(dim=0)
            fij += torch.einsum("mia,mjb->iajb", batch_one_hot * batch_weights.reshape(-1, 1, 1), batch_one_hot)
        fi /= self.weights.sum()
        fij /= self.weights.sum()
        torch.clamp_(fi, min=0.0)
        torch.clamp_(fij, min=0.0)
        # Add pseudocount
        if pseudocount > 0.0:
            fi = (1 - pseudocount) * fi + pseudocount / q
            fij = (1 - pseudocount) * fij + pseudocount / (q * q)
        # Set diagonal elements fij[i, a, i, b] = fi[i, a] * delta[a, b]
        for i in range(L):
            fij[i, :, i, :] = torch.diag(fi[i, :])
            
        return fi, fij
                