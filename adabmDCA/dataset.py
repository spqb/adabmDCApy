import warnings
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from adabmDCA.alignment import Alignment
from adabmDCA.input_loading import (
    AlignmentLoadConfig,
    LoadedAlignment,
    WeightInput,
    load_alignment,
    load_sequence_weights,
)

CPU_DEVICE = torch.device("cpu")


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
        path_data: str | Path | Alignment,
        path_weights: WeightInput | None = None,
        alphabet: str = "protein",
        clustering_th: float = 0.8,
        no_reweighting: bool = False,
        remove_duplicates: bool = False,
        filter_sequences: bool = False,
        message: bool = True,
        device: torch.device = CPU_DEVICE,
        dtype: torch.dtype = torch.float32,
    ):
        warnings.warn(
            "DatasetDCA(path_data=...) is deprecated; use DatasetDCA.from_alignment() "
            "or the high-level train_model/sample_sequences/predict_contacts APIs.",
            DeprecationWarning,
            stacklevel=2,
        )
        loaded = load_alignment(
            path_data,
            config=AlignmentLoadConfig(
                alphabet=alphabet,
                invalid_sequences="drop" if filter_sequences else "error",
                remove_duplicates=remove_duplicates,
            ),
        )
        weights = load_sequence_weights(
            path_weights,
            loaded_alignment=loaded,
            no_reweighting=no_reweighting,
            clustering_seqid=clustering_th,
            device=device,
            dtype=dtype,
        )
        self._initialize(loaded, weights, device=device, dtype=dtype)
        if message:
            print(
                "Multi-sequence alignment imported: "
                f"M = {self.data.shape[0]}, L = {self.data.shape[1]}, "
                f"q = {len(self.tokens)}, M_eff = {int(self.weights.sum())}."
            )

    def _initialize(
        self,
        loaded: LoadedAlignment,
        weights: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.names = np.asarray(loaded.alignment.names, dtype=str)
        self.data = torch.as_tensor(
            loaded.encoded_sequences,
            dtype=torch.int64,
            device=device,
        )
        self.weights = weights.to(device=device, dtype=dtype)
        self.tokens = loaded.tokens
        self.device = device
        self.dtype = dtype
        self.load_result = loaded

    @classmethod
    def from_loaded_alignment(
        cls,
        loaded: LoadedAlignment,
        *,
        weights: WeightInput | None = None,
        clustering_th: float = 0.8,
        no_reweighting: bool = False,
        device: torch.device = CPU_DEVICE,
        dtype: torch.dtype = torch.float32,
        allow_signed_weights: bool = False,
    ) -> "DatasetDCA":
        """Materialize an in-memory dataset from a validated alignment."""
        resolved_weights = load_sequence_weights(
            weights,
            loaded_alignment=loaded,
            no_reweighting=no_reweighting,
            clustering_seqid=clustering_th,
            device=device,
            dtype=dtype,
            allow_negative=allow_signed_weights,
        )
        dataset = cls.__new__(cls)
        dataset._initialize(loaded, resolved_weights, device=device, dtype=dtype)
        return dataset

    @classmethod
    def from_alignment(
        cls,
        alignment: str | Path | Alignment,
        *,
        weights: WeightInput | None = None,
        load_config: AlignmentLoadConfig | None = None,
        clustering_th: float = 0.8,
        no_reweighting: bool = False,
        device: torch.device = CPU_DEVICE,
        dtype: torch.dtype = torch.float32,
        allow_signed_weights: bool = False,
    ) -> "DatasetDCA":
        """Load and materialize an alignment without constructor-side policy."""
        loaded = load_alignment(alignment, config=load_config)
        return cls.from_loaded_alignment(
            loaded,
            weights=weights,
            clustering_th=clustering_th,
            no_reweighting=no_reweighting,
            device=device,
            dtype=dtype,
            allow_signed_weights=allow_signed_weights,
        )

    @classmethod
    def from_path(cls, path: str | Path, **kwargs) -> "DatasetDCA":
        """Compatibility factory mirroring :meth:`from_alignment`.

        .. deprecated:: 0.7.8
           Use :meth:`from_alignment` instead.
        """
        warnings.warn(
            "DatasetDCA.from_path() is deprecated; use DatasetDCA.from_alignment().",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_alignment(path, **kwargs)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        """Shuffles the dataset."""
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

    def get_frequencies(self, pseudocount: float = 0.0, batch_size: int = 10000) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the single-site and two-site frequencies of the dataset. When there are too many sequences, computing the frequencies directly from the one-hot encoding can be memory-intensive.
        Therefore, we compute the frequencies using batched operations.

        Args:
            pseudocount (float, optional): Pseudocount to be added to the frequencies. Defaults to 0.0.
            batch_size (int, optional): Batch size to use when computing the frequencies. Defaults to 10000.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Single-site frequencies fi of shape (L, q) and two-site frequencies fij of shape (L, q, L, q).
        """
        L, q = self.get_num_residues(), self.get_num_states()
        fi = torch.zeros((L, q), device=self.device, dtype=self.dtype)
        fij = torch.zeros((L, q, L, q), device=self.device, dtype=self.dtype)
        num_batches = (len(self.data) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch_data = self.data[i * batch_size : (i + 1) * batch_size]
            batch_weights = self.weights[i * batch_size : (i + 1) * batch_size]
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
