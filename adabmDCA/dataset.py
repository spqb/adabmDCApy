from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from adabmDCA.alignment import Alignment
from adabmDCA.input_loading import (
    AlignmentLoadConfig,
    WeightInput,
    load_alignment,
    load_sequence_weights,
)

CPU_DEVICE = torch.device("cpu")


class DatasetDCA(Dataset):
    """Encoded alignment with sequence weights, as a PyTorch dataset.

    Build one with :meth:`from_alignment`. Items are ``(sequence, weight)`` pairs
    of encoded sequences.

    Attributes:
        names: Sequence names.
        data: Encoded sequences, shape ``(M, L)``, integer states.
        weights: Weight of each sequence.
        tokens: Ordered alphabet.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "DatasetDCA cannot be built from a path; use DatasetDCA.from_alignment(), "
            "or the high-level train_model/sample_sequences/predict_contacts APIs."
        )

    def _initialize(
        self,
        loaded: Alignment,
        weights: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.names = np.asarray(loaded.names, dtype=str)
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
        """Load an alignment and its weights into a dataset.

        An ``Alignment`` already returned by :func:`load_alignment` is reused
        when ``load_config`` is omitted, preserving its filtering provenance.
        Pass ``load_config`` to apply a new loading policy.

        Args:
            alignment: Path, :class:`Alignment` or sequences.
            weights: Sequence weights (file, array or tensor), or ``None`` to compute them.
            load_config: Loading policy; see :class:`AlignmentLoadConfig`.
            clustering_th: Identity threshold of the computed weights.
            no_reweighting: Give every sequence weight 1.
            device: Device of the tensors.
            dtype: Precision of the weights and one-hot encodings.
            allow_signed_weights: Accept negative weights (experimental reintegration).

        Returns:
            A :class:`DatasetDCA`.

        Example:
            >>> dataset = DatasetDCA.from_alignment("family.fasta", load_config=AlignmentLoadConfig(alphabet="rna"))
            >>> fi, fij = dataset.get_frequencies(pseudocount=1 / dataset.get_effective_size())
        """
        loaded = (
            alignment
            if isinstance(alignment, Alignment) and alignment.tokens is not None and load_config is None
            else load_alignment(alignment, config=load_config)
        )
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

    def get_effective_size(self) -> float:
        """Returns the effective size (Meff) of the dataset.

        Returns:
            float: Sum of the sequence weights.
        """
        return float(self.weights.sum().item())

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
