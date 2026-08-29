from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot

from adabmDCA.alignment import Alignment, read_alignment, write_alignment
from adabmDCA.alphabet import get_tokens
from adabmDCA.fasta import decode_sequence, encode_sequence, validate_alphabet
from adabmDCA.utils import get_mask_save


def load_chains(
    fname: str,
    tokens: str,
    load_weights: bool = False,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, ...]:
    """Loads the sequences from a fasta file and returns the one-hot encoded version.
    If the sequences are weighted, the log-weights are also returned. If the sequences are not weighted, the log-weights are set to 0.
    
    Args:
        fname (str): Path to the file containing the sequences.
        tokens (str): "protein", "dna", "rna" or another string with the alphabet to be used.
        load_weights (bool, optional): If True, the log-weights are loaded and returned. Defaults to False.
        device (torch.device, optional): Device where to store the sequences. Defaults to "cpu".
        dtype (torch.dtype, optional): Data type of the sequences. Defaults to torch.float32
    
    Return:
        Tuple[torch.Tensor, ...]: One-hot encoded sequences and log-weights if load_weights is True.
    """
    def parse_header(header: str):
        h = header.split("|")
        if len(h) == 2:
            log_weight = float(h[1].split("=")[1])
            return log_weight
        else:
            return 0.0
    
    alignment = read_alignment(fname, format="fasta")
    headers = np.asarray(alignment.names, dtype=str)
    sequences = np.asarray(alignment.sequences, dtype=str)
    validate_alphabet(sequences, tokens=tokens)
    encoded_sequences = encode_sequence(sequences, tokens=tokens)
    encoded_sequences = torch.tensor(encoded_sequences, dtype=torch.int64)
    sequences_oh = one_hot(encoded_sequences, num_classes=len(tokens)).to(device=device, dtype=dtype)
    
    if load_weights:
        log_weights = np.vectorize(parse_header)(headers)
        log_weights = torch.tensor(log_weights, device=device, dtype=dtype)
        return (sequences_oh, log_weights)
    else:
        return (sequences_oh,)


def save_chains(
    fname: str,
    chains: Union[list, np.ndarray, torch.Tensor],
    tokens: str,
    log_weights: Union[torch.Tensor, np.ndarray, None] = None
) -> None:
    """Saves the chains in a fasta file.

    Args:
        fname (str): Path to the file where to save the chains.
        chains (Union[list, np.ndarray, torch.Tensor]): Iterable with sequences in string, categorical or one-hot encoded format.
        tokens (str): "protein", "dna", "rna" or another string with the alphabet to be used.
        log_weights (Union[torch.Tensor, np.ndarray, None], optional): Log-weights of the chains. Defaults to None.
    """
    if log_weights is not None:
        if isinstance(log_weights, torch.Tensor):
            log_weights = log_weights.cpu().numpy()
        headers = [f"chain_{i}|log_weight={log_weights[i]}" for i in range(len(chains))]
    else:
        headers = [f"chain_{i}" for i in range(len(chains))]
    resolved_tokens = get_tokens(tokens)
    if isinstance(chains, torch.Tensor):
        chains = chains.detach().cpu().numpy()
    decoded = decode_sequence(np.asarray(chains), resolved_tokens)
    if isinstance(decoded, str):
        decoded = [decoded]
    write_alignment(
        Alignment(
            names=tuple(headers),
            sequences=tuple(str(sequence) for sequence in decoded),
        ),
        fname,
        format="fasta",
    )


def load_params(
    fname: str,
    tokens: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Import parameters from the established ``J``/``h`` text format.

    The file is parsed in two streaming passes so memory use is bounded by the
    final tensors and a small coupling chunk. Files containing one or both
    coupling triangles are supported.

    Args:
        fname (str): Path of the file that stores the parameters.
        tokens (str): "protein", "dna", "rna" or another string with a compatible alphabet to be used.
        device (torch.device): Device where to store the parameters.
        dtype (torch.dtype): Data type of the parameters. Defaults to torch.float32.

    Returns:
        Dict[str, torch.Tensor]: Parameters of the model.
            - "bias": Tensor of shape (L, q) - local biases.
            - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
    """
    tokens = get_tokens(tokens)
    token_to_idx = {token: idx for idx, token in enumerate(tokens)}
    q = len(tokens)
    max_position = -1
    num_biases = 0

    # First pass: validate the compact text records and determine L without
    # retaining the file or millions of Python objects in memory.
    with open(fname, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            parts = raw_line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "J":
                if len(parts) != 6:
                    raise ValueError(f"Malformed coupling record on line {line_number}.")
                idx0, idx1 = int(parts[1]), int(parts[2])
                if idx0 < 0 or idx1 < 0:
                    raise ValueError(f"Parameter positions cannot be negative (line {line_number}).")
                if parts[3] not in token_to_idx or parts[4] not in token_to_idx:
                    raise ValueError(f"Unknown coupling token on line {line_number}.")
                max_position = max(max_position, idx0, idx1)
            elif parts[0] == "h":
                if len(parts) != 4:
                    raise ValueError(f"Malformed bias record on line {line_number}.")
                idx0 = int(parts[1])
                if idx0 < 0:
                    raise ValueError(f"Parameter positions cannot be negative (line {line_number}).")
                if parts[2] not in token_to_idx:
                    raise ValueError(f"Unknown bias token on line {line_number}.")
                max_position = max(max_position, idx0)
                num_biases += 1

    if num_biases == 0 or max_position < 0:
        raise ValueError("The parameter file contains no bias records.")

    L = max_position + 1
    numpy_dtype = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
    }.get(dtype, np.float32)
    h = np.zeros((L, q), dtype=numpy_dtype)
    J = np.zeros((L, q, L, q), dtype=numpy_dtype)

    chunk_size = 65_536
    j_idx0: list[int] = []
    j_idx1: list[int] = []
    j_idx2: list[int] = []
    j_idx3: list[int] = []
    j_values: list[float] = []

    def flush_couplings() -> None:
        if not j_values:
            return
        J[
            np.asarray(j_idx0, dtype=np.intp),
            np.asarray(j_idx2, dtype=np.intp),
            np.asarray(j_idx1, dtype=np.intp),
            np.asarray(j_idx3, dtype=np.intp),
        ] = np.asarray(j_values, dtype=numpy_dtype)
        j_idx0.clear()
        j_idx1.clear()
        j_idx2.clear()
        j_idx3.clear()
        j_values.clear()

    # Second pass: populate the final memory layout in bounded chunks.
    with open(fname, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            parts = raw_line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "J":
                j_idx0.append(int(parts[1]))
                j_idx1.append(int(parts[2]))
                j_idx2.append(token_to_idx[parts[3]])
                j_idx3.append(token_to_idx[parts[4]])
                try:
                    j_values.append(float(parts[5]))
                except ValueError as exc:
                    raise ValueError(f"Invalid coupling value on line {line_number}.") from exc
                if len(j_values) >= chunk_size:
                    flush_couplings()
            elif parts[0] == "h":
                try:
                    h[int(parts[1]), token_to_idx[parts[2]]] = float(parts[3])
                except ValueError as exc:
                    raise ValueError(f"Invalid bias value on line {line_number}.") from exc
    flush_couplings()

    # Preserve the historical behavior for files containing either one or
    # both coupling triangles, while using only q×q temporary blocks.
    for idx0 in range(L):
        diagonal = J[idx0, :, idx0, :].copy()
        J[idx0, :, idx0, :] = diagonal + diagonal.T
        for idx1 in range(idx0 + 1, L):
            block = J[idx0, :, idx1, :] + J[idx1, :, idx0, :].T
            J[idx0, :, idx1, :] = block
            J[idx1, :, idx0, :] = block.T

    bias = torch.from_numpy(h)
    couplings = torch.from_numpy(J)
    return {
        "bias": bias.to(device=device, dtype=dtype),
        "coupling_matrix": couplings.to(device=device, dtype=dtype),
    }
    
    
def load_params_old(
    fname: str,
    tokens: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Import the parameters of the model from a file.

    Args:
        fname (str): Path of the file that stores the parameters.
        tokens (str): "protein", "dna", "rna" or another string with a compatible alphabet to be used.
        device (torch.device): Device where to store the parameters.
        dtype (torch.dtype): Data type of the parameters. Defaults to torch.float32.

    Returns:
        Dict[str, torch.Tensor]: Parameters of the model.
            - "bias": Tensor of shape (L, q) - local biases.
            - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
    """
    # deprecation warning
    import warnings
    warnings.warn(
        "load_params_old is deprecated and will be removed in a future version. "
        "Please use load_params instead.",
        DeprecationWarning
    )
    
    param_labels = pd.read_csv(fname, sep=" ", usecols=[0,]).to_numpy()
    skiprows = (param_labels == "J").sum() + 1
    skipfooter = len(param_labels) - skiprows + 1

    df_J = pd.read_csv(
        fname,
        sep=" ",
        names=["param", "idx0", "idx1", "idx2", "idx3", "val"],
        skipfooter=skipfooter,
        engine="python"
    ).astype({"idx0" : int, "idx1" : int, "idx2" : str, "idx3" : str, "val" : float})
    
    df_h = pd.read_csv(
        fname,
        sep=" ",
        names=["param", "idx0", "idx1", "val"],
        skiprows=skiprows
    ).astype({"idx0" : int, "idx1" : str, "val" : float})
    
    # Convert from amino acid format to numeric format
    tokens = get_tokens(tokens)
    validate_alphabet(df_h["idx1"].to_numpy(), tokens=tokens)
    df_J["idx2"] = encode_sequence(df_J["idx2"].to_numpy(), tokens=tokens)
    df_J["idx3"] = encode_sequence(df_J["idx3"].to_numpy(), tokens=tokens)
    df_h["idx1"] = encode_sequence(df_h["idx1"].to_numpy(), tokens=tokens)
    

    h_idx = df_h.loc[:, ["idx0", "idx1"]].to_numpy()
    L, q = h_idx.max(0) + 1
    h_val = df_h.loc[:, "val"].to_numpy()

    h = np.zeros(shape=(L * q,))
    h_idx_flat = h_idx @ np.array([q, 1])
    h[h_idx_flat] = h_val
    h = h.reshape(L, q)
    
    J_idx = df_J.loc[:, ["idx0", "idx1", "idx2", "idx3"]].to_numpy()
    J_val = df_J.loc[:, "val"].to_numpy()

    J = np.zeros(shape=(L**2 * q**2,))
    J_idx_flat = J_idx @ np.array([L * q**2, q**2, q, 1])
    J[J_idx_flat] = J_val
    
    # Only the upper-triangular part of J is filled
    J = J.reshape(L, L, q, q).transpose(0, 2, 1, 3).reshape(L * q, L * q)
    J = (J + J.T).reshape(L, q, L, q)

    return {
        "bias" : torch.tensor(h, dtype=dtype, device=device),
        "coupling_matrix" : torch.tensor(J, dtype=dtype, device=device),
        }


def save_params(
    fname: str,
    params: Dict[str, torch.Tensor],
    tokens: str,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """Save parameters in the established ``J``/``h`` text format.

    Couplings are streamed in bounded chunks using the canonical ``i < j``
    triangle. A supplied symmetric mask is collapsed onto that triangle.

    Args:
        fname (str): Path to the file where to save the parameters.
        params (Dict[str, torch.Tensor]): Parameters of the model.
            - "bias": Tensor of shape (L, q) - local biases.
            - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
        tokens (str): "protein", "dna", "rna" or another string with a compatible alphabet to be used.
        mask (Optional[torch.Tensor]): Tensor of shape (L, q, L, q) - Mask of the coupling matrix that determines which are the non-zero entries.
            If None, the lower-triangular part of the coupling matrix is masked. Defaults to None.
    """
    tokens = get_tokens(tokens)
    if "bias" not in params or "coupling_matrix" not in params:
        raise ValueError("params must contain 'bias' and 'coupling_matrix'.")
    bias = params["bias"].detach().cpu()
    couplings = params["coupling_matrix"].detach().cpu()
    if bias.ndim != 2:
        raise ValueError("params['bias'] must have shape (L, q).")
    L, q = bias.shape
    if len(tokens) != q:
        raise ValueError(f"The alphabet has {len(tokens)} states but the parameters have {q} states.")
    if tuple(couplings.shape) != (L, q, L, q):
        raise ValueError("params['coupling_matrix'] must have shape (L, q, L, q).")

    # NumPy does not expose every torch dtype (notably bfloat16).
    try:
        bias_np = bias.numpy()
        couplings_np = couplings.numpy()
    except TypeError:
        bias_np = bias.float().numpy()
        couplings_np = couplings.float().numpy()

    mask_np = None
    if mask is not None:
        if tuple(mask.shape) != (L, q, L, q):
            raise ValueError("mask must have shape (L, q, L, q).")
        mask_np = mask.detach().to(device="cpu", dtype=torch.bool).numpy()

    state0_all = np.repeat(np.arange(q), q)
    state1_all = np.tile(np.arange(q), q)
    buffer: list[str] = []
    chunk_size = 65_536

    with open(fname, "w", encoding="utf-8", buffering=1024 * 1024) as handle:
        def flush_lines() -> None:
            if buffer:
                handle.write("".join(buffer))
                buffer.clear()

        # The established format stores only canonical i < j coupling
        # records. A supplied symmetric mask is collapsed onto that triangle.
        for idx0 in range(L):
            for idx1 in range(idx0 + 1, L):
                if mask_np is None:
                    state0 = state0_all
                    state1 = state1_all
                else:
                    pair_mask = mask_np[idx0, :, idx1, :] | mask_np[idx1, :, idx0, :].T
                    state0, state1 = np.nonzero(pair_mask)
                values = couplings_np[idx0, state0, idx1, state1]
                for aa0, aa1, value in zip(state0, state1, values):
                    buffer.append(f"J {idx0} {idx1} {tokens[aa0]} {tokens[aa1]} {value!s}\n")
                if len(buffer) >= chunk_size:
                    flush_lines()

        for idx0 in range(L):
            for state in range(q):
                buffer.append(f"h {idx0} {tokens[state]} {bias_np[idx0, state]!s}\n")
                if len(buffer) >= chunk_size:
                    flush_lines()
        flush_lines()
    
    
def load_params_oldformat(
    fname: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Import the parameters of the model from a file. Assumes the old DCA format.

    Args:
        fname (str): Path of the file that stores the parameters.
        device (torch.device): Device where to store the parameters.
        dtype (torch.dtype): Data type of the parameters. Defaults to torch.float32.

    Returns:
        Dict[str, torch.Tensor]: Parameters of the model.
            - "bias": Tensor of shape (L, q) - local biases.
            - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
    """
    df = pd.read_csv(fname, sep=" ", names=["param", "idx0", "idx1", "idx2", "idx3", "val"])
    df_J = df.loc[df["param"] == "J", ["idx0", "idx1", "idx2", "idx3", "val"]].astype({"idx0" : int, "idx1" : int, "idx2" : int, "idx3" : int, "val" : float})
    df_h = df.loc[df["param"] == "h", ["idx0", "idx1", "idx2"]].astype({"idx0" : int, "idx1" : int, "idx2" : float}).rename(columns={"idx2" : "val"})

    h_idx = df_h.loc[:, ["idx0", "idx1"]].to_numpy()
    L, q = h_idx.max(0) + 1
    h_val = df_h.loc[:, "val"].to_numpy()

    h = np.zeros(shape=(L * q,))
    h_idx_flat = h_idx @ np.array([q, 1])
    h[h_idx_flat] = h_val
    h = h.reshape(L, q)
    
    J_idx = df_J.loc[:, ["idx0", "idx1", "idx2", "idx3"]].to_numpy()
    J_val = df_J.loc[:, "val"].to_numpy()

    J = np.zeros(shape=(L**2 * q**2,))
    J_idx_flat = J_idx @ np.array([L * q**2, q**2, q, 1])
    J[J_idx_flat] = J_val
    
    # Only the upper-triangular part of J is filled
    J = J.reshape(L, L, q, q).transpose(0, 2, 1, 3).reshape(L * q, L * q)
    J = (J + J.T).reshape(L, q, L, q)

    return {
        "bias" : torch.tensor(h, dtype=dtype, device=device),
        "coupling_matrix" : torch.tensor(J, dtype=dtype, device=device),
        }


def save_params_oldformat(
    fname: str,
    params: Dict[str, torch.Tensor],
    mask: Optional[torch.Tensor] = None,
) -> None:
    """Saves the parameters of the model in a file. Assumes the old DCA format.

    Args:
        fname (str): Path to the file where to save the parameters.
        params (Dict[str, torch.Tensor]): Parameters of the model.
            - "bias": Tensor of shape (L, q) - local biases.
            - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
        mask (Optional[torch.Tensor]): Tensor of shape (L, q, L, q) - Mask of the coupling matrix that determines which are the non-zero entries.
            If None, the lower-triangular part of the coupling matrix is masked. Defaults to None.
    """
    L, q = params["bias"].shape
    if mask is None:
        mask = get_mask_save(L, q, device=torch.device("cpu"))
    mask_np = mask.cpu().numpy()
    params_np = {k : v.cpu().numpy() for k, v in params.items()}
    
    L, q, *_ = mask.shape
    idx0 = np.arange(L * q).reshape(L * q) // q
    idx1 = np.arange(L * q).reshape(L * q) % q
    df_h = pd.DataFrame.from_dict({"param" : np.full(L * q, "h"), "idx0" : idx0, "idx1" : idx1, "idx2" : params_np["bias"].flatten()}, orient="columns")

    maskt = np.transpose(mask_np, (0, 2, 1, 3)) # Transpose mask and coupling matrix from (L, q, L, q) to (L, L, q, q)
    Jt = np.transpose(params_np["coupling_matrix"], (0, 2, 1, 3))
    idx0, idx1, idx2, idx3 = maskt.nonzero()
    J_val = Jt[idx0, idx1, idx2, idx3]
    df_J = pd.DataFrame.from_dict({"param" : np.full(len(J_val), "J"), "idx0" : idx0, "idx1" : idx1, "idx2" : idx2, "idx3" : idx3, "val" : J_val}, orient="columns")
    df_J.to_csv(fname, sep=" ", header=False, index=False)
    df_h.to_csv(fname, sep=" ", header=False, index=False, mode="a")
