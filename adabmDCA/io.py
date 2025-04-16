from typing import Dict, Tuple
import pandas as pd
import numpy as np

import torch

from adabmDCA.fasta import (
    write_fasta,
    encode_sequence,
    import_from_fasta,
    validate_alphabet,
    get_tokens,
)
from adabmDCA.utils import get_mask_save


def load_chains(
    fname: str,
    tokens: str,
    load_weights: bool = False
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Loads the sequences from a fasta file and returns the numeric-encoded version.
    If the sequences are weighted, the log-weights are also returned. If the sequences are not weighted, the log-weights are set to 0.
    
    Args:
        fname (str): Path to the file containing the sequences.
        tokens (str): "protein", "dna", "rna" or another string with the alphabet to be used.
        load_weights (bool, optional): If True, the log-weights are loaded and returned. Defaults to False.
    
    Return:
        np.ndarray | Tuple[np.ndarray, np.ndarray]: Numeric-encoded sequences and log-weights if load_weights is True.
    """
    def parse_header(header: str):
        h = header.split("|")
        if len(h) == 2:
            log_weigth = float(h[1].split("=")[1])
            return log_weigth
        else:
            return 0.
    
    headers, sequences = import_from_fasta(fasta_name=fname)
    validate_alphabet(sequences, tokens=tokens)
    encoded_sequences = encode_sequence(sequences, tokens=tokens)
    
    if load_weights:
        log_weights = np.vectorize(parse_header)(headers)
        return encoded_sequences, log_weights
    else:
        return encoded_sequences
    

def save_chains(
    fname: str,
    chains: torch.Tensor,
    tokens: str,
    log_weights: torch.Tensor = None
) -> None:
    """Saves the chains in a fasta file.

    Args:
        fname (str): Path to the file where to save the chains.
        chains (torch.Tensor): Chains.
        tokens (str): "protein", "dna", "rna" or another string with the alphabet to be used.
        log_weights (torch.Tensor, optional): Log-weights of the chains. Defaults to None.
    """
    
    # Check if chains is a 3D tensor
    if chains.ndim != 2:
        raise ValueError("chains must be a 2D tensor")
    
    chains = chains.cpu().numpy()
    if log_weights is not None:
        log_weigth = log_weights.cpu().numpy()
        headers = [f"chain_{i}|log_weight={log_weigth[i]}" for i in range(len(chains))]
    else:
        headers = [f"chain_{i}" for i in range(len(chains))]
    write_fasta(
        fname=fname,
        headers=headers,
        sequences=chains,
        numeric_input=True,
        remove_gaps=False,
        tokens=tokens,
    )
    
    
def load_params(
    fname: str,
    tokens: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Import the parameters of the model from a file.

    Args:
        fname (str): Path of the file that stores the parameters.
        tokens (str): "protein", "dna", "rna" or another string with the alphabet to be used.
        device (torch.device): Device where to store the parameters.
        dtype (torch.dtype): Data type of the parameters. Defaults to torch.float32.

    Returns:
        Dict[str, torch.Tensor]: Parameters of the model.
    """
    
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
    mask: torch.Tensor | None = None,
) -> None:
    """Saves the parameters of the model in a file.

    Args:
        fname (str): Path to the file where to save the parameters.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        tokens (str): "protein", "dna", "rna" or another string with the alphabet to be used.
        mask (torch.Tensor | None): Mask of the coupling matrix that determines which are the non-zero entries.
            If None, the lower-triangular part of the coupling matrix is masked. Defaults to None.
    """
    tokens = get_tokens(tokens)
    if mask is None:
        mask = get_mask_save(L, q, device="cpu")
    mask = mask.cpu().numpy()
    params = {k : v.cpu().numpy() for k, v in params.items()}
    
    L, q, *_ = mask.shape
    idx0 = np.arange(L * q).reshape(L * q) // q
    idx1 = np.arange(L * q).reshape(L * q) % q
    idx1_aa = np.vectorize(lambda n, tokens : tokens[n], excluded=["tokens"])(idx1, tokens).astype(str)
    df_h = pd.DataFrame(
        {
            "param" : np.full(L * q, "h"),
            "idx0" : idx0,
            "idx1" : idx1_aa,
            "idx2" : params["bias"].flatten(),
        }
    )
    
    
    maskt = mask.transpose(0, 2, 1, 3) # Transpose mask and coupling matrix from (L, q, L, q) to (L, L, q, q)
    Jt = params["coupling_matrix"].transpose(0, 2, 1, 3)
    idx0, idx1, idx2, idx3 = maskt.nonzero()
    idx2_aa = np.vectorize(lambda n, tokens : tokens[n], excluded=["tokens"])(idx2, tokens).astype(str)
    idx3_aa = np.vectorize(lambda n, tokens : tokens[n], excluded=["tokens"])(idx3, tokens).astype(str)
    J_val = Jt[idx0, idx1, idx2, idx3]
    df_J = pd.DataFrame(
        {
            "param" : np.full(len(J_val), "J").tolist(),
            "idx0" : idx0,
            "idx1" : idx1,
            "idx2" : idx2_aa,
            "idx3" : idx3_aa,
            "val" : J_val,
        }
    )
    df_J.to_csv(fname, sep=" ", header=False, index=False)
    df_h.to_csv(fname, sep=" ", header=False, index=False, mode="a")
    
    
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
    mask: torch.Tensor | None = None,
) -> None:
    """Saves the parameters of the model in a file. Assumes the old DCA format.

    Args:
        fname (str): Path to the file where to save the parameters.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        mask (torch.Tensor): Mask of the coupling matrix that determines which are the non-zero entries.
            If None, the lower-triangular part of the coupling matrix is masked. Defaults to None.
    """
    if mask is None:
        mask = get_mask_save(L, q, device="cpu")
    mask = mask.cpu().numpy()
    params = {k : v.cpu().numpy() for k, v in params.items()}
    
    L, q, *_ = mask.shape
    idx0 = np.arange(L * q).reshape(L * q) // q
    idx1 = np.arange(L * q).reshape(L * q) % q
    df_h = pd.DataFrame.from_dict({"param" : np.full(L * q, "h"), "idx0" : idx0, "idx1" : idx1, "idx2" : params["bias"].flatten()}, orient="columns")
    
    maskt = mask.transpose(0, 2, 1, 3) # Transpose mask and coupling matrix from (L, q, L, q) to (L, L, q, q)
    Jt = params["coupling_matrix"].transpose(0, 2, 1, 3)
    idx0, idx1, idx2, idx3 = maskt.nonzero()
    J_val = Jt[idx0, idx1, idx2, idx3]
    df_J = pd.DataFrame.from_dict({"param" : np.full(len(J_val), "J"), "idx0" : idx0, "idx1" : idx1, "idx2" : idx2, "idx3" : idx3, "val" : J_val}, orient="columns")
    df_J.to_csv(fname, sep=" ", header=False, index=False)
    df_h.to_csv(fname, sep=" ", header=False, index=False, mode="a")