from typing import Dict, Callable

import torch
from torch.nn.functional import one_hot


def sampling_profile(
    params: Dict[str, torch.Tensor],
    nsamples: int,
    beta: float,
) -> torch.Tensor:
    """Samples from the profile model defined by the local biases only.
    
    Args:
        params (Dict[str, torch.Tensor]): Parameters of the model.
            - "bias": Tensor of shape (L, q) - local biases.
        nsamples (int): Number of samples to generate.
        beta (float): Inverse temperature.
        
    Returns:
        torch.Tensor: Sampled one-hot encoded sequences of shape (nsamples, L, q).
    """
    L, q = params["bias"].shape
    device = params["bias"].device
    dtype = params["bias"].dtype
    logits = beta * params["bias"].unsqueeze(0).expand(nsamples, -1, -1)  # Shape: (nsamples, L, q)
    sampled_indices = torch.multinomial(torch.softmax(logits.view(-1, q), dim=-1), num_samples=1).squeeze(-1)
    sampled_sequences = one_hot(sampled_indices, num_classes=q).view(nsamples, L, q).to(dtype).to(device)
    
    return sampled_sequences


def gibbs_step_uniform_sites(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    """Performs a single mutation using the Gibbs sampler. In this version, the mutation is attempted at the same sites for all chains.

    Args:
        chains (torch.Tensor): One-hot encoded sequences of shape (batch_size, L, q).
        params (Dict[str, torch.Tensor]): Parameters of the model.
            - "bias": Tensor of shape (L, q) - local biases.
            - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        torch.Tensor: Updated chains.
    """
    N, L, q = chains.shape
    device = chains.device
    dtype = chains.dtype
    idx = torch.randint(0, L, (1,), device=device)[0]
    couplings_residue = params["coupling_matrix"][idx].reshape(q, L * q)
    logit_residue = beta * (params["bias"][idx].unsqueeze(0) + chains.reshape(N, L * q) @ couplings_residue.T) # (N, q)
    new_residues = one_hot(torch.multinomial(torch.softmax(logit_residue, dim=-1), num_samples=1).squeeze(-1), num_classes=q).to(dtype)
    chains[:, idx] = new_residues

    return chains


def gibbs_step_independent_sites(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    """Performs a single mutation using the Gibbs sampler. This version selects different random sites for each chain. It is
    less efficient than the 'gibbs_step_uniform_sites' function, but it is more suitable for mutating starting from the same wild-type sequence since mutations are independent across chains.

    Args:
        chains (torch.Tensor): One-hot encoded sequences of shape (batch_size, L, q).
        params (Dict[str, torch.Tensor]): Parameters of the model.
            - "bias": Tensor of shape (L, q) - local biases.
            - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        torch.Tensor: Updated chains.
    """
    N, L, q = chains.shape
    device = chains.device
    dtype = chains.dtype
    # Select a different random site for each sequence in the batch
    idx_batch = torch.randint(0, L, (N,), device=device)
    biases = params["bias"][idx_batch]  # Shape: (N, q)
    couplings_batch = params["coupling_matrix"][idx_batch]  # Shape: (N, q, L, q)
    chains_flat = chains.reshape(N, L * q, 1)
    couplings_flat = couplings_batch.reshape(N, q, L * q)
    coupling_term = torch.bmm(couplings_flat, chains_flat).squeeze(-1)  # (N, q, L*q) @ (N, L*q, 1) -> (N, q, 1) -> (N, q)
    logits = beta * (biases + coupling_term)
    new_residues = one_hot(torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(-1), num_classes=q).to(dtype)
    # Create an index for the batch dimension
    batch_arange = torch.arange(N, device=device)
    chains[batch_arange, idx_batch] = new_residues

    return chains


def gibbs_sampling(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Gibbs sampling. Attempts L * nsweeps mutations to each sequence in 'chains'.
    
    Args:
        chains (torch.Tensor): Initial one-hot encoded samples of size (batch_size, L, q).
        params (Dict[str, torch.Tensor]): Parameters of the model.
            - "bias": Tensor of shape (L, q) - local biases.
            - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
        nsweeps (int): Number of sweeps, where one sweep corresponds to attempting L mutations.
        beta (float, optional): Inverse temperature. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Updated chains.
    """
    L = params["bias"].shape[0]
    chains_mutate = chains.clone() # avoids to modify the chains inplace
    num_steps = nsweeps * L
    for _ in torch.arange(num_steps):
        chains_mutate = gibbs_step_uniform_sites(chains_mutate, params, beta)

    return chains_mutate


def metropolis_step_uniform_sites(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    """Performs a single mutation using the Metropolis sampler. In this version, the mutation is attempted at the same sites for all chains.

    Args:
        chains (torch.Tensor): One-hot encoded sequences of shape (batch_size, L, q).
        params (Dict[str, torch.Tensor]): Parameters of the model.
            - "bias": Tensor of shape (L, q) - local biases.
            - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        torch.Tensor: Updated chains.
    """
    device = chains.device
    dtype = chains.dtype
    N, L, q = chains.shape
    idx = torch.randint(0, L, (1,), device=chains.device)[0]
    res_old = chains[:, idx, :] # shape (N, q)
    # Propose a random new residue
    res_new = one_hot(torch.randint(0, q, (N,), device=chains.device), num_classes=q).to(dtype)
    # Compute local fields
    biases = params["bias"][idx].unsqueeze(0) # shape (1, q)
    couplings_residue = params["coupling_matrix"][idx].reshape(q, L * q)
    chains_flat = chains.reshape(N, L * q)
    coupling_term = chains_flat @ couplings_residue.T # shape (N, q), background
    local_field = biases + coupling_term
    # Metropolis acceptance step
    delta_E = torch.sum((res_old - res_new) * local_field, dim=-1) # shape (N,)
    accept_prob = torch.exp(- beta * delta_E).unsqueeze(-1)
    chains[:, idx, :] = torch.where(accept_prob > torch.rand((N, 1), device=device, dtype=dtype), res_new, res_old)

    return chains


def metropolis_step_independent_sites(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    """Performs a single mutation using the Metropolis sampler. This version selects different random sites for each chain. It is
    less efficient than the 'metropolis_step_uniform_sites' function, but it is more suitable for mutating starting from the same wild-type sequence since mutations are independent across chains.

    Args:
        chains (torch.Tensor): One-hot encoded sequences of shape (batch_size, L, q).
        params (Dict[str, torch.Tensor]): Parameters of the model.
            - "bias": Tensor of shape (L, q) - local biases.
            - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        torch.Tensor: Updated chains.
    """
    N, L, q = chains.shape
    device = chains.device
    idx_batch = torch.randint(0, L, (N,), device=device)
    res_new = one_hot(torch.randint(0, q, (N,), device=device), num_classes=q).type(chains.dtype)
    batch_arange = torch.arange(N, device=device)
    res_old = chains[batch_arange, idx_batch]
    # Compute local fields
    biases = params["bias"][idx_batch]
    couplings_batch = params["coupling_matrix"][idx_batch]
    chains_flat = chains.reshape(N, L * q, 1)
    couplings_flat = couplings_batch.reshape(N, q, L * q)
    coupling_term = torch.bmm(couplings_flat, chains_flat).squeeze(-1)
    local_field = biases + coupling_term # Shape: (N, q)
    # Metropolis acceptance step
    delta_E = torch.sum((res_old - res_new) * local_field, dim=-1) # Shape: (N,)
    acceptance_prob = torch.exp(- beta * delta_E)
    random_uniform = torch.rand(N, device=device, dtype=chains.dtype)
    accept_mask = (random_uniform < acceptance_prob) # Shape: (N,)
    final_residues = torch.where(accept_mask.unsqueeze(-1), res_new, res_old)
    chains[batch_arange, idx_batch] = final_residues

    return chains
    

def metropolis_sampling(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Metropolis sampling. Attempts L * nsweeps mutations to each sequence in 'chains'.

    Args:
        chains (torch.Tensor): One-hot encoded sequences of shape (batch_size, L, q).
        params (Dict[str, torch.Tensor]): Parameters of the model.
            - "bias": Tensor of shape (L, q) - local biases.
            - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
        nsweeps (int): Number of sweeps to be performed, where one sweep corresponds to attempting L mutations.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        torch.Tensor: Updated chains.
    """
    L = params["bias"].shape[0]
    chains_mutate = chains.clone() # avoids to modify the chains inplace
    num_steps = nsweeps * L
    for _ in torch.arange(num_steps):
        chains_mutate = metropolis_step_uniform_sites(chains_mutate, params, beta)

    return chains_mutate


def get_sampler(sampling_method: str) -> Callable:
    """Returns the sampling function corresponding to the chosen method.

    Args:
        sampling_method (str): String indicating the sampling method. Choose between 'metropolis' and 'gibbs'.

    Raises:
        KeyError: Unknown sampling method.

    Returns:
        Callable: Sampling function.
    """
    if sampling_method == "gibbs":
        return gibbs_sampling
    elif sampling_method == "metropolis":
        return metropolis_sampling
    else:
        raise KeyError("Unknown sampling method. Choose between 'metropolis' and 'gibbs'.")
