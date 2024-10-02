from typing import Dict
import torch
from adabmDCA.custom_fn import one_hot


def gibbs_sampling(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Gibbs sampling.
    
    Args:
        chains (torch.Tensor): Initial chains.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        nsweeps (int): Number of sweeps.
        beta (float, optional): Inverse temperature. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Updated chains.
    """
    L = params["bias"].shape[0]
    
    @torch.jit.script
    def do_sweep(
        chains: torch.Tensor,
        residue_idxs: torch.Tensor,
        params: Dict[str, torch.Tensor],
        beta: float,
    ) -> torch.Tensor:
        N, L, q = chains.shape
        for i in residue_idxs:
            # Select the couplings attached to the residue we are considering (i) and flatten along the other residues ({j})
            couplings_residue = params["coupling_matrix"][i].view(q, L * q)
            # Update the chains
            logit_residue = beta * (params["bias"][i].unsqueeze(0) + chains.reshape(N, L * q) @ couplings_residue.T) # (N, q)
            chains[:, i, :] = one_hot(torch.multinomial(torch.softmax(logit_residue, -1), 1), num_classes=q).squeeze(1)
            
        return chains
    
    for t in torch.arange(nsweeps):
        # Random permutation of the residues
        residue_idxs = torch.randperm(L)
        chains = do_sweep(chains, residue_idxs, params, beta)
        
    return chains


def metropolis(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    L, q = params["coupling_matrix"].shape[:2]

    def get_deltaE(
        idx: int,
        chain: torch.Tensor,
        residue_old: torch.Tensor,
        residue_new: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> float:
        coupling_residue = torch.flatten(chain) @ torch.flatten(params["coupling_matrix"][:, :, idx, :], start_dim=0, end_dim=1)
        E_old = - params["bias"][idx] @ residue_old - coupling_residue @ residue_old
        E_new = - params["bias"][idx] @ residue_new - coupling_residue @ residue_new
        
        return E_new - E_old

    def do_sweep(chain: torch.Tensor) -> torch.Tensor:
        residue_idxs = torch.randperm(L)
        for i in residue_idxs:
            res_old = chain[i]
            res_new = one_hot(torch.randint(0, q, (1,)), num_classes=q).float().squeeze()
            delta_E = get_deltaE(idx=i, chain=chain, residue_old=res_old, residue_new=res_new, params=params)
            accept_prob = torch.exp(- beta * delta_E)
            if accept_prob > torch.rand(1):
                chain[i] = res_new
                
        return chain

    for _ in range(nsweeps):
        chains = do_sweep(chains)

    return chains

def metropolis(
    key: torch.Generator,
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    it_mcmc: int,
    beta: float = 1.0,
) -> torch.Tensor:
    num_chains = chains.shape[0]
    updated_chains = torch.stack([metropolis_(key, chains[i], params, it_mcmc, beta) for i in range(num_chains)])
    return updated_chains


def get_sampler(sampling_method: str):
    if sampling_method == "gibbs":
        return gibbs_sampling
    elif sampling_method == "metropolis":
        return metropolis
    else:
        raise KeyError("Unknown sampling method. Choose between 'metropolis' and 'gibbs'.")