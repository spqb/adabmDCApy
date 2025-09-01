from typing import Dict, Callable
from tqdm.autonotebook import tqdm
import torch
from adabmDCA.dca import get_seqid


def compute_mixing_time(
    sampler: Callable,
    data: torch.Tensor,
    params: Dict[str, torch.Tensor],
    n_max_sweeps: int,
    beta: float,
) -> Dict[str, list]:
    """Computes the mixing time using the t and t/2 method. The sampling will halt when the mixing time is reached or
    the limit of `n_max_sweeps` sweeps is reached.

    Args:
        sampler (Callable): Sampling function.
        data (torch.Tensor): Initial data.
        params (Dict[str, torch.Tensor]): Parameters for the sampling.
        n_max_sweeps (int): Maximum number of sweeps.
        beta (float): Inverse temperature for the sampling.

    Returns:
        Dict[str, list]: Results of the mixing time analysis.
    """

    torch.manual_seed(0)
    
    L, _ = params["bias"].shape
    # Initialize chains at random
    sample_t = data
    # Copy sample_t to a new variable sample_t_half
    sample_t_half = sample_t.clone()

    # Initialize variables
    results = {
        "seqid_t": [],
        "std_seqid_t": [],
        "seqid_t_t_half": [],
        "std_seqid_t_t_half": [],
        "t_half": [],
    }

    # Loop through sweeps
    pbar = tqdm(
        total=n_max_sweeps,
        colour="red",
        dynamic_ncols=True,
        leave=False,
        ascii="-#",
    )
    pbar.set_description("Iterating until the mixing time is reached")
        
    for i in range(1, n_max_sweeps + 1):
        pbar.update(1)
        # Set the seed to i
        torch.manual_seed(i)
        # Perform a sweep on sample_t
        sample_t = sampler(chains=sample_t, params=params, nsweeps=1, beta=beta)

        if i % 2 == 0:
            # Set the seed to i/2
            torch.manual_seed(i // 2)
            # Perform a sweep on sample_t_half
            sample_t_half = sampler(chains=sample_t_half, params=params, nsweeps=1, beta=beta)

            # Calculate the average distance between sample_t and itself shuffled
            perm = torch.randperm(len(sample_t))
            seqid_t, std_seqid_t = get_seqid(sample_t, sample_t[perm], average=True)
            seqid_t, std_seqid_t = seqid_t / L, std_seqid_t / L

            # Calculate the average distance between sample_t and sample_t_half
            seqid_t_t_half, std_seqid_t_t_half = get_seqid(sample_t, sample_t_half, average=True)
            seqid_t_t_half, std_seqid_t_t_half = seqid_t_t_half / L, std_seqid_t_t_half / L

            # Store the results
            results["seqid_t"].append(seqid_t.item())
            results["std_seqid_t"].append(std_seqid_t.item())
            results["seqid_t_t_half"].append(seqid_t_t_half.item())
            results["std_seqid_t_t_half"].append(std_seqid_t_t_half.item())
            results["t_half"].append(i // 2)

            # Check if they have crossed
            if torch.abs(seqid_t - seqid_t_t_half) / torch.sqrt(std_seqid_t**2 + std_seqid_t_t_half**2) < 0.1:
                break

        if i == n_max_sweeps:
            print(f"Mixing time not reached within {n_max_sweeps // 2} sweeps.")
            
    pbar.close()

    return results