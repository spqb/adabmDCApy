import argparse
from pathlib import Path

import torch
import numpy as np
import copy
from tqdm import tqdm
import time

from adabmDCA.fasta import get_tokens
from adabmDCA.utils import init_parameters, init_chains, get_device, get_dtype
from adabmDCA.io import load_params, load_chains, import_from_fasta
# from adabmDCA.fasta import encode_sequence, decode_sequence
from adabmDCA.functional import one_hot
from adabmDCA.sampling import get_sampler
from adabmDCA.statmech import compute_energy
from adabmDCA.parser import add_args_tdint
from adabmDCA.resampling import compute_seqID 
from adabmDCA.checkpoint import Log_checkpoint


# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Computes the Entropy of the given model using a Thermodynamical Integration.")
    parser = add_args_tdint(parser)
    
    return parser


def main():
    print("\n" + "".join(["*"] * 10) + f" Computing Entropy " + "".join(["*"] * 10) + "\n")
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    # Check if the data file exists
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file {args.data} not found.")
    
    # Check if the parameters file exists
    if not Path(args.path_params).exists():
        raise FileNotFoundError(f"Parameters file {args.path_params} not found.")

    # Check if the target-sequence file exists
    if not Path(args.path_targetseq).exists():
        raise FileNotFoundError(f"Target Sequence file {args.path_targetseq} not found.")

    
    # Create the folder where to save the model
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    
    if args.label is not None:
        file_paths = {"log" : folder / Path(f"{args.label}.log")}      
    else:
        file_paths = {"log" : folder / Path(f"adabmDCA.log")}

    # select sampler
    sampler = get_sampler("gibbs")

    # read and encode natural data
    tokens = get_tokens(args.alphabet)
    _, nat_data = import_from_fasta(args.data, tokens=tokens, filter_sequences=True)
    M, L, q = len(nat_data), len(nat_data[0]), len(tokens)
    nat_data = one_hot(torch.tensor(nat_data, device=device, dtype=torch.int32), num_classes=q).to(dtype)
    print(f"Number of sequences in the MSA: M={M}")
    print(f"Length of the MSA: L={L}")
    print(f"Number of Potts states: q={q}\n")

    # read parameters
    params = load_params(args.path_params, tokens=tokens, device=device, dtype=dtype)

    # read chains
    if args.path_chains is None:
        chains = init_chains(args.ngen, L, q, device=device)
    else:   
        chains = load_chains(args.path_chains, tokens=tokens)
        chains = one_hot(torch.tensor(chains, device=device, dtype=torch.int32), num_classes=q).to(dtype)
        if chains.shape[0] != args.ngen:
            chains = resample_sequences(chains, weights=torch.ones(chains.shape[0])/chains.shape[0], nextract=args.ngen)
        
    # target sequence
    _, targetseq = import_from_fasta(args.path_targetseq, tokens=tokens, filter_sequences=True) 
    targetseq = one_hot(torch.tensor(targetseq, device=device, dtype=torch.int32), num_classes=q).to(dtype).squeeze(dim=0)

    # Sampling to thermalize at theta = 0
    chains_0 = sampler(chains, params, args.nsweeps_zero) 
    ave_energy_0 = torch.mean(compute_energy(chains_0, params))
    seqID_0 = compute_seqID(chains_0, targetseq)
    print(f"Average seqID at theta = 0: {seqID_0.mean():.2f}")
    print(f"Average energy at theta = 0: {ave_energy_0:.2f}")

    # Sampling to thermalize at theta = theta_max
    theta_max = args.theta_max
    params_theta = copy.deepcopy(params)
    params_theta["bias"] += theta_max * targetseq
    chains_theta = one_hot(torch.randint(0, q, size=(args.ngen, L), device=device), num_classes=q)
    chains_theta = sampler(chains_theta, params_theta, args.nsweeps_theta)
    energy_theta = compute_energy(chains_theta, params)
    ave_energy_theta = torch.mean(energy_theta)
    seqID_max = compute_seqID(chains_theta, targetseq)
    print(f"\nAverage seqID at theta = {theta_max}: {seqID_max.mean():.2f}")
    print(f"Average energy at theta = {theta_max}: {ave_energy_theta:.2f}\n")

    # initialize checkpoint
    checkpoint = Log_checkpoint(
            file_paths=file_paths,
            tokens=tokens,
            args=args,
            use_wandb=False,
        )  
            
    # Find theta_max to generate 10% WT sequences
    p_wt =  (seqID_max == L).sum().item() / args.ngen
    nsweep_find_theta = 100
    while p_wt <= 0.1:
        theta_max += 0.01 * theta_max
        print(f"Number of sequences collapsed to WT is less than 10%. Increasing theta max to: {theta_max:.2f}", flush=True)
        params_theta["bias"] = params["bias"] + theta_max * targetseq
        chains_theta = sampler(chains_theta, params_theta, nsweep_find_theta)
        seqID = compute_seqID(chains_theta, targetseq)
        p_wt = (seqID == L).sum().item() / args.ngen
        print(f"{(p_wt * 100):.2f}% sequences collapse to wt", flush=True)
    
    # Thermodynamic Integration
    int_step = args.nsteps
    nsweeps = args.nsweeps
    F_max = np.log(p_wt) + torch.mean(compute_energy(chains_theta[seqID == L], params_theta))
    thetas = torch.linspace(0, theta_max, int_step) 
    factor = theta_max / (2*int_step)
    F, S, integral = F_max, 0, 0
    torch.set_printoptions(precision=2)

    # initialize progress bar
    pbar = tqdm(
        initial=max(0, thetas[0]),
        total=theta_max,
        colour="red",
        dynamic_ncols=True,
        leave=False,
        ascii="-#",
        bar_format="{desc} {percentage:.2f}%[{bar}] Theta: {n:.3f}/{total_fmt} [{elapsed}]"
    )
    pbar.set_description(f"Theta: {0} - Entropy: {0:.2f}")

    time_start = time.time()

    for i, theta in enumerate(thetas):
        print(f"\nstep n:{i}, theta={theta:.2f}")
        # sampling and compute seqID
        params_theta["bias"] = params["bias"] + theta * targetseq
        chains_theta = sampler(chains_theta, params_theta, nsweeps)
        seqID = compute_seqID(chains_theta, targetseq)
        mean_seqID = seqID.mean()
        print(f"average seqID: {mean_seqID:.3f}", flush=True)
        
        # step of integration to compute entropy
        if i == 0 or i == int_step - 1:
            F += factor * torch.mean(seqID) 
            integral += factor * mean_seqID
        else:
            F += 2 * factor * mean_seqID
            integral += 2 * factor * mean_seqID
        S = ave_energy_0 - F
        print(f"Entropy: {S:.3f}")
   
        # progress bar
        pbar.n = min(max(0, float(theta)), theta_max)
        pbar.set_description(f"Theta: {theta} - Entropy: {S:.2f}")

        # checkpoint
        checkpoint.log({   
                        "Theta": theta,
                        "Free Energy": F,
                        "Entropy": S,
                        "Time": time.time() - time_start,
                    })
        checkpoint.save_log()
        
    print(f"Process completed. Results saved in {file_paths['log']}")


if __name__ == "__main__":
    main()