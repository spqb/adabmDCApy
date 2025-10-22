import argparse
import os

import torch
import numpy as np
from tqdm import tqdm
import time

from adabmDCA.fasta import get_tokens
from adabmDCA.utils import init_chains, get_device, get_dtype, resample_sequences
from adabmDCA.io import load_params, load_chains, import_from_fasta
from adabmDCA.functional import one_hot
from adabmDCA.sampling import get_sampler
from adabmDCA.statmech import compute_energy
from adabmDCA.parser import add_args_tdint
from adabmDCA.dca import get_seqid


# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Computes the Entropy of the given model using a Thermodynamical Integration.")
    parser = add_args_tdint(parser)
    
    return parser


def main():
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    print("\n" + "".join(["*"] * 10) + f" Computing model's entropy " + "".join(["*"] * 10) + "\n")
    
    # Create output folder
    folder = args.output
    os.makedirs(folder, exist_ok=True)

    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    # Check if the data file exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file {args.data} not found.")
    
    # Check if the parameters file exists
    if not os.path.exists(args.path_params):
        raise FileNotFoundError(f"Parameters file {args.path_params} not found.")

    # Check if the target-sequence file exists
    if not os.path.exists(args.path_targetseq):
        raise FileNotFoundError(f"Target Sequence file {args.path_targetseq} not found.")
    
    if args.label is not None:
        file_log = os.path.join(folder, f"{args.label}.log")
    else:
        file_log = os.path.join(folder, "td_integration.log")

    # select sampler
    sampler = torch.jit.script(get_sampler(args.sampler))

    # read and encode natural data
    tokens = get_tokens(args.alphabet)
    _, nat_data = import_from_fasta(args.data, tokens=tokens, filter_sequences=True)
    if len(nat_data) == 0:
        raise ValueError(f"No valid sequences found in the input MSA after filtering. Consider changing the alphabet, currently set to {args.alphabet}.")
    M, L, q = len(nat_data), len(nat_data[0]), len(tokens)
    nat_data = one_hot(torch.tensor(nat_data, device=device, dtype=torch.int32), num_classes=q).to(dtype)
    print(f"Number of sequences in the MSA: M={M}")
    print(f"Length of the MSA: L={L}")
    print(f"Number of Potts states: q={q}\n")

    # read parameters
    print(f"Loading parameters from {args.path_params}...")
    params = load_params(args.path_params, tokens=tokens, device=device, dtype=dtype)

    # read chains
    if args.path_chains is None:
        chains = init_chains(args.nchains, L, q, device=device, dtype=dtype)
    else:   
        chains = load_chains(args.path_chains, tokens=tokens, device=device, dtype=dtype)[0]
        if chains.shape[0] != args.nchains:
            chains = resample_sequences(chains, weights=torch.ones(chains.shape[0])/chains.shape[0], nextract=args.nchains)
    print(f"Number of chains set to {args.nchains}.")
        
    # target sequence
    _, targetseq = import_from_fasta(args.path_targetseq, tokens=tokens, filter_sequences=True, remove_duplicates=True)
    targetseq = one_hot(torch.tensor(targetseq, device=device, dtype=torch.int32), num_classes=q).to(dtype)
    if len(targetseq) != 1:
        print(f"Target sequence file contains more than one sequence. Using the first sequence as target sequence.")
        targetseq = targetseq[0]

    # initialize checkpoint
    template = "{0:<20} {1:<50}\n"  
    with open(file_log, "w") as f:
        if args.label is not None:
            f.write(template.format("label:", args.label))
        else:
            f.write(template.format("label:", "N/A"))
        
        f.write(template.format("input MSA:", str(args.data)))
        f.write(template.format("model path:", str(args.path_params)))
        f.write(template.format("target sequence:", str(args.path_targetseq)))
        f.write(template.format("alphabet:", args.alphabet))
        f.write(template.format("nchains:", args.nchains))
        f.write(template.format("nsweeps:", args.nsweeps))
        if args.nsweeps_theta is not None:
            f.write(template.format("nsweeps  theta:", str(args.nsweeps_theta)))
        if args.nsweeps_zero is not None:
            f.write(template.format("nsweeps zero:", str(args.nsweeps_zero)))
        f.write(template.format("nsteps:", args.nsteps))
        f.write(template.format("data type:", args.dtype))
        f.write(template.format("random seed:", args.seed))
        f.write("\n")
        # write the header of the log file
        logs = {
            "Epoch": 0,
            "Theta": 0.0,
            "Free Energy": 0.0,
            "Entropy": 0.0,
            "Time": 0.0
        }
        header_string = " ".join([f"{key:<15}" for key in logs.keys()])
        f.write(header_string + "\n")
    
    # Sampling to thermalize at theta = 0
    print("Thermalizing at theta = 0...")
    chains_0 = sampler(chains, params, args.nsweeps_zero) 
    ave_energy_0 = torch.mean(compute_energy(chains_0, params))

    # Sampling to thermalize at theta = theta_max
    print("Thermalizing at theta = theta_max...")
    theta_max = args.theta_max
    params_theta = {k : v.clone() for k, v in params.items()}
    params_theta["bias"] += theta_max * targetseq
    chains_theta = init_chains(args.nchains, L, q, device=device, dtype=dtype)
    chains_theta = sampler(chains_theta, params_theta, args.nsweeps_theta)
    seqID_max = get_seqid(chains_theta, targetseq)
            
    # Find theta_max to generate 10% target sequences in the sample
    print("Finding theta_max to generate 10% target sequences in the sample...")
    p_wt =  (seqID_max == L).sum().item() / args.nchains # percentage of targetseq in the sample
    nsweep_find_theta = 100
    while p_wt <= 0.1:
        theta_max += 0.01 * theta_max
        print(f"{(p_wt * 100):.2f}% sequences collapse to WT")
        print(f"Number of sequences collapsed to WT is less than 10%. Increasing theta max to: {theta_max:.2f}...")
        params_theta["bias"] = params["bias"] + theta_max * targetseq
        chains_theta = sampler(chains_theta, params_theta, nsweep_find_theta)
        seqID = get_seqid(chains_theta, targetseq)
        p_wt = (seqID == L).sum().item() / args.nchains
    
    # initiaize Thermodynamic Integration
    print("Starting Thermodynamic Integration...")
    int_step = args.nsteps
    nsweeps = args.nsweeps
    seqID = get_seqid(chains_theta, targetseq)
    F_max = np.log(p_wt) + torch.mean(compute_energy(chains_theta[seqID == L], params_theta))
    thetas = torch.linspace(0, theta_max, int_step) 
    factor = theta_max / (2 * int_step)
    F, S, integral = F_max, 0, 0
    torch.set_printoptions(precision=2)

    # initialize progress bar
    pbar = tqdm(
        initial=max(0, thetas[0].item()),
        total=round(theta_max, 3),
        colour="red",
        dynamic_ncols=True,
        leave=False,
        ascii="-#",
        bar_format="{desc} {percentage:.2f}%[{bar}] Theta: {n:.3f}/{total_fmt} [{elapsed}]"
    )
    pbar.set_description(f"Step: {0} - SeqID: Nan - Entropy: Nan")

    time_start = time.time()

    for i, theta in enumerate(thetas):
        
        # sampling and compute seqID
        params_theta["bias"] = params["bias"] + theta * targetseq
        chains_theta = sampler(chains_theta, params_theta, nsweeps)
        seqID = get_seqid(chains_theta, targetseq)
        mean_seqID = seqID.mean()
        
        # step of integration to compute entropy
        if i == 0 or i == int_step - 1:
            F += factor * torch.mean(seqID) 
            integral += factor * mean_seqID
        else:
            F += 2 * factor * mean_seqID
            integral += 2 * factor * mean_seqID
        S = ave_energy_0 - F
   
        # progress bar
        pbar.n = min(max(0, float(theta)), theta_max)
        pbar.set_description(f"Step: {i} - SeqID: {mean_seqID:.3f} - Entropy: {S:.2f}")

        # checkpoint
        logs["Epoch"] = i
        logs["Theta"] = float(theta)
        logs["Free Energy"] = F.item()
        logs["Entropy"] = S.item()
        logs["Time"] = time.time() - time_start
        with open(file_log, "a") as f:
            f.write(" ".join([f"{value:<15.3f}" if isinstance(value, float) else f"{value:<15}" for value in logs.values()]) + "\n")
        
    pbar.close()
    print(f"Process completed. Results saved in {file_log}.")


if __name__ == "__main__":
    main()