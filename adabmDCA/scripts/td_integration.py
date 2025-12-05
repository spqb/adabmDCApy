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
    
    print("\n" + "="*80)
    print("  THERMODYNAMIC INTEGRATION - ENTROPY COMPUTATION")
    print("="*80 + "\n")
    
    # Create output folder
    folder = args.output
    os.makedirs(folder, exist_ok=True)

    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    
    # Configuration section
    print("[CONFIGURATION]")
    print("-" * 80)
    template = "  {0:<28} {1:<50}"
    print(template.format("Input MSA:", args.data))
    print(template.format("Parameters file:", args.path_params))
    print(template.format("Target sequence:", args.path_targetseq))
    print(template.format("Output folder:", str(folder)))
    if args.label is not None:
        print(template.format("Label:", args.label))
    print(template.format("Alphabet:", args.alphabet))
    print(template.format("Number of chains:", args.nchains))
    print(template.format("Sweeps per step:", args.nsweeps))
    print(template.format("Integration steps:", args.nsteps))
    print(template.format("Initial theta_max:", args.theta_max))
    print(template.format("Sampler:", args.sampler))
    print(template.format("Device:", str(device)))
    print(template.format("Data type:", args.dtype))
    print("-" * 80 + "\n")

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
    print("[DATA LOADING]")
    print("-" * 80)
    print(f"  Loading MSA from: {args.data}")
    tokens = get_tokens(args.alphabet)
    _, nat_data = import_from_fasta(args.data, tokens=tokens, filter_sequences=True)
    if len(nat_data) == 0:
        raise ValueError(f"No valid sequences found in the input MSA after filtering. Consider changing the alphabet, currently set to {args.alphabet}.")
    M, L, q = len(nat_data), len(nat_data[0]), len(tokens)
    nat_data = one_hot(torch.tensor(nat_data, device=device, dtype=torch.int32), num_classes=q).to(dtype)
    print(f"  ✓ MSA loaded")
    print(f"    • Sequences (M): {M}")
    print(f"    • Length (L): {L}")
    print(f"    • States (q): {q}")

    # read parameters
    print(f"  Loading parameters from: {args.path_params}")
    params = load_params(args.path_params, tokens=tokens, device=device, dtype=dtype)
    print(f"  ✓ Parameters loaded")

    # read chains
    if args.path_chains is None:
        print(f"  Initializing {args.nchains} random chains...")
        chains = init_chains(args.nchains, L, q, device=device, dtype=dtype)
        print(f"  ✓ Chains initialized")
    else:
        print(f"  Loading chains from: {args.path_chains}")
        chains = load_chains(args.path_chains, tokens=tokens, device=device, dtype=dtype)[0]
        if chains.shape[0] != args.nchains:
            chains = resample_sequences(chains, weights=torch.ones(chains.shape[0])/chains.shape[0], nextract=args.nchains)
        print(f"  ✓ Chains loaded ({args.nchains} chains)")
        
    # target sequence
    print(f"  Loading target sequence from: {args.path_targetseq}")
    _, targetseq = import_from_fasta(args.path_targetseq, tokens=tokens, filter_sequences=True, remove_duplicates=True)
    targetseq = one_hot(torch.tensor(targetseq, device=device, dtype=torch.int32), num_classes=q).to(dtype)
    if len(targetseq) != 1:
        print(f"  ⚠ Multiple sequences found, using first as target")
        targetseq = targetseq[0]
    else:
        targetseq = targetseq[0]
    print(f"  ✓ Target sequence loaded")
    print("-" * 80 + "\n")

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
    print("[THERMALIZATION]")
    print("-" * 80)
    print(f"  Thermalizing at θ = 0 ({args.nsweeps_zero} sweeps)...")
    chains_0 = sampler(chains, params, args.nsweeps_zero) 
    ave_energy_0 = torch.mean(compute_energy(chains_0, params))
    print(f"  ✓ Average energy at θ=0: {ave_energy_0:.3f}")

    # Sampling to thermalize at theta = theta_max
    print(f"  Thermalizing at θ = θ_max ({args.nsweeps_theta} sweeps)...")
    theta_max = args.theta_max
    params_theta = {k : v.clone() for k, v in params.items()}
    params_theta["bias"] += theta_max * targetseq
    chains_theta = init_chains(args.nchains, L, q, device=device, dtype=dtype)
    chains_theta = sampler(chains_theta, params_theta, args.nsweeps_theta)
    seqID_max = get_seqid(chains_theta, targetseq)
    print(f"  ✓ Thermalized at θ_max")
            
    # Find theta_max to generate 10% target sequences in the sample
    print(f"\n  Optimizing θ_max for 10% target sequence coverage...")
    p_wt =  (seqID_max == L).sum().item() / args.nchains # percentage of targetseq in the sample
    nsweep_find_theta = 100
    iteration = 0
    while p_wt <= 0.1:
        iteration += 1
        theta_max += 0.01 * theta_max
        print(f"    Iteration {iteration}: {(p_wt * 100):.2f}% WT coverage | θ_max = {theta_max:.4f}")
        params_theta["bias"] = params["bias"] + theta_max * targetseq
        chains_theta = sampler(chains_theta, params_theta, nsweep_find_theta)
        seqID = get_seqid(chains_theta, targetseq)
        p_wt = (seqID == L).sum().item() / args.nchains
    print(f"  ✓ Optimal θ_max: {theta_max:.4f} ({(p_wt * 100):.2f}% WT coverage)")
    print("-" * 80 + "\n")
    
    # initiaize Thermodynamic Integration
    print("[THERMODYNAMIC INTEGRATION]")
    print("-" * 80)
    print(f"  Integration range: [0, {theta_max:.4f}]")
    print(f"  Number of steps: {args.nsteps}")
    print(f"  Sweeps per step: {args.nsweeps}")
    print("-" * 80)
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
        bar_format="  {desc} [{bar}] θ: {n:.3f}/{total_fmt} [{elapsed}]"
    )
    pbar.set_description(f"Step {0:4d} | SeqID: ---- | S: ----    ")

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
        pbar.set_description(f"Step {i:4d} | SeqID: {mean_seqID:.3f} | S: {S:8.2f}")

        # checkpoint
        logs["Epoch"] = i
        logs["Theta"] = float(theta)
        logs["Free Energy"] = F.item()
        logs["Entropy"] = S.item()
        logs["Time"] = time.time() - time_start
        with open(file_log, "a") as f:
            f.write(" ".join([f"{value:<15.3f}" if isinstance(value, float) else f"{value:<15}" for value in logs.values()]) + "\n")
        
    pbar.close()
    
    print("\n" + "-" * 80)
    print("  INTEGRATION COMPLETED")
    print("-" * 80)
    print(f"  Final entropy: {S:.4f}")
    print(f"  Final free energy: {F:.4f}")
    print(f"  Total integration steps: {int_step}")
    print(f"\n  Results saved: {file_log}")
    print("-" * 80 + "\n")
    
    print("=" * 80)
    print("  THERMODYNAMIC INTEGRATION COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()