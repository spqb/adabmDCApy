import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

from adabmDCA.fasta_utils import get_tokens, write_fasta, compute_weights, import_clean_dataset, encode_sequence
from adabmDCA.resampling import compute_mixing_time
from adabmDCA.io import load_params
from adabmDCA.utils import init_chains, resample_sequences, get_device
from adabmDCA.sampling import get_sampler
from adabmDCA.functional import one_hot
from adabmDCA.statmech import compute_energy
from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from adabmDCA.parser import add_args_sample


# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description='Samples from a DCA model.')
    parser = add_args_sample(parser)
    
    return parser


def main():       
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create output folder
    filename = Path(args.output)
    folder = filename.parent / Path(filename.name)
    # Create the folder where to save the samples
    folder.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "".join(["*"] * 10) + f" Sampling from DCA model " + "".join(["*"] * 10) + "\n")
    # Set the device
    device = get_device(args.device)
    tokens = get_tokens(args.alphabet)
        
    # Import parameters
    print(f"Loading parameters from {args.path_params}...")
    params = load_params(fname=args.path_params, tokens=tokens, device=device)
    L, q = params["bias"].shape
    print(f"L = {L}, q = {q}")
    
    # Select the sampler
    sampler = get_sampler(args.sampler)
    
    # Import data
    print(f"Loading data from {args.data}...")
    headers, sequences = import_clean_dataset(args.data, tokens)
    data = encode_sequence(sequences, tokens)
    data = torch.tensor(data, device=device)
    
    if args.weights is None:
        print("Computing the weights...")
        weights = compute_weights(data, device=device).view(-1)
    else:
        weights = torch.tensor(np.loadtxt(args.weights), device=device, dtype=torch.float32)
    
    nmeasure = min(args.nmeasure, len(data))
    data = one_hot(data, num_classes=len(tokens))
    data_resampled = resample_sequences(data, weights, nmeasure)
    
    if args.pseudocount is None:
        args.pseudocount = 1. / weights.sum()
    print(f"Using pseudocount: {args.pseudocount}...")
    
    # Compute the mixing time
    print("Computing the mixing time...")
    results_mix = compute_mixing_time(
        sampler=sampler,
        data=data_resampled,
        params=params,
        n_max_sweeps=args.max_nsweeps,
        beta=args.beta,
    )
    mixing_time = results_mix["t_half"][-1]
    print(f"Measured mixing time (if converged): {mixing_time} sweeps")
    
    # Sample from random initialization
    print(f"Sampling for {args.nmix} * t_mix sweeps...")
    pbar = tqdm(
        total=args.nmix * mixing_time,
        colour="red",
        dynamic_ncols=True,
        leave=False,
        ascii="-#",
    )
    
    # Initialize the chains at random
    samples = init_chains(
        num_chains=args.ngen,
        L=L,
        q=q,
        device=device,
    )
    
    # Compute single and two-site frequencies of the data
    fi = get_freq_single_point(data=data, weights=weights, pseudo_count=args.pseudocount)
    fij = get_freq_two_points(data=data, weights=weights, pseudo_count=args.pseudocount)
    
    # Dictionary to store Pearson coefficient and slope along the sampling
    results_sampling = {
        "nsweeps" : [],
        "pearson" : [],
        "slope" : [],
    }
    
    # Sample for (args.nmix * mixing_time) sweeps starting from random initialization
    for i in range(args.nmix * mixing_time):
        pbar.update(1)
        samples = sampler(chains=samples, params=params, nsweeps=1, beta=args.beta)
        pi = get_freq_single_point(data=samples, weights=None, pseudo_count=0.)
        pij = get_freq_two_points(data=samples, weights=None, pseudo_count=0.)
        pearson, slope = get_correlation_two_points(fi=fi, pi=pi, fij=fij, pij=pij)
        results_sampling["nsweeps"].append(i)
        results_sampling["pearson"].append(pearson)
        results_sampling["slope"].append(slope)
    pbar.close()
    
    # Compute the energy of the samples
    print("Computing the energy of the samples...")
    energies = compute_energy(samples, params=params).cpu().numpy()
    
    print("\nSaving the samples...")
    headers = [f"sequence {i+1} | DCAenergy: {energies[i]:.3f}" for i in range(args.ngen)]
    write_fasta(
        fname=folder / Path(f"{args.label}_samples.fasta"),
        headers=headers,
        sequences=samples.argmax(-1).cpu().numpy(),
        numeric_input=True,
        remove_gaps=False,
        alphabet=tokens,
    )
    
    print("Writing sampling log...")
    df_mix_log = pd.DataFrame.from_dict(results_mix)    
    df_mix_log.to_csv(
        folder / Path(f"{args.label}_mix.log"),
        index=False
    )
    df_samp_log = pd.DataFrame.from_dict(results_sampling)    
    df_samp_log.to_csv(
        folder / Path(f"{args.label}_sampling.log"),
        index=False
    )
    
    print(f"\nDone, results saved in {str(folder)}")
    
    
if __name__ == "__main__":
    main()