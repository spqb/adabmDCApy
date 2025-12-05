import argparse
import os
import pandas as pd
from tqdm import tqdm
import torch

from adabmDCA.fasta import (
    get_tokens,
    write_fasta,
)
from adabmDCA.resampling import compute_mixing_time
from adabmDCA.io import load_params
from adabmDCA.dataset import DatasetDCA
from adabmDCA.utils import init_chains, resample_sequences, get_device, get_dtype
from adabmDCA.sampling import get_sampler
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
    
    print("\n" + "="*80)
    print("  SAMPLING FROM DCA MODEL")
    print("="*80 + "\n")
    
    # Create output folder
    folder = args.output
    # Create the folder where to save the samples
    os.makedirs(folder, exist_ok=True)

    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    tokens = get_tokens(args.alphabet)
    
    # Configuration section
    print("[CONFIGURATION]")
    print("-" * 80)
    template = "  {0:<28} {1:<50}"
    print(template.format("Parameters file:", args.path_params))
    if args.data is not None:
        print(template.format("Reference data:", args.data))
    print(template.format("Output folder:", str(folder)))
    print(template.format("Output label:", args.label))
    print(template.format("Number of samples:", args.ngen))
    print(template.format("Sampler:", args.sampler))
    print(template.format("Beta (temperature):", args.beta))
    print(template.format("Device:", str(device)))
    print(template.format("Data type:", args.dtype))
    print("-" * 80 + "\n")
    
    # Check that the data file exists
    if args.data is not None and not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file {args.data} not found.")
    
    # Check that the parameters file exists
    if not os.path.exists(args.path_params):
        raise FileNotFoundError(f"Parameters file {args.path_params} not found.")
        
    # Import parameters
    print("[MODEL LOADING]")
    print("-" * 80)
    print(f"  Loading parameters from: {args.path_params}")
    params = load_params(fname=args.path_params, tokens=tokens, device=device, dtype=dtype)
    L, q = params["bias"].shape
    print(f"  ✓ Parameters loaded (L={L}, q={q})")
    
    # Select the sampler
    print(f"  Initializing sampler: {args.sampler}")
    sampler = torch.jit.script(get_sampler(args.sampler))
    print(f"  ✓ Sampler ready")
    
    # Initialize the chains at random
    print(f"  Initializing {args.ngen} random chains...")
    samples = init_chains(
        num_chains=args.ngen,
        L=L,
        q=q,
        device=device,
        dtype=dtype,
    )
    print(f"  ✓ Chains initialized")
    print("-" * 80 + "\n")
    
    # Import data
    if args.data is not None:
        print("[DATA ANALYSIS]")
        print("-" * 80)
        print(f"  Loading reference data from: {args.data}")
        dataset = DatasetDCA(
            path_data=args.data,
            path_weights=args.weights,
            alphabet=tokens,
            clustering_th=args.clustering_seqid,
            no_reweighting=args.no_reweighting,
            filter_sequences=True,
            remove_duplicates=True,
            device=device,
            dtype=dtype,
            message=False,
        )
        nmeasure = min(args.nmeasure, len(dataset))
        data_resampled = resample_sequences(dataset.data, dataset.weights, nmeasure)
        print(f"  ✓ Data loaded ({len(dataset)} sequences)")
    
        if args.pseudocount is None:
            args.pseudocount = 1. / dataset.weights.sum().item()
        print(f"  Pseudocount: {args.pseudocount:.6f}")
        
        # Compute the mixing time
        print(f"  Computing mixing time (max {args.max_nsweeps} sweeps)...")
        results_mix = compute_mixing_time(
            sampler=sampler,
            data=data_resampled,
            params=params,
            n_max_sweeps=args.max_nsweeps,
            beta=args.beta,
        )
        mixing_time = results_mix["t_half"][-1]
        print(f"  ✓ Mixing time: {mixing_time} sweeps")
        print("-" * 80 + "\n")
        
        # Sample from random initialization
        num_sweeps = args.nmix * mixing_time
        print("[SAMPLING]")
        print("-" * 80)
        print(f"  Total sweeps: {num_sweeps} ({args.nmix} × mixing time)")
        print(f"  Running MCMC sampling...")
    
        pbar = tqdm(
            total=num_sweeps,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
            bar_format="  {desc}: [{bar}] {n}/{total} sweeps [{elapsed}]",
        )
        pbar.set_description("Sampling")

        # Compute single and two-site frequencies of the data
        fi = get_freq_single_point(data=dataset.data, weights=dataset.weights, pseudo_count=args.pseudocount)
        fij = get_freq_two_points(data=dataset.data, weights=dataset.weights, pseudo_count=args.pseudocount)

        # Dictionary to store Pearson coefficient and slope along the sampling
        results_sampling = {
            "nsweeps" : [],
            "pearson" : [],
            "slope" : [],
        }

        # Sample for (args.nmix * mixing_time) sweeps starting from random initialization
        for i in range(num_sweeps):
            pbar.update(1)
            samples = sampler(chains=samples, params=params, nsweeps=1, beta=args.beta)
            pi = get_freq_single_point(data=samples, weights=None, pseudo_count=0.)
            pij = get_freq_two_points(data=samples, weights=None, pseudo_count=0.)
            pearson, slope = get_correlation_two_points(fi=fi, pi=pi, fij=fij, pij=pij)
            results_sampling["nsweeps"].append(i)
            results_sampling["pearson"].append(pearson)
            results_sampling["slope"].append(slope)
        pbar.close()
        print(f"  ✓ Sampling completed")
        print(f"  Final Pearson correlation: {pearson:.4f}")
        print("-" * 80 + "\n")
    
    else:
        num_sweeps = args.max_nsweeps
        print("[SAMPLING]")
        print("-" * 80)
        print(f"  Total sweeps: {num_sweeps}")
        print(f"  Running MCMC sampling...")
        pbar = tqdm(
            total=num_sweeps,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
            bar_format="  {desc}: [{bar}] {n}/{total} sweeps [{elapsed}]",
        )
        pbar.set_description("Sampling")
        for i in range(num_sweeps):
            pbar.update(1)
            samples = sampler(chains=samples, params=params, nsweeps=1, beta=args.beta)
        pbar.close()
        print(f"  ✓ Sampling completed")
        print("-" * 80 + "\n")
        results_mix = {}
        results_sampling = {}
        
    
    # Compute the energy of the samples
    print("[OUTPUT]")
    print("-" * 80)
    print("  Computing sample energies...")
    energies = compute_energy(samples, params=params).cpu().numpy()
    mean_energy = energies.mean()
    std_energy = energies.std()
    print(f"  ✓ Mean energy: {mean_energy:.3f} ± {std_energy:.3f}")
    
    print("  Saving samples...")
    headers = [f"sequence {i+1} | DCAenergy: {energies[i]:.3f}" for i in range(args.ngen)]
    fasta_file = os.path.join(folder, f"{args.label}_samples.fasta")
    write_fasta(
        fname=fasta_file,
        headers=headers,
        sequences=samples,
        remove_gaps=False,
        tokens=tokens,
    )
    print(f"  ✓ Samples saved: {fasta_file}")
    
    print("  Saving logs...")
    df_mix_log = pd.DataFrame.from_dict(results_mix)
    mix_log_file = os.path.join(folder, f"{args.label}_mix.log")
    df_mix_log.to_csv(mix_log_file, index=False)
    
    df_samp_log = pd.DataFrame.from_dict(results_sampling)
    samp_log_file = os.path.join(folder, f"{args.label}_sampling.log")
    df_samp_log.to_csv(samp_log_file, index=False)
    print(f"  ✓ Logs saved")
    print("-" * 80 + "\n")
    
    print("=" * 80)
    print("  SAMPLING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n  Results saved in: {str(folder)}")
    print(f"    • Samples: {fasta_file}")
    if results_mix:
        print(f"    • Mixing time log: {mix_log_file}")
    if results_sampling:
        print(f"    • Sampling log: {samp_log_file}")
    print("\n" + "=" * 80 + "\n")
    
    
if __name__ == "__main__":
    main()