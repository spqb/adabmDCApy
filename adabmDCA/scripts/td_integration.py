import argparse
from pathlib import Path

import torch
import numpy as np
import copy

from adabmDCA.fasta import get_tokens
from adabmDCA.utils import init_parameters, init_chains, get_device, get_dtype
from adabmDCA.io import load_params, load_chains, import_from_fasta
# from adabmDCA.fasta import encode_sequence, decode_sequence
from adabmDCA.functional import one_hot
from adabmDCA.sampling import get_sampler
from adabmDCA.statmech import compute_energy
from adabmDCA.parser import add_args_tdint
from adabmDCA.resampling import compute_seqID 




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

    # select sampler
    sampler = get_sampler("gibbs")

    # read and encode natural data
    tokens = get_tokens(args.alphabet)
    _, sequences = import_from_fasta(args.data, tokens=tokens, filter_sequences=True)
    M, L, q = len(sequences), len(sequences[0]), len(tokens)
    nat_data = one_hot(torch.tensor(sequences, device=device, dtype=torch.int32), num_classes=q).to(dtype)
    print(f"Number of sequences in the MSA: M={M}")
    print(f"Length of the MSA: L={L}")
    print(f"Number of Potts states: q={q}\n")

    # read parameters
    params = load_params(args.path_params, tokens=tokens, device=device, dtype=dtype)

    # read chains
    if chains_path is None:
        chains = init_chains(args.ngen, L, q)
    else:   
        chains = load_chains(args.path_chains, tokens=tokens)
        chains = one_hot(torch.tensor(chains, device=device, dtype=torch.int32), num_classes=q).to(dtype)
        if chains.shape[0] != args.ngen:
            chains = resample_sequences(chains, weights=torch.ones(chains.shape[0])/chains.shape[0], nextract=args.ngen)
        
    # target sequence
    _, targetseq = import_from_fasta(args.path_targetseq, tokens=tokens, filter_sequences=True) 
    targetseq = one_hot(torch.tensor(targetseq, device=device, dtype=torch.int32), num_classes=q).to(dtype).squeeze(dim=0)

    # Sampling to thermalize at theta = 0
    n_sweep_0 = 10
    chains_0 = sampler(chains, params, n_sweep_0) 
    ave_energy_0 = torch.mean(compute_energy(chains_0, params))
    seqID_0 = compute_seqID(chains_0, targetseq)
    print(f"Average seqID at theta = 0: {seqID_0.mean():.2f}")
    print(f"Average energy at theta = 0: {ave_energy_0:.2f}")

    # Sampling to thermalize at theta = theta_max
    params_theta = copy.deepcopy(params)
    params_theta["bias"] += theta_max * targetseq
    
    chains_theta = one_hot(torch.randint(0, q, size=(args.ngen, L), device=device), num_classes=q)
    nsweep_theta_max = 100
    chains_theta = sampler(chains_theta, params_theta, nsweep_theta_max)
    ave_energy_theta = torch.mean(energy_theta)
    seqID_max = compute_seqID(chains_theta, targetseq)
    print(f"Average seqID at theta = {theta_max}: {seqID_max.mean():.2f}")
    print(f"Average energy at theta = {theta_max}: {ave_energy_theta:.2f}\n")

    # Find theta_max to generate 10% WT sequences
    p_wt =  (seqID_max == L).sum().item() / args.ngen
    nsweep_find_theta = 100
    while p_wt <= 0.1:
        theta_max += 0.05 * theta_max
        print(f"Number of sequences collapsed to WT is less than 10%. Increasing theta max to: {theta_max:.2f}", flush=True)
        params_theta["bias"] = params["bias"] + theta_max * targetseq
        chains_theta = sampler(chains_theta, params_theta, nsweep_find_theta)
        seqID = compute_seqID(chains_theta, targetseq)
        p_wt = (seqID == L).sum().item() / args.ngen
        energy_theta = compute_energy(chains_theta, params)
        ave_energy_theta = torch.mean(energy_theta)
        print(f"{(p_wt * 100):.2f}% sequences collapse to wt", flush=True)
    
    # Thermodynamic Integration
    int_step = 200
    nsweeps = 100

    F_max = np.log(p_wt) + torch.mean(compute_energy(chains_theta[seqID == L], params_theta))
    thetas = torch.linspace(0, theta_max, int_step) 
    factor = theta_max / (2*int_step)

    t_start = time.time()
    F, S, integral = F_max, 0, 0
    torch.set_printoptions(precision=2)

    for i, theta in enumerate(thetas):
        print(f"\nstep n:{i}, theta={theta:.2f}")
        params_theta["bias"] = params["bias"] + theta * targetseq
        chains_theta = sampler(chains_theta, params_theta, nsweeps)
        seqID = compute_seqID(chains_theta, targetseq)
        mean_seqID = seqID.mean()
        print(f"average seqID: {mean_seqID:.3f}", flush=True)
        if i == 0 or i == int_step - 1:
            F += factor * torch.mean(seqID) 
            integral += factor * mean_seqID
        else:
            F += 2 * factor * mean_seqID
            integral += 2 * factor * mean_seqID

        S = ave_energy_0 - F
        print(f"Free energy: {F:.3f}")
        print(f"Integral: {integral:.3f}")
        print(f"Entropy: {S:.3f}")
        










def main():
    print("\n" + "".join(["*"] * 10) + f" Generating DMS " + "".join(["*"] * 10) + "\n")
    
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















    # import data and parameters
    tokens = get_tokens(args.alphabet)
    names, sequences = import_from_fasta(args.data)
    wt_name = names[0]
    # remove non-alphanumeric characters from wt name
    wt_name = "".join(e for e in wt_name if e.isalnum())
    wt_seq = torch.tensor(encode_sequence(sequences[0], tokens))
    
    print(f"Loading parameters from {args.path_params}...")
    params = load_params(args.path_params, tokens=tokens, device=device, dtype=dtype)
    L, q = params["bias"].shape
    
    # generate DMS
    dms = []
    site_list = []
    old_residues = []
    new_residues = []
    
    print("Generating the single mutant library...")
    for i in range(L):
        for a in range(q):
            if wt_seq[i] != a:
                seq = wt_seq.clone()
                seq[i] = a
                dms.append(seq)
                site_list.append(i)
                old_residues.append(tokens[wt_seq[i]])
                new_residues.append(tokens[a])
    
    print("Computing the DCA scores...")
    dms = torch.vstack(dms).to(device=device)
    dms = one_hot(dms, num_classes=q).to(dtype)
    energies = compute_energy(dms, params)
    energy_wt = compute_energy(one_hot(wt_seq.view(1, -1).to(device), num_classes=q).to(dtype), params)
    deltaE = energies - energy_wt
    
    dms = torch.argmax(dms, -1).cpu().numpy()
    dms_decoded = decode_sequence(dms, tokens)
    
    print("Saving the results...")
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    fname_out = folder / Path(f"{wt_name}_DMS.fasta")
    
    with open(fname_out, "w") as f:
        for i, res_old, res_new, e, seq in zip(site_list, old_residues, new_residues, deltaE, dms_decoded):
            f.write(f">{res_old}{i}{res_new} | DCAscore: {e:.3f}\n")
            f.write(seq + "\n")
    
    print(f"Process completed. Results saved in {fname_out}")
    

if __name__ == "__main__":
    main()