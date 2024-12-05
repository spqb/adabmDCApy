import argparse
from pathlib import Path
import numpy as np

import torch

from adabmDCA.fasta import get_tokens
from adabmDCA.utils import get_device, get_dtype
from adabmDCA.io import load_params, import_from_fasta
from adabmDCA.fasta import encode_sequence, decode_sequence
from adabmDCA.functional import one_hot
from adabmDCA.statmech import compute_energy
from adabmDCA.parser import add_args_dms


# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Generates the Deep Mutational Scanning of a given wild type.")
    parser = add_args_dms(parser)
    
    return parser


def main():    
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    
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