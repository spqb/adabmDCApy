import argparse
import os

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
    
    print("\n" + "="*80)
    print("  DEEP MUTATIONAL SCANNING (DMS)")
    print("="*80 + "\n")
    
    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    
    # Configuration section
    print("[CONFIGURATION]")
    print("-" * 80)
    template = "  {0:<28} {1:<50}"
    print(template.format("Wild-type sequence:", args.data))
    print(template.format("Parameters file:", args.path_params))
    print(template.format("Output folder:", args.output))
    print(template.format("Alphabet:", args.alphabet))
    print(template.format("Device:", str(device)))
    print(template.format("Data type:", args.dtype))
    print("-" * 80 + "\n")
    
    # Check if the data file exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file {args.data} not found.")
    
    # Check if the parameters file exists
    if not os.path.exists(args.path_params):
        raise FileNotFoundError(f"Parameters file {args.path_params} not found.")
    
    # import data and parameters
    print("[DATA LOADING]")
    print("-" * 80)
    print(f"  Loading wild-type sequence from: {args.data}")
    tokens = get_tokens(args.alphabet)
    names, sequences = import_from_fasta(args.data)
    wt_name = names[0]
    # remove non-alphanumeric characters from wt name
    wt_name = "".join(e for e in wt_name if e.isalnum())
    wt_seq = torch.tensor(encode_sequence(sequences[0], tokens))
    L_wt = len(wt_seq)
    print(f"  ✓ Wild-type loaded: {wt_name}")
    print(f"    • Length: {L_wt}")
    
    print(f"  Loading parameters from: {args.path_params}")
    params = load_params(args.path_params, tokens=tokens, device=device, dtype=dtype)
    L, q = params["bias"].shape
    print(f"  ✓ Parameters loaded (L={L}, q={q})")
    print("-" * 80 + "\n")
    
    # generate DMS
    print("[MUTANT LIBRARY GENERATION]")
    print("-" * 80)
    dms = []
    site_list = []
    old_residues = []
    new_residues = []
    
    print(f"  Generating single-point mutant library...")
    for i in range(L):
        for a in range(q):
            if wt_seq[i] != a:
                seq = wt_seq.clone()
                seq[i] = a
                dms.append(seq)
                site_list.append(i)
                old_residues.append(tokens[wt_seq[i]])
                new_residues.append(tokens[a])
    
    n_mutants = len(dms)
    print(f"  ✓ Mutant library generated: {n_mutants} single mutants")
    print("-" * 80 + "\n")
    
    print("[ENERGY COMPUTATION]")
    print("-" * 80)
    print(f"  Computing DCA scores for {n_mutants} mutants...")
    dms = torch.vstack(dms).to(device=device)
    dms = one_hot(dms, num_classes=q).to(dtype)
    energies = compute_energy(dms, params)
    energy_wt = compute_energy(one_hot(wt_seq.view(1, -1).to(device), num_classes=q).to(dtype), params)
    deltaE = energies - energy_wt
    print(f"  ✓ DCA scores computed")
    print(f"    • Wild-type energy: {energy_wt.item():.3f}")
    print(f"    • ΔE range: [{deltaE.min().item():.3f}, {deltaE.max().item():.3f}]")
    print(f"    • Mean ΔE: {deltaE.mean().item():.3f}")
    
    dms = torch.argmax(dms, -1).cpu().numpy()
    dms_decoded = decode_sequence(dms, tokens)
    print("-" * 80 + "\n")
    
    print("[OUTPUT]")
    print("-" * 80)
    folder = args.output
    os.makedirs(folder, exist_ok=True)
    fname_out = os.path.join(folder, f"{wt_name}_DMS.fasta")
    
    print("  Saving DMS results...")
    with open(fname_out, "w") as f:
        for i, res_old, res_new, e, seq in zip(site_list, old_residues, new_residues, deltaE, dms_decoded):
            f.write(f">{res_old}{i}{res_new} | DCAscore: {e:.3f}\n")
            f.write(seq + "\n")
    print(f"  ✓ Results saved: {fname_out}")
    print("-" * 80 + "\n")
    
    print("=" * 80)
    print("  DEEP MUTATIONAL SCANNING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n  Results saved in: {fname_out}")
    print(f"    • Wild-type: {wt_name}")
    print(f"    • Total mutants: {n_mutants}")
    print(f"    • Sites scanned: {L}")
    print("\n" + "=" * 80 + "\n")
    

if __name__ == "__main__":
    main()