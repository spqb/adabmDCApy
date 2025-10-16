import argparse
import os

import torch
from torch.nn.functional import one_hot

from adabmDCA.fasta import get_tokens
from adabmDCA.utils import get_device, get_dtype
from adabmDCA.io import load_params, import_from_fasta
from adabmDCA.fasta import decode_sequence
from adabmDCA.statmech import compute_energy
from adabmDCA.parser import add_args_energies


# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Computes the DCA energies of a sequence dataset.")
    parser = add_args_energies(parser)
    
    return parser


def main():    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    print("\n" + "".join(["*"] * 10) + f" Computing DCA energies " + "".join(["*"] * 10) + "\n")
    
    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    
    # Check if the data file exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file {args.data} not found.")
    
    # Check if the parameters file exists
    if not os.path.exists(args.path_params):
        raise FileNotFoundError(f"Parameters file {args.path_params} not found.")
    
    # import data
    tokens = get_tokens(args.alphabet)
    names, data = import_from_fasta(args.data, tokens=tokens, remove_duplicates=True, filter_sequences=True)
    sequences = decode_sequence(data, tokens)
    data = torch.tensor(data, dtype=torch.int64)
    
    # import parameters and compute DCA energies
    print(f"Loading parameters from {args.path_params}...")
    params = load_params(args.path_params, tokens=tokens, device=device, dtype=dtype)
    q = params["bias"].shape[1]
    data = one_hot(data, num_classes=q).to(dtype=dtype, device=device)
    print(f"Computing DCA energies...")
    energies = compute_energy(data, params).cpu().numpy()
    
    # Save results in a file
    print("Saving results...")
    folder = args.output
    os.makedirs(folder, exist_ok=True)
    fname_out = os.path.join(folder, f"{os.path.splitext(os.path.basename(args.data))[0]}_energies.fasta")

    with open(fname_out, "w") as f:
        for n, s, e in zip(names, sequences, energies):
            f.write(f">{n} | DCAenergy: {e:.3f}\n")
            f.write(f"{s}\n")
    
    print(f"Process completed. Output saved in {fname_out}")
    
    
if __name__ == "__main__":
    main()