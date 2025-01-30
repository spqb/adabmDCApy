import argparse
from pathlib import Path
import numpy as np

import torch

from adabmDCA.fasta import get_tokens
from adabmDCA.io import load_params
from adabmDCA.utils import set_zerosum_gauge, get_device, get_dtype
from adabmDCA.parser import add_args_contacts

# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description='Computes the Frobenius norm matrix extracted from a DCA model.')
    parser = add_args_contacts(parser)
    
    return parser


def main():
    print("\n" + "".join(["*"] * 10) + f" Computing the Frobenius norm " + "".join(["*"] * 10) + "\n")
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    
    # Check if the parameters file exists
    if not Path(args.path_params).exists():
        raise FileNotFoundError(f"Parameters file {args.path_params} not found.")
    
    # Import parameters
    tokens = get_tokens(args.alphabet)
    print(f"Loading parameters from {args.path_params}...")
    params = load_params(args.path_params, tokens=tokens, device=device, dtype=dtype)
    L, q = params["bias"].shape
    
    # Zero-sum gauge
    params = set_zerosum_gauge(params)
    
    # Get index of the gap symbol
    gap_idx = tokens.index("-")
    
    cm_reduced = params["coupling_matrix"]
    # Take all the entries of the coupling matrix except where the gap is involved
    mask = torch.arange(q) != gap_idx
    cm_reduced = cm_reduced[:, mask, :, :][:, :, :, mask]
    
    # Compute the Frobenius norm
    print("Computing the Frobenius norm...")
    F = torch.sqrt(torch.square(cm_reduced).sum([1, 3]))
    
    # Compute the average-product corrected Frobenius norm
    Fapc = F - torch.outer(F.sum(1), F.sum(0)) / F.sum()
    # Set to zero the diagonal
    Fapc = Fapc - torch.diag(Fapc.diag())
    
    # Save the results
    print("Saving results...")
    if args.label is not None:
        fname_out = args.output / Path(f"{args.label}_frobenius.txt")
    else:
        fname_out = args.output / Path(f"frobenius.txt")
      
    with open(fname_out, "w") as f:  
        for i in range(Fapc.shape[0]):
            for j in range(Fapc.shape[1]):
                f.write(f"{i},{j},{Fapc[i, j]}\n")

    print(f"Process completed. Results saved in {fname_out}")
    
if __name__ == "__main__":
    main()     
    