import argparse
from pathlib import Path
import numpy as np

import torch

from adabmDCA.fasta_utils import get_tokens
from adabmDCA.io import load_params
from adabmDCA.utils import set_zerosum_gauge


# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description='Samples from a DCA model.')
    parser.add_argument("-p", "--path_params",  type=Path,   required=True,          help="Path to the file containing the parameters of DCA model to sample from.")
    parser.add_argument("-o", "--output",       type=Path,   required=True,          help="Path to the folder where to save the output.")
    parser.add_argument("--label",              type=str,    default=None,           help="(Defaults to None). If provoded, adds a label to the output files inside the output folder.")
    parser.add_argument("--alphabet",           type=str,    default="protein",      help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument("--device",             type=str,    default="cuda",         help="(Defaults to cuda). Device to be used. Choose among ['cpu', 'cuda'].")
    
    return parser


if __name__ == '__main__':        
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Import parameters
    tokens = get_tokens(args.alphabet)
    params = load_params(args.path_params, tokens=tokens, device=args.device)
    
    # Zero-sum gauge
    params = set_zerosum_gauge(params)
    
    # Get index of the gap symbol
    gap_idx = np.where(np.array([c for c in tokens]) == "-")[0]
    
    cm_reduced = params["coupling_matrix"]
    cm_reduced = cm_reduced[:, gap_idx, :, :]
    cm_reduced = cm_reduced[:, :, :, gap_idx]
    
    # Compute the Frobenius norm
    F = torch.sqrt(torch.square(cm_reduced).sum([1, 3]))
    
    # Compute the average-product corrected Frobenius norm
    Fapc = F - (F.sum(1) * F.sum(0) / F.sum())
    
    # Save the results
    if args.label is not None:
        fname_out = args.output / Path(f"{args.label}_frobenius.txt")
    else:
        fname_out = args.output / Path(f"frobenius.txt")
      
    with open(fname_out, "w") as f:  
        for i in range(Fapc.shape[0]):
            for j in range(Fapc.shape[1]):
                f.write(f"{i},{j},{Fapc[i, j]}\n")

    print(f"Process completed. Results saved in {fname_out}")            
    