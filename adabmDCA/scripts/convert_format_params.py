# Converts the old DCA parameters format in the new human-readable format that does not require to specify the alphabet
# (i, j, a, b, value) -> (i, j, aa(a), aa(b), value)

import argparse
from pathlib import Path
import numpy as np

from adabmDCA.fasta import get_tokens
from adabmDCA.io import load_params_oldformat, save_params


# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Converts the old DCA parameters format in the new human-readable format that does not require to specify the alphabet (i, j, a, b, value) -> (i, j, aa(a), aa(b), value)")
    parser.add_argument("-p", "--path_params",  type=Path,   required=True,      help="Path to the file containing the parameters of DCA model to sample from.")
    parser.add_argument("--alphabet",           type=str,    default="protein",  help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    return parser


def main():       
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Import parameters in the old format
    params = load_params_oldformat(args.path_params, device="cpu") # This function will be modified
    tokens = get_tokens(args.alphabet)
    
    # Save parameters in the new format
    fname_out = args.path_params.parent / Path(args.path_params.stem + "_newformat.dat")
    L, q, *_ = params["coupling_matrix"].shape
    mask1 = np.zeros(shape=(L, q, L, q))
    mask1[np.nonzero(params["coupling_matrix"])] = 1
    mask2 = np.ones(shape=(L, q, L, q))
    idx1_rm, idx2_rm = np.tril_indices(L, k=0)
    mask2[idx1_rm, :, idx2_rm, :] = np.zeros(shape=(q, q))
    save_params(fname_out, params, np.logical_and(mask1, mask2), tokens=tokens)
    print(f"Completed. Output saved in {fname_out}")
    
    
if __name__ == '__main__':
    main()