import argparse
from pathlib import Path
import numpy as np

import torch

from adabmDCA.fasta_utils import get_tokens
from adabmDCA.methods import compute_energy
from adabmDCA.io import load_params, import_from_fasta
from adabmDCA.fasta_utils import encode_sequence
from adabmDCA.custom_fn import one_hot


# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Computes the DCA energies of a sequence dataset.")
    parser.add_argument("-d", "--data",         type=Path,   required=True,      help="Path to the fasta file containing the sequences.")
    parser.add_argument("-p", "--path_params",  type=Path,   required=True,      help="Path to the file containing the parameters of DCA model to sample from.")
    parser.add_argument("-o", "--output",       type=Path,   required=True,      help="Path to the folder where to save the output.")
    parser.add_argument("--alphabet",           type=str,    default="protein",  help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument("--device",             type=str,    default="cuda",     help="(Defaults to cuda). Device to be used. Choose among ['cpu', 'cuda'].")
    return parser


if __name__ == '__main__':        
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # import data
    tokens = get_tokens(args.alphabet)
    names, sequences = import_from_fasta(args.data)
    data = np.vectorize(encode_sequence, excluded=["tokens"], signature="(), () -> (n)")(sequences, tokens)
    data = torch.tensor(data, device=args.device)
    
    # import parameters and compute DCA energies
    params = load_params(args.path_params, tokens=tokens, device=args.device)
    q = params["bias"].shape[1]
    data = one_hot(data, num_classes=q)
    energies = compute_energy(data, params).cpu().numpy()
    
    # Save results in a file
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    fname_out = folder / Path(args.data.stem + "_energies.fasta")
    with open(fname_out, "w") as f:
        for n, s, e in zip(names, sequences, energies):
            f.write(f">{n} | DCAenergy: {e:.3f}\n")
            f.write(f"{s}\n")
    
    print(f"Process completed. Output saved in {fname_out}")