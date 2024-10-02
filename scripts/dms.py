import argparse
from pathlib import Path
import numpy as np

import torch

from adabmDCA.fasta_utils import get_tokens
from adabmDCA.methods import compute_energy
from adabmDCA.io import load_params, import_from_fasta
from adabmDCA.fasta_utils import encode_sequence, decode_sequence
from adabmDCA.custom_fn import one_hot


# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Generates the Deep Mutational Scanning of a given wild type.")
    parser.add_argument("-d", "--data",         type=Path,   required=True,      
                        help="Path to the fasta file containing wild type sequence. If more than one sequence is present, the first one is used.")
    parser.add_argument("-p", "--path_params",  type=Path,   required=True,      help="Path to the file containing the parameters of DCA model to sample from.")
    parser.add_argument("-o", "--output",       type=Path,   required=True,      help="Path to the folder where to save the output.")
    parser.add_argument("--alphabet",           type=str,    default="protein",  help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument("--device",             type=str,    default="cuda",     help="(Defaults to cuda). Device to be used. Choose among ['cpu', 'cuda'].")
    return parser


if __name__ == '__main__':        
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # import data and parameters
    tokens = get_tokens(args.alphabet)
    names, sequences = import_from_fasta(args.data)
    wt_name = names[0]
    # remove non-alphanumeric characters from wt name
    wt_name = "".join(e for e in wt_name if e.isalnum())
    wt_seq = encode_sequence(sequences[0], tokens)
    
    params = load_params(args.path_params, tokens=tokens, device=args.device)
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
                seq = wt_seq.copy()
                seq[i] = a
                dms.append(seq)
                site_list.append(i)
                old_residues.append(tokens[wt_seq[i]])
                new_residues.append(tokens[a])
    
    print("Computing the DCA scores...")
    dms = one_hot(torch.vstack(dms), num_classes=q).to(args.device)
    energies = compute_energy(dms, params)
    energy_wt = compute_energy(one_hot(torch.tensor([wt_seq], device=args.device), num_classes=q), params)
    deltaE = energies - energy_wt
    
    dms = torch.argmax(dms, -1).cpu().numpy()
    dms_decoded = np.vectorize(decode_sequence, excluded=["tokens"], signature="(l), () -> ()")(dms, tokens)
    
    print("Saving the results...")
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    fname_out = folder / Path(f"{wt_name}_DMS.fasta")
    
    with open(fname_out, "w") as f:
        for i, res_old, res_new, e, seq in zip(site_list, old_residues, new_residues, deltaE, dms_decoded):
            f.write(f">{res_old}{i}{res_new} | DCAscore : {e:.3f}\n")
            f.write(seq + "\n")
    
    print(f"Process completed. Results saved in {fname_out}")