import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from adabmDCA.fasta import get_tokens
from adabmDCA.io import load_params
from adabmDCA.utils import get_device, get_dtype
from adabmDCA.parser import add_args_contacts
from adabmDCA.dca import get_contact_map
from adabmDCA.plot import plot_contact_map

# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description='Computes the Frobenius norm matrix extracted from a DCA model.')
    parser = add_args_contacts(parser)
    
    return parser


def main():    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    print("\n" + "".join(["*"] * 10) + f" Computing the Frobenius norm " + "".join(["*"] * 10) + "\n")
    
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
    Fapc = get_contact_map(params, tokens)
    
    # Save the results
    print("Saving results...")
    if args.label is not None:
        fname_out = args.output / Path(f"{args.label}_contact_map.txt")
    else:
        fname_out = args.output / Path(f"contact_map.txt")
      
    with open(fname_out, "w") as f:  
        for i in range(Fapc.shape[0]):
            for j in range(Fapc.shape[1]):
                f.write(f"{i},{j},{Fapc[i, j]}\n")
                
    # plot the contact map into a file
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(dpi=150, figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax = plot_contact_map(ax, Fapc)
    fig.tight_layout()
    fig.savefig(fname_out.with_suffix(".png"))

    print(f"Process completed. Results saved in {fname_out}")
    
if __name__ == "__main__":
    main()     
    