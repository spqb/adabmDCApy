import argparse
import os
import matplotlib.pyplot as plt

from adabmDCA.fasta import get_tokens
from adabmDCA.io import load_params
from adabmDCA.utils import get_device, get_dtype
from adabmDCA.parser import add_args_contacts
from adabmDCA.dca import get_contact_map, get_mf_contact_map
from adabmDCA.plot import plot_contact_map
from adabmDCA.dataset import DatasetDCA

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
    
    # Either the data file or the parameters file must be provided
    if args.path_params is None and args.data is None:
        raise ValueError("Either the data file or the parameters file must be provided.")
    
    # Check if the parameters file exists
    if args.path_params is not None and not os.path.exists(args.path_params):
        raise FileNotFoundError(f"Parameters file {args.path_params} not found.")
    
    # Import parameters
    tokens = get_tokens(args.alphabet)
    if args.path_params is None:
        print("Using the mean-field approximation for contact prediction.")
        dataset = DatasetDCA(
            path_data=args.data,
            path_weights=None,
            alphabet=args.alphabet,
            device=device,
            dtype=dtype,
        )
        Fapc = get_mf_contact_map(dataset.data, tokens=tokens, weights=dataset.weights)
    else:
        print(f"Loading parameters from {args.path_params}...")
        params = load_params(args.path_params, tokens=tokens, device=device, dtype=dtype)
        Fapc = get_contact_map(params, tokens)
    
    # Save the results
    print("Saving results...")
    if args.label is not None:
        fname_out = os.path.join(args.output, f"{args.label}_contact_map")
    else:
        fname_out = os.path.join(args.output, "contact_map")

    with open(fname_out + ".txt", "w") as f:
        for i in range(Fapc.shape[0]):
            for j in range(Fapc.shape[1]):
                f.write(f"{i},{j},{Fapc[i, j]}\n")
                
    # plot the contact map into a file
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(dpi=150, figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax = plot_contact_map(ax, Fapc)
    fig.tight_layout()
    fig.savefig(fname_out + ".png")

    print(f"Process completed. Results saved in {fname_out}")
    
if __name__ == "__main__":
    main()     
    