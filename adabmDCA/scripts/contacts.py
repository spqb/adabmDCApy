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
    
    print("\n" + "="*80)
    print("  CONTACT MAP PREDICTION")
    print("="*80 + "\n")
    
    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    
    # Either the data file or the parameters file must be provided
    if args.path_params is None and args.data is None:
        raise ValueError("Either the data file or the parameters file must be provided.")
    
    # Check if the parameters file exists
    if args.path_params is not None and not os.path.exists(args.path_params):
        raise FileNotFoundError(f"Parameters file {args.path_params} not found.")
    
    # Configuration section
    print("[CONFIGURATION]")
    print("-" * 80)
    template = "  {0:<28} {1:<50}"
    if args.path_params is not None:
        print(template.format("Method:", "DCA model"))
        print(template.format("Parameters file:", args.path_params))
    else:
        print(template.format("Method:", "Mean-field approximation"))
        print(template.format("Data file:", args.data))
    print(template.format("Output folder:", args.output))
    if args.label is not None:
        print(template.format("Label:", args.label))
    print(template.format("Alphabet:", args.alphabet))
    print(template.format("Device:", str(device)))
    print(template.format("Data type:", args.dtype))
    print("-" * 80 + "\n")
    
    # Import parameters
    print("[CONTACT MAP COMPUTATION]")
    print("-" * 80)
    tokens = get_tokens(args.alphabet)
    if args.path_params is None:
        print("  Using mean-field approximation...")
        print(f"  Loading data from: {args.data}")
        dataset = DatasetDCA(
            path_data=args.data,
            path_weights=None,
            alphabet=args.alphabet,
            device=device,
            dtype=dtype,
            remove_duplicates=True,
            filter_sequences=True,
            message=False,
        )
        print(f"  ✓ Data loaded ({len(dataset)} sequences)")
        print("  Computing Frobenius norm matrix...")
        Fapc = get_mf_contact_map(dataset.data, tokens=tokens, weights=dataset.weights)
    else:
        print(f"  Loading parameters from: {args.path_params}")
        params = load_params(args.path_params, tokens=tokens, device=device, dtype=dtype)
        L, q = params["bias"].shape
        print(f"  ✓ Parameters loaded (L={L}, q={q})")
        print("  Computing Frobenius norm matrix...")
        Fapc = get_contact_map(params, tokens)
    
    map_size = Fapc.shape[0]
    max_score = Fapc.max()
    min_score = Fapc.min()
    mean_score = Fapc.mean()
    print(f"  ✓ Contact map computed")
    print(f"    • Matrix size: {map_size} × {map_size}")
    print(f"    • Score range: [{min_score:.4f}, {max_score:.4f}]")
    print(f"    • Mean score: {mean_score:.4f}")
    print("-" * 80 + "\n")
    
    # Save the results
    print("[OUTPUT]")
    print("-" * 80)
    if args.label is not None:
        fname_out = os.path.join(args.output, f"{args.label}_contact_map")
    else:
        fname_out = os.path.join(args.output, "contact_map")
    
    print("  Saving contact map matrix...")
    with open(fname_out + ".txt", "w") as f:
        for i in range(Fapc.shape[0]):
            for j in range(Fapc.shape[1]):
                f.write(f"{i},{j},{Fapc[i, j]}\n")
    print(f"  ✓ Matrix saved: {fname_out}.txt")
                
    # plot the contact map into a file
    print("  Generating contact map plot...")
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(dpi=150, figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax = plot_contact_map(ax, Fapc)
    fig.tight_layout()
    fig.savefig(fname_out + ".png")
    print(f"  ✓ Plot saved: {fname_out}.png")
    print("-" * 80 + "\n")

    print("=" * 80)
    print("  CONTACT MAP PREDICTION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n  Results saved in: {args.output}")
    print(f"    • Matrix file: {fname_out}.txt")
    print(f"    • Plot file:   {fname_out}.png")
    print(f"    • Matrix size: {map_size} × {map_size}")
    print("\n" + "=" * 80 + "\n")
    
if __name__ == "__main__":
    main()     
    