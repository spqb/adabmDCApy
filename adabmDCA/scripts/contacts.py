import argparse

from adabmDCA.parser import add_args_contacts
from adabmDCA.scripts._utils import ensure_output_dir, require_file

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

    import matplotlib.pyplot as plt

    from adabmDCA.api.contacts import predict_contacts
    from adabmDCA.plot import plot_contact_map
    from adabmDCA.utils import get_device, get_dtype
    
    print("\n" + "="*80)
    print("  CONTACT MAP PREDICTION")
    print("="*80 + "\n")
    
    # Set the device
    device = get_device(args.device)
    get_dtype(args.dtype)
    
    # Either the data file or the parameters file must be provided
    if args.path_params is None and args.data is None:
        raise ValueError("Either the data file or the parameters file must be provided.")
    
    if args.path_params is not None:
        require_file(args.path_params, "Parameters file")
    if args.path_params is None:
        require_file(args.data, "Data file")

    output_dir = ensure_output_dir(args.output)
    
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
        print(template.format("Pseudocount:", args.pseudocount))
    print(template.format("Output folder:", args.output))
    if args.label is not None:
        print(template.format("Label:", args.label))
    print(template.format("Alphabet:", args.alphabet))
    print(template.format("Device:", str(device)))
    print(template.format("Data type:", args.dtype))
    print("-" * 80 + "\n")
    
    # Compute through the public application API
    print("[CONTACT MAP COMPUTATION]")
    print("-" * 80)
    if args.path_params is None:
        print("  Using mean-field approximation...")
        print(f"  Loading data from: {args.data}")
        print("  Computing Frobenius norm matrix...")
        result = predict_contacts(
            fasta_path=args.data,
            alphabet=args.alphabet,
            pseudocount=args.pseudocount,
            device=str(device),
            dtype=args.dtype,
        )
    else:
        print(f"  Loading parameters from: {args.path_params}")
        result = predict_contacts(
            model=args.path_params,
            alphabet=args.alphabet,
            device=str(device),
            dtype=args.dtype,
        )
        L = result.model.length
        q = result.model.num_states
        print(f"  ✓ Parameters loaded (L={L}, q={q})")
        print("  Computing Frobenius norm matrix...")
    Fapc = result.scores
    
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
        fname_out = output_dir / f"{args.label}_contact_map"
    else:
        fname_out = output_dir / "contact_map"
    
    print("  Saving contact map matrix...")
    matrix_file = fname_out.with_suffix(".txt")
    plot_file = fname_out.with_suffix(".png")
    result.save_matrix(matrix_file)
    print(f"  ✓ Matrix saved: {matrix_file}")
                
    # plot the contact map into a file
    print("  Generating contact map plot...")
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(dpi=150, figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax = plot_contact_map(ax, Fapc)
    fig.tight_layout()
    fig.savefig(plot_file)
    plt.close(fig)
    print(f"  ✓ Plot saved: {plot_file}")
    print("-" * 80 + "\n")

    print("=" * 80)
    print("  CONTACT MAP PREDICTION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n  Results saved in: {args.output}")
    print(f"    • Matrix file: {matrix_file}")
    print(f"    • Plot file:   {plot_file}")
    print(f"    • Matrix size: {map_size} × {map_size}")
    print("\n" + "=" * 80 + "\n")
    
if __name__ == "__main__":
    main()     
    
