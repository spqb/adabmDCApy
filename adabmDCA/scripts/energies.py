import argparse
import os

from adabmDCA.parser import add_args_energies
from adabmDCA.scripts._utils import ensure_output_dir, require_file


# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Computes the DCA energies of a sequence dataset.")
    parser = add_args_energies(parser)
    
    return parser


def main():    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    from adabmDCA.api.scoring import score_sequences
    from adabmDCA.utils import get_device, get_dtype
    
    print("\n" + "="*80)
    print("  DCA ENERGY COMPUTATION")
    print("="*80 + "\n")
    
    # Set the device
    device = get_device(args.device)
    get_dtype(args.dtype)
    
    # Configuration section
    print("[CONFIGURATION]")
    print("-" * 80)
    template = "  {0:<28} {1:<50}"
    print(template.format("Input data:", args.data))
    print(template.format("Parameters file:", args.path_params))
    print(template.format("Output folder:", args.output))
    print(template.format("Alphabet:", args.alphabet))
    print(template.format("Device:", str(device)))
    print(template.format("Data type:", args.dtype))
    print("-" * 80 + "\n")
    
    require_file(args.data, "Data file")
    require_file(args.path_params, "Parameters file")
    
    # Load data and compute energies through the public application API
    print("[DATA LOADING]")
    print("-" * 80)
    print(f"  Loading sequences from: {args.data}")
    result = score_sequences(
        model=args.path_params,
        fasta_path=args.data,
        alphabet=args.alphabet,
        device=str(device),
        dtype=args.dtype,
        remove_duplicates=True,
    )
    names = result.names
    sequences = result.sequences
    energies = result.energies
    n_sequences = len(sequences)
    seq_length = len(sequences[0]) if n_sequences > 0 else 0
    print(f"  ✓ Sequences loaded")
    print(f"    • Number of sequences: {n_sequences}")
    print(f"    • Sequence length: {seq_length}")
    
    print(f"  Loading parameters from: {args.path_params}")
    L = result.model.length
    q = result.model.num_states
    print(f"  ✓ Parameters loaded (q={q}, L={L})")
    print("-" * 80 + "\n")
    
    print("[ENERGY COMPUTATION]")
    print("-" * 80)
    print(f"  Computing DCA energies for {n_sequences} sequences...")
    mean_energy = energies.mean()
    std_energy = energies.std()
    min_energy = energies.min()
    max_energy = energies.max()
    print(f"  ✓ Energies computed")
    print(f"    • Mean: {mean_energy:.3f}")
    print(f"    • Std:  {std_energy:.3f}")
    print(f"    • Min:  {min_energy:.3f}")
    print(f"    • Max:  {max_energy:.3f}")
    print("-" * 80 + "\n")
    
    # Save results in a file
    print("[OUTPUT]")
    print("-" * 80)
    folder = ensure_output_dir(args.output)
    fname_out = folder / f"{os.path.splitext(os.path.basename(args.data))[0]}_energies.fasta"
    
    print("  Saving results...")
    result.to_fasta(fname_out)
    print(f"  ✓ Results saved: {fname_out}")
    print("-" * 80 + "\n")
    
    print("=" * 80)
    print("  ENERGY COMPUTATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n  Results saved in: {fname_out}")
    print(f"    • Sequences processed: {n_sequences}")
    print(f"    • Mean energy: {mean_energy:.3f}")
    print("\n" + "=" * 80 + "\n")
    
    
if __name__ == "__main__":
    main()
