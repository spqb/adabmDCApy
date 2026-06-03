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

    import torch
    from torch.nn.functional import one_hot

    from adabmDCA.fasta import decode_sequence, get_tokens
    from adabmDCA.io import import_from_fasta, load_params
    from adabmDCA.statmech import compute_energy
    from adabmDCA.utils import get_device, get_dtype
    
    print("\n" + "="*80)
    print("  DCA ENERGY COMPUTATION")
    print("="*80 + "\n")
    
    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    
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
    
    # import data
    print("[DATA LOADING]")
    print("-" * 80)
    print(f"  Loading sequences from: {args.data}")
    tokens = get_tokens(args.alphabet)
    names, data = import_from_fasta(args.data, tokens=tokens, remove_duplicates=True, filter_sequences=True)
    sequences = decode_sequence(data, tokens)
    data = torch.tensor(data, dtype=torch.int64)
    n_sequences = len(data)
    seq_length = len(data[0]) if n_sequences > 0 else 0
    print(f"  ✓ Sequences loaded")
    print(f"    • Number of sequences: {n_sequences}")
    print(f"    • Sequence length: {seq_length}")
    
    # import parameters and compute DCA energies
    print(f"  Loading parameters from: {args.path_params}")
    params = load_params(args.path_params, tokens=tokens, device=device, dtype=dtype)
    L = params["bias"].shape[0]
    q = params["bias"].shape[1]
    print(f"  ✓ Parameters loaded (q={q}, L={L})")
    if seq_length != L:
        raise ValueError(
            f"Sequence length ({seq_length}) does not match model length ({L})."
        )
    print("-" * 80 + "\n")
    
    print("[ENERGY COMPUTATION]")
    print("-" * 80)
    data = one_hot(data, num_classes=q).to(dtype=dtype, device=device)
    print(f"  Computing DCA energies for {n_sequences} sequences...")
    energies = compute_energy(data, params).cpu().numpy()
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
    with fname_out.open("w") as f:
        for n, s, e in zip(names, sequences, energies):
            f.write(f">{n} | DCAenergy: {e:.3f}\n")
            f.write(f"{s}\n")
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
