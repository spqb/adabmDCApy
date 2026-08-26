import argparse

from adabmDCA.parser import add_args_dms
from adabmDCA.scripts._utils import ensure_output_dir, require_file


# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Generates the Deep Mutational Scanning of a given wild type.")
    parser = add_args_dms(parser)
    
    return parser


def main():    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    from adabmDCA.api.mutations import scan_mutations
    from adabmDCA.io import import_from_fasta
    from adabmDCA.utils import get_device, get_dtype
    
    print("\n" + "="*80)
    print("  DEEP MUTATIONAL SCANNING (DMS)")
    print("="*80 + "\n")
    
    # Set the device
    device = get_device(args.device)
    get_dtype(args.dtype)
    
    # Configuration section
    print("[CONFIGURATION]")
    print("-" * 80)
    template = "  {0:<28} {1:<50}"
    print(template.format("Wild-type sequence:", args.data))
    print(template.format("Parameters file:", args.path_params))
    print(template.format("Output folder:", args.output))
    print(template.format("Alphabet:", args.alphabet))
    print(template.format("Device:", str(device)))
    print(template.format("Data type:", args.dtype))
    print("-" * 80 + "\n")
    
    require_file(args.data, "Data file")
    require_file(args.path_params, "Parameters file")
    
    # import data and parameters
    print("[DATA LOADING]")
    print("-" * 80)
    print(f"  Loading wild-type sequence from: {args.data}")
    names, sequences = import_from_fasta(args.data)
    wt_name = names[0]
    # remove non-alphanumeric characters from wt name
    wt_name = "".join(e for e in wt_name if e.isalnum())
    wild_type = str(sequences[0])
    L_wt = len(wild_type)
    print(f"  ✓ Wild-type loaded: {wt_name}")
    print(f"    • Length: {L_wt}")
    
    print(f"  Loading parameters from: {args.path_params}")
    result = scan_mutations(
        wild_type,
        model=args.path_params,
        name=wt_name,
        alphabet=args.alphabet,
        device=str(device),
        dtype=args.dtype,
    )
    L = result.model.length
    q = result.model.num_states
    print(f"  ✓ Parameters loaded (L={L}, q={q})")
    print("-" * 80 + "\n")
    
    # generate DMS
    print("[MUTANT LIBRARY GENERATION]")
    print("-" * 80)
    print(f"  Generating single-point mutant library...")
    n_mutants = len(result.mutations)
    print(f"  ✓ Mutant library generated: {n_mutants} single mutants")
    print("-" * 80 + "\n")
    
    print("[ENERGY COMPUTATION]")
    print("-" * 80)
    print(f"  Computing DCA scores for {n_mutants} mutants...")
    energy_wt = result.wild_type_energy
    deltaE = result.delta_energies
    print(f"  ✓ DCA scores computed")
    print(f"    • Wild-type energy: {energy_wt:.3f}")
    print(f"    • ΔE range: [{deltaE.min():.3f}, {deltaE.max():.3f}]")
    print(f"    • Mean ΔE: {deltaE.mean():.3f}")
    print("-" * 80 + "\n")
    
    print("[OUTPUT]")
    print("-" * 80)
    folder = ensure_output_dir(args.output)
    fname_out = folder / f"{wt_name}_DMS.fasta"
    
    print("  Saving DMS results...")
    result.to_fasta(fname_out)
    print(f"  ✓ Results saved: {fname_out}")
    print("-" * 80 + "\n")
    
    print("=" * 80)
    print("  DEEP MUTATIONAL SCANNING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n  Results saved in: {fname_out}")
    print(f"    • Wild-type: {wt_name}")
    print(f"    • Total mutants: {n_mutants}")
    print(f"    • Sites scanned: {L}")
    print("\n" + "=" * 80 + "\n")
    

if __name__ == "__main__":
    main()
