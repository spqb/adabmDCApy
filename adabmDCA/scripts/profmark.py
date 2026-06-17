import argparse
from pathlib import Path

from adabmDCA.parser import add_args_profmark
from adabmDCA.scripts._utils import require_file

# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Splits a multi-sequence alignment FASTA file into training and test sets using the Cobalt algorithm described in Petti et al (2022).")
    parser = add_args_profmark(parser)
    
    return parser

def main(args=None):
    if args is None:
        args = create_parser().parse_args()

    import torch

    from adabmDCA.cobalt import run_cobalt
    from adabmDCA.fasta import import_from_fasta, get_tokens, write_fasta
    from adabmDCA.utils import get_device

    print("\n" + "="*80)
    print("  DATASET SPLITTING - COBALT ALGORITHM")
    print("="*80 + "\n")
    
    device = get_device(args.device, message=False)
    
    # Configuration section
    print("[CONFIGURATION]")
    print("-" * 80)
    template = "  {0:<28} {1:<50}"
    print(template.format("Input MSA:", args.input_msa))
    print(template.format("Output prefix:", args.output_prefix))
    print(template.format("Alphabet:", args.alphabet))
    print(template.format("Sequence identity t1:", args.t1))
    print(template.format("Sequence identity t2:", args.t2))
    print(template.format("Sequence identity t3:", args.t3))
    print(template.format("Max train sequences:", args.maxtrain if args.maxtrain is not None else "None"))
    print(template.format("Max test sequences:", args.maxtest if args.maxtest is not None else "None"))
    print(template.format("Best-of iterations:", args.bestof))
    print(template.format("Random seed:", args.seed))
    print(template.format("Device:", str(device)))
    print("-" * 80 + "\n")
    
    output_prefix = Path(args.output_prefix)
    if output_prefix.parent != Path("."):
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
    require_file(args.input_msa, "Input MSA file")
    
    # load the MSA
    print("[DATA LOADING]")
    print("-" * 80)
    print(f"  Loading MSA from: {args.input_msa}")
    tokens = get_tokens(args.alphabet)
    headers, msa = import_from_fasta(args.input_msa, tokens)
    msa = torch.tensor(msa, device=device)
    n_sequences = len(headers)
    seq_length = len(msa[0])
    print(f"  ✓ MSA loaded")
    print(f"    • Sequences: {n_sequences}")
    print(f"    • Length: {seq_length}")
    print("-" * 80 + "\n")
    
    # run the Cobalt algorithm
    print("[COBALT ALGORITHM]")
    print("-" * 80)
    print(f"  Running {args.bestof} iteration(s) to find optimal split...")
    print("-" * 80)
    geom_mean = 0
    rnd_gen = torch.Generator(device=device).manual_seed(args.seed)
    for i in range(args.bestof):
        headers_train_prop, train_prop, headers_test_prop, test_prop = run_cobalt(
            headers,
            msa,
            args.t1,
            args.t2,
            args.t3,
            args.maxtrain,
            args.maxtest,
            rnd_gen,
        )
        geom_mean_prop = len(train_prop) * len(test_prop)
        print(f"  Iteration {i+1:2d}: train={len(train_prop):4d} | test={len(test_prop):4d} | score={geom_mean_prop:8d}")
        if geom_mean_prop > geom_mean:
            geom_mean = geom_mean_prop
            headers_train, train, headers_test, test = (
                headers_train_prop,
                train_prop,
                headers_test_prop,
                test_prop,
            )
    
    print("-" * 80)
    if geom_mean == 0:
        print("  ⚠ FAILED: No valid split found")
        print(f"    Parameters: t1={args.t1}, t2={args.t2}, t3={args.t3}")
        print("-" * 80 + "\n")
        return
    
    print(f"  ✓ Best split found")
    print(f"    • Training set: {len(train)} sequences")
    print(f"    • Test set: {len(test)} sequences")
    print(f"    • Geometric mean: {geom_mean}")
    print("-" * 80 + "\n")
    
    # write the training and test sets to files
    print("[OUTPUT]")
    print("-" * 80)
    train_file = output_prefix.with_suffix(output_prefix.suffix + ".train.fasta")
    test_file = output_prefix.with_suffix(output_prefix.suffix + ".test.fasta")
    
    print("  Saving training set...")
    write_fasta(
        train_file,
        headers_train,
        train,
        tokens=tokens,
    )
    print(f"  ✓ Training set saved: {train_file}")
    
    print("  Saving test set...")
    write_fasta(
        test_file,
        headers_test,
        test,
        tokens=tokens,
    )
    print(f"  ✓ Test set saved: {test_file}")
    print("-" * 80 + "\n")
    
    print("=" * 80)
    print("  DATASET SPLITTING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n  Results saved in: {output_prefix.parent}")
    print(f"    • Training set: {train_file} ({len(train)} sequences)")
    print(f"    • Test set:     {test_file} ({len(test)} sequences)")
    print("\n" + "=" * 80 + "\n")
    
if __name__ == "__main__":
    main()
