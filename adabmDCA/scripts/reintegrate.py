import argparse
from contextlib import contextmanager
import sys

from adabmDCA.parser import add_args_train, add_args_reintegration
from adabmDCA.scripts._utils import ensure_output_dir, require_file


@contextmanager
def training_argv(args: list[str]):
    previous_argv = sys.argv[:]
    sys.argv = ["adabmDCA train", *args]
    try:
        yield
    finally:
        sys.argv = previous_argv

def create_parser():
    parser = argparse.ArgumentParser(description='Reintegrate a DCA model.')
    parser = add_args_train(parser)
    parser = add_args_reintegration(parser)
    return parser

def main():    
    parser = create_parser()
    args = parser.parse_args()

    import numpy as np
    import torch

    from adabmDCA.dataset import DatasetDCA
    from adabmDCA.fasta import get_tokens, write_fasta
    from adabmDCA.utils import get_device, get_dtype
    
    print("\n" + "="*80)
    print("  REINTEGRATED DCA MODEL TRAINING")
    print("="*80 + "\n")
    
    device = get_device(args.device, message=False)
    dtype = get_dtype(args.dtype)
    tokens = get_tokens(args.alphabet)
    
    # Configuration section
    print("[CONFIGURATION]")
    print("-" * 80)
    template = "  {0:<28} {1:<50}"
    print(template.format("Natural data:", args.data))
    print(template.format("Reintegration data:", args.reint))
    print(template.format("Adjustment vector:", args.adj))
    if args.lambda_ is not None:
        print(template.format("Lambda value:", args.lambda_))
    print(template.format("Output folder:", args.output))
    print(template.format("Model type:", args.model))
    print(template.format("Alphabet:", args.alphabet))
    print(template.format("Device:", str(device)))
    print(template.format("Data type:", args.dtype))
    print("-" * 80 + "\n")
    require_file(args.data, "Natural data file")
    require_file(args.reint, "Reintegration data file")
    require_file(args.adj, "Adjustment vector file")
    if args.path_params is not None:
        require_file(args.path_params, "Parameters file")
    if args.path_chains is not None:
        require_file(args.path_chains, "Chains file")
    
    # Create the folder where to save the model
    folder = ensure_output_dir(args.output)
    
    print("[DATA LOADING]")
    print("-" * 80)
    print("  Loading natural dataset...")
    
    dataset_nat = DatasetDCA(
        path_data=args.data,
        path_weights=args.weights,
        alphabet=args.alphabet,
        no_reweighting=args.no_reweighting,
        clustering_th=args.clustering_seqid,
        filter_sequences=True,
        remove_duplicates=True,
        device=device,
        dtype=dtype,
        message=False,
    )
    M_nat = len(dataset_nat)
    print(f"  ✓ Natural dataset loaded ({M_nat} sequences)")
    
    print("  Loading reintegration dataset...")
    dataset_reint = DatasetDCA(
        path_data=args.reint,
        path_weights=args.adj,
        alphabet=args.alphabet,
        filter_sequences=True,
        remove_duplicates=True,
        device=device,
        dtype=dtype,
        message=False,
    )
    M_reint = len(dataset_reint)
    print(f"  ✓ Reintegration dataset loaded ({M_reint} sequences)")
    print("-" * 80 + "\n")

    # Concatenate the two datasets
    print("[DATASET GENERATION]")
    print("-" * 80)
    print("  Merging natural and reintegration datasets...")
    msa = torch.cat((dataset_nat.to_one_hot(), dataset_reint.to_one_hot()), dim=0)
    msa_names = np.append(dataset_nat.names, dataset_reint.names)
    print(f"  ✓ Combined dataset: {len(msa)} sequences")
    
    if args.lambda_ is None:
        span_adjust = torch.abs(dataset_reint.weights).max()
        if torch.isclose(span_adjust, torch.zeros_like(span_adjust)):
            raise ValueError("Cannot infer lambda_: adjustment vector contains only zeros.")
        lambda_ = 1 / span_adjust
        print(f"  Lambda (auto): {lambda_:.6f} (1 / max|adjust|)")
    else:
        lambda_ = args.lambda_
        print(f"  Lambda: {lambda_:.6f}")
    
    # Compute the scaling factor k
    k = (lambda_ * dataset_nat.get_effective_size()) / len(dataset_reint)
    print(f"  Scaling factor k: {k:.6f}")
    
    # Concatenate the weights
    weights = torch.cat((dataset_nat.weights, k * dataset_reint.weights), dim=0)
    print(f"  ✓ Reintegration weights computed")
    
    # Save the new dataset
    args.label = f"{args.label}-lambda_{lambda_:.2f}" if args.label is not None else f"lambda_{lambda_}"
    
    print("  Saving reintegrated dataset...")
    path_msa = folder / f"{args.label}_msa.fasta"
    write_fasta(
        fname=path_msa,
        headers=msa_names,
        sequences=msa,
        remove_gaps=False,
        tokens=tokens,
    )
    print(f"  ✓ MSA saved: {path_msa}")
    
    path_weights = folder / f"{args.label}_weights.dat"
    np.savetxt(path_weights, weights.cpu().numpy())
    print(f"  ✓ Weights saved: {path_weights}")
    print("-" * 80 + "\n")
    
    # launch the training
    print("[LAUNCHING TRAINING]")
    print("-" * 80)
    print("  Starting DCA training with reintegrated dataset...")
    print("-" * 80 + "\n")
    train_args = [
        "--data", str(path_msa),
        "--weights", str(path_weights),
        "--output", str(folder),
        "--device", str(args.device),
        "--dtype", str(args.dtype),
        "--alphabet", str(args.alphabet),
        "--model", str(args.model),
        "--clustering_seqid", str(args.clustering_seqid),
        "--lr", str(args.lr),
        "--label", str(args.label),
        "--nsweeps", str(args.nsweeps),
        "--nchains", str(args.nchains),
        "--target", str(args.target),
        "--nepochs", str(args.nepochs),
        "--seed", str(args.seed),
        "--gsteps", str(args.gsteps),
        "--factivate", str(args.factivate),
        "--density", str(args.density),
        "--drate", str(args.drate),
    ]
    if args.pseudocount is None:
        pseudocount_default = 0.1 if args.model == "edgeDCA" else 1e-6
        train_args.append("--pseudocount")
        train_args.append(str(pseudocount_default))
    else:
        train_args.append("--pseudocount")
        train_args.append(str(args.pseudocount))
    if args.no_reweighting:
        train_args.append("--no_reweighting")
    if args.wandb:
        train_args.append("--wandb")
    if args.path_params is not None:
        train_args.append("--path_params")
        train_args.append(str(args.path_params))
    if args.path_chains is not None:
        train_args.append("--path_chains")
        train_args.append(str(args.path_chains))
    
    from adabmDCA.scripts import train as train_script

    with training_argv(train_args):
        train_script.main()
    
if __name__ == "__main__":
    main()
    
