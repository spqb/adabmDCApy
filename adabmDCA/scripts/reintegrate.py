import argparse
import os
import numpy as np
import subprocess
import torch
from adabmDCA.fasta import get_tokens, write_fasta
from adabmDCA.parser import add_args_train, add_args_reintegration
from adabmDCA.utils import get_device, get_dtype
from adabmDCA.dataset import DatasetDCA

def create_parser():
    parser = argparse.ArgumentParser(description='Reintegrate a DCA model.')
    parser = add_args_train(parser)
    parser = add_args_reintegration(parser)
    return parser

def main():    
    parser = create_parser()
    args = parser.parse_args()
    
    print("\n" + "".join(["*"] * 10) + f" Training a reintegrated DCA model " + "".join(["*"] * 10) + "\n")
    print("Generating the reintegrated dataset...")
    
    device = get_device(args.device, message=False)
    dtype = get_dtype(args.dtype)
    tokens = get_tokens(args.alphabet)
    
    # Create the folder where to save the model
    folder = args.output
    os.makedirs(folder, exist_ok=True)
    
    dataset_nat = DatasetDCA(
        path_data=args.data,
        path_weights=args.weights,
        alphabet=args.alphabet,
        clustering_th=args.clustering_seqid,
        no_reweighting=args.no_reweighting,
        filter_sequences=True,
        remove_duplicates=True,
        device=device,
        dtype=dtype,
        message=False,
    )
    
    dataset_reint = DatasetDCA(
        path_data=args.reint,
        alphabet=args.alphabet,
        no_reweighting=True,
        filter_sequences=False,
        remove_duplicates=False,
        device=device,
        dtype=dtype,
        message=False,
    )

    # Concatenate the two datasets
    msa = torch.cat((dataset_nat.data, dataset_reint.data), dim=0)
    msa_names = np.append(dataset_nat.names, dataset_reint.names)
    
    # Import the adjustment vector
    with open(args.adj, "r") as f:
        adjust = torch.tensor([float(x) for x in f.read().split()], device=device, dtype=dtype)
    if args.lambda_ is None:
        span_adjust = torch.abs(adjust).max()
        lambda_ = 1 / span_adjust
        print(f"No lambda value provided. Using lambda = {lambda_:.4f} (1 / max(|adjustment vector|))")
    else:
        lambda_ = args.lambda_
    # Compute the scaling factor k
    k = (lambda_ * dataset_nat.get_effective_size()) / len(dataset_reint)
    
    # Concatenate the weights
    weights = torch.cat((dataset_nat.weights.view(-1), k * adjust), dim=0).unsqueeze(1)
    
    # Save the new dataset
    args.label = f"{args.label}-lambda_{args.lambda_}" if args.label is not None else f"lambda_{args.lambda_}"
    path_msa = os.path.join(folder, f"{args.label}_msa.fasta")
    write_fasta(
        fname=path_msa,
        headers=msa_names,
        sequences=msa,
        remove_gaps=False,
        tokens=tokens,
    )
    path_weights = os.path.join(folder, f"{args.label}_weights.dat")
    np.savetxt(path_weights, weights.cpu().numpy())
    
    # launch the training
    train_command = [
        "adabmDCA",
        "train",
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
        pseudocount_default = 1e-6
        train_command.append("--pseudocount")
        train_command.append(str(pseudocount_default))
    else:
        train_command.append("--pseudocount")
        train_command.append(str(args.pseudocount))
    if args.no_reweighting:
        train_command.append("--no_reweighting")
    if args.wandb:
        train_command.append("--wandb")
    if args.path_params is not None:
        train_command.append("--path_params")
        train_command.append(str(args.path_params))
    if args.path_chains is not None:
        train_command.append("--path_chains")
        train_command.append(str(args.path_chains))
    
    subprocess.run(train_command, check=True)
    
if __name__ == "__main__":
    main()
    