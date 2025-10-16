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
        device=device,
        dtype=dtype,
        message=False,
    )
    
    dataset_reint = DatasetDCA(
        path_data=args.reint,
        alphabet=args.alphabet,
        no_reweighting=True,
        device=device,
        dtype=dtype,
        message=False,
    )
    
    # concatenate the two datasets
    msa = torch.cat((dataset_nat.data, dataset_reint.data), dim=0)
    msa_names = np.append(dataset_nat.names, dataset_reint.names)
    k = (args.lambda_ * dataset_nat.get_effective_size()) / len(dataset_reint)
    # Import the adjustment vector
    with open(args.adj, "r") as f:
        adjust = torch.tensor([float(x) for x in f.read().split()], device=device, dtype=dtype)
    # Conncatenate the weights
    weights = torch.cat((dataset_nat.weights.view(-1), k * adjust), dim=0).unsqueeze(1)
    
    # Save the new dataset
    args.label = f"{args.label}-lambda_{args.lambda_}" if args.label is not None else f"lambda_{args.lambda_}"
    path_msa = os.path.join(folder, f"{args.label}_msa.fasta")
    write_fasta(
        fname=path_msa,
        headers=msa_names,
        sequences=msa,
        numeric_input=True,
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
        "--pseudocount", str(0),
        "--seed", str(args.seed),
        "--checkpoints", str(args.checkpoints),
        "--target_acc_rate", str(args.target_acc_rate),
        "--gsteps", str(args.gsteps),
        "--factivate", str(args.factivate),
        "--density", str(args.density),
        "--drate", str(args.drate),
    ]
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
    