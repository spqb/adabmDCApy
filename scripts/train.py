#!/usr/bin/python3

from pathlib import Path
import importlib
import argparse
import numpy as np

import torch

from adabmDCA.dataset import DatasetDCA
from adabmDCA.fasta_utils import get_tokens
from adabmDCA.io import load_chains, load_params
from adabmDCA.stats import get_freq_single_point, get_freq_two_points
from adabmDCA.methods import init_chains, init_parameters
from adabmDCA.parser import add_args_dca, add_args_eaDCA, add_args_edDCA
from adabmDCA.sampling import get_sampler
from adabmDCA.custom_fn import one_hot


# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description='Train a DCA model.')
    parser = add_args_dca(parser)
    parser = add_args_eaDCA(parser)
    parser = add_args_edDCA(parser)    
    
    return parser


if __name__ == '__main__':
    
    # Load parser, training dataset and DCA model
    parser = create_parser()
    args = parser.parse_args()
    
    print("\n" + "".join(["*"] * 10) + f" Training {args.model} model " + "".join(["*"] * 10) + "\n")
    print(f"Input MSA:\t\t{args.data}")
    print(f"Output folder:\t\t{args.output}")
    print(f"Alphabet:\t\t{args.alphabet}")
    print(f"Learning rate:\t\t{args.lr}")
    print(f"Number of sweeps:\t{args.nsweeps}")
    print(f"Sampler:\t\t{args.sampler}")
    print(f"Target Pearson Cij:\t{args.target}")
    if args.pseudocount is not None:
        print(f"Pseudocount:\t\t{args.pseudocount}")
    print(f"Random seed:\t\t{args.seed}")
    print("\n")
    
    # Create the folder where to save the model
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    
    if args.label is not None:
        file_paths = {
            "log" : folder / Path(f"{args.label}_adabmDCA.log"),
            "params" : folder / Path(f"{args.label}_params.dat"),
            "chains" : folder / Path(f"{args.label}_chains.fasta")
        }
        
    else:
        file_paths = {
            "log" : folder / Path(f"adabmDCA.log"),
            "params" : folder / Path(f"params.dat"),
            "chains" : folder / Path(f"chains.fasta")
        }
    
    # Import dataset
    print("Importing dataset...")
    dataset = DatasetDCA(path_data=args.data, path_weights=args.weights, alphabet=args.alphabet)
    DCA_model = importlib.import_module(f"adabmDCA.models.{args.model}")
    tokens = get_tokens(args.alphabet)
    
    # Save the weights if not already provided
    if args.weights is None:
        if args.label is not None:
            path_weights = folder / f"{args.label}_weights.dat"
        else:
            path_weights = folder / "weights.dat"
        np.savetxt(path_weights, dataset.weights.cpu().numpy())
        print(f"Weights saved in {path_weights}")
        
    # Set the random seed
    torch.manual_seed(args.seed)
        
    # Shuffle the dataset
    dataset.shuffle()
    
    # Compute statistics of the data
    L = dataset.get_num_residues()
    q = dataset.get_num_states()
    
    if args.pseudocount is None:
        args.pseudocount = 1. / dataset.get_effective_size()
        print(f"Pseudocount automatically set to {args.pseudocount}.")
        
    data_oh = one_hot(
        torch.tensor(dataset.data, device=args.device),
        num_classes=q,
    ).float()
    
    fi_target = get_freq_single_point(data=data_oh, weights=dataset.weights, pseudo_count=args.pseudocount)
    fij_target = get_freq_two_points(data=data_oh, weights=dataset.weights, pseudo_count=args.pseudocount) 
    
    # Initialize parameters and chains
    if args.path_params:
        tokens = get_tokens(args.alphabet)
        params = load_params(fname=args.path_params, tokens=tokens, device=args.device)
        mask = torch.zeros(shape=(L, q, L, q), device=args.device)
        mask[torch.nonzero(params["coupling_matrix"])] = 1
        
    else:
        params = init_parameters(fi=fi_target)
        
        if args.model in ["bmDCA", "edDCA"]:
            mask = torch.ones(size=(L, q, L, q), device=args.device)
            mask[torch.arange(L), :, torch.arange(L), :] = 0
            
        else:
            mask = torch.zeros(size=(L, q, L, q), device=args.device)
    
    if args.path_chains:
        chains = one_hot(
            torch.tensor(load_chains(fname=args.path_chains, tokens=dataset.tokens), device=args.device),
            num_classes=q,
        ).float()
        
    else:
        if args.nchains is None:
            args.nchains = min(5000, dataset.get_effective_size())
            print(f"Number of chains automatically set to {args.nchains}.")
            
        else:
            print(f"Number of chains set to {args.nchains}.")
            
        chains = init_chains(num_chains=args.nchains, L=L, q=q, fi=fi_target, device=args.device)
        
    # Select the sampling function
    sampler = get_sampler(args.sampler)
    
    print("\n")
        
    # Save the hyperparameters of the model     
    with open(file_paths["log"], "w") as f:
        if args.label is not None:
            f.write(f"label:\t\t\t\t{args.label}\n")
        else:
            f.write(f"label:\t\t\t\tN/A\n")
        f.write(f"model:\t\t\t\t{args.model}\n")
        f.write(f"input MSA:\t\t\t{args.data}\n")
        f.write(f"alphabet:\t\t\t{args.alphabet}\n")
        f.write(f"sampler:\t\t\t{args.sampler}\n")
        f.write(f"nchains:\t\t\t{args.nchains}\n")
        f.write(f"nsweeps:\t\t\t{args.nsweeps}\n")
        f.write(f"lr:\t\t\t\t\t{args.lr}\n")
        f.write(f"pseudo count:\t\t{args.pseudocount}\n")
        f.write(f"target Pearson Cij:\t{args.target}\n")
        if args.model == "eaDCA":
            f.write(f"gsteps:\t\t\t\t{args.gsteps}\n")
            f.write(f"factivate:\t\t\t{args.factivate}\n")
        elif args.model == "edDCA":
            f.write(f"target density:\t\t{args.density}\n")
            f.write(f"drate:\t\t\t\t{args.drate}\n")
        f.write(f"random seed:\t\t{args.seed}\n")
        f.write(f"\nepoch\t\tPearson\t\tDensity\t\ttime [s]\n")

    
    DCA_model.fit(
        sampler=sampler,
        fij_target=fij_target,
        fi_target=fi_target,
        params=params,
        mask=mask,
        chains=chains,
        tokens=tokens,
        target_pearson=args.target,
        pseudo_count=args.pseudocount,
        nsweeps=args.nsweeps,
        nepochs=args.nepochs,
        lr=args.lr,
        factivate=args.factivate,
        gsteps=args.gsteps,
        drate=args.drate,
        target_density=args.density,
        file_paths=file_paths,
        device=args.device,
    )