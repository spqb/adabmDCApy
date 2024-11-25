from pathlib import Path
import importlib
import argparse
import numpy as np

import torch

from adabmDCA.dataset import DatasetDCA
from adabmDCA.fasta_utils import get_tokens
from adabmDCA.io import load_chains, load_params
from adabmDCA.stats import get_freq_single_point, get_freq_two_points
from adabmDCA.utils import init_chains, init_parameters, get_device, get_dtype
from adabmDCA.parser import add_args_train
from adabmDCA.sampling import get_sampler
from adabmDCA.functional import one_hot


# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description='Train a DCA model.')
    parser = add_args_train(parser)
    
    return parser


def main():
    
    # Load parser, training dataset and DCA model
    parser = create_parser()
    args = parser.parse_args()
    
    print("\n" + "".join(["*"] * 10) + f" Training {args.model} model " + "".join(["*"] * 10) + "\n")
    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    print("\n")
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
    print(f"Data type:\t\t{args.dtype}")
    print("\n")
    
    # Create the folder where to save the model
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    
    if args.label is not None:
        file_paths = {
            "log" : folder / Path(f"{args.label}.log"),
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
    dataset = DatasetDCA(
        path_data=args.data,
        path_weights=args.weights,
        alphabet=args.alphabet,
        clustering_th=args.clustering_seqid,
        device=device,
        dtype=dtype,
    )
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
        torch.tensor(dataset.data, device=device),
        num_classes=q,
    ).to(dtype)
    
    fi_target = get_freq_single_point(data=data_oh, weights=dataset.weights, pseudo_count=args.pseudocount)
    fij_target = get_freq_two_points(data=data_oh, weights=dataset.weights, pseudo_count=args.pseudocount) 
    
    # Initialize parameters and chains
    if args.path_params:
        tokens = get_tokens(args.alphabet)
        params = load_params(fname=args.path_params, tokens=tokens, device=device, dtype=dtype)
        mask = torch.zeros(size=(L, q, L, q), dtype=torch.bool, device=device)
        mask[torch.nonzero(params["coupling_matrix"])] = 1
        
    else:
        params = init_parameters(fi=fi_target)
        
        if args.model in ["bmDCA", "edDCA"]:
            mask = torch.ones(size=(L, q, L, q), dtype=torch.bool, device=device)
            mask[torch.arange(L), :, torch.arange(L), :] = 0
            
        else:
            mask = torch.zeros(size=(L, q, L, q), device=device, dtype=torch.bool)
    
    if args.path_chains:
        chains, log_weights = load_chains(fname=args.path_chains, tokens=dataset.tokens, load_weights=True)
        chains = one_hot(
            torch.tensor(chains, device=device),
            num_classes=q,
        ).to(dtype)
        log_weights = torch.tensor(log_weights, device=device, dtype=dtype)
        args.nchains = chains.shape[0]
        print(f"Loaded {args.nchains} chains.")
        
    else:
        print(f"Number of chains set to {args.nchains}.")
        chains = init_chains(num_chains=args.nchains, L=L, q=q, fi=fi_target, device=device, dtype=dtype)
        log_weights = torch.zeros(size=(args.nchains,), device=device, dtype=dtype)
        
    # Select the sampling function
    sampler = get_sampler(args.sampler)
    
    print("\n")
        
    # Save the hyperparameters of the model
    template = "{0:20} {1:10}\n"  
    with open(file_paths["log"], "w") as f:
        if args.label is not None:
            f.write(template.format("label:", args.label))
        else:
            f.write(template.format("label:", "N/A"))
            
        f.write(template.format("model:", str(args.model)))
        f.write(template.format("input MSA:", str(args.data)))
        f.write(template.format("alphabet:", args.alphabet))
        f.write(template.format("sampler:", args.sampler))
        f.write(template.format("nchains:", args.nchains))
        f.write(template.format("nsweeps:", args.nsweeps))
        f.write(template.format("lr:", args.lr))
        f.write(template.format("pseudo count:", args.pseudocount))
        f.write(template.format("data type:", args.dtype))
        f.write(template.format("target Pearson Cij:", args.target))
        if args.model == "eaDCA":
            f.write(template.format("gsteps:", args.gsteps))
            f.write(template.format("factivate:", args.factivate))
        f.write(template.format("random seed:", args.seed))
        f.write("\n")
        template = "{0:10} {1:10} {2:10} {3:10} {4:10} {5:10} {6:10}\n"
        f.write(template.format("Epoch", "Pearson", "Slope", "LL", "Entropy", "Density", "Time [s]"))

    
    DCA_model.fit(
        sampler=sampler,
        fij_target=fij_target,
        fi_target=fi_target,
        params=params,
        mask=mask,
        chains=chains,
        log_weights=log_weights,
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
    )
    
    
if __name__ == "__main__":
    main()