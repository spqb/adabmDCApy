import importlib
import os
import argparse
import numpy as np
import torch

from adabmDCA.dataset import DatasetDCA
from adabmDCA.fasta import get_tokens
from adabmDCA.io import load_chains, load_params
from adabmDCA.stats import get_freq_single_point, get_freq_two_points
from adabmDCA.utils import init_chains, init_parameters, get_device, get_dtype
from adabmDCA.parser import add_args_train
from adabmDCA.sampling import get_sampler
from adabmDCA.checkpoint import Checkpoint


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
    template = "{0:<30} {1:<50}"
    print(template.format("Input MSA:", str(args.data)))
    print(template.format("Output folder:", str(args.output)))
    print(template.format("Alphabet:", args.alphabet))
    print(template.format("Learning rate:", args.lr))
    print(template.format("Number of sweeps:", args.nsweeps))
    print(template.format("Sampler:", args.sampler))
    print(template.format("Target Pearson Cij:", args.target))
    if args.pseudocount is not None:
        print(template.format("Pseudocount:", args.pseudocount))
    print(template.format("Random seed:", args.seed))
    print(template.format("Data type:", args.dtype))
    print("\n")
    
    # Check if the data file exist
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file {args.data} not found.")
    
    if args.test is not None:
        if not os.path.exists(args.test):
            raise FileNotFoundError(f"Test file {args.test} not found.")
    
    # Create the folder where to save the model
    folder = args.output
    os.makedirs(folder, exist_ok=True)

    if args.label is not None:
        file_paths = {
            "log" : os.path.join(folder, f"{args.label}.log"),
            "params" : os.path.join(folder, f"{args.label}_params.dat"),
            "chains" : os.path.join(folder, f"{args.label}_chains.fasta")
        }
        
    else:
        file_paths = {
            "log" : os.path.join(folder, f"adabmDCA.log"),
            "params" : os.path.join(folder, f"params.dat"),
            "chains" : os.path.join(folder, f"chains.fasta")
        }
    
    # Import dataset
    print("Importing dataset...")
    dataset = DatasetDCA(
        path_data=args.data,
        path_weights=args.weights,
        alphabet=args.alphabet,
        clustering_th=args.clustering_seqid,
        no_reweighting=args.no_reweighting,
        device=device,
        dtype=dtype,
    )
    
    # Import the test dataset if provided
    if args.test is not None:
        print("Importing test dataset...")
        test_dataset = DatasetDCA(
            path_data=args.test,
            path_weights=None,
            alphabet=args.alphabet,
            clustering_th=args.clustering_seqid,
            no_reweighting=args.no_reweighting,
            device=device,
            dtype=dtype,
        )
        pseudocount_test = 1. / test_dataset.get_effective_size()
        fi_test = get_freq_single_point(data=test_dataset.data, weights=test_dataset.weights, pseudo_count=pseudocount_test)
        fij_test = get_freq_two_points(data=test_dataset.data, weights=test_dataset.weights, pseudo_count=pseudocount_test)
    else:
        fi_test = None
        fij_test = None
    
    DCA_model = importlib.import_module(f"adabmDCA.models.{args.model}")
    tokens = get_tokens(args.alphabet)
    
    # Save the weights if not already provided
    if args.weights is None:
        if args.label is not None:
            path_weights = os.path.join(folder, f"{args.label}_weights.dat")
        else:
            path_weights = os.path.join(folder, "weights.dat")
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
    fi_target = get_freq_single_point(data=dataset.data, weights=dataset.weights, pseudo_count=args.pseudocount)
    fij_target = get_freq_two_points(data=dataset.data, weights=dataset.weights, pseudo_count=args.pseudocount)

    # Initialize parameters and chains
    if args.path_params:
        print("Loading parameters...")
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
        print("Loading chains...")
        chains, log_weights = load_chains(fname=args.path_chains, tokens=dataset.tokens, load_weights=True, device=device, dtype=dtype)
        log_weights = torch.tensor(log_weights, device=device, dtype=dtype)
        args.nchains = chains.shape[0]
        print(f"Loaded {args.nchains} chains.")
        
    else:
        print(f"Number of chains set to {args.nchains}.")
        chains = init_chains(num_chains=args.nchains, L=L, q=q, fi=fi_target, device=device, dtype=dtype)
        log_weights = torch.zeros(size=(args.nchains,), device=device, dtype=dtype)
        
    # Select the sampling function
    sampler = torch.jit.script(get_sampler(args.sampler))
    print("\n")
    
    checkpoint = Checkpoint(
        file_paths=file_paths,
        tokens=tokens,
        args=vars(args),
        params=params,
        chains=chains,
        use_wandb=args.wandb,
    )

    DCA_model.fit(
        sampler=sampler,
        fij_target=fij_target,
        fi_target=fi_target,
        fi_test=fi_test,
        fij_test=fij_test,
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
        checkpoint=checkpoint,
    )
    
    
if __name__ == "__main__":
    main()