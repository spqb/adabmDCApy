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
from adabmDCA.graph import compute_density


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
    
    print("\n" + "="*80)
    print(f"  TRAINING {args.model.upper()} MODEL")
    print("="*80 + "\n")
    
    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    
    # Configuration section
    print("[CONFIGURATION]")
    print("-" * 80)
    template = "  {0:<28} {1:<50}"
    print(template.format("Input MSA:", str(args.data)))
    print(template.format("Output folder:", str(args.output)))
    print(template.format("Model type:", args.model))
    print(template.format("Alphabet:", args.alphabet))
    print(template.format("Learning rate:", args.lr))
    print(template.format("Number of sweeps:", args.nsweeps))
    print(template.format("Sampler:", args.sampler))
    print(template.format("Target Pearson Cij:", args.target))
    if args.pseudocount is not None:
        print(template.format("Pseudocount:", args.pseudocount))
    print(template.format("Random seed:", args.seed))
    print(template.format("Device:", str(device)))
    print(template.format("Data type:", args.dtype))
    print("-" * 80 + "\n")
    
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
    
    print("[OUTPUT FILES]")
    print("-" * 80)
    print(f"  Log file:        {file_paths['log']}")
    print(f"  Parameters file: {file_paths['params']}")
    print(f"  Chains file:     {file_paths['chains']}")
    print("-" * 80 + "\n")
    
    # Import dataset
    print("[DATA LOADING]")
    print("-" * 80)
    print("  Importing training dataset...")
    dataset = DatasetDCA(
        path_data=args.data,
        path_weights=args.weights,
        alphabet=args.alphabet,
        clustering_th=args.clustering_seqid,
        no_reweighting=args.no_reweighting,
        device=device,
        dtype=dtype,
        message=False,
        filter_sequences=True,
        remove_duplicates=True,
    )
    
    # Import the test dataset if provided
    if args.test is not None:
        print("  Importing test dataset...")
        test_dataset = DatasetDCA(
            path_data=args.test,
            path_weights=None,
            alphabet=args.alphabet,
            clustering_th=args.clustering_seqid,
            no_reweighting=args.no_reweighting,
            device=device,
            dtype=dtype,
            message=False,
            filter_sequences=True,
            remove_duplicates=True,
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
        print(f"  ✓ Weights saved: {path_weights}")
        
    # Set the random seed
    torch.manual_seed(args.seed)
        
    # Shuffle the dataset
    dataset.shuffle()
    
    # Compute statistics of the data
    L = dataset.get_num_residues()
    q = dataset.get_num_states()
    M = len(dataset)
    M_eff = dataset.get_effective_size()
    
    print(f"\n  Dataset statistics:")
    print(f"    • Sequence length (L): {L}")
    print(f"    • Alphabet size (q): {q}")
    print(f"    • Number of sequences (M): {M}")
    print(f"    • Effective sequences (M_eff): {M_eff}")
    
    if args.pseudocount is None:
        args.pseudocount = 1. / dataset.get_effective_size()
        print(f"    • Pseudocount (auto): {args.pseudocount:.6f}")
    print("-" * 80 + "\n")
    fi_target = get_freq_single_point(data=dataset.data, weights=dataset.weights, pseudo_count=args.pseudocount)
    fij_target = get_freq_two_points(data=dataset.data, weights=dataset.weights, pseudo_count=args.pseudocount)

    # Initialize parameters and chains
    print("[INITIALIZATION]")
    print("-" * 80)
    if args.path_params:
        print(f"  Loading parameters from: {args.path_params}")
        tokens = get_tokens(args.alphabet)
        params = load_params(fname=args.path_params, tokens=tokens, device=device, dtype=dtype)
        print("  ✓ Parameters loaded")
        mask = ~ torch.isclose(params["coupling_matrix"], torch.zeros_like(params["coupling_matrix"]))
        density = compute_density(mask) * 100
        print(f"  ✓ Model density: {density:.3f}%")
        
    else:
        print("  Initializing parameters from data statistics...")
        params = init_parameters(fi=fi_target)
        print("  ✓ Parameters initialized")
        
        if args.model in ["bmDCA", "edDCA"]:
            mask = torch.ones(size=(L, q, L, q), dtype=torch.bool, device=device)
            mask[torch.arange(L), :, torch.arange(L), :] = 0
            
        else:
            mask = torch.zeros(size=(L, q, L, q), device=device, dtype=torch.bool)
    
    if args.path_chains:
        print(f"  Loading chains from: {args.path_chains}")
        chains, log_weights = load_chains(fname=args.path_chains, tokens=dataset.tokens, load_weights=True, device=device, dtype=dtype)
        log_weights = log_weights.to(device=device, dtype=dtype)
        args.nchains = chains.shape[0]
        print(f"  ✓ Loaded {args.nchains} chains")
        
    else:
        print(f"  Initializing {args.nchains} chains from data statistics...")
        chains = init_chains(num_chains=args.nchains, L=L, q=q, fi=fi_target, device=device, dtype=dtype)
        log_weights = torch.zeros(size=(args.nchains,), device=device, dtype=dtype)
        print(f"  ✓ Chains initialized")
        
    # Select the sampling function
    print(f"  Setting up sampler: {args.sampler}")
    sampler = torch.jit.script(get_sampler(args.sampler))
    print("  ✓ Sampler ready")
    print("-" * 80 + "\n")
    
    print("[TRAINING]")
    print("-" * 80)
    
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
    
    print("\n" + "=" * 80)
    print("  TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n  Results saved in: {folder}")
    print(f"    \u2713 Parameters: {file_paths['params']}")
    print(f"    \u2713 Chains:     {file_paths['chains']}")
    print(f"    \u2713 Log file:   {file_paths['log']}")
    print("\n" + "=" * 80 + "\n")
    
    
if __name__ == "__main__":
    main()