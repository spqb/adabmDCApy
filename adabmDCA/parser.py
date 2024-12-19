import argparse
from pathlib import Path

def add_args_dca(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    dca_args = parser.add_argument_group("General DCA arguments")
    dca_args.add_argument("-d", "--data",         type=Path,  required=True,        help="Filename of the dataset to be used for training the model.")
    dca_args.add_argument("-o", "--output",       type=Path,  default='DCA_model',  help="(Defaults to DCA_model). Path to the folder where to save the model.")
    dca_args.add_argument("-m", "--model",        type=str,   default="bmDCA",      help="(Defaults to bmDCA). Type of model to be trained.", choices=["bmDCA", "eaDCA", "edDCA"])
    # Optional arguments
    dca_args.add_argument("-t", "--test",         type=Path,  default=None,         help="(Defaults to None). Filename of the dataset to be used for testing the model.")
    dca_args.add_argument("-p", "--path_params",  type=Path,  default=None,         help="(Defaults to None) Path to the file containing the model's parameters. Required for restoring the training.")
    dca_args.add_argument("-c", "--path_chains",  type=Path,  default=None,         help="(Defaults to None) Path to the fasta file containing the model's chains. Required for restoring the training.")
    dca_args.add_argument("-l", "--label",        type=str,   default=None,         help="(Defaults to None). If provoded, adds a label to the output files inside the output folder.")
    dca_args.add_argument("--alphabet",           type=str,   default="protein",    help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    dca_args.add_argument("--lr",                 type=float, default=0.05,         help="(Defaults to 0.05). Learning rate.")
    dca_args.add_argument("--nsweeps",            type=int,   default=10,           help="(Defaults to 10). Number of sweeps for each gradient estimation.")
    dca_args.add_argument("--sampler",            type=str,   default="gibbs",      help="(Defaults to gibbs). Sampling method to be used.", choices=["metropolis", "gibbs"])
    dca_args.add_argument("--nchains",            type=int,   default=10000,        help="(Defaults to 10000). Number of Markov chains to run in parallel.")
    dca_args.add_argument("--target",             type=float, default=0.95,         help="(Defaults to 0.95). Pearson correlation coefficient on the two-sites statistics to be reached.")
    dca_args.add_argument("--nepochs",            type=int,   default=50000,        help="(Defaults to 50000). Maximum number of epochs allowed.")
    dca_args.add_argument("--pseudocount",        type=float, default=None,         help="(Defaults to None). Pseudo count for the single and two-sites statistics. Acts as a regularization. If None, it is set to 1/Meff.")
    dca_args.add_argument("--seed",               type=int,   default=0,            help="(Defaults to 0). Seed for the random number generator.")
    dca_args.add_argument("--wandb",              action="store_true",              help="If provided, logs the training on Weights and Biases.")
    dca_args.add_argument("--device",             type=str,   default="cuda",       help="(Defaults to cuda). Device to be used.")
    dca_args.add_argument("--dtype",              type=str,   default="float32",    help="(Defaults to float32). Data type to be used.")
    
    return parser


def add_args_reweighting(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    weight_args = parser.add_argument_group("Sequence reweighting arguments")
    weight_args.add_argument("-w", "--weights",      type=Path,  default=None,         help="(Defaults to None). Path to the file containing the weights of the sequences. If None, the weights are computed automatically.")
    weight_args.add_argument("--clustering_seqid",   type=float, default=0.8,          help="(Defaults to 0.8). Sequence Identity threshold for clustering. Used only if 'weights' is not provided.")
    weight_args.add_argument("--no_reweighting",     action="store_true",              help="If provided, the reweighting of the sequences is not performed.")

    return parser


def add_args_checkpoint(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    checkpoint_args = parser.add_argument_group("Checkpoint arguments")
    checkpoint_args.add_argument("--checkpoints",     type=str,   default="linear",     help="(Defaults to 'linear'). Choses the type of checkpoint criterium to be used.", choices=["linear", "acceptance"])
    checkpoint_args.add_argument("--target_acc_rate", type=float, default=0.5,          help="(Defaults to 0.5). Target acceptance rate for deciding when to save the model when the 'acceptance' checkpoint is used.")
    
    return parser


def add_args_eaDCA(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    eadca_args = parser.add_argument_group("eaDCA arguments")
    eadca_args.add_argument("--gsteps",           type=int,   default=10,           help="(Defaults to 10). Number of gradient updates to be performed on a given graph.")
    eadca_args.add_argument("--factivate",        type=float, default=0.001,        help="(Defaults to 0.001). Fraction of inactive couplings to be proposed for activation at each graph update.")

    return parser


def add_args_edDCA(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    eddca_args = parser.add_argument_group("edDCA arguments")
    eddca_args.add_argument("--density",          type=float, default=0.02,         help="(Defaults to 0.02). Target density to be reached.")
    eddca_args.add_argument("--drate",            type=float, default=0.01,         help="(Defaults to 0.01). Fraction of remaining couplings to be pruned at each decimation step.")

    return parser


def add_args_train(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_args_dca(parser)
    parser = add_args_eaDCA(parser)
    parser = add_args_edDCA(parser)
    parser = add_args_reweighting(parser)
    parser = add_args_checkpoint(parser)
    
    return parser


def add_args_energies(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-d", "--data",         type=Path,   required=True,      help="Path to the fasta file containing the sequences.")
    parser.add_argument("-p", "--path_params",  type=Path,   required=True,      help="Path to the file containing the parameters of DCA model.")
    parser.add_argument("-o", "--output",       type=Path,   required=True,      help="Path to the folder where to save the output.")
    parser.add_argument("--alphabet",           type=str,    default="protein",  help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument("--device",             type=str,    default="cuda",     help="(Defaults to cuda). Device to be used.")
    parser.add_argument("--dtype",              type=str,    default="float32",  help="(Defaults to float32). Data type to be used.")

    return parser


def add_args_contacts(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-p", "--path_params",  type=Path,   required=True,          help="Path to the file containing the parameters of DCA model to sample from.")
    parser.add_argument("-o", "--output",       type=Path,   required=True,          help="Path to the folder where to save the output.")
    parser.add_argument("-l", "--label",        type=str,    default=None,           help="(Defaults to None). If provoded, adds a label to the output files inside the output folder.")
    parser.add_argument("--alphabet",           type=str,    default="protein",      help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument("--device",             type=str,    default="cuda",         help="(Defaults to cuda). Device to be used.")
    parser.add_argument("--dtype",              type=str,    default="float32",      help="(Defaults to float32). Data type to be used.")
    
    return parser


def add_args_dms(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-d", "--data",         type=Path,   required=True,      
                        help="Path to the fasta file containing wild type sequence. If more than one sequence is present, the first one is used.")
    parser.add_argument("-p", "--path_params",  type=Path,   required=True,      help="Path to the file containing the parameters of DCA model to sample from.")
    parser.add_argument("-o", "--output",       type=Path,   required=True,      help="Path to the folder where to save the output.")
    parser.add_argument("--alphabet",           type=str,    default="protein",  help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument("--device",             type=str,    default="cuda",     help="(Defaults to cuda). Device to be used.")
    parser.add_argument("--dtype",              type=str,    default="float32",  help="(Defaults to float32). Data type to be used.")

    return parser


def add_args_sample(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-p", "--path_params",  type=Path,   required=True,      help="Path to the file containing the parameters of DCA model to sample from.")
    parser.add_argument("-d", "--data",         type=Path,   required=True,      help="Path to the file containing the data to sample from.")
    parser.add_argument("-o", "--output",       type=Path,   required=True,      help="Path to the folder where to save the output.")
    parser.add_argument("--ngen",               type=int,    required=True,      help="Number of sequences to be generated.") 
    
    # Optional arguments
    parser.add_argument("-l", "--label",        type=str,    default="sampling", help="(Defaults to 'sampling'). Label to be used for the output files.")
    parser.add_argument("--nmeasure",           type=int,    default=10000,      help="(Defaults to min(10000, len(data)). Number of data sequences to use for computing the mixing time.")
    parser.add_argument("--nmix",               type=int,    default=2,          help="(Defaults to 2). Number of mixing times used to generate 'ngen' sequences starting from random.")
    parser.add_argument("--max_nsweeps",        type=int,    default=1000,       help="(Defaults to 1000). Maximum number of chain updates.")
    parser.add_argument("--alphabet",           type=str,    default="protein",  help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument("--sampler",            type=str,    default="gibbs",    help="(Defaults to gibbs). Sampling method to be used. Choose between 'metropolis' and 'gibbs'.")
    parser.add_argument("--beta",               type=float,  default=1.0,        help="(Defaults to 1.0). Inverse temperature for the sampling.")
    parser.add_argument("--pseudocount",        type=float,  default=None,       help="(Defaults to None). Pseudocount for the single and two points statistics used during the training. If None, 1/Meff is used.")
    parser.add_argument("--device",             type=str,    default="cuda",     help="(Defaults to cuda). Device to perform computations on.")
    parser.add_argument("--dtype",              type=str,    default="float32",  help="(Defaults to float32). Data type to be used.")
    
    parser = add_args_reweighting(parser)
    
    return parser