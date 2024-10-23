import argparse
from pathlib import Path

def add_args_dca(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    dca_args = parser.add_argument_group("General DCA arguments")
    dca_args.add_argument("-d", "--data",         type=Path,  required=True,        help="Filename of the dataset to be used for training the model.")
    dca_args.add_argument("-o", "--output",       type=Path,  default='DCA_model',  help="(Defaults to DCA_model). Path to the folder where to save the model.")
    dca_args.add_argument("-m", "--model",        type=str,   default="bmDCA",      help="(Defaults to bmDCA). Type of model to be trained.", choices=["bmDCA", "eaDCA", "edDCA"])
    # Optional arguments
    dca_args.add_argument("-w", "--weights",      type=Path,  default=None,         help="(Defaults to None). Path to the file containing the weights of the sequences. If None, the weights are computed automatically.")
    dca_args.add_argument("-p", "--path_params",  type=Path,  default=None,         help="(Defaults to None) Path to the file containing the model's parameters. Required for restoring the training.")
    dca_args.add_argument("-c", "--path_chains",  type=Path,  default=None,         help="(Defaults to None) Path to the fasta file containing the model's chains. Required for restoring the training.")
    dca_args.add_argument("-l", "--label",        type=str,   default=None,         help="(Defaults to None). If provoded, adds a label to the output files inside the output folder.")
    dca_args.add_argument("--alphabet",           type=str,   default="protein",    help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    dca_args.add_argument("--lr",                 type=float, default=0.01,         help="(Defaults to 0.01). Learning rate.")
    dca_args.add_argument("--nsweeps",            type=int,   default=10,           help="(Defaults to 10). Number of sweeps for each gradient estimation.")
    dca_args.add_argument("--sampler",            type=str,   default="gibbs",      help="(Defaults to gibbs). Sampling method to be used.", choices=["metropolis", "gibbs"])
    dca_args.add_argument("--nchains",            type=int,   default=10000,        help="(Defaults to 10000). Number of Markov chains to run in parallel.")
    dca_args.add_argument("--target",             type=float, default=0.95,         help="(Defaults to 0.95). Pearson correlation coefficient on the two-sites statistics to be reached.")
    dca_args.add_argument("--nepochs",            type=int,   default=50000,        help="(Defaults to 50000). Maximum number of epochs allowed.")
    dca_args.add_argument("--pseudocount",        type=float, default=None,         help="(Defaults to None). Pseudo count for the single and two-sites statistics. Acts as a regularization. If None, it is set to 1/Meff.")
    dca_args.add_argument("--seed",               type=int,   default=0,            help="(Defaults to 0). Seed for the random number generator.")
    dca_args.add_argument("--device",             type=str,   default="cuda",       help="(Defaults to cuda). Device to be used.")
    
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