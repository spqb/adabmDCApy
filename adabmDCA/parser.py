import argparse

def add_args_dca(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    dca_args = parser.add_argument_group("General DCA arguments")
    dca_args.add_argument("-d", "--data",         type=str,   required=True,        help="Filename of the fasta file to be used for training the model.")
    dca_args.add_argument("-o", "--output",       type=str,   default='DCA_model',  help="(Defaults to 'DCA_model'). Path to the folder where to save the model.")
    dca_args.add_argument("-m", "--model",        type=str,   default="bmDCA",      help="(Defaults to 'bmDCA'). Type of model to be trained.", choices=["bmDCA", "eaDCA", "edDCA"])
    dca_args.add_argument("-t", "--test",         type=str,   default=None,         help="(Defaults to None). Filename of the fasta file to be used for testing the model. If provided, the test log-likelihood is computed at each checkpoint.")
    dca_args.add_argument("-p", "--path_params",  type=str,   default=None,         help="(Defaults to None) Path to the file containing the model's parameters. Required for restoring an old training.")
    dca_args.add_argument("-c", "--path_chains",  type=str,   default=None,         help="(Defaults to None) Path to the fasta file containing the model's chains. Required for restoring an old training.")
    dca_args.add_argument("-l", "--label",        type=str,   default=None,         help="(Defaults to None). If provided, adds a label to the output files inside the output folder.")
    dca_args.add_argument("--alphabet",           type=str,   default="protein",    help="(Defaults to 'protein'). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    dca_args.add_argument("--lr",                 type=float, default=0.01,         help="(Defaults to 0.01). Learning rate.")
    dca_args.add_argument("--nsweeps",            type=int,   default=10,           help="(Defaults to 10). Number of sweeps per Markov chain for gradient estimation.")
    dca_args.add_argument("--sampler",            type=str,   default="gibbs",      help="(Defaults to 'gibbs'). Sampling method to be used.", choices=["metropolis", "gibbs"])
    dca_args.add_argument("--nchains",            type=int,   default=10000,        help="(Defaults to 10000). Number of Markov chains to run in parallel.")
    dca_args.add_argument("--target",             type=float, default=0.95,         help="(Defaults to 0.95). Target Pearson correlation coefficient on the two-sites statistics to be reached.")
    dca_args.add_argument("--nepochs",            type=int,   default=50000,        help="(Defaults to 50000). Maximum number of epochs allowed.")
    dca_args.add_argument("--pseudocount",        type=float, default=None,         help="(Defaults to None). Pseudo count for the single and two-sites statistics. Acts as a regularization. If None, it is set to 1/Meff.")
    dca_args.add_argument("--seed",               type=int,   default=0,            help="(Defaults to 0). Seed for the random number generator.")
    dca_args.add_argument("--wandb",              action="store_true",              help="If provided, logs the training on Weights and Biases.")
    dca_args.add_argument("--device",             type=str,   default="cuda",       help="(Defaults to 'cuda'). Device to be used.")
    dca_args.add_argument("--dtype",              type=str,   default="float32",    help="(Defaults to 'float32'). Data type to be used.")

    return parser


def add_args_reweighting(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    weight_args = parser.add_argument_group("Sequence reweighting arguments")
    weight_args.add_argument("-w", "--weights",      type=str,  default=None,          help="(Defaults to None). Path to the file containing the weights of the sequences. If None, the weights are computed automatically.")
    weight_args.add_argument("--clustering_seqid",   type=float, default=0.8,          help="(Defaults to 0.8). Sequence Identity threshold for clustering. Used only if 'weights' is not provided.")
    weight_args.add_argument("--no_reweighting",     action="store_true",              help="If provided, the reweighting of the sequences is not performed.")

    return parser


def add_args_eaDCA(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    eadca_args = parser.add_argument_group("eaDCA arguments")
    eadca_args.add_argument("--gsteps",           type=int,   default=10,           help="(Defaults to 10). Number of gradient updates to be performed on a given graph.")
    eadca_args.add_argument("--factivate",        type=float, default=0.001,        help="(Defaults to 0.001). Fraction of inactive couplings to be proposed for activation at each graph update.")

    return parser


def add_args_edDCA(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    eddca_args = parser.add_argument_group("edDCA arguments")
    eddca_args.add_argument("--density",          type=float, default=0.02,         help="(Defaults to 0.02). Target density to be reached.")
    eddca_args.add_argument("--drate",            type=float, default=0.01,         help="(Defaults to 0.01). Fraction of remaining couplings to be pruned at each decimation step.")

    return parser


def add_args_train(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_args_dca(parser)
    parser = add_args_eaDCA(parser)
    parser = add_args_edDCA(parser)
    parser = add_args_reweighting(parser)
    
    return parser


def add_args_energies(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-d", "--data",         type=str,   required=True,      help="Path to the fasta file containing the sequences.")
    parser.add_argument("-p", "--path_params",  type=str,   required=True,      help="Path to the file containing the parameters of DCA model.")
    parser.add_argument("-o", "--output",       type=str,   required=True,      help="Path to the folder where to save the output.")
    parser.add_argument("--alphabet",           type=str,    default="protein",  help="(Defaults to 'protein'). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument("--device",             type=str,    default="cuda",     help="(Defaults to 'cuda'). Device to be used.")
    parser.add_argument("--dtype",              type=str,    default="float32",  help="(Defaults to 'float32'). Data type to be used.")

    return parser


def add_args_contacts(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-o", "--output",       type=str,    required=True,          help="Path to the folder where to save the output.")
    parser.add_argument("-p", "--path_params",  type=str,    default=None,           help="(Defaults to None). Path to the file containing the parameters of DCA model to use for contact prediction. If None, the mean-field approximation is used.")
    parser.add_argument("-d", "--data",         type=str,    default=None,           help="(Defaults to None). Path to the file containing the data. Used for the mean-field approximation only.")
    parser.add_argument("-l", "--label",        type=str,    default=None,           help="(Defaults to None). If provided, adds a label to the output files inside the output folder.")
    parser.add_argument("--alphabet",           type=str,    default="protein",      help="(Defaults to 'protein'). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument("--device",             type=str,    default="cuda",         help="(Defaults to 'cuda'). Device to be used.")
    parser.add_argument("--dtype",              type=str,    default="float32",      help="(Defaults to 'float32'). Data type to be used.")

    return parser


def add_args_dms(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-d", "--data",         type=str,   required=True,       help="Path to the fasta file containing wild type sequence. If more than one sequence is present, the first one is used.")
    parser.add_argument("-p", "--path_params",  type=str,   required=True,       help="Path to the file containing the parameters of DCA model to sample from.")
    parser.add_argument("-o", "--output",       type=str,   required=True,       help="Path to the folder where to save the output.")
    parser.add_argument("--alphabet",           type=str,    default="protein",  help="(Defaults to 'protein'). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument("--device",             type=str,    default="cuda",     help="(Defaults to 'cuda'). Device to be used.")
    parser.add_argument("--dtype",              type=str,    default="float32",  help="(Defaults to 'float32'). Data type to be used.")

    return parser


def add_args_sample(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-p", "--path_params",  type=str,    required=True,      help="Path to the file containing the parameters of DCA model to sample from.")
    parser.add_argument("-o", "--output",       type=str,    required=True,      help="Path to the folder where to save the output.")
    parser.add_argument("--ngen",               type=int,    required=True,      help="Number of sequences to be generated.") 
    
    # Optional arguments
    parser.add_argument("-d", "--data",         type=str,    default=None,       help="Path to the file containing the natural data. If provided, the mixing time of the model is computed. Defaults to None.")
    parser.add_argument("-l", "--label",        type=str,    default="sampling", help="(Defaults to 'sampling'). Label to be used for the output files.")
    parser.add_argument("--nmeasure",           type=int,    default=10000,      help="(Defaults to min(10000, len(data))). Number of data sequences to use for computing the mixing time.")
    parser.add_argument("--nmix",               type=int,    default=2,          help="(Defaults to 2). Number of mixing times used to generate 'ngen' sequences starting from random.")
    parser.add_argument("--max_nsweeps",        type=int,    default=5000,       help="(Defaults to 5000). Maximum number of chain updates.")
    parser.add_argument("--alphabet",           type=str,    default="protein",  help="(Defaults to 'protein'). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument("--sampler",            type=str,    default="gibbs",    help="(Defaults to 'gibbs'). Sampling method to be used. Choose between 'metropolis' and 'gibbs'.")
    parser.add_argument("--beta",               type=float,  default=1.0,        help="(Defaults to 1.0). Inverse temperature for the sampling.")
    parser.add_argument("--pseudocount",        type=float,  default=None,       help="(Defaults to None). Pseudocount for the single and two-sites statistics used during the training. If None, 1/Meff is used.")
    parser.add_argument("--device",             type=str,    default="cuda",     help="(Defaults to 'cuda'). Device to perform computations on.")
    parser.add_argument("--dtype",              type=str,    default="float32",  help="(Defaults to 'float32'). Data type to be used.")

    parser = add_args_reweighting(parser)
    
    return parser

def add_args_tdint(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-p", "--path_params",      type=str,   required=True,           help="Path to the file containing the parameters of DCA model to sample from.")
    parser.add_argument("-d", "--data",             type=str,   required=True,           help="Path to the file containing the data to sample from.")
    parser.add_argument("-t", "--path_targetseq",   type=str,   required=True,           help="Path to the file containing the target sequence.")
    parser.add_argument("-o", "--output",           type=str,   default='DCA_model',     help="Path to the folder where to save the output.")

    # Optional arguments
    parser.add_argument("-c", "--path_chains",  type=str,    default=None,            help="(Defaults to None). Path to the fasta file containing the model's chains.")
    parser.add_argument("-l", "--label",        type=str,    default="entropy",       help="(Defaults to 'entropy'). Label to be used for the output files.")
    parser.add_argument("--nchains",            type=int,    default=10000,           help="(Defaults to 10000). Number of chains to be used.") 
    parser.add_argument("--theta_max",          type=float,  default=5,               help="(Defaults to 5). Maximum integration strength") 
    parser.add_argument("--nsteps",             type=int,    default=100,             help="(Defaults to 100). Number of integration steps.")
    parser.add_argument("--nsweeps",            type=int,    default=100,             help="(Defaults to 100). Number of chain updates for each integration step.")
    parser.add_argument("--nsweeps_theta",      type=int,    default=100,             help="(Defaults to 100). Number of chain updates to equilibrate chains at theta_max.")
    parser.add_argument("--nsweeps_zero",       type=int,    default=100,             help="(Defaults to 100). Number of chain updates to equilibrate chains at theta = 0.")
    parser.add_argument("--alphabet",           type=str,    default="protein",       help="(Defaults to 'protein'). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument("--nepochs",            type=int,    default=50000,           help="(Defaults to 50000). Maximum number of epochs allowed.")
    parser.add_argument("--sampler",            type=str,    default="gibbs",         help="(Defaults to 'gibbs'). Sampling method to be used. Choose between 'metropolis' and 'gibbs'.")
    parser.add_argument("--seed",               type=int,    default=0,               help="(Defaults to 0). Seed for the random number generator.")
    parser.add_argument("--device",             type=str,    default="cuda",          help="(Defaults to 'cuda'). Device to perform computations on.")
    parser.add_argument("--dtype",              type=str,    default="float32",       help="(Defaults to 'float32'). Data type to be used.")

    return parser

def add_args_reintegration(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--reint",  type=str,       required=True,  help="Path to the fasta file containing the reintegrated sequences.")
    parser.add_argument("--adj",    type=str,       required=True,  help="Path to the file containing the adjustment vector.")
    parser.add_argument("--lambda_", type=float,    required=True,  help="Reintegration strength parameter.")

    return parser

def add_args_profmark(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("output_prefix", type=str,                      help="Prefix for the output files.")
    parser.add_argument("input_msa",     type=str,                      help="Fasta file containing the multiple sequence alignment.")
    parser.add_argument("-t1",           type=float, default=0.5,       help="(Defaults to 0.5) No sequence in S (the candidate training set) has more than this fraction of its residues identical to any sequence in T (the candidate test set).")
    parser.add_argument("-t2",           type=float, default=0.5,       help="(Defaults to 0.5) No pair of test sequences has more than this value fractional identity.")
    parser.add_argument("-t3",           type=float, default=1.0,       help="(Defaults to 1.0) No pair of training sequences has more than this value fractional identity.")
    parser.add_argument("--bestof",      type=int,   default=1,         help="(Defaults to 1) Runs the algorithm n times and returns the one that maximizes |S| * |T|.")
    parser.add_argument("--maxtrain",    type=int,   default=None,      help="(Defaults to None) Maximum number of sequences in the training set.")
    parser.add_argument("--maxtest",     type=int,   default=None,      help="(Defaults to None) Maximum number of sequences in the test set.")
    parser.add_argument("--alphabet",    type=str,   default="protein", help="(Defaults to 'protein') Alphabet to use for encoding the sequences. Choose among 'protein', 'rna', 'dna' or a user-defined alphabet.")
    parser.add_argument("--seed",        type=int,   default=0,         help="(Defaults to 0) Random seed for reproducibility.")
    parser.add_argument("--device",      type=str,   default="cuda",    help="(Defaults to 'cuda') Device to use for computation.")

    return parser