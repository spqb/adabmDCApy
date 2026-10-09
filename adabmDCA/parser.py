import argparse

from adabmDCA.training_config import (
    DEFAULT_ACTIVATION_FRACTION,
    DEFAULT_ACTIVATION_STEPS,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_CLUSTERING_SEQID,
    DEFAULT_DECIMATION_RATE,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_L2_REGULARIZATION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MODEL_TYPE,
    DEFAULT_N_CHAINS,
    DEFAULT_N_SWEEPS,
    DEFAULT_SAMPLER,
    DEFAULT_SEED,
    DEFAULT_TARGET_DENSITY,
    DEFAULT_TARGET_PEARSON,
    MODEL_TYPES,
    SAMPLERS,
)


def add_args_dca(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    dca_args = parser.add_argument_group("General DCA arguments")
    dca_args.add_argument(
        "-d", "--data", type=str, required=True, help="Filename of the fasta file to be used for training the model."
    )
    dca_args.add_argument(
        "-o",
        "--output",
        type=str,
        default="DCA_model",
        help="(Defaults to 'DCA_model'). Path to the folder where to save the model.",
    )
    dca_args.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        help="(Defaults to 'bmDCA'). Type of model to be trained.",
        choices=MODEL_TYPES,
    )
    dca_args.add_argument(
        "-v",
        "--validation",
        "--val",
        dest="val",
        type=str,
        default=None,
        help="(Defaults to None). Filename of the fasta file to be used for validating the model. If provided, validation metrics are computed at each checkpoint. --val is an older alias.",
    )
    dca_args.add_argument(
        "-p",
        "--path_params",
        type=str,
        default=None,
        help="(Defaults to None) Path to the file containing the model's parameters. Required for restoring an old training.",
    )
    dca_args.add_argument(
        "-c",
        "--path_chains",
        type=str,
        default=None,
        help="(Defaults to None) Path to the fasta file containing the model's chains. Required for restoring an old training.",
    )
    dca_args.add_argument(
        "-l",
        "--label",
        type=str,
        default=None,
        help="(Defaults to None). If provided, adds a label to the output files inside the output folder.",
    )
    dca_args.add_argument(
        "--alphabet",
        type=str,
        default="auto",
        help="(Defaults to auto). Detect protein, dna, or rna; otherwise specify a custom string of tokens. DNA wins ambiguous nucleotide matches.",
    )
    dca_args.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="(Defaults to 0.01). Learning rate.")
    dca_args.add_argument(
        "--nsweeps",
        type=int,
        default=DEFAULT_N_SWEEPS,
        help="(Defaults to 10). Number of sweeps per Markov chain for gradient estimation.",
    )
    dca_args.add_argument(
        "--sampler",
        type=str,
        default=DEFAULT_SAMPLER,
        help="(Defaults to 'metropolized_gibbs'). Sampling method to be used.",
        choices=SAMPLERS,
    )
    dca_args.add_argument(
        "--nchains",
        type=int,
        default=DEFAULT_N_CHAINS,
        help="(Defaults to 2000). Number of Markov chains to run in parallel.",
    )
    dca_args.add_argument(
        "--target",
        type=float,
        default=DEFAULT_TARGET_PEARSON,
        help="(Defaults to 0.95). Target Pearson correlation coefficient on the two-sites statistics to be reached.",
    )
    dca_args.add_argument(
        "--nepochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="(Defaults to 50000). Compatibility limit: gradient steps for bmDCA and PTT edgeDCA; structure steps (graph activations/decimations) for eaDCA, including PTT eaDCA, and edDCA.",
    )
    dca_args.add_argument(
        "--max-gradient-steps",
        type=int,
        default=None,
        help="Optional limit on parameter updates, including PTT edge corrections and nested eaDCA/edDCA optimization. PTT eaDCA may stop inside a --gsteps block.",
    )
    dca_args.add_argument(
        "--max-structure-steps", type=int, default=None, help="Optional limit on graph activation or decimation steps."
    )
    dca_args.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"Save parameters and chains every N training steps (default: {DEFAULT_CHECKPOINT_INTERVAL}). The final state is also saved.",
    )
    dca_args.add_argument(
        "--pseudocount",
        type=float,
        default=None,
        help="(Defaults to None). Pseudo count for the single and two-sites statistics. Acts as a regularization. If None, it is set to 0.1 for edgeDCA and 1/Meff otherwise.",
    )
    dca_args.add_argument(
        "--l2_reg",
        type=float,
        default=DEFAULT_L2_REGULARIZATION,
        help="(Defaults to 0.0). L2 regularization coefficient for the parameters.",
    )
    dca_args.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="(Defaults to 0). Seed for the random number generator."
    )
    dca_args.add_argument("--wandb", action="store_true", help="If provided, logs the training on Weights and Biases.")
    dca_args.add_argument("--no-progress", action="store_true", help="Disable the terminal training progress bar.")
    dca_args.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="(Defaults to 'auto'). Device to use: CUDA if available, otherwise MPS, otherwise CPU.",
    )
    dca_args.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=("float32", "float64", "bfloat16"),
        help=(
            "Training precision (default: float32). bfloat16 uses BF16 sampling couplings "
            "with FP32 master parameters; requires Ampere+ CUDA and Triton."
        ),
    )

    return parser


def add_args_reweighting(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    weight_args = parser.add_argument_group("Sequence reweighting arguments")
    weight_args.add_argument(
        "-w",
        "--weights",
        type=str,
        default=None,
        help="(Defaults to None). Path to the file containing the weights of the sequences. If None, the weights are computed automatically.",
    )
    weight_args.add_argument(
        "--clustering_seqid",
        type=float,
        default=DEFAULT_CLUSTERING_SEQID,
        help="(Defaults to 0.8). Sequence Identity threshold for clustering. Used only if 'weights' is not provided.",
    )
    weight_args.add_argument(
        "--no_reweighting", action="store_true", help="If provided, the reweighting of the sequences is not performed."
    )

    return parser


def add_args_eaDCA(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    eadca_args = parser.add_argument_group("eaDCA arguments")
    eadca_args.add_argument(
        "--gsteps",
        type=int,
        default=DEFAULT_ACTIVATION_STEPS,
        help="(Defaults to 10). Number of gradient updates to be performed on a given graph.",
    )
    eadca_args.add_argument(
        "--factivate",
        type=float,
        default=DEFAULT_ACTIVATION_FRACTION,
        help="(Defaults to 0.001). Fraction of the inactive coupling entries, counted once per symmetric pair, activated at each graph update.",
    )

    return parser


def add_args_edDCA(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    eddca_args = parser.add_argument_group("edDCA arguments")
    eddca_args.add_argument(
        "--density",
        type=float,
        default=DEFAULT_TARGET_DENSITY,
        help="(Defaults to 0.02). Target density to be reached.",
    )
    eddca_args.add_argument(
        "--drate",
        type=float,
        default=DEFAULT_DECIMATION_RATE,
        help="(Defaults to 0.01). Fraction of remaining couplings to be pruned at each decimation step.",
    )

    return parser


def add_args_ptt(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ptt_args = parser.add_argument_group(
        "Parallel Trajectory Tempering arguments",
        "PTT defaults: Metropolized Gibbs, 10 sweeps per exchange round, 2000 chains, 50000 updates.",
    )
    parser.add_argument("--strategy", choices=("ptt", "pcd"), default="ptt",
                        help="Training strategy (default: ptt).")
    ptt_args.add_argument("--ptt-resume", help="Resume a full PTT HDF5 training archive.")
    ptt_args.add_argument("--ptt-swaps-factor", dest="ptt_swaps", type=int, default=None,
                          help="PTT exchange-round factor; updates use int(factor * sqrt(active replicas)) rounds (default: 1).")
    ptt_args.add_argument("--ptt-target-acceptance", type=float, default=None, help="PTT target acceptance (default: 0.25).")
    ptt_args.add_argument("--ptt-max-replicas", type=int, default=None, help="Refresh the reservoir above this active ladder size after held-snapshot replacement (default: 2).")
    ptt_args.add_argument("--ptt-target-replicas", type=int, default=None, help="Active replicas retained after reservoir refresh (default: 2).")
    ptt_args.add_argument("--ptt-reservoir-size", type=int, default=None, help="Reservoir population size (default: 10 * nchains).")
    ptt_args.add_argument("--ptt-full-sampler", action="store_true", default=None, help="Generate new reservoirs using the full historical ladder.")
    ptt_args.add_argument("--ptt-min-acceptance", type=float, default=None, help="Acceptance below this starts a mixing diagnostic (default: 0.1).")
    ptt_args.add_argument("--ptt-mixing-chains", type=int, default=None, help="Chains per replica in the mixing experiment (default: 100).")
    ptt_args.add_argument("--ptt-mixing-initial-rounds", type=int, default=None, help="Initial mixing trajectory length (default: 100).")
    ptt_args.add_argument("--ptt-mixing-thermalization-rounds", type=int, default=None, help="Warmup before the mixing experiment (default: 1000).")
    ptt_args.add_argument("--ptt-mixing-max-rounds", type=int, default=None, help="Maximum mixing trajectory length before recovery (default: 20000).")
    ptt_args.add_argument("--ptt-mixing-window-factor", type=float, default=None, help="Required trajectory length divided by max(tau_int, tau_exp) (default: 20).")
    ptt_args.add_argument(
        "--ptt-mixing-method", choices=("renewal", "autocorrelation"), default=None,
        help=("How PTT training checks that the ladder mixes (default: renewal). 'renewal' tracks when "
              "each configuration entered the ladder from the exact profile or the reservoir, requires the "
              "ladder to be repopulated twice within --ptt-mixing-max-rounds, and spaces reservoir batches "
              "by full renewal of the collection replica; 'autocorrelation' fits tau_int and tau_exp of "
              "each configuration's replica position."),
    )
    ptt_args.add_argument(
        "--ptt-renewal-tolerance", type=float, default=None,
        help="Maximum fraction of old configurations, per chains of one replica, at renewal (default: 0.01).",
    )
    ptt_args.add_argument(
        "--diagnostic", action="store_true",
        help=("Show detailed PTT progress, including learning rates, trust-region, lag and mixing "
              "diagnostics, train/validation LL per residue, and adjacent swap acceptance rates."),
    )
    ptt_args.add_argument("--ptt-max-recoveries", type=int, default=None, help="Maximum learning-rate reductions after mixing failures (default: 3).")
    ptt_args.add_argument("--ptt-min-learning-rate", type=float, default=None, help="Effective-step floor during mixing recovery: learning rate for bmDCA, 1 - pseudocount for edgeDCA (default: 1e-8).")
    ptt_args.add_argument("--ptt-optimizer", choices=("adaptive", "sgd"), default=None,
                          help=("PTT optimizer: adaptive bounds each update by a KL trust radius and pauses when endpoint "
                                "chains lag; sgd uses fixed --lr for bmDCA or fixed --pseudocount for edgeDCA "
                                "(default: adaptive)."))
    ptt_args.add_argument("--ptt-trust-radius", type=float, default=None,
                        help="Adaptive optimizer: maximum local KL divergence per update (default: 0.01).")
    ptt_args.add_argument("--ptt-lag-tolerance", type=float, default=None,
                        help=("Adaptive optimizer: largest lag of the endpoint chains behind the model along the "
                              "drift and its tails, in standard deviations of the population, before training "
                              "pauses (default: 0.25)."))
    ptt_args.add_argument("--ptt-lag-horizon", type=int, default=None,
                        help="Adaptive optimizer: memory of past updates, in gradient steps (default: 200).")
    ptt_args.add_argument("--ptt-lag-pause-rounds", type=int, default=None,
                        help="Adaptive optimizer: maximum exchange rounds of one pause (default: 100).")
    ptt_args.add_argument("--ptt-activation", choices=("fixed", "adaptive"), default=None,
                        help=("eaDCA: couplings activated per graph update. fixed activates --factivate of the "
                              "inactive entries; adaptive activates at most that fraction, only statistically "
                              "significant candidates, as many as fit the KL budget of the first update "
                              "(default: fixed)."))
    ptt_args.add_argument("--ptt-activation-kl-share", type=float, default=None,
                        help=("Adaptive activation: share of --ptt-trust-radius available to the first update of "
                              "the new couplings; halved by each mixing recovery (default: 0.5)."))
    ptt_args.add_argument("--ptt-activation-significance", type=float, default=None,
                        help=("Adaptive activation: candidates need |f - p| above this many standard errors of "
                              "the data and chain frequencies; 0 disables the test (default: 3)."))
    ptt_args.add_argument("--ptt-validation-stop", action=argparse.BooleanOptionalAction, default=None,
                        help=("Stop when the validation log-likelihood per site plateaus instead of at --target; "
                              "requires --validation. Default: on when --validation is given, off otherwise."))
    ptt_args.add_argument("--ptt-validation-window", type=int, default=None,
                        help="Validation stop: updates per window of the compared medians (default: 100).")
    ptt_args.add_argument("--ptt-validation-min-gain", type=float, default=None,
                        help=("Validation stop: stop when the median validation log-likelihood per site of the last "
                              "window exceeds that of the window before by less than this (default: 0)."))
    ptt_args.add_argument("--ptt-equilibration-rounds", type=int, default=None,
                        help="PTT rounds after a ladder change (default: 10).")
    ptt_args.add_argument("--ptt-initialization-rounds", type=int, default=None,
                        help="PTT initialization rounds (default: 100).")
    return parser


def add_args_train(parser: argparse.ArgumentParser, *, ptt: bool = True) -> argparse.ArgumentParser:
    """Add the training options; ``ptt=False`` omits ``--strategy`` and the PTT options (PCD-only commands)."""
    parser = add_args_dca(parser)
    parser = add_args_eaDCA(parser)
    parser = add_args_edDCA(parser)
    parser = add_args_reweighting(parser)
    if ptt:
        parser = add_args_ptt(parser)

    return parser


def add_args_energies(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-d", "--data", type=str, required=True, help="Path to the fasta file containing the sequences."
    )
    parser.add_argument(
        "-p", "--path_params", type=str, required=True, help="Path to the file containing the parameters of DCA model."
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the folder where to save the output.")
    parser.add_argument(
        "--alphabet",
        type=str,
        default="auto",
        help="(Defaults to auto). Detect protein, dna, or rna; otherwise specify a custom string of tokens. DNA wins ambiguous nucleotide matches.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="(Defaults to 'auto'). Device to use: CUDA if available, otherwise MPS, otherwise CPU.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="(Defaults to 'float32'). Data type to be used.")
    parser.add_argument("--local-lambda", type=float, default=None,
                        help="Coefficient of summed per-residue CDE in local free energy E - lambda * CDE. "
                             "By default, local free energy is not computed. Obtain a fitted value using the sampling script.")

    return parser


def add_args_contacts(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the folder where to save the output.")
    parser.add_argument(
        "-p",
        "--path_params",
        type=str,
        default=None,
        help="(Defaults to None). Path to the file containing the parameters of DCA model to use for contact prediction. If None, the mean-field approximation is used.",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=None,
        help="(Defaults to None). Path to the file containing the data. Used for the mean-field approximation only.",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default=None,
        help="(Defaults to None). If provided, adds a label to the output files inside the output folder.",
    )
    parser.add_argument(
        "--alphabet",
        type=str,
        default="auto",
        help="(Defaults to auto). Detect protein, dna, or rna; otherwise specify a custom string of tokens. DNA wins ambiguous nucleotide matches.",
    )
    parser.add_argument(
        "--pseudocount",
        type=float,
        default=0.5,
        help="(Defaults to 0.5). Pseudocount used to regularize the empirical frequencies in the mean-field approximation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="(Defaults to 'auto'). Device to use: CUDA if available, otherwise MPS, otherwise CPU.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="(Defaults to 'float32'). Data type to be used.")

    return parser


def add_args_dms(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="Path to the fasta file containing wild type sequence. If more than one sequence is present, the first one is used.",
    )
    parser.add_argument(
        "-p",
        "--path_params",
        type=str,
        required=True,
        help="Path to the file containing the parameters of DCA model to sample from.",
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the folder where to save the output.")
    parser.add_argument(
        "--alphabet",
        type=str,
        default="auto",
        help="(Defaults to auto). Detect protein, dna, or rna; otherwise specify a custom string of tokens. DNA wins ambiguous nucleotide matches.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="(Defaults to 'auto'). Device to use: CUDA if available, otherwise MPS, otherwise CPU.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="(Defaults to 'float32'). Data type to be used.")

    return parser


def add_args_sample(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    io_args = parser.add_argument_group("Input and output")
    io_args.add_argument(
        "-p", "--path_params", type=str, required=True,
        help="DCA model to sample from: a parameter file (plain or gzip-compressed) or, with --strategy ptt, a PTT archive.",
    )
    io_args.add_argument("-o", "--output", type=str, required=True, help="Path to the folder where to save the output.")
    io_args.add_argument(
        "--ngen", type=int, default=None,
        help=("Number of sequences to be generated. Defaults to the number of chains per model used in "
              "training with --strategy ptt, and to 2000 with --strategy pcd."),
    )
    io_args.add_argument(
        "-l", "--label", type=str, default=None, help="(Defaults to None). Label to be used for the output files."
    )
    io_args.add_argument(
        "--alphabet", type=str, default="auto",
        help=("(Defaults to auto). Detect protein, dna, or rna; otherwise specify a custom string of tokens. "
              "DNA wins ambiguous nucleotide matches."),
    )
    io_args.add_argument(
        "--seed", type=int, default=0, help="(Defaults to 0). Seed for reproducible sequence generation."
    )
    io_args.add_argument(
        "--device", type=str, default="auto",
        help="(Defaults to 'auto'). Device to use: CUDA if available, otherwise MPS, otherwise CPU.",
    )
    io_args.add_argument(
        "--dtype", type=str, default="float32", choices=("float32", "float64", "bfloat16"),
        help=(
            "(Defaults to 'float32'). Sampling precision. bfloat16 stores sampling couplings in BF16 "
            "while retaining FP32 model state and requires an Ampere-or-newer CUDA GPU with Triton."
        ),
    )

    standard_args = parser.add_argument_group("PCD sampling (--strategy pcd)")
    standard_args.add_argument(
        "--sampler", type=str, default=DEFAULT_SAMPLER, choices=SAMPLERS,
        help="(Defaults to 'metropolized_gibbs'). Sampling method to be used: 'metropolis', 'gibbs' or 'metropolized_gibbs'.",
    )
    standard_args.add_argument(
        "--beta", type=float, default=1.0, help="(Defaults to 1.0). Inverse temperature for the sampling."
    )
    standard_args.add_argument(
        "--nmix", type=int, default=2,
        help="(Defaults to 2). Number of mixing times used to generate 'ngen' sequences starting from random.",
    )
    standard_args.add_argument(
        "--max_nsweeps", type=int, default=5000,
        help="(Defaults to 5000). Maximum chain updates. Ignored with --strategy ptt; use --ptt-max-rounds.",
    )

    ptt_args = parser.add_argument_group("PTT sampling (--strategy ptt)")
    parser.add_argument("--strategy", choices=("ptt", "pcd"), default="ptt",
                        help="Sampling strategy (default: ptt).")
    ptt_args.add_argument("--ptt-local-sweeps", type=int, default=10, help="Local sweeps per PTT exchange round (default: 10).")
    ptt_args.add_argument(
        "--ptt-local-kernel", choices=("archived", "metropolis", "gibbs", "metropolized_gibbs"),
        default="metropolized_gibbs",
        help=(
            "Local update for PTT generation (default: metropolized_gibbs). Every kernel samples the same "
            "distributions. 'metropolized_gibbs' proposes from the site conditional excluding the current state; "
            "'archived' keeps the training kernel."
        ),
    )
    ptt_args.add_argument(
        "--ptt-mixing-method", choices=("renewal", "autocorrelation"), default="renewal",
        help=(
            "How PTT decides the ladder is equilibrated (default: renewal). 'renewal' tracks when each "
            "configuration was drawn at the exact bottom model and waits until the initial populations have "
            "been replaced (once, or twice with --ptt-stationary); 'autocorrelation' fits tau_int and tau_exp "
            "of replica labels."
        ),
    )
    ptt_args.add_argument(
        "--ptt-renewal-tolerance", type=float, default=0.01,
        help="Maximum endpoint fraction of configurations older than the reference round (default: 0.01).",
    )
    ptt_args.add_argument(
        "--ptt-stationary", action="store_true",
        help=("Renewal method: after the warmup, which replaces the initial populations once, renew the ladder "
              "a second time, so that every sample descends from draws made after the ladder had forgotten "
              "its start. About doubles the sampling time."),
    )
    ptt_args.add_argument(
        "--ptt-max-rounds", type=int, default=20_000,
        help=("Maximum PTT equilibration rounds (default: 20000). With --ptt-mixing-method autocorrelation, "
              "the budget excludes its 1000 warm-up rounds."),
    )

    reference_args = parser.add_argument_group("Reference alignment")
    reference_args.add_argument(
        "-d", "--data", type=str, default=None,
        help=("Natural alignment used as reference: for the mixing time, the Pearson of the generated sequences and "
              "the plots. Required with --strategy ptt, where it also gives the ladder log-likelihoods."),
    )
    reference_args.add_argument(
        "--pseudocount", type=float, default=None,
        help="(Defaults to None). Pseudocount for the reference single and two-site statistics. If None, 1/Meff is used.",
    )
    reference_args.add_argument(
        "--nmeasure", type=int, default=10000,
        help="(Defaults to min(10000, len(data))). Reference sequences used for the mixing time and the PCA.",
    )
    reference_args.add_argument(
        "-v", "--validation", "--test", dest="test", metavar="FASTA", type=str, default=None,
        help=("Held-out alignment, e.g. the validation set used in training (--test is an older alias). "
              "Used with --plot: its distances to the nearest --data sequence are the yardstick for those of "
              "the generated sequences in distances.png (overfitting check); it also enables the PRIVET "
              "test (beta). Without it, only distances to and within --data are plotted."),
    )
    reference_args.add_argument(
        "--privet-window", type=float, nargs=2, default=[0.01, 0.5], metavar=("Q1", "Q2"),
        help=("(Defaults to 0.01 0.5). Beta feature: quantiles of the training nearest-neighbour distances "
              "fitted by the extreme-value law of the PRIVET memorization test; check the fit in privet.png."),
    )
    parser = add_args_reweighting(parser)

    report_args = parser.add_argument_group("Reporting")
    report_args.add_argument(
        "--plot", action="store_true",
        help=(
            "Save mixing/renewal, final Cij scatter, and natural-versus-generated PCA plots; "
            "standard sampling also saves its Pearson trajectory. Requires a reference alignment supplied with --data."
        ),
    )
    report_args.add_argument(
        "--diagnostic", action="store_true",
        help=(
            "In PTT mode, print renewal fractions periodically (renewal method), or tau_int, tau_exp and "
            "the updated round target after every mixing estimate (autocorrelation method), together with "
            "all adjacent-model acceptance rates."
        ),
    )
    return parser


def add_args_tdint(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--strategy", choices=("ptt", "pcd"), default="ptt",
                        help="Entropy strategy (default: ptt). PTT reads an archive; PCD uses MCMC thermodynamic integration.")
    parser.add_argument(
        "-p",
        "--path_params",
        type=str,
        required=True,
        help="Path to the file containing the parameters of DCA model to sample from.",
    )
    parser.add_argument(
        "-d", "--data", type=str, default=None, help="Natural alignment (required with --strategy pcd)."
    )
    parser.add_argument(
        "-t", "--path_targetseq", type=str, default=None,
        help="Path to the target alignment. Uses the first valid sequence and warns if more than one is present."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="DCA_model", help="Path to the folder where to save the output."
    )

    # Optional arguments
    parser.add_argument(
        "-c",
        "--path_chains",
        type=str,
        default=None,
        help="(Defaults to None). Path to the fasta file containing the model's chains.",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default="entropy",
        help="(Defaults to 'entropy'). Label to be used for the output files.",
    )
    parser.add_argument("--nchains", type=int, default=10000,
                        help="(Defaults to 10000). Integration chain count; ignored with --strategy ptt because the archive owns the chains.")
    parser.add_argument("--theta_max", type=float, default=5, help="(Defaults to 5). Maximum integration strength")
    parser.add_argument("--nsteps", type=int, default=100, help="(Defaults to 100). Number of integration steps.")
    parser.add_argument(
        "--nsweeps", type=int, default=100, help="(Defaults to 100). Number of chain updates for each integration step."
    )
    parser.add_argument(
        "--nsweeps_theta",
        type=int,
        default=100,
        help="(Defaults to 100). Number of chain updates to equilibrate chains at theta_max.",
    )
    parser.add_argument(
        "--nsweeps_zero",
        type=int,
        default=100,
        help="(Defaults to 100). Number of chain updates to equilibrate chains at theta = 0.",
    )
    parser.add_argument(
        "--alphabet",
        type=str,
        default="auto",
        help="(Defaults to auto). Detect protein, dna, or rna; otherwise specify a custom string of tokens. DNA wins ambiguous nucleotide matches.",
    )
    parser.add_argument(
        "--nepochs", type=int, default=50000, help="(Defaults to 50000). Maximum number of epochs allowed."
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default=DEFAULT_SAMPLER,
        help="(Defaults to 'metropolized_gibbs'). Sampling method to be used: 'metropolis', 'gibbs' or 'metropolized_gibbs'.",
        choices=SAMPLERS,
    )
    parser.add_argument("--seed", type=int, default=0, help="(Defaults to 0). Seed for the random number generator.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="(Defaults to 'auto'). Device to use: CUDA if available, otherwise MPS, otherwise CPU.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="(Defaults to 'float32'). Data type to be used.")

    return parser


def add_args_reintegration(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--reint", type=str, required=True, help="Path to the fasta file containing the reintegrated sequences."
    )
    parser.add_argument("--adj", type=str, required=True, help="Path to the file containing the adjustment vector.")
    parser.add_argument(
        "--lambda_",
        type=float,
        default=None,
        help="(Defaults to None)Reintegration strength parameter. If None, it is set to 1 / max|adjust|",
    )

    return parser


def add_args_split_data(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("output_prefix", type=str, help="Prefix for the output files.")
    parser.add_argument("input_msa", type=str, help="FASTA file containing the multiple sequence alignment.")
    parser.add_argument("--method", choices=("clustering", "cobalt"), default="clustering",
                        help="Splitting method (default: clustering).")

    clustering = parser.add_argument_group(
        "clustering options",
        "Assign whole clusters from an MMseqs2-like sequence clustering approach to training or test.",
    )
    clustering.add_argument("--identity", type=float, default=0.8,
                            help="Minimum sequence identity for clustering (default: 0.8).")
    clustering.add_argument("--train-fraction", type=float, default=0.8,
                            help="Target fraction of sequences in training (default: 0.8).")

    cobalt = parser.add_argument_group(
        "cobalt options",
        "Apply hard thresholds to inter-set and intra-set sequence identities "
        "(Petti & Eddy, PLoS Comput Biol 18(3):e1009492, 2022).",
    )
    cobalt.add_argument(
        "-t1", type=float, default=0.5,
        help="Maximum identity between training and test sequences (default: 0.5).",
    )
    cobalt.add_argument(
        "-t2", type=float, default=0.5,
        help="Maximum identity between test sequences (default: 0.5).",
    )
    cobalt.add_argument(
        "-t3", type=float, default=1.0,
        help="Maximum identity between training sequences (default: 1.0).",
    )
    cobalt.add_argument(
        "--bestof", type=int, default=1,
        help="Number of attempts; select the split maximizing train size × test size (default: 1).",
    )
    cobalt.add_argument("--maxtrain", type=int, default=None,
                        help="Maximum number of training sequences (default: no limit).")
    cobalt.add_argument("--maxtest", type=int, default=None,
                        help="Maximum number of test sequences (default: no limit).")

    parser.add_argument(
        "--alphabet", type=str, default="auto",
        help="(Defaults to auto). Detect protein, dna, or rna; otherwise specify a custom string of tokens. DNA wins ambiguous nucleotide matches.",
    )
    parser.add_argument("--seed", type=int, default=0, help="(Defaults to 0) Random seed for reproducibility.")
    parser.add_argument(
        "--device", type=str, default="auto",
        help="(Defaults to 'auto'). Device to use: CUDA if available, otherwise MPS, otherwise CPU.",
    )
    return parser

