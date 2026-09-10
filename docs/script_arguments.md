# <span id="script_arguments">Script Arguments</span>
 
These tables summarize the most frequently used command-line arguments for
`adabmDCA 2.0`. Run `adabmDCA <command> --help` for the complete, authoritative
option list for the installed release.

## Train a DCA model

| Command                   | Default value | Description |
|---------------------------|--------------|-------------|
| `-d, --data`             | N/A          | Filename of the dataset to be used for training the model. |
| `-o, --output`           | DCA_model    | Path to the folder where to save the model. |
| `-m, --model`            | bmDCA        | Type of model to be trained. Possible options are `bmDCA`, `eaDCA`, `edDCA`, and `edgeDCA`. |
| `-v, --val`             | None         | Filename of the validation dataset. If provided, validation metrics are computed at each checkpoint. |
| `-w, --weights`          | None         | Path to the file containing the weights of the sequences. If `None`, the weights are computed automatically. |
| `--clustering_seqid`     | 0.8          | Sequence identity threshold to be used for computing the sequence weights. |
| `--no_reweighting`       | N/A          | If this flag is used, the routine assigns uniform weights to the sequences. |
| `-p, --path_params`      | None         | Path to the file containing the model's parameters. Required for restoring the training. |
| `-c, --path_chains`      | None         | Path to the FASTA file containing the model's chains. Required for restoring the training. |
| `-l, --label`           | None         | A label to identify different algorithm runs. It prefixes the output files with this label. |
| `--alphabet`            | auto         | Detect a standard protein, DNA, or RNA alphabet, or provide a custom token string. |
| `--lr`                  | 0.01         | Learning rate. Ignored when `--model edgeDCA` is used. |
| `--nsweeps`             | 10           | Number of sweeps for each gradient estimation. |
| `--sampler`             | metropolis   | Sampling method to be used. Possible options are `gibbs` and `metropolis`. |
| `--nchains`             | 10000        | Number of Markov chains to run in parallel. |
| `--target`              | 0.95         | Pearson correlation coefficient on the two-sites statistics to be reached. |
| `--nepochs`             | 50000        | Compatibility limit: gradient steps for bmDCA, structure steps for sparse models. |
| `--max-gradient-steps`  | None         | Global limit on parameter-gradient updates, including nested eaDCA/edDCA optimization. |
| `--max-structure-steps` | None         | Limit on graph activation or decimation steps. |
| `--checkpoint-interval` | 100         | Save parameters and chains every N training steps. Final states are also saved. |
| `--pseudocount`        | None         | Pseudo count for the single and two-sites statistics. Acts as a regularization. If `None`, it is set to `0.1` for `edgeDCA` and $1/M_{\mathrm{eff}}$ otherwise. |
| `--l2_reg`             | 0.0          | L2 regularization coefficient for biases and couplings. |
| `--seed`               | 0            | Random seed. |
| `--wandb`              | off          | Log training metrics to Weights & Biases. |
| `--no-progress`        | off          | Disable the interactive terminal progress bar. |
| `--nthreads`¹         | 1            | Number of threads used in the Julia multithreaded version. |
| `--device`¹           | auto         | Select CUDA when available, otherwise MPS, otherwise CPU. |
| `--dtype`¹            | float32      | Training precision: float32, float64, or bfloat16 (BF16 sampling couplings with FP32 master state; Ampere+ CUDA and Triton required). Used in the Python version. |

### eaDCA options

| Command                | Default value | Description |
|------------------------|--------------|-------------|
| `--gsteps`            | 10           | Number of gradient updates to be performed on a given graph. |
| `--factivate`         | 0.001        | Fraction of inactive couplings to try to activate at each graph update. |

### edDCA options

| Command                | Default value | Description |
|------------------------|--------------|-------------|
| `--gsteps`            | 10           | The number of gradient updates applied at each step of the graph convergence process. |
| `--density`          | 0.02         | Target density to be reached. |
| `--drate`            | 0.01         | Fraction of remaining couplings to be pruned at each decimation step. |

### edgeDCA notes

`edgeDCA` uses the standard training options above, except that `--lr` is ignored. It starts from an empty coupling graph and activates one complete residue-residue edge at each graph update, choosing the inactive edge with the largest KL discrepancy between empirical and model two-site statistics. If `--pseudocount` is not provided, its default is `0.1`.

For `edgeDCA`, `--pseudocount` acts as an effective learning rate for edge activation: the closer it is to `1`, the smaller the effective learning rate is. On some datasets, values up to `0.95` can be used.

                                                        
## Sampling from a DCA model 

| Command                   | Default value | Description |
|---------------------------|--------------|-------------|
| `-p, --path_params`      | N/A          | Path to the file containing the parameters of the DCA model to sample from. |
| `-d, --data`             | None         | Reference MSA used to estimate mixing time and sampling convergence. Without it, generation runs for exactly `--max_nsweeps` sweeps. |
| `-o, --output`           | N/A          | Path to the folder where to save the output. |
| `--ngen`                 | N/A          | Required number of samples to generate. |
| `-l, --label`           | None         | A label to identify different algorithm runs. It prefixes the output files with this label. |
| `-w, --weights`          | None         | Path to the file containing the weights of the sequences. If `None`, the weights are computed automatically. |
| `--clustering_seqid`     | 0.8          | Sequence identity threshold to be used for computing the sequence weights. |
| `--no_reweighting`       | N/A          | If this flag is used, the routine assigns uniform weights to the sequences. |
| `--nmeasure`            | 10000        | Number of data sequences to use for computing the mixing time. The value min(`nmeasure`, len(data)) is taken. |
| `--nmix`                | 2            | Number of mixing times used to generate 'ngen' sequences starting from random. |
| `--max_nsweeps`         | 5000         | Maximum number of sweeps allowed. |
| `--alphabet`            | auto         | Detect a standard protein, DNA, or RNA alphabet from the reference and parameter symbols, or provide a custom token string. |
| `--sampler`             | metropolis   | Sampling method to be used. Possible options are `gibbs` and `metropolis`. |
| `--beta`               | 1.0          | Inverse temperature to be used for the sampling. |
| `--seed`               | 0            | Random seed for reproducible sequence generation. |
| `--pseudocount`        | None         | Pseudo count for the single and two-sites statistics. Acts as a regularization. If `None`, it is set to $1/M_{\mathrm{eff}}$. |
| `--device`¹            | auto         | Select CUDA when available, otherwise MPS, otherwise CPU. |
| `--dtype`¹             | float32      | Sampling precision: float32, float64, or bfloat16. BF16 stores the fixed coupling matrix in reduced precision while keeping chains, statistics, energies, and model parameters in FP32; it requires an Ampere-or-newer CUDA GPU and Triton. |
| `--plot`               | off          | Save PNG plots of the mixing autocorrelation, Pearson correlation during sampling, final reference-versus-generated $C_{ij}$, and natural-versus-generated PC1–PC2 and PC3–PC4 projections with marginal histograms. Requires `--data`. |


## Computing DCA energies of a MSA

| Command                   | Default value | Description |
|---------------------------|--------------|-------------|
| `-d, --data`             | N/A          | Filename of the input MSA. |
| `-p, --path_params`      | N/A          | Path to the file containing the parameters of the DCA model. |
| `-o, --output`           | N/A          | Path to the folder where to save the output. |
| `--alphabet`            | auto         | Detect a standard alphabet from the data and parameter symbols, or provide a custom token string. |
| `--device`¹            | auto         | Select CUDA when available, otherwise MPS, otherwise CPU. |
| `--dtype`¹             | float32      | Data type to be used between float32 and float64. Used in the Python version. |

## Generate a Deep Mutational Scan (DMS) from a wild type

| Command                   | Default value | Description |
|---------------------------|--------------|-------------|
| `-d, --data`             | N/A          | Filename of the input MSA containing the wild type. If multiple sequences are present, the first one is used. |
| `-p, --path_params`      | N/A          | Path to the file containing the parameters of the DCA model. |
| `-o, --output`           | N/A          | Path to the folder where to save the output. |
| `--alphabet`            | auto         | Detect a standard alphabet from the data and parameter symbols, or provide a custom token string. |
| `--device`¹            | auto         | Select CUDA when available, otherwise MPS, otherwise CPU. |
| `--dtype`¹             | float32      | Data type to be used between float32 and float64. Used in the Python version. |



## Compute the Frobenius contact matrix

| Command                   | Default value | Description |
|---------------------------|--------------|-------------|
| `-p, --path_params`      | N/A          | Path to the file containing the parameters of the DCA model. |
| `-d, --data`             | None         | Path to the data MSA. Used to compute contacts with the mean-field DCA approximation when `--path_params` is not provided. |
| `-o, --output`           | N/A          | Path to the folder where to save the output. |
| `-l, --label`           | None         | If provided, adds a label to the output files inside the output folder. |
| `--alphabet`            | auto         | Detect a standard alphabet from the available alignment and parameter symbols, or provide a custom token string. |
| `--pseudocount`         | 0.5          | Pseudocount used to regularize empirical frequencies in the mean-field approximation. |
| `--device`¹            | auto         | Select CUDA when available, otherwise MPS, otherwise CPU. |
| `--dtype`¹             | float32      | Data type to be used between float32 and float64. Used in the Python version. |

¹ Python-specific runtime option.

Automatic alphabet detection accepts only the standard DNA, RNA, and protein alphabets (including gaps). DNA is preferred over RNA, then protein when multiple alphabets fit; A/C/G-only data therefore selects DNA. Model parameter symbols are also considered when provided. Nonstandard data requires an explicit custom `--alphabet` string. Preprocessing detects the alphabet after its transformations.

## Other workflows

The remaining commands have focused guides and complete built-in help:

| Command | Purpose | Guide |
|---------|---------|-------|
| `preprocess` | Convert and clean FASTA or Stockholm alignments. | [Alignment preprocessing](alignment_preprocessing.md) |
| `entropy` | Estimate free energy and entropy by thermodynamic integration. | [Thermodynamic integration](thermodynamic_integration.md) |
| `reintegrate` | Train with experimental-feedback adjustments. | [Applications](applications.md#reintegration) |
| `profmark` | Create profile-aware training/test splits. | [Applications](applications.md#profmark) |
| `plot-training-log` | Render metrics from a structured training log. | [Training](training.md#training-output-and-logs) |
