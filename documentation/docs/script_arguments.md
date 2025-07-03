# <span id="script_arguments">Script Arguments</span>
 
In this section we list all the possible command-line arguments for the main routines of `adabmDCA 2.0`.

## Train a DCA model

| Command                   | Default value | Description |
|---------------------------|--------------|-------------|
| `-d, --data`             | N/A          | Filename of the dataset to be used for training the model. |
| `-o, --output`           | DCA_model    | Path to the folder where to save the model. |
| `-m, --model`            | bmDCA        | Type of model to be trained. Possible options are `bmDCA`, `eaDCA`, and `edDCA`. |
| `-w, --weights`          | None         | Path to the file containing the weights of the sequences. If `None`, the weights are computed automatically. |
| `--clustering_seqid`     | 0.8          | Sequence identity threshold to be used for computing the sequence weights. |
| `--no_reweighting`       | N/A          | If this flag is used, the routine assigns uniform weights to the sequences. |
| `-p, --path_params`      | None         | Path to the file containing the model's parameters. Required for restoring the training. |
| `-c, --path_chains`      | None         | Path to the FASTA file containing the model's chains. Required for restoring the training. |
| `-l, --label`           | None         | A label to identify different algorithm runs. It prefixes the output files with this label. |
| `--alphabet`            | protein      | Type of encoding for the sequences. Choose among `protein`, `rna`, `dna`, or a user-defined string of tokens. |
| `--lr`                  | 0.05         | Learning rate. |
| `--nsweeps`             | 10           | Number of sweeps for each gradient estimation. |
| `--sampler`             | gibbs        | Sampling method to be used. Possible options are `gibbs` and `metropolis`. |
| `--nchains`             | 10000        | Number of Markov chains to run in parallel. |
| `--target`              | 0.95         | Pearson correlation coefficient on the two-sites statistics to be reached. |
| `--nepochs`             | 50000        | Maximum number of epochs allowed. |
| `--pseudocount`        | None         | Pseudo count for the single and two-sites statistics. Acts as a regularization. If `None`, it is set to $1/M_{\mathrm{eff}}$. |
| `--seed`               | 0            | Random seed. |
| `--nthreads`¹         | 1            | Number of threads used in the Julia multithreaded version. |
| `--device`¹           | cuda         | Device to be used between cuda (GPU) and CPU. Used in the Python version. |
| `--dtype`¹            | float32      | Data type to be used between float32 and float64. Used in the Python version. |

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

                                                        
## Sampling from a DCA model 

| Command                   | Default value | Description |
|---------------------------|--------------|-------------|
| `-p, --path_params`      | N/A          | Path to the file containing the parameters of the DCA model to sample from. |
| `-d, --data`             | N/A          | Filename of the dataset MSA. |
| `-o, --output`           | N/A          | Path to the folder where to save the output. |
| `--ngen`                 | None         | Number of samples to generate. |
| `-l, --label`           | None         | A label to identify different algorithm runs. It prefixes the output files with this label. |
| `-w, --weights`          | None         | Path to the file containing the weights of the sequences. If `None`, the weights are computed automatically. |
| `--clustering_seqid`     | 0.8          | Sequence identity threshold to be used for computing the sequence weights. |
| `--no_reweighting`       | N/A          | If this flag is used, the routine assigns uniform weights to the sequences. |
| `--nmeasure`            | 10000        | Number of data sequences to use for computing the mixing time. The value min(`nmeasure`, len(data)) is taken. |
| `--nmix`                | 2            | Number of mixing times used to generate 'ngen' sequences starting from random. |
| `--max_nsweeps`         | 10000        | Maximum number of sweeps allowed. |
| `--alphabet`            | protein      | Type of encoding for the sequences. Choose among `protein`, `rna`, `dna`, or a user-defined string of tokens. |
| `--sampler`             | gibbs        | Sampling method to be used. Possible options are `gibbs` and `metropolis`. |
| `--beta`               | 1.0          | Inverse temperature to be used for the sampling. |
| `--pseudocount`        | None         | Pseudo count for the single and two-sites statistics. Acts as a regularization. If `None`, it is set to $1/M_{\mathrm{eff}}$. |
| `--device`¹            | cuda         | Device to be used between cuda (GPU) and CPU. Used in the Python version. |
| `--dtype`¹             | float32      | Data type to be used between float32 and float64. Used in the Python version. |


## Computing DCA energies of a MSA

| Command                   | Default value | Description |
|---------------------------|--------------|-------------|
| `-d, --data`             | N/A          | Filename of the input MSA. |
| `-p, --path_params`      | N/A          | Path to the file containing the parameters of the DCA model. |
| `-o, --output`           | N/A          | Path to the folder where to save the output. |
| `--alphabet`            | protein      | Type of encoding for the sequences. Choose among `protein`, `rna`, `dna`, or a user-defined string of tokens. |
| `--device`¹            | cuda         | Device to be used between cuda (GPU) and CPU. Used in the Python version. |
| `--dtype`¹             | float32      | Data type to be used between float32 and float64. Used in the Python version. |

## Generate a Deep Mutational Scan (DMS) from a wild type

| Command                   | Default value | Description |
|---------------------------|--------------|-------------|
| `-d, --data`             | N/A          | Filename of the input MSA containing the wild type. If multiple sequences are present, the first one is used. |
| `-p, --path_params`      | N/A          | Path to the file containing the parameters of the DCA model. |
| `-o, --output`           | N/A          | Path to the folder where to save the output. |
| `--alphabet`            | protein      | Type of encoding for the sequences. Choose among `protein`, `rna`, `dna`, or a user-defined string of tokens. |
| `--device`¹            | cuda         | Device to be used between cuda (GPU) and CPU. Used in the Python version. |
| `--dtype`¹             | float32      | Data type to be used between float32 and float64. Used in the Python version. |



## Compute the Frobenius contact matrix

| Command                   | Default value | Description |
|---------------------------|--------------|-------------|
| `-p, --path_params`      | N/A          | Path to the file containing the parameters of the DCA model. |
| `-o, --output`           | N/A          | Path to the folder where to save the output. |
| `-l, --label`           | None         | If provided, adds a label to the output files inside the output folder. |
| `--alphabet`            | protein      | Type of encoding for the sequences. Choose among `protein`, `rna`, `dna`, or a user-defined string of tokens. |
| `--device`¹            | cuda         | Device to be used between cuda (GPU) and CPU. Used in the Python version. |
| `--dtype`¹             | float32      | Data type to be used between float32 and float64. Used in the Python version. |

¹ Used in specific versions of the software.

