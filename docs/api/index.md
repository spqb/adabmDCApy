<!-- markdownlint-disable -->

# API Overview

## Modules

- [`cobalt`](./cobalt.md#module-cobalt)
- [`dataset`](./dataset.md#module-dataset)
- [`fasta`](./fasta.md#module-fasta)
- [`functional`](./functional.md#module-functional)
- [`io`](./io.md#module-io)
- [`sampling`](./sampling.md#module-sampling)
- [`statmech`](./statmech.md#module-statmech)
- [`stats`](./stats.md#module-stats)
- [`utils`](./utils.md#module-utils)

## Classes

- [`dataset.DatasetDCA`](./dataset.md#class-datasetdca): Dataset class for handling multi-sequence alignments data.

## Functions

- [`cobalt.prune_redundant_sequences`](./cobalt.md#function-prune_redundant_sequences): Prunes sequences from X such that no sequence has more than 'seqid_th' fraction of its residues identical to any other sequence in the set.
- [`cobalt.run_cobalt`](./cobalt.md#function-run_cobalt): Runs the Cobalt algorithm to split the input MSA into training and test sets.
- [`cobalt.split_train_test`](./cobalt.md#function-split_train_test): Splits X into two sets, T and S, such that no sequence in S has more than
- [`fasta.compute_weights`](./fasta.md#function-compute_weights): Computes the weight to be assigned to each sequence 's' in 'data' as 1 / n_clust, where 'n_clust' is the number of sequences
- [`fasta.decode_sequence`](./fasta.md#function-decode_sequence): Takes a numeric sequence or list of seqences in input an returns the corresponding string encoding.
- [`fasta.encode_sequence`](./fasta.md#function-encode_sequence): Encodes a sequence or a list of sequences into a numeric format.
- [`fasta.get_tokens`](./fasta.md#function-get_tokens): Converts the alphabet into the corresponding tokens.
- [`fasta.import_from_fasta`](./fasta.md#function-import_from_fasta): Import sequences from a fasta file. The following operations are performed:
- [`fasta.validate_alphabet`](./fasta.md#function-validate_alphabet): Check if the chosen alphabet is compatible with the input sequences.
- [`fasta.write_fasta`](./fasta.md#function-write_fasta): Generate a fasta file with the input sequences.
- [`functional.one_hot`](./functional.md#function-one_hot): A fast one-hot encoding function faster than the PyTorch one working with torch.int32 and returning a float Tensor.
- [`io.load_params`](./io.md#function-load_params): Import the parameters of the model from a file.
- [`io.load_params_oldformat`](./io.md#function-load_params_oldformat): Import the parameters of the model from a file. Assumes the old DCA format.
- [`io.save_chains`](./io.md#function-save_chains): Saves the chains in a fasta file.
- [`io.save_params`](./io.md#function-save_params): Saves the parameters of the model in a file.
- [`io.save_params_oldformat`](./io.md#function-save_params_oldformat): Saves the parameters of the model in a file. Assumes the old DCA format.
- [`sampling.get_sampler`](./sampling.md#function-get_sampler): Returns the sampling function corresponding to the chosen method.
- [`sampling.gibbs_sampling`](./sampling.md#function-gibbs_sampling): Gibbs sampling.
- [`sampling.metropolis`](./sampling.md#function-metropolis): Metropolis sampling.
- [`statmech.compute_energy`](./statmech.md#function-compute_energy): Compute the DCA energy of the sequences in X.
- [`statmech.compute_entropy`](./statmech.md#function-compute_entropy): Compute the entropy of the DCA model.
- [`statmech.compute_logZ_exact`](./statmech.md#function-compute_logz_exact): Compute the log-partition function of the model.
- [`statmech.compute_log_likelihood`](./statmech.md#function-compute_log_likelihood): Compute the log-likelihood of the model.
- [`statmech.enumerate_states`](./statmech.md#function-enumerate_states): Enumerate all possible states of a system of L sites and q states.
- [`statmech.iterate_tap`](./statmech.md#function-iterate_tap): Iterates the TAP equations until convergence.
- [`stats.extract_Cij_from_freq`](./stats.md#function-extract_cij_from_freq): Extracts the lower triangular part of the covariance matrices of the data and chains starting from the frequencies.
- [`stats.extract_Cij_from_seqs`](./stats.md#function-extract_cij_from_seqs): Extracts the lower triangular part of the covariance matrices of the data and chains starting from the sequences.
- [`stats.generate_unique_triplets`](./stats.md#function-generate_unique_triplets): Generates a set of unique triplets of positions. Used to compute the 3-points statistics.
- [`stats.get_correlation_two_points`](./stats.md#function-get_correlation_two_points): Computes the Pearson coefficient and the slope between the two-point frequencies of data and chains.
- [`stats.get_covariance_matrix`](./stats.md#function-get_covariance_matrix): Computes the weighted covariance matrix of the input multi sequence alignment.
- [`stats.get_freq_single_point`](./stats.md#function-get_freq_single_point): Computes the single point frequencies of the input MSA.
- [`stats.get_freq_three_points`](./stats.md#function-get_freq_three_points): Computes the 3-body connected correlation statistics of the input MSAs.
- [`stats.get_freq_two_points`](./stats.md#function-get_freq_two_points): Computes the 2-points statistics of the input MSA.
- [`utils.get_device`](./utils.md#function-get_device): Returns the device where to store the tensors.
- [`utils.get_dtype`](./utils.md#function-get_dtype): Returns the data type of the tensors.
- [`utils.get_mask_save`](./utils.md#function-get_mask_save): Returns the mask to save the upper-triangular part of the coupling matrix.
- [`utils.init_chains`](./utils.md#function-init_chains): Initialize the chains of the DCA model. If 'fi' is provided, the chains are sampled from the
- [`utils.init_parameters`](./utils.md#function-init_parameters): Initialize the parameters of the DCA model.
- [`utils.resample_sequences`](./utils.md#function-resample_sequences): Extracts nextract sequences from data with replacement according to the weights.
- [`utils.set_zerosum_gauge`](./utils.md#function-set_zerosum_gauge): Sets the zero-sum gauge on the coupling matrix.