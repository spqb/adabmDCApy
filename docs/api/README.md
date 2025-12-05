<!-- markdownlint-disable -->

# API Overview

## Modules

- [`adabmDCA.checkpoint`](./adabmDCA.checkpoint.md#module-adabmdcacheckpoint)
- [`adabmDCA.cobalt`](./adabmDCA.cobalt.md#module-adabmdcacobalt)
- [`adabmDCA.dataset`](./adabmDCA.dataset.md#module-adabmdcadataset)
- [`adabmDCA.dca`](./adabmDCA.dca.md#module-adabmdcadca)
- [`adabmDCA.fasta`](./adabmDCA.fasta.md#module-adabmdcafasta)
- [`adabmDCA.functional`](./adabmDCA.functional.md#module-adabmdcafunctional)
- [`adabmDCA.graph`](./adabmDCA.graph.md#module-adabmdcagraph)
- [`adabmDCA.io`](./adabmDCA.io.md#module-adabmdcaio)
- [`adabmDCA.plot`](./adabmDCA.plot.md#module-adabmdcaplot)
- [`adabmDCA.resampling`](./adabmDCA.resampling.md#module-adabmdcaresampling)
- [`adabmDCA.sampling`](./adabmDCA.sampling.md#module-adabmdcasampling)
- [`adabmDCA.statmech`](./adabmDCA.statmech.md#module-adabmdcastatmech)
- [`adabmDCA.stats`](./adabmDCA.stats.md#module-adabmdcastats)
- [`adabmDCA.training`](./adabmDCA.training.md#module-adabmdcatraining)
- [`adabmDCA.utils`](./adabmDCA.utils.md#module-adabmdcautils)

## Classes

- [`checkpoint.Checkpoint`](./adabmDCA.checkpoint.md#class-checkpoint): Helper class to save the model's parameters and chains at regular intervals during training and to log the
- [`dataset.DatasetDCA`](./adabmDCA.dataset.md#class-datasetdca): Dataset class for handling multi-sequence alignments data.

## Functions

- [`cobalt.prune_redundant_sequences`](./adabmDCA.cobalt.md#function-prune_redundant_sequences): Prunes sequences from X such that no sequence has more than 'seqid_th' fraction of its residues identical to any other sequence in the set.
- [`cobalt.run_cobalt`](./adabmDCA.cobalt.md#function-run_cobalt): Runs the Cobalt algorithm to split the input MSA into training and test sets.
- [`cobalt.split_train_test`](./adabmDCA.cobalt.md#function-split_train_test): Splits X into two sets, T and S, such that no sequence in S has more than
- [`dca.get_contact_map`](./adabmDCA.dca.md#function-get_contact_map): Computes the contact map from the model coupling matrix.
- [`dca.get_mf_contact_map`](./adabmDCA.dca.md#function-get_mf_contact_map): Computes the contact map using mean-field approximation from the data.
- [`dca.get_seqid`](./adabmDCA.dca.md#function-get_seqid): Returns a tensor containing the sequence identities between two sets of one-hot encoded sequences.
- [`dca.get_seqid_stats`](./adabmDCA.dca.md#function-get_seqid_stats): - If s2 is provided, computes the mean and the standard deviation of the mean sequence identity between two sets of one-hot encoded sequences.
- [`dca.set_zerosum_gauge`](./adabmDCA.dca.md#function-set_zerosum_gauge): Sets the zero-sum gauge on the coupling matrix.
- [`fasta.compute_weights`](./adabmDCA.fasta.md#function-compute_weights): Computes the weight to be assigned to each sequence 's' in 'data' as 1 / n_clust, where 'n_clust' is the number of sequences
- [`fasta.decode_sequence`](./adabmDCA.fasta.md#function-decode_sequence): Takes a numeric sequence or list of seqences in input an returns the corresponding string encoding.
- [`fasta.encode_sequence`](./adabmDCA.fasta.md#function-encode_sequence): Encodes a sequence or a list of sequences into a numeric format.
- [`fasta.get_tokens`](./adabmDCA.fasta.md#function-get_tokens): Converts a known alphabet into the corresponding tokens, otherwise returns the custom alphabet.
- [`fasta.import_from_fasta`](./adabmDCA.fasta.md#function-import_from_fasta): Import sequences from a fasta or compressed fasta (.fas.gz) file. The following operations are performed:
- [`fasta.validate_alphabet`](./adabmDCA.fasta.md#function-validate_alphabet): Check if the chosen alphabet is compatible with the input sequences.
- [`fasta.write_fasta`](./adabmDCA.fasta.md#function-write_fasta): Generate a fasta file with the input sequences.
- [`functional.one_hot`](./adabmDCA.functional.md#function-one_hot): A fast one-hot encoding function faster than the PyTorch one working with torch.int32 and returning a float Tensor.
- [`graph.decimate_graph`](./adabmDCA.graph.md#function-decimate_graph): Performs one decimation step and updates the parameters and mask.
- [`graph.update_mask_activation`](./adabmDCA.graph.md#function-update_mask_activation): Updates the mask by removing the nactivate couplings with the smallest Dkl.
- [`graph.update_mask_decimation`](./adabmDCA.graph.md#function-update_mask_decimation): Updates the mask by removing the n_remove couplings with the smallest Dkl.
- [`io.load_chains`](./adabmDCA.io.md#function-load_chains): Loads the sequences from a fasta file and returns the one-hot encoded version.
- [`io.load_params`](./adabmDCA.io.md#function-load_params): Import the parameters of the model from a text file.
- [`io.load_params_old`](./adabmDCA.io.md#function-load_params_old): Import the parameters of the model from a file.
- [`io.load_params_oldformat`](./adabmDCA.io.md#function-load_params_oldformat): Import the parameters of the model from a file. Assumes the old DCA format.
- [`io.save_chains`](./adabmDCA.io.md#function-save_chains): Saves the chains in a fasta file.
- [`io.save_params`](./adabmDCA.io.md#function-save_params): Saves the parameters of the model in a file.
- [`io.save_params_oldformat`](./adabmDCA.io.md#function-save_params_oldformat): Saves the parameters of the model in a file. Assumes the old DCA format.
- [`plot.plot_PCA`](./adabmDCA.plot.md#function-plot_pca): Makes the scatter plot of the components (pc1, pc2) of the input data and shows the histograms of the components.
- [`plot.plot_autocorrelation`](./adabmDCA.plot.md#function-plot_autocorrelation): Plots the time-autocorrelation curve of the sequence identity and the generated and data sequence identities.
- [`plot.plot_contact_map`](./adabmDCA.plot.md#function-plot_contact_map): Plots the contact map.
- [`plot.plot_pearson_sampling`](./adabmDCA.plot.md#function-plot_pearson_sampling): Plots the Pearson correlation coefficient over sampling time.
- [`plot.plot_scatter_correlations`](./adabmDCA.plot.md#function-plot_scatter_correlations): Plots the scatter plot of the data and generated Cij and Cijk values.
- [`resampling.compute_mixing_time`](./adabmDCA.resampling.md#function-compute_mixing_time): Computes the mixing time using the t and t/2 method. The sampling will halt when the mixing time is reached or
- [`sampling.get_sampler`](./adabmDCA.sampling.md#function-get_sampler): Returns the sampling function corresponding to the chosen method.
- [`sampling.gibbs_sampling`](./adabmDCA.sampling.md#function-gibbs_sampling): Gibbs sampling. Attempts L * nsweeps mutations to each sequence in 'chains'.
- [`sampling.gibbs_step_independent_sites`](./adabmDCA.sampling.md#function-gibbs_step_independent_sites): Performs a single mutation using the Gibbs sampler. This version selects different random sites for each chain. It is
- [`sampling.gibbs_step_uniform_sites`](./adabmDCA.sampling.md#function-gibbs_step_uniform_sites): Performs a single mutation using the Gibbs sampler. In this version, the mutation is attempted at the same sites for all chains.
- [`sampling.metropolis_sampling`](./adabmDCA.sampling.md#function-metropolis_sampling): Metropolis sampling. Attempts L * nsweeps mutations to each sequence in 'chains'.
- [`sampling.metropolis_step_independent_sites`](./adabmDCA.sampling.md#function-metropolis_step_independent_sites): Performs a single mutation using the Metropolis sampler. This version selects different random sites for each chain. It is
- [`sampling.metropolis_step_uniform_sites`](./adabmDCA.sampling.md#function-metropolis_step_uniform_sites): Performs a single mutation using the Metropolis sampler. In this version, the mutation is attempted at the same sites for all chains.
- [`sampling.sampling_profile`](./adabmDCA.sampling.md#function-sampling_profile): Samples from the profile model defined by the local biases only.
- [`statmech.compute_energy`](./adabmDCA.statmech.md#function-compute_energy): Compute the DCA energy for a batch of sequences.
- [`statmech.compute_entropy`](./adabmDCA.statmech.md#function-compute_entropy): Compute the entropy of the DCA model.
- [`statmech.compute_logZ_exact`](./adabmDCA.statmech.md#function-compute_logz_exact): Compute the log-partition function of the model.
- [`statmech.compute_log_likelihood`](./adabmDCA.statmech.md#function-compute_log_likelihood): Compute the log-likelihood of the model.
- [`statmech.enumerate_states`](./adabmDCA.statmech.md#function-enumerate_states): Enumerate all possible states of a system of L sites and q states.
- [`statmech.iterate_tap`](./adabmDCA.statmech.md#function-iterate_tap): Iterates the TAP equations until convergence.
- [`stats.extract_Cij_from_freq`](./adabmDCA.stats.md#function-extract_cij_from_freq): Extracts the lower triangular part of the covariance matrices of the natural data and generated data starting from the frequencies.
- [`stats.extract_Cij_from_seqs`](./adabmDCA.stats.md#function-extract_cij_from_seqs): Extracts the lower triangular part of the covariance matrices of the natural data and generated data starting from the sequences.
- [`stats.generate_unique_triplets`](./adabmDCA.stats.md#function-generate_unique_triplets): Generates a set of unique triplets of positions. Used to compute the 3-points statistics.
- [`stats.get_correlation_two_points`](./adabmDCA.stats.md#function-get_correlation_two_points): Computes the Pearson coefficient and the slope between the two-point frequencies of data and chains.
- [`stats.get_covariance_matrix`](./adabmDCA.stats.md#function-get_covariance_matrix): Computes the weighted covariance matrix of the input multi sequence alignment.
- [`stats.get_freq_single_point`](./adabmDCA.stats.md#function-get_freq_single_point): Computes the single point frequencies of the input MSA.
- [`stats.get_freq_three_points`](./adabmDCA.stats.md#function-get_freq_three_points): Computes the 3-body connected correlation statistics of the input MSAs.
- [`stats.get_freq_two_points`](./adabmDCA.stats.md#function-get_freq_two_points): Computes the 2-points statistics of the input MSA.
- [`training.train_eaDCA`](./adabmDCA.training.md#function-train_eadca): Fits an eaDCA model on the training data and saves the results in a file.
- [`training.train_edDCA`](./adabmDCA.training.md#function-train_eddca): Fits an edDCA model on the training data and saves the results in a file.
- [`training.train_graph`](./adabmDCA.training.md#function-train_graph): Trains the model on a given graph until the target Pearson correlation is reached or the maximum number of epochs is exceeded.
- [`training.update_params`](./adabmDCA.training.md#function-update_params): Updates the parameters of the model.
- [`utils.get_device`](./adabmDCA.utils.md#function-get_device): Returns the device where to store the tensors.
- [`utils.get_dtype`](./adabmDCA.utils.md#function-get_dtype): Returns the data type of the tensors.
- [`utils.get_mask_save`](./adabmDCA.utils.md#function-get_mask_save): Returns the mask to save the upper-triangular part of the coupling matrix.
- [`utils.init_chains`](./adabmDCA.utils.md#function-init_chains): Initialize the Markov chains of the DCA model. If 'fi' is provided, the chains are sampled from the
- [`utils.init_parameters`](./adabmDCA.utils.md#function-init_parameters): Initialize the parameters of the DCA model. The bias terms are initialized
- [`utils.resample_sequences`](./adabmDCA.utils.md#function-resample_sequences): Extracts nextract sequences from data with replacement according to the weights.


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
