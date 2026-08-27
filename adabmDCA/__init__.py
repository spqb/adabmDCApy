from importlib import import_module

__version__ = "0.7.8"

_EXPORTS = {
    "Alignment": "adabmDCA.alignment",
    "AlignmentConversionResult": "adabmDCA.alignment",
    "detect_alignment_format": "adabmDCA.alignment",
    "read_alignment": "adabmDCA.alignment",
    "write_alignment": "adabmDCA.alignment",
    "convert_alignment": "adabmDCA.alignment",
    "convert_stockholm_to_fasta": "adabmDCA.alignment",
    "AlignmentProcessingConfig": "adabmDCA.preprocessing",
    "AlignmentProcessingReport": "adabmDCA.preprocessing",
    "AlignmentFilterResult": "adabmDCA.preprocessing",
    "AlignmentProcessingResult": "adabmDCA.preprocessing",
    "remove_insertions": "adabmDCA.preprocessing",
    "normalize_gap_symbols": "adabmDCA.alignment",
    "filter_gap_fraction": "adabmDCA.preprocessing",
    "preprocess_alignment": "adabmDCA.preprocessing",
    "DCAModel": "adabmDCA.api.model",
    "load_model": "adabmDCA.api.model",
    "inspect_model": "adabmDCA.api.model",
    "compute_energies": "adabmDCA.api.scoring",
    "score_sequences": "adabmDCA.api.scoring",
    "compute_contact_map": "adabmDCA.api.contacts",
    "predict_contacts": "adabmDCA.api.contacts",
    "scan_mutations": "adabmDCA.api.mutations",
    "sample_sequences": "adabmDCA.api.sampling",
    "generate_sequences": "adabmDCA.api.sampling",
    "train_model": "adabmDCA.api.training",
    "split_alignment": "adabmDCA.api.splitting",
    "reintegrate_model": "adabmDCA.api.reintegration",
    "estimate_entropy": "adabmDCA.api.entropy",
    "StopReason": "adabmDCA.training_control",
    "TrainingLimits": "adabmDCA.training_control",
    "TrainingCounters": "adabmDCA.training_control",
    "TrainingConfig": "adabmDCA.training_config",
    "ConfigurationError": "adabmDCA.training_config",
    "AlignmentLoadConfig": "adabmDCA.input_loading",
    "LoadedAlignment": "adabmDCA.input_loading",
    "load_alignment": "adabmDCA.input_loading",
    "load_sequence_weights": "adabmDCA.input_loading",
    "AdabmDCAError": "adabmDCA.api.exceptions",
    "InputValidationError": "adabmDCA.api.exceptions",
    "InputLoadError": "adabmDCA.api.exceptions",
    "WeightLoadError": "adabmDCA.api.exceptions",
    "ChainLoadError": "adabmDCA.api.exceptions",
    "OutputSerializationError": "adabmDCA.api.exceptions",
    "ComputationError": "adabmDCA.api.exceptions",
    "ConvergenceError": "adabmDCA.api.exceptions",
    "ModelCompatibilityError": "adabmDCA.api.exceptions",
    "ModelLoadError": "adabmDCA.api.exceptions",
    "OperationCancelledError": "adabmDCA.api.exceptions",
    "AlignmentError": "adabmDCA.api.exceptions",
    "AlignmentLoadError": "adabmDCA.api.exceptions",
    "AlignmentFormatError": "adabmDCA.api.exceptions",
    "AlignmentLengthError": "adabmDCA.api.exceptions",
    "import_from_fasta": "adabmDCA.fasta",
    "get_tokens": "adabmDCA.alphabet",
    "write_fasta": "adabmDCA.fasta",
    "encode_sequence": "adabmDCA.fasta",
    "decode_sequence": "adabmDCA.fasta",
    "compute_weights": "adabmDCA.fasta",
    "get_freq_single_point": "adabmDCA.stats",
    "get_freq_two_points": "adabmDCA.stats",
    "get_freq_three_points": "adabmDCA.stats",
    "get_correlation_two_points": "adabmDCA.stats",
    "load_params": "adabmDCA.io",
    "save_params": "adabmDCA.io",
    "get_sampler": "adabmDCA.sampling",
    "gibbs_sampling": "adabmDCA.sampling",
    "metropolis_sampling": "adabmDCA.sampling",
    "sampling_profile": "adabmDCA.sampling",
    "gibbs_step_independent_sites": "adabmDCA.sampling",
    "metropolis_step_independent_sites": "adabmDCA.sampling",
    "gibbs_step_uniform_sites": "adabmDCA.sampling",
    "metropolis_step_uniform_sites": "adabmDCA.sampling",
    "one_hot": "adabmDCA.functional",
    "compute_energy": "adabmDCA.statmech",
    "get_seqid": "adabmDCA.dca",
    "get_seqid_stats": "adabmDCA.dca",
    "get_contact_map": "adabmDCA.dca",
    "get_mf_contact_map": "adabmDCA.dca",
    "init_chains": "adabmDCA.utils",
    "init_parameters": "adabmDCA.utils",
    "parse_log_file": "adabmDCA.utils",
    "resample_sequences": "adabmDCA.utils",
    "DatasetDCA": "adabmDCA.dataset",
    "ProfileSplitResult": "adabmDCA.api.results",
    "ReintegrationResult": "adabmDCA.api.results",
    "ThermodynamicIntegrationProgress": "adabmDCA.api.results",
    "ThermodynamicIntegrationResult": "adabmDCA.api.results",
    "RESULT_SCHEMA_VERSION": "adabmDCA.api.serialization",
    "to_jsonable": "adabmDCA.api.serialization",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'adabmDCA' has no attribute {name!r}")

    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted([*globals(), *_EXPORTS])
