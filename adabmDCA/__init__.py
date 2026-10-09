from importlib import import_module

from adabmDCA.api import _EXPORTS as _API_EXPORTS

__version__ = "1.0.0"

# Everything in adabmDCA.api, plus the lower-level building blocks.
_EXPORTS = {
    **_API_EXPORTS,
    "MixingEstimate": "adabmDCA.ptt.mixing",
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
    "get_tokens": "adabmDCA.alphabet",
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
    "metropolized_gibbs_sampling": "adabmDCA.sampling",
    "metropolized_gibbs_step_uniform_sites": "adabmDCA.sampling",
    "sampling_profile": "adabmDCA.sampling",
    "gibbs_step_independent_sites": "adabmDCA.sampling",
    "metropolis_step_independent_sites": "adabmDCA.sampling",
    "gibbs_step_independent_sites_triton": "adabmDCA.sampling_triton",
    "metropolis_step_independent_sites_triton": "adabmDCA.sampling_triton",
    "gibbs_step_uniform_sites": "adabmDCA.sampling",
    "metropolis_step_uniform_sites": "adabmDCA.sampling",
    "one_hot": "adabmDCA.functional",
    "compute_energy": "adabmDCA.statmech",
    "get_cde": "adabmDCA.statmech",
    "get_seqid": "adabmDCA.dca",
    "get_seqid_stats": "adabmDCA.dca",
    "get_contact_map": "adabmDCA.dca",
    "get_mf_contact_map": "adabmDCA.dca",
    "init_chains": "adabmDCA.utils",
    "init_parameters": "adabmDCA.utils",
    "parse_log_file": "adabmDCA.utils",
    "resample_sequences": "adabmDCA.utils",
    "DatasetDCA": "adabmDCA.dataset",
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
