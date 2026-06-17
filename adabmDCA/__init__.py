from importlib import import_module

__version__ = "0.7.8"

_EXPORTS = {
    "import_from_fasta": "adabmDCA.fasta",
    "get_tokens": "adabmDCA.fasta",
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
