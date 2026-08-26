"""High-level, transport-neutral adabmDCA application API."""

from importlib import import_module


_EXPORTS = {
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
    "AdabmDCAError": "adabmDCA.api.exceptions",
    "InputValidationError": "adabmDCA.api.exceptions",
    "ModelCompatibilityError": "adabmDCA.api.exceptions",
    "ModelLoadError": "adabmDCA.api.exceptions",
    "OperationCancelledError": "adabmDCA.api.exceptions",
    "AlignmentError": "adabmDCA.api.exceptions",
    "AlignmentFormatError": "adabmDCA.api.exceptions",
    "AlignmentLengthError": "adabmDCA.api.exceptions",
    "ContactMapResult": "adabmDCA.api.results",
    "EnergyResult": "adabmDCA.api.results",
    "ModelMetadata": "adabmDCA.api.results",
    "MutationRecord": "adabmDCA.api.results",
    "MutationScanResult": "adabmDCA.api.results",
    "SamplingProgress": "adabmDCA.api.results",
    "SamplingResult": "adabmDCA.api.results",
    "TrainingProgress": "adabmDCA.api.results",
    "TrainingResult": "adabmDCA.api.results",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'adabmDCA.api' has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted([*globals(), *_EXPORTS])
