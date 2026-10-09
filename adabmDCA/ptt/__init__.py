"""Parallel Trajectory Tempering: sampler, archives, mixing diagnostics and training.

Modules: ``sampler`` (the replica ladder), ``kernels`` (local, replica and
exchange kernels), ``archive`` (HDF5 persistence), ``mixing`` and ``health``
(diagnostics), ``config``, and for training ``optim`` (parameter updates),
``policies`` (model-specific graph handling) and ``training`` (the loop).
"""

from importlib import import_module

_EXPORTS = {
    "PTTConfig": "adabmDCA.ptt.config",
    "PTTSampler": "adabmDCA.ptt.sampler",
    "PartitionEstimate": "adabmDCA.ptt.sampler",
    "bridge_increment": "adabmDCA.ptt.sampler",
    "load_ptt_endpoint": "adabmDCA.ptt.archive",
    "MixingEstimate": "adabmDCA.ptt.mixing",
    "RenewalEstimate": "adabmDCA.ptt.mixing",
    "train_ptt": "adabmDCA.ptt.training",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'adabmDCA.ptt' has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted([*globals(), *_EXPORTS])
