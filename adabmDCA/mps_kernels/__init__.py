"""Fused Apple Silicon kernels, enabled automatically on supported MPS devices.

Modules:
    sampling: Gibbs, Metropolis and Metropolized Gibbs updates and adapters.
    exchange: Categorical replica exchange probabilities.
    swaps: Fused PTT chain/metadata swaps and population permutations.
    sparse: Cached neighbour tables and nonzero coupling blocks.
    random: Philox shader helpers and PyTorch MPS random keys.
    runtime: Availability and cached, lazy shader compilation.

Set ADABMDCA_MPS=0 to select the original PyTorch samplers. Importing this
package is safe on CPU/CUDA hosts and older PyTorch versions without Metal
shader compilation. PTT uses categorical local and stacked-replica samplers
and a fused exchange kernel, with float32 state on the GPU.
"""

from importlib import import_module

from adabmDCA.mps_kernels.runtime import is_mps_available

_EXPORTS = {
    "sparse_coupling_layout": "adabmDCA.mps_kernels.sparse",
    "exchange_log_acceptance": "adabmDCA.mps_kernels.exchange",
    "replica_sampler": "adabmDCA.mps_kernels.sampling",
    "sample_replicas": "adabmDCA.mps_kernels.sampling",
    "METHODS": "adabmDCA.mps_kernels.sampling",
    "sample_categorical": "adabmDCA.mps_kernels.sampling",
    "categorical_sampler": "adabmDCA.mps_kernels.sampling",
    "onehot_sampler": "adabmDCA.mps_kernels.sampling",
}

__all__ = list(_EXPORTS) + ["is_mps_available"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'adabmDCA.mps_kernels' has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
