"""Local, stacked-replica and exchange kernels used by the PTT sampler."""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch

from adabmDCA.sampling import prepare_sampler
from adabmDCA.statmech import compute_energy

_TIMING_KEYS = ("local_sampling_seconds", "exchange_seconds", "permutation_seconds")
LOCAL_KERNELS = ("metropolis", "gibbs", "metropolized_gibbs")


def _one_hot(chains, params):
    if chains.ndim == 3:
        return chains
    return torch.nn.functional.one_hot(chains.long(), params["bias"].shape[1]).to(params["bias"].dtype)



def _numba_categorical_sampler(name, device):
    """The multithreaded Numba kernel for ``name`` on CPU, when Numba is installed."""
    if device.type != "cpu":
        return None
    from adabmDCA.numba_kernels import categorical_sampler, is_numba_available

    return categorical_sampler(name) if is_numba_available() else None


def _prepare_categorical_sampler(name, device):
    if name == "metropolized_gibbs":
        return _prepare_metropolized_gibbs_sampler(device)
    numba_sampler = _numba_categorical_sampler(name, device)
    if numba_sampler is not None:
        return numba_sampler
    if device.type == "cuda":
        try:
            from adabmDCA.sampling_triton import (
                gibbs_sampling_categorical_triton,
                is_triton_available,
                metropolis_sampling_categorical_triton,
            )
            if is_triton_available():
                return (
                    gibbs_sampling_categorical_triton
                    if name == "gibbs" else metropolis_sampling_categorical_triton
                )
        except ImportError:
            pass
    one_hot_sampler = prepare_sampler(name, device)

    def categorical_fallback(chains, params, nsweeps, beta=1.0):
        sampled = one_hot_sampler(_one_hot(chains, params), params, nsweeps, beta)
        return sampled.argmax(-1).to(torch.int32)

    return categorical_fallback


def _prepare_metropolized_gibbs_sampler(device):
    """Single-model Metropolized Gibbs: Triton on CUDA, Numba on CPU when installed, else PyTorch."""
    from adabmDCA.sampling import metropolized_gibbs_sampling_categorical

    numba_sampler = _numba_categorical_sampler("metropolized_gibbs", device)
    if numba_sampler is not None:
        return numba_sampler

    if device.type == "cuda":
        try:
            from adabmDCA.sampling_triton import is_triton_available, metropolized_gibbs_sampling_categorical_triton

            if is_triton_available():
                return metropolized_gibbs_sampling_categorical_triton
        except ImportError:
            pass
    return metropolized_gibbs_sampling_categorical


class _ReplicaKernel:
    """Stacked-replica local kernel with its coupling layout and use threshold.

    ``transform_couplings`` maps one (L, q, L, q) coupling tensor to the view
    stacked for the kernel; ``min_length`` is the smallest L for which the
    stacked kernel is preferred over per-replica local kernels. ``always``
    marks kernels that pay off in every advance, training included, because
    the per-replica kernels would copy the couplings on every call anyway;
    otherwise the stacked couplings are built only for long sampling runs.
    """

    def __init__(self, function, *, transform_couplings=None, min_length=1, always=False):
        self.function = function
        self.transform_couplings = transform_couplings or (lambda couplings: couplings)
        self.min_length = min_length
        self.always = always

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def _prepare_replica_sampler(name, device):
    if device.type == "cpu" and name in ("gibbs", "metropolis", "metropolized_gibbs"):
        from adabmDCA.numba_kernels import is_numba_available

        if not is_numba_available():
            return None
        from adabmDCA.numba_kernels import replica_sampler

        # The Numba sampler prepares its own layout (dense or sparse) from the stacked couplings.
        return _ReplicaKernel(replica_sampler(name), always=True)
    if name not in ("metropolis", "metropolized_gibbs") or device.type != "cuda":
        return None
    try:
        from adabmDCA.sampling_triton import (
            is_triton_available,
            metropolis_sampling_replicas_categorical_triton,
            metropolized_gibbs_sampling_replicas_triton,
        )
        if not is_triton_available():
            return None
        if name == "metropolis":
            return _ReplicaKernel(metropolis_sampling_replicas_categorical_triton, min_length=200)
        # Takes the stacked couplings as they are and prepares a dense or sparse layout itself.
        return _ReplicaKernel(metropolized_gibbs_sampling_replicas_triton)
    except ImportError:
        pass
    return None


def _prepare_exchange_kernel(device):
    def dense_categorical(lower, upper, lower_chains, upper_chains):
        delta = {
            "bias": upper["bias"] - lower["bias"],
            "coupling_matrix": upper["coupling_matrix"] - lower["coupling_matrix"],
        }
        log_acceptance = (
            compute_energy(_one_hot(upper_chains, upper), delta).double()
            - compute_energy(_one_hot(lower_chains, lower), delta).double()
        )
        return log_acceptance.clamp_max_(0.0)

    if device.type == "cpu":
        from adabmDCA.numba_kernels import is_numba_available

        if is_numba_available():
            from adabmDCA.numba_kernels import exchange_log_acceptance

            # Gathers beat the one-hot products for every alphabet on CPU
            # (equal at q=5, about six times faster at q=21).
            return exchange_log_acceptance
    if device.type == "cuda":
        try:
            from adabmDCA.sampling_triton import (
                exchange_log_acceptance_categorical_triton,
                exchange_log_acceptance_sparse_triton,
                is_triton_available,
            )
            if is_triton_available():
                def cuda_dispatch(lower, upper, lower_chains, upper_chains):
                    # On the RTX A5000 the gather kernel wins from q=10
                    # upward; small RNA alphabets remain faster through GEMM,
                    # even for sparse graphs. With q >= 10, sparse graphs read
                    # only their coupled pairs: 1.8x faster than the gather
                    # kernel at 5% of the pairs (L=279, q=21), as fast at 15%.
                    if lower["bias"].shape[1] >= 10:
                        sparse = exchange_log_acceptance_sparse_triton(lower, upper, lower_chains, upper_chains)
                        if sparse is not None:
                            return sparse
                        return exchange_log_acceptance_categorical_triton(
                            lower, upper, lower_chains, upper_chains
                        )
                    return dense_categorical(lower, upper, lower_chains, upper_chains)

                return cuda_dispatch
        except ImportError:
            pass
    return dense_categorical


class _PhaseTimer:
    def __init__(self, device):
        self.device = device
        self.cpu_totals = {key: 0.0 for key in _TIMING_KEYS}
        self.events = {key: [] for key in _TIMING_KEYS}

    @contextmanager
    def measure(self, key):
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self.events[key].append((start, end))
        else:
            start = time.perf_counter()
            try:
                yield
            finally:
                self.cpu_totals[key] += time.perf_counter() - start

    def finish(self):
        if self.device.type == "cuda" and any(self.events.values()):
            torch.cuda.synchronize(self.device)
            for key, pairs in self.events.items():
                self.cpu_totals[key] = sum(start.elapsed_time(end) for start, end in pairs) / 1000.0
        return self.cpu_totals
