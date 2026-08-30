"""Compare the current samplers with the former torch.arange loops.

Run on a CUDA machine with::

    conda run -n dca python benchmarks/benchmark_sampling.py --device cuda
"""

import argparse
import time
from typing import Callable, Dict

import torch

from adabmDCA.sampling import (
    gibbs_sampling,
    gibbs_step_independent_sites,
    gibbs_step_uniform_sites,
    metropolis_sampling,
    metropolis_step_independent_sites,
    metropolis_step_uniform_sites,
)
from adabmDCA.sampling_triton import (
    gibbs_sampling_triton,
    gibbs_step_independent_sites_triton,
    is_triton_available,
    metropolis_sampling_triton,
    metropolis_step_independent_sites_triton,
)


def baseline_gibbs(chains: torch.Tensor, params: Dict[str, torch.Tensor], nsweeps: int) -> torch.Tensor:
    result = chains.clone()
    for _ in torch.arange(nsweeps * chains.shape[1]):
        result = gibbs_step_uniform_sites(result, params)
    return result


def baseline_metropolis(chains: torch.Tensor, params: Dict[str, torch.Tensor], nsweeps: int) -> torch.Tensor:
    result = chains.clone()
    for _ in torch.arange(nsweeps * chains.shape[1]):
        result = metropolis_step_uniform_sites(result, params)
    return result


def elapsed(function, *, device, repeats=5):
    for _ in range(2):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return min(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chains", type=int, default=2048)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--states", type=int, default=21)
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    torch.manual_seed(0)
    state = torch.randint(args.states, (args.chains, args.length), device=device)
    dtype = getattr(torch, args.dtype)
    chains = torch.nn.functional.one_hot(state, args.states).to(dtype)
    bias = torch.randn(args.length, args.states, device=device, dtype=dtype) * 0.1
    couplings = torch.randn(
        args.length,
        args.states,
        args.length,
        args.states,
        device=device,
        dtype=dtype,
    ) * 0.01
    couplings = 0.5 * (couplings + couplings.permute(2, 3, 0, 1))
    site = torch.arange(args.length, device=device)
    couplings[site, :, site, :] = 0
    params = {"bias": bias, "coupling_matrix": couplings}

    # Training scripts both implementations, so benchmark their scripted forms
    # instead of timing Python loop overhead.
    cases: tuple[tuple[str, Callable, Callable], ...] = (
        ("gibbs", torch.jit.script(gibbs_sampling), torch.jit.script(baseline_gibbs)),
        ("metropolis", torch.jit.script(metropolis_sampling), torch.jit.script(baseline_metropolis)),
    )
    print(f"device={device} N={args.chains} L={args.length} q={args.states} sweeps={args.sweeps}")
    for name, optimized, old_sampler in cases:
        old = elapsed(lambda: old_sampler(chains, params, args.sweeps), device=device)
        new = elapsed(lambda: optimized(chains, params, args.sweeps), device=device)
        print(f"{name:10s} baseline={old:8.4f}s optimized={new:8.4f}s speedup={old / new:6.2f}x")
    if device.type == "cuda" and is_triton_available():
        old_sampler = torch.jit.script(baseline_gibbs)
        old = elapsed(lambda: old_sampler(chains, params, args.sweeps), device=device)
        fused = elapsed(lambda: gibbs_sampling_triton(chains, params, args.sweeps), device=device)
        print(f"{'triton-g':10s} baseline={old:8.4f}s optimized={fused:8.4f}s speedup={old / fused:6.2f}x")
        old_sampler = torch.jit.script(baseline_metropolis)
        old = elapsed(lambda: old_sampler(chains, params, args.sweeps), device=device)
        fused = elapsed(lambda: metropolis_sampling_triton(chains, params, args.sweeps), device=device)
        print(f"{'triton-m':10s} baseline={old:8.4f}s optimized={fused:8.4f}s speedup={old / fused:6.2f}x")
        independent_cases = (
            (
                "indep-g",
                torch.jit.script(gibbs_step_independent_sites),
                gibbs_step_independent_sites_triton,
            ),
            (
                "indep-m",
                torch.jit.script(metropolis_step_independent_sites),
                metropolis_step_independent_sites_triton,
            ),
        )
        for name, baseline_step, fused_step in independent_cases:
            old = elapsed(lambda: baseline_step(chains.clone(), params), device=device)
            fused = elapsed(lambda: fused_step(chains.clone(), params), device=device)
            print(f"{name:10s} baseline={old:8.4f}s optimized={fused:8.4f}s speedup={old / fused:6.2f}x")


if __name__ == "__main__":
    main()
