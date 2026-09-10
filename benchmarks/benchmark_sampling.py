"""End-to-end old/new sampler benchmarks, including preparation and RNG.

    python -m benchmarks.benchmark_sampling --device cuda --output /tmp/gpu.json
    python -m benchmarks.benchmark_sampling --device cpu --threads 4 --output /tmp/cpu.json

The frozen Triton baseline predates the layout and multi-step optimizations.
Compilation is warmed up; old/new order alternates to reduce timing bias.
"""

import argparse
import json
import platform
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Dict

import torch

from adabmDCA import sampling, sampling_triton
from benchmarks import sampling_triton_baseline


def baseline_gibbs(
    chains: torch.Tensor, params: Dict[str, torch.Tensor], nsweeps: int, beta: float = 1.0
) -> torch.Tensor:
    result = chains.clone()
    for _ in torch.arange(nsweeps * chains.shape[1]):
        result = sampling.gibbs_step_uniform_sites(result, params, beta)
    return result


def baseline_metropolis(
    chains: torch.Tensor, params: Dict[str, torch.Tensor], nsweeps: int, beta: float = 1.0
) -> torch.Tensor:
    result = chains.clone()
    for _ in torch.arange(nsweeps * chains.shape[1]):
        result = sampling.metropolis_step_uniform_sites(result, params, beta)
    return result


def benchmark_pair(old, new, device, repeats):
    for _ in range(2):
        old()
        new()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = [[], []]
    for repeat in range(repeats):
        for index in (0, 1) if repeat % 2 == 0 else (1, 0):
            start = time.perf_counter()
            (old, new)[index]()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples[index].append(1000 * (time.perf_counter() - start))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--chains", type=int, default=2048)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--states", type=int, default=21)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--steps-per-launch", type=int, default=16)
    parser.add_argument("--no-transpose", action="store_true")
    parser.add_argument("--independent", action="store_true", help="also compare old/new independent-site Triton steps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.chains, args.length, args.states, args.sweeps, args.threads, args.repeats, args.steps_per_launch) < 1:
        parser.error("shape, sweep, thread, repeat and launch sizes must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    state = torch.randint(args.states, (args.chains, args.length), device=device)
    chains = torch.nn.functional.one_hot(state, args.states).to(dtype)
    bias = torch.randn(args.length, args.states, device=device, dtype=dtype) * 0.1
    coupling = torch.randn(args.length, args.states, args.length, args.states, device=device, dtype=dtype) * 0.01
    coupling = 0.5 * (coupling + coupling.permute(2, 3, 0, 1))
    site = torch.arange(args.length, device=device)
    coupling[site, :, site, :] = 0
    params = {"bias": bias, "coupling_matrix": coupling}
    cases = [
        ("torch-gibbs", torch.jit.script(baseline_gibbs), torch.jit.script(sampling.gibbs_sampling), {}),
        ("torch-metropolis", torch.jit.script(baseline_metropolis), torch.jit.script(sampling.metropolis_sampling), {}),
    ]
    if device.type == "cuda":
        if not sampling_triton.is_triton_available():
            raise SystemExit("Triton is unavailable")
        cases.extend(
            [
                (
                    "triton-gibbs",
                    sampling_triton_baseline.gibbs_sampling_triton,
                    sampling_triton.gibbs_sampling_triton,
                    {"steps_per_launch": args.steps_per_launch, "transpose_couplings": not args.no_transpose},
                ),
                (
                    "triton-metropolis",
                    sampling_triton_baseline.metropolis_sampling_triton,
                    sampling_triton.metropolis_sampling_triton,
                    {"steps_per_launch": args.steps_per_launch},
                ),
            ]
        )
    report = {
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "triton": sampling_triton.triton.__version__ if sampling_triton.is_triton_available() else None,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "threads": torch.get_num_threads(),
        },
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "results": [],
    }
    print(json.dumps(report, indent=2), flush=True)
    for name, old_sampler, new_sampler, options in cases:
        old = partial(old_sampler, chains, params, args.sweeps)
        new = partial(new_sampler, chains, params, args.sweeps, **options)
        torch.manual_seed(123)
        expected = old()
        torch.manual_seed(123)
        actual = new()
        mismatches = (expected.argmax(-1) != actual.argmax(-1)).sum().item()
        torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual[..., 0]), rtol=0, atol=0)
        if mismatches:
            raise AssertionError(f"{name}: {mismatches} differing residues with identical RNG seed")
        old_samples, new_samples = benchmark_pair(old, new, device, args.repeats)
        old_ms, new_ms = statistics.median(old_samples), statistics.median(new_samples)
        row = {
            "name": name,
            "old_ms": old_ms,
            "new_ms": new_ms,
            "speedup": old_ms / new_ms,
            "mismatched_residues": mismatches,
            "old_samples_ms": old_samples,
            "new_samples_ms": new_samples,
        }
        report["results"].append(row)
        print(
            f"{name:18s} old={old_ms:10.3f}ms new={new_ms:10.3f}ms speedup={old_ms / new_ms:6.2f}x exact_match=True",
            flush=True,
        )
    if device.type == "cuda" and args.independent:
        for name in ("gibbs", "metropolis"):
            before = getattr(sampling_triton_baseline, name + "_step_independent_sites_triton")
            after = getattr(sampling_triton, name + "_step_independent_sites_triton")
            def old(sampler=before):
                return sampler(chains.clone(), params)

            def new(sampler=after):
                return sampler(chains.clone(), params)
            torch.manual_seed(123)
            expected = old()
            torch.manual_seed(123)
            actual = new()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            old_samples, new_samples = benchmark_pair(old, new, device, args.repeats)
            old_ms, new_ms = statistics.median(old_samples), statistics.median(new_samples)
            report["results"].append(
                {
                    "name": "independent-" + name,
                    "old_ms": old_ms,
                    "new_ms": new_ms,
                    "speedup": old_ms / new_ms,
                    "mismatched_residues": 0,
                    "old_samples_ms": old_samples,
                    "new_samples_ms": new_samples,
                }
            )
            print(
                f"independent-{name:10s} old={old_ms:.3f}ms new={new_ms:.3f}ms speedup={old_ms / new_ms:.2f}x exact_match=True",
                flush=True,
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
