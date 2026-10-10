"""Reproduce MPS/Numba/PyTorch sampler and PCD training timings on local splits.

Run from the repository root with an editable install and the cpu extra:
    python tools/benchmark_mps.py --output /tmp/mps-benchmark.json

GPU timings synchronize before and after each call. Compilation is warmed up;
input/output conversions and changed coupling layouts are included. The models
are short training runs, not converged models or a mixing benchmark. Protein
splits are local example data and can be supplied with --protein-train/validation.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

from adabmDCA import load_alignment, train_model
from adabmDCA.mps_kernels import is_mps_available
from adabmDCA.numba_kernels import is_numba_available
from adabmDCA.sampling import prepare_sampler


@contextlib.contextmanager
def backend(name):
    previous = os.environ.get("ADABMDCA_MPS")
    os.environ["ADABMDCA_MPS"] = "0" if name == "torch_mps" else "1"
    try:
        yield torch.device("cpu" if name == "numba_cpu" else "mps")
    finally:
        if previous is None:
            os.environ.pop("ADABMDCA_MPS", None)
        else:
            os.environ["ADABMDCA_MPS"] = previous


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def measure(function, device, repeats):
    function()  # compilation, layout caches, and allocator warmup
    timings = []
    for _ in range(repeats):
        sync(device)
        started = time.perf_counter()
        function()
        sync(device)
        timings.append(time.perf_counter() - started)
    return {"median_seconds": statistics.median(timings), "seconds": timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chains", type=int, default=2000)
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--training-steps", type=int, default=20)
    parser.add_argument("--training-chains", type=int, default=2000)
    parser.add_argument("--training-sweeps", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rna-train", type=Path, default=Path("example_data/RF00379/splits/RF00379_train.fasta"))
    parser.add_argument("--rna-validation", type=Path,
                        default=Path("example_data/RF00379/splits/RF00379_validation.fasta"))
    parser.add_argument("--protein-train", type=Path,
                        default=Path("example_data/cm_russ_natural/splits/cm_russ_natural.train.fasta"))
    parser.add_argument("--protein-validation", type=Path,
                        default=Path("example_data/cm_russ_natural/splits/cm_russ_natural.test.fasta"))
    args = parser.parse_args()
    if not is_mps_available() or not is_numba_available():
        parser.error("Requires Metal shaders and Numba; enable both backends.")
    torch.set_num_threads(args.threads)
    report = {"platform": platform.platform(), "torch": torch.__version__,
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "families": {}}
    for family, train_path, val_path in (("RF00379", args.rna_train, args.rna_validation),
                                         ("cm_russ_natural", args.protein_train, args.protein_validation)):
        train, validation = load_alignment(train_path), load_alignment(val_path)
        training_options = {"validation_path": validation, "alphabet": train.tokens, "n_chains": args.training_chains,
                            "n_sweeps": args.training_sweeps, "target_pearson": 0.999999, "seed": 42,
                            "no_reweighting": True, "checkpoint_interval": 1}
        summary = {"length": train.sequence_length, "q": len(train.tokens),
                   "training_sequences": len(train), "validation_sequences": len(validation),
                   "training": {}, "sampling": {}}
        report["families"][family] = summary
        cpu_model = None
        for name in ("numba_cpu", "metal_mps", "torch_mps"):
            print(f"{family}: PCD training with {name}", flush=True)
            with backend(name) as device, contextlib.redirect_stdout(io.StringIO()):
                # Warm compilation and device-specific paths, then time a complete run,
                # including input statistics and validation (no disk output).
                train_model(train, device=str(device), max_gradient_steps=2, **training_options)
                sync(device)
                started = time.perf_counter()
                result = train_model(train, device=str(device), max_gradient_steps=args.training_steps,
                                     **training_options)
                sync(device)
                elapsed = time.perf_counter() - started
                assert result.gradient_steps == args.training_steps
                assert all(torch.isfinite(v).all() for v in result.model.params.values())
                summary["training"][name] = {"seconds": elapsed, "gradient_steps": result.gradient_steps,
                                             "metrics": result.final_metrics}
                if name == "numba_cpu":
                    cpu_model = result.model.params
            print(f"  {elapsed:.3f}s, Pearson {result.final_metrics.get('Pearson')}, "
                  f"validation {result.final_metrics.get('Pearson_val')}", flush=True)
        length, q = summary["length"], summary["q"]
        torch.manual_seed(9)
        initial = torch.nn.functional.one_hot(torch.randint(q, (args.chains, length)), q).float()
        for method in ("gibbs", "metropolis", "metropolized_gibbs"):
            summary["sampling"][method] = {}
            for name in ("numba_cpu", "metal_mps", "torch_mps"):
                with backend(name) as device:
                    sampler = prepare_sampler(method, device)
                    p = {k: v.to(device) for k, v in cpu_model.items()}
                    x = initial.to(device)
                    timing = measure(lambda sampler=sampler, x=x, p=p: sampler(x, p, args.sweeps), device, args.repeats)
                    # Training changes couplings on every call: include layout rebuilding.
                    def changed_model(p=p, sampler=sampler, x=x):
                        p["coupling_matrix"].add_(0)
                        return sampler(x, p, args.sweeps)
                    changed = measure(changed_model, device, args.repeats) if name != "torch_mps" else None
                summary["sampling"][method][name] = {**timing, "changed_model": changed}
                print(f"{family}: {method} {name}: {timing['median_seconds']:.4f}s", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
