"""Compare FP32 and BF16 sampling within the actual FP32-master training workflow.

python -m benchmarks.benchmark_bfloat16_training --output /tmp/bf16_training.json
python -m benchmarks.benchmark_bfloat16_training --data example_data/RF00379.fasta --alphabet rna
"""

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch

from adabmDCA import train_model
from adabmDCA.alignment import Alignment, read_alignment
from adabmDCA.fasta import get_tokens
from adabmDCA.sampling import prepare_training_sampler
from adabmDCA.sampling_triton import triton
from adabmDCA.statmech import compute_energy


def synthetic_alignment(n, length, tokens):
    generator = torch.Generator().manual_seed(79)
    states = torch.randint(len(tokens), (n, length), generator=generator)
    # Correlated pairs give training a signal beyond sampling noise.
    states[:, 1::2] = states[:, : length - 1 : 2]
    return Alignment(
        names=tuple(f"s{i}" for i in range(n)),
        sequences=tuple("".join(tokens[x] for x in row) for row in states.tolist()),
    )


def compare(functions, repeats):
    for fn in functions:
        fn()
    torch.cuda.synchronize()
    samples, peaks, results = [[], []], [[], []], [None, None]
    for repeat in range(repeats):
        for index in (0, 1) if repeat % 2 == 0 else (1, 0):
            # Do not retain the previous result while measuring memory.
            results[index] = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline_memory = torch.cuda.memory_allocated()
            start = time.perf_counter()
            result = functions[index]()
            torch.cuda.synchronize()
            samples[index].append(1000 * (time.perf_counter() - start))
            peaks[index].append((torch.cuda.max_memory_allocated() - baseline_memory) / 2**20)
            results[index] = result
    medians = [statistics.median(values) for values in samples]
    return {
        "fp32_ms": medians[0],
        "bf16_ms": medians[1],
        "speedup": medians[0] / medians[1],
        "samples_ms": dict(zip(("float32", "bfloat16"), samples, strict=True)),
        "incremental_peak_mib": dict(zip(("float32", "bfloat16"), peaks, strict=True)),
    }, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--alphabet", default="protein")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--sequences", type=int, default=512)
    parser.add_argument("--chains", type=int, default=2048)
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, default=Path("/tmp/bfloat16_training.json"))
    args = parser.parse_args()
    if min(args.length, args.sequences, args.chains, args.sweeps, args.epochs, args.repeats) < 1:
        parser.error("shape, sweep, epoch and repeat counts must be positive")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    # Validate before allocating benchmark tensors.
    prepare_training_sampler("gibbs", device, "bfloat16")
    tokens = get_tokens(args.alphabet)
    data = read_alignment(args.data) if args.data else synthetic_alignment(args.sequences, args.length, tokens)
    length, q = len(data.sequences[0]), len(tokens)
    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "threads": torch.get_num_threads(),
        },
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "shape": {"sequences": len(data.sequences), "chains": args.chains, "length": length, "q": q},
        "results": [],
    }
    print(json.dumps(report, indent=2), flush=True)
    for sampler in ("gibbs", "metropolis"):
        torch.manual_seed(args.seed)
        chains = torch.nn.functional.one_hot(torch.randint(q, (args.chains, length), device=device), q).float()
        coupling = torch.randn(length, q, length, q, device=device) * 0.01
        coupling = (coupling + coupling.permute(2, 3, 0, 1)) / 2
        idx = torch.arange(length, device=device)
        coupling[idx, :, idx, :] = 0
        params = {"bias": torch.randn(length, q, device=device) * 0.1, "coupling_matrix": coupling}
        functions = [
            partial(prepare_training_sampler(sampler, device, dtype), chains, params, args.sweeps)
            for dtype in ("float32", "bfloat16")
        ]
        sampling, sampled = compare(functions, args.repeats)
        del sampled, functions
        # Quantify rounding error in the single-site transition probabilities.
        field = params["bias"][0] + chains.flatten(1) @ coupling[0].reshape(q, -1).T
        rounded_field = params["bias"][0] + chains.flatten(1) @ coupling[0].bfloat16().float().reshape(q, -1).T
        if sampler == "gibbs":
            error = 0.5 * (field.softmax(-1) - rounded_field.softmax(-1)).abs().sum(-1)
            rounding = {
                "mean_conditional_total_variation": error.mean().item(),
                "max_conditional_total_variation": error.max().item(),
            }
        else:
            batch = torch.arange(args.chains, device=device)
            old = chains[:, 0].argmax(-1)
            proposed = torch.randint(q, (args.chains,), device=device)
            accept = (field[batch, proposed] - field[batch, old]).exp().clamp_max(1)
            rounded_accept = (rounded_field[batch, proposed] - rounded_field[batch, old]).exp().clamp_max(1)
            rounding = {
                "mean_acceptance_probability_error": (accept - rounded_accept).abs().mean().item(),
                "max_acceptance_probability_error": (accept - rounded_accept).abs().max().item(),
            }
        del params, coupling, chains, field, rounded_field
        functions = [
            partial(
                train_model,
                data,
                model_type="bmDCA",
                alphabet=args.alphabet,
                dtype=dtype,
                device="cuda",
                sampler=sampler,
                n_chains=args.chains,
                n_sweeps=args.sweeps,
                max_epochs=args.epochs,
                target_pearson=0.999999,
                no_reweighting=True,
                seed=args.seed,
            )
            for dtype in ("float32", "bfloat16")
        ]
        training, results = compare(functions, args.repeats)
        old, new = results
        assert old.gradient_steps == new.gradient_steps == args.epochs
        assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in new.model.params.values())
        delta = new.model.params["coupling_matrix"] - old.model.params["coupling_matrix"]
        quality = {
            "coupling_rms_difference": delta.square().mean().sqrt().item(),
            "coupling_max_difference": delta.abs().max().item(),
            "fp32_final_pearson": old.final_metrics["Pearson"],
            "bf16_final_pearson": new.final_metrics["Pearson"],
            "fp32_final_likelihood": old.final_metrics["LL_train"],
            "bf16_final_likelihood": new.final_metrics["LL_train"],
            "energy_rms_difference_per_site": (
                (compute_energy(old.chains, old.model.params) - compute_energy(old.chains, new.model.params)) / length
            )
            .square()
            .mean()
            .sqrt()
            .item(),
            "completed_gradient_steps": new.gradient_steps,
        }
        row = {
            "sampler": sampler,
            "sampling": sampling,
            "training": training,
            "rounding": rounding,
            "training_comparison": quality,
        }
        report["results"].append(row)
        print(json.dumps(row, indent=2), flush=True)
        del old, new, results, delta, functions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
