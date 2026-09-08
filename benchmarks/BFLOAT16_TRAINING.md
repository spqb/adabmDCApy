# Optional BF16 training: A5000 results — 2026-09-08

`train_model(..., device="cuda", dtype="bfloat16")` and `adabmDCA train ... --device cuda --dtype bfloat16` now enable BF16 coupling storage inside the sampler. Master parameters, biases, parameter updates, chains, statistics, energy/AIS calculations and saved models remain FP32. The option works with all four training models and both samplers; it requires CUDA, Triton and an Ampere-or-newer NVIDIA GPU.

## Implementation

- Gibbs copies the master coupling matrix directly to the transposed BF16 sampling layout in a single conversion/copy. Metropolis makes a BF16 copy in the original layout. Copies are refreshed every sampler call, so parameter updates are never hidden by a stale cache.
- Kernel loads are converted to FP32 before reductions or subtraction. Logits, softmax and acceptance arithmetic remain FP32. Uniform random numbers also remain FP32, including for direct low-level BF16-chain calls.
- Existing FP32/FP64 defaults remain intact. The training API resolves BF16 mode to FP32 master tensors and records the requested mode in `TrainingResult.config.dtype`; model metadata accurately reports FP32.
- Keeping persistent chains in FP32 avoids repeated conversions in the statistics, energy and checkpoint paths. Consequently this is mixed-precision sampling within training, not whole-model BF16 training or a promise of half the training memory.

## Measurement method

NVIDIA RTX A5000 (24 GiB), PyTorch 2.10.0+cu128, Triton 3.6.0; four CPU intra-op threads. Each pair has one untimed warm-up per mode and five timed repetitions in alternating FP32/BF16 order. CUDA is synchronized around measurements. Compilation is excluded. Sampler timings include the fresh coupling conversion, transpose, RNG and output construction. End-to-end timings call the public `train_model` API from initialization through all requested gradient steps, including target/model statistics, parameter updates and energy/AIS/diagnostic calculations; checkpoints and W&B are disabled. Target Pearson is 0.999999 and completed step counts are asserted equal. Sequence reweighting is disabled.

The A5000 was shared: another Python process was at 97% GPU utilization and the GPU temperature was 87°C in the pre-run snapshot. Absolute times and small percentage differences are sensitive to contention/thermal variation. Raw measurements are saved alongside the report; the FP32/BF16 modes were interleaved rather than benchmarked in separate long batches.

The two protein cases use deterministic synthetic alignments with correlated neighboring residues (512 sequences, q=21). The RNA case uses all 3,827 input sequences in `example_data/RF00379.fasta` (L=136, q=5), subject to the standard input deduplication. These are short fixed-step comparisons, not training-to-convergence studies.

## End-to-end training

| Data | Chains × L × q | Gradient steps | Sampler | FP32 ms | BF16 ms | Speedup |
|---|---|---:|---|---:|---:|---:|
| protein | 2048 × 100 × 21 | 20 | gibbs | 1717.21 | 1701.79 | 1.009× |
| protein | 2048 × 100 × 21 | 20 | metropolis | 1254.88 | 1197.73 | 1.048× |
| protein_large | 4096 × 200 × 21 | 5 | gibbs | 2154.56 | 2062.61 | 1.045× |
| protein_large | 4096 × 200 × 21 | 5 | metropolis | 1248.99 | 1116.60 | 1.119× |
| rna | 2048 × 136 × 5 | 20 | gibbs | 1057.25 | 1075.23 | 0.983× |
| rna | 2048 × 136 × 5 | 20 | metropolis | 885.76 | 896.91 | 0.988× |

Each gradient step runs 10 sampling sweeps. The largest observed end-to-end gain is approximately 12% for Metropolis on the larger protein case. Gibbs gains are small. The RNA runs show slight regressions (about 1–2%), so BF16 remains opt-in.

## Sampling alone

| Data shape | Sampler | FP32 ms | BF16 ms | Speedup |
|---|---|---:|---:|---:|
| protein | gibbs | 40.41 | 40.47 | 0.999× |
| protein | metropolis | 13.97 | 12.43 | 1.124× |
| protein_large | gibbs | 288.72 | 284.07 | 1.016× |
| protein_large | metropolis | 109.46 | 81.08 | 1.350× |
| rna | gibbs | 21.83 | 21.72 | 1.005× |
| rna | metropolis | 9.28 | 8.81 | 1.054× |

These sampler microbenchmarks use random symmetric FP32 couplings with zero self-couplings and standard deviation approximately 0.0071 (before BF16 rounding); they do not time samples from the trained models. BF16 mainly helps Metropolis here. After the preceding layout optimization, Gibbs does not automatically benefit from halving coupling element size: conversion and FP32 reduction work remain, and cache reuse reduces the importance of DRAM bandwidth.

## Numerical checks

Full repository suite: **127 passed** on CPU/GPU. New tests cover both samplers, shared/independent sites, BF16 one-hot outputs with FP32 RNG, exact comparison to FP32 kernels using identically rounded couplings/biases, unmodified FP32 master parameters, conversion refresh after updates, all four training models, checkpoint output and FP32 model reload. Existing FP32/FP64 exact-output regressions also pass.

| Data | Sampler | Coupling RMS difference | Max coupling difference | Absolute final Pearson difference | Energy RMS difference/site |
|---|---|---:|---:|---:|---:|
| protein | gibbs | 2.22e-07 | 9.77e-06 | 3.94e-06 | 4.78e-07 |
| protein | metropolis | 1.13e-07 | 4.88e-06 | 5.9e-06 | 2.79e-07 |
| protein_large | gibbs | 4.34e-08 | 2.44e-06 | 3.11e-06 | 1.59e-07 |
| protein_large | metropolis | 2.32e-08 | 2.44e-06 | 1.42e-06 | 8.37e-08 |
| rna | gibbs | 1.93e-06 | 1.95e-05 | 0.000337 | 1.4e-05 |
| rna | metropolis | 1.54e-06 | 1.95e-05 | 6.75e-06 | 1.32e-05 |

These compare the final FP32 master models after equal work and the same initial seed. Energies for both models are evaluated on the FP32 run’s final chains. They do not establish equivalence after convergence. BF16 coupling rounding changes transition probabilities and can change trajectories. The exactness tests deliberately compare against FP32 evaluation of the *same rounded parameters*, not the original unrounded FP32 model.

For the separate random-coupling sampling fixtures, Gibbs conditional-distribution total-variation errors and Metropolis acceptance-probability errors are recorded in each JSON `rounding` section. They measure quantization at one site (all chains), not full stationary-distribution error.

## Memory

| Data | Sampler | FP32 incremental training peak MiB | BF16 incremental training peak MiB |
|---|---|---:|---:|
| protein | gibbs | 176.91 | 176.91 |
| protein | metropolis | 176.56 | 176.91 |
| protein_large | gibbs | 640.40 | 640.03 |
| protein_large | metropolis | 640.07 | 640.03 |
| rna | gibbs | 43.41 | 43.94 |
| rna | metropolis | 51.08 | 52.99 |

Memory numbers are median increases in `torch.cuda.max_memory_allocated()` above the allocation baseline before each call, not reserved memory or total device usage. Persistent FP32 master tensors, energy workspaces and statistics dominate the peak. Metropolis adds a transient BF16 copy; Gibbs replaces an FP32 transpose with a BF16 transpose. Total training memory is not halved.

## Usage and reproduction

```python
from adabmDCA import train_model

result = train_model(
    "alignment.fasta",
    device="cuda", dtype="bfloat16",
    sampler="metropolis",
)
assert result.model.metadata.dtype == "float32"
```

```bash
adabmDCA train -d alignment.fasta -o model --device cuda --dtype bfloat16
python -m pytest -q
python -m benchmarks.benchmark_bfloat16_training --epochs 20 --output /tmp/protein.json
python -m benchmarks.benchmark_bfloat16_training --length 200 --chains 4096 --output /tmp/protein_large.json
python -m benchmarks.benchmark_bfloat16_training --data example_data/RF00379.fasta --alphabet rna --epochs 20 --output /tmp/rna.json
```

Raw results: `results/bfloat16_2026-09-08/{protein,protein_large,rna}.json`. GPU tests/benchmarks ran outside the local execution sandbox to expose the A5000. Tests used temporary pytest dependencies under `/tmp/adabmdca-testdeps`.
