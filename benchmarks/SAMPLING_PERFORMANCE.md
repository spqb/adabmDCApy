# Sampling optimization measurements — 2026-09-08

The optimized Triton samplers preserve the categorical update algorithms and random-number generation order. Gibbs now reads contiguous candidate-state vectors from a transposed coupling matrix. Both samplers retain chain states in registers for up to 16 sequential updates per launch. Output one-hot tensors are written directly in the requested dtype; independent-site updates write only the selected site for contiguous inputs. The CPU samplers use `range` instead of allocating a `torch.arange` loop iterator.

## Environment and measurement method

- NVIDIA RTX A5000, 24 GiB; AMD Ryzen Threadripper PRO 5955WX, 16 cores / 32 threads.
- PyTorch 2.10.0+cu128, Triton 3.6.0, Python 3.10.13. CPU intra-op threads fixed at 4.
- Two sweeps per call, two warm-up calls per implementation, seven timed repetitions; table values are medians in milliseconds.
- Timings include coupling transposition, categorical conversion, random-number generation, all sampling updates, allocation and output conversion. Compilation is excluded by warm-up. CUDA is synchronized after each timed call. Old/new order alternates each repetition.
- The original Triton implementation is frozen in `sampling_triton_baseline.py` from commit `cb8dc8b0dbea84ac034d9aef97622f53bb8fa3e1`. CPU comparisons use scripted old/new PyTorch loops.
- This is a desktop GPU with intermittent competing compute activity. Another Python workload was at 100% utilization during early experiments and absent at a later check. Absolute timings varied between runs. Interpret each interleaved old/new pair together; do not combine absolute times from different runs. These are local measurements, not universal GPU tuning results.
- Every benchmark asserts exact old/new sample equality with seed 123 before timing. All assertions passed. Raw repetitions and configurations are in `results/sampling_2026-09-08/*.json`.

## GPU: original versus optimized Triton

| N × L × q | dtype | sampler | original ms | optimized ms | speedup |
|---|---|---|---:|---:|---:|
| 256 × 32 × 5 | float32 | gibbs | 0.894 | 0.183 | 4.87× |
| 256 × 32 × 5 | float32 | metropolis | 0.906 | 0.175 | 5.18× |
| 2048 × 100 × 21 | float32 | gibbs | 16.318 | 3.412 | 4.78× |
| 2048 × 100 × 21 | float32 | metropolis | 3.805 | 1.218 | 3.12× |
| 4096 × 200 × 21 | float32 | gibbs | 142.202 | 27.116 | 5.24× |
| 4096 × 200 × 21 | float32 | metropolis | 13.765 | 10.399 | 1.32× |
| 2048 × 100 × 21 | float64 | gibbs | 24.671 | 22.471 | 1.10× |
| 2048 × 100 × 21 | float64 | metropolis | 3.258 | 2.581 | 1.26× |

For the medium float32 case, scripted PyTorch Gibbs took 34.512 ms versus 3.412 ms for optimized Triton in the same benchmark run. For the large case these were 103.987 ms versus 27.116 ms.

## CPU: scripted original versus optimized PyTorch

Triton kernels target CUDA here; they are not run or emulated for CPU performance measurements. Removing the tensor loop iterator does not materially change CPU performance.

| N × L × q | dtype | sampler | original ms | optimized ms | speedup |
|---|---|---|---:|---:|---:|
| 256 × 32 × 5 | float32 | gibbs | 7.843 | 7.732 | 1.01× |
| 256 × 32 × 5 | float32 | metropolis | 5.843 | 5.855 | 1.00× |
| 2048 × 100 × 21 | float32 | gibbs | 531.358 | 532.376 | 1.00× |
| 2048 × 100 × 21 | float32 | metropolis | 385.511 | 380.961 | 1.01× |
| 2048 × 100 × 21 | float64 | gibbs | 630.684 | 632.032 | 1.00× |
| 2048 × 100 × 21 | float64 | metropolis | 487.847 | 488.493 | 1.00× |

## Independent-site GPU updates

These timings include cloning the input for each call, as these APIs mutate their input.

| N × L × q | dtype | sampler | original ms | optimized ms | speedup |
|---|---|---|---:|---:|---:|
| 256 × 32 × 5 | float32 | gibbs | 0.150 | 0.122 | 1.23× |
| 256 × 32 × 5 | float32 | metropolis | 0.159 | 0.130 | 1.22× |
| 2048 × 100 × 21 | float32 | gibbs | 0.655 | 0.445 | 1.47× |
| 2048 × 100 × 21 | float32 | metropolis | 0.423 | 0.208 | 2.03× |
| 4096 × 200 × 21 | float32 | gibbs | 3.513 | 2.829 | 1.24× |
| 4096 × 200 × 21 | float32 | metropolis | 1.938 | 0.774 | 2.50× |
| 2048 × 100 × 21 | float64 | gibbs | 1.346 | 1.010 | 1.33× |
| 2048 × 100 × 21 | float64 | metropolis | 0.943 | 0.635 | 1.49× |

## What produced the gain

At N=2048, L=100, q=21, float32, transposition with one update per launch gives 4.49× Gibbs speedup over the original. Fusion without transposition gives 0.91× (a regression). Thus the contiguous candidate layout is the important Gibbs change. Metropolis benefits mainly from reducing launches. Sixteen updates per launch is a bounded default, exposed as `steps_per_launch`; it is not claimed optimal for every shape or GPU.

Transposition allocates one additional coupling-sized tensor per sampler call (L²q² elements), reuses it across random-number chunks, and never caches it across parameter updates. Set `transpose_couplings=False` when that memory cost is unacceptable. Independent-site calls keep the original coupling layout to avoid a full matrix transpose for one update. Strided one-hot inputs use the existing reconstruction/copy fallback.

The old `N*L <= 250_000` Gibbs dispatch threshold has been removed because it described the old gather kernel: the optimized kernel wins in the measured case above that threshold too. `prepare_sampler` uses Triton on CUDA when available, with scripted PyTorch retained for CPU/no-Triton environments. Call the scripted PyTorch sampler explicitly when reproducing an old run that used that fallback. A separate dense-GEMM-plus-fused-draw backend was not added: the new gather path already outperformed the existing dense baseline in every measured configuration. Warp/block autotuning across GPU models remains future work.

## Correctness and compatibility

- Full repository suite: **110 passed**, including actual GPU tests. Existing TorchScript deprecation warnings remain.
- Exact categorical-state comparisons against the original kernels for float32/float64, L/q = 1/1, 17/5 and 33/21; launch chunks of 1, 4, 16 and 64; repeated sites; 35 updates including a partial final chunk; non-multiple chain counts; asymmetric couplings and nonzero self-couplings.
- Same-seed public old/new samplers preserve both samples and post-call RNG state in the regression cases. CPU eager and scripted implementations are checked too.
- Existing controlled-random PyTorch-reference tests pass. Added tests cover zero sweeps, empty batches, strided inputs, independent-site in-place identity, input preservation, parameter changes between calls and backend dispatch.
- Intentional correctness fix: mask Gibbs padding after temperature scaling. The original produces NaNs for beta=0 and invalid padded logits for negative beta when q is not a power of two. The new path matches a controlled PyTorch reference. The sampled index is also clamped to q-1 against cumulative-sum rounding at the upper endpoint.
- Exact test matches are not a universal bitwise guarantee for floating-point reductions near sampling thresholds. PyTorch uses a different categorical RNG algorithm from Triton, so switching backends (including formerly large-input fallback calls) changes seeded trajectories while preserving the intended transition probabilities.

## Reproduce

Run from the repository root in an environment with PyTorch, Triton and pytest installed:

```bash
python -m pytest -q
python -m benchmarks.benchmark_sampling --device cuda --independent --output /tmp/gpu_medium.json
python -m benchmarks.benchmark_sampling --device cuda --chains 256 --length 32 --states 5 --independent
python -m benchmarks.benchmark_sampling --device cuda --chains 4096 --length 200 --independent
python -m benchmarks.benchmark_sampling --device cuda --dtype float64 --independent
python -m benchmarks.benchmark_sampling --device cpu --threads 4 --output /tmp/cpu_medium.json
python -m benchmarks.benchmark_sampling --device cpu --chains 256 --length 32 --states 5 --threads 4
python -m benchmarks.benchmark_sampling --device cpu --dtype float64 --threads 4
python -m benchmarks.benchmark_sampling --device cuda --steps-per-launch 1
python -m benchmarks.benchmark_sampling --device cuda --no-transpose
```

Here, GPU access required running outside the execution sandbox. Test-only pytest/ruff dependencies were installed under `/tmp/adabmdca-testdeps`, without changing the project environment. The full suite was run with `PYTHONPATH=/tmp/adabmdca-testdeps:. OMP_NUM_THREADS=4 /home/lorenzo/miniconda3/envs/dca/bin/python -m pytest -q`.
