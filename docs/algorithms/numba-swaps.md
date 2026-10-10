# CPU PTT swap fusion

With the `cpu` extra installed, unsteered PTT generation on a full ladder of
exactly seven models automatically uses a fused Numba movement kernel. The
kernel lives in `adabmDCA/numba_kernels/swaps.py`. Training, ladders of other
sizes, reservoir ladders and steering retain the existing movement path.
See [Installation](../installation.md#faster-cpu-sampling) for the CPU extra.

For each adjacent pair, the kernel applies the acceptance mask and both
population permutations in one pass over chain rows. It carries int64 lineage
and birth rounds, boolean top-reached flags and optional float32/float64 lag
memory with each configuration. Separate output buffers prevent permutation
races, and rows run in parallel using PyTorch's configured thread count.
NumPy views share the input CPU tensors without copying them.

Exchange probabilities and local sampling kernels are unchanged. Adjacent
pairs still execute sequentially, and uniforms and permutations are drawn
from PyTorch in their original order. Acceptance means retain CPU float64
precision and their original accumulation order. Tests and the benchmark
preserve chains, metadata, acceptance statistics and private RNG states exactly.

`ADABMDCA_NUMBA_SWAPS=0` disables only fused movement, keeping Numba local and
exchange kernels. `ADABMDCA_NUMBA=0` disables all Numba dispatch. Fused
permutation work is included in the PTT `exchange_seconds` phase rather than
`permutation_seconds`.

## Measurements

Measured on an Apple M4 with 16 GB memory, macOS 26.6.2, PyTorch 2.13.0,
Numba 0.68.0 and four CPU threads, on 2026-10-09. Both paths use the same
Numba local samplers and exchange probabilities. Compilation and parameter
layouts are warmed up; three paired repetitions alternate execution order.

Each seven-model ladder interpolates the couplings of a three-update CPU-fitted
model, with fixed fields, an exact profile at the bottom and birth/top-reached
tracking. The fits use the RNA/protein train and validation splits in
`example_data/`. Sparse cases retain regular graphs with roughly 15% pair
density. These short models measure throughput, rather than convergence or
mixing of mature models.

Wall time per five-round `advance`, including profile redraws and ten local
Metropolized Gibbs sweeps per round, with 2,000 chains per model:

| Family | Graph | Previous movement | Fused movement | Speedup |
|---|---|---:|---:|---:|
| RF00379 | Dense | 6.044 s | 5.962 s | 1.014× |
| RF00379 | Sparse | 2.351 s | 2.338 s | 1.006× |
| cm_russ_natural | Dense | 10.999 s | 10.708 s | 1.027× |
| cm_russ_natural | Sparse | 3.215 s | 3.201 s | 1.004× |

Whole-call gains are small and overlap the observed timing variation. The
isolated exchange/movement/profile workload improves by roughly 3–10%, but
local sampling dominates ten-sweep calls. CPU fusion saves allocations and
memory passes; it does not remove GPU dispatches or device-to-host waits.
These results therefore do not establish a substantial whole-sampling gain.
Raw repeated timings and settings:
[`numba-swaps-m4.json`](../assets/benchmarks/numba-swaps-m4.json).

A separate check with 128 chains, one local sweep and five alternating
repetitions finds roughly 1–6% whole-call gains and no consistent slowdown
at the smaller population:
[`numba-swaps-small-m4.json`](../assets/benchmarks/numba-swaps-small-m4.json).
Cases labelled `0_sweeps` replace local sampling with an identity function to
isolate exchanges, movement and profile redraws; the public API still requires
positive local sweeps. Public sequence generation, including renewal and
reference energies, is separately tested against the previous path.

```bash
python -m tools.benchmark_numba_swaps --output /tmp/numba-swaps.json
python -m tools.benchmark_numba_swaps --chains 128 --sweeps 1 --repeats 5 --output /tmp/numba-swaps-small.json
python -m pytest -q tests/test_swaps_numba.py
```

The benchmark accepts `--dtype float64`, `--threads`, and alternative alignment
paths through `--rna-train`, `--rna-validation`, `--protein-train` and
`--protein-validation`. The default protein files are local and not tracked.

## Correctness checks

Tests cover all optional metadata combinations, populations of 1, 129 and
2,000 chains, float32/float64 memory, identifiers beyond 2⁴⁰, strict acceptance
thresholds, noncontiguous inputs and movement with inference tensors. Dense
and sparse seven-model trajectories match the previous path with one or four
threads and both parameter dtypes. Partial round counts, global RNG isolation,
public generation and dispatch exclusions are also checked. The tests skip
when Numba is absent or disabled.
