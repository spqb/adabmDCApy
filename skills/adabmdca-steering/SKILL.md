---
name: adabmdca-steering
description: Generate sequences from an adabmDCA Potts model with Python SteeringConfig when the request involves substitution windows, alphabet restrictions, Hamming biases, scoring callbacks or low temperature. Covers MCMC versus PTT and steered entropy; use the main adabmdca skill for unconstrained sampling.
---

# Steered generation with adabmDCA

Steering samples a trained model under constraints or extra costs. It is Python only: `sample_sequences(..., steering=SteeringConfig(...))`. For training, plain sampling and the CLI, use the `adabmdca` skill.

The target on the allowed sequences is

    u(x) = beta * H(x) + sum_k b_k * d_H(x, r_k) + sum_j lambda_j * V_j(x)

lower `u` more probable. Hard constraints define the allowed set; the three terms are the cooled Potts energy, linear distance biases and weighted callbacks.

## Choose the strategy first

Both strategies sample the same target with the same constrained kernel; they differ in how they cross barriers.

| Problem | Use | Why |
| --- | --- | --- |
| Easy constraints: a shell or small substitution budget around a reference, at `beta = 1`, with no or weak callbacks | **Plain MCMC** (`ptt=False`, budget `n_sweeps`) | Often cheaper for local targets; PTT pays for a whole ladder |
| Hard targets: low temperature (`beta > 1`), strong callbacks, wide windows far from the data, multimodal targets | **PTT** (`ptt=True`, budget `ptt_max_rounds`), from the `ptt.h5` archive when there is one | Replicas carry configurations across barriers that single chains do not cross |
| The entropy or log Z of the steered distribution is needed | **PTT** with `entropy=True` (an error with MCMC) | Only the ladder has an exact bottom to normalize from |

Measured on the Azoarcus group I intron model (L = 193, RNA, gap-free ACGU):

| Target | MCMC | PTT |
| --- | --- | --- |
| 5–10 substitutions from a start, 25–30 from wild type | converged in **5 s** (608 rounds) | converged in 80 s |
| same shell + ViennaRNA structure potential, weight 1 | budget exhausted, R-hat 1.71 | budget exhausted, R-hat 1.23 (converged at weight 0.25 with ~1,300 rounds) |
| `beta = 1/0.3`, at most 15 substitutions | not converged after 585 s, R-hat 2.02 | nearly converged in 585 s, R-hat 1.02, bulk ESS 183 |
| `beta = 1/0.3`, no window | R-hat 3.70 after 585 s | R-hat 1.49 after 595 s (neither converged) |

These results support starting with MCMC for easy local design and considering PTT for colder or more strongly biased targets. They are examples from one model, not runtime guarantees or universal switching thresholds; use diagnostics for the actual target. Cold global sampling may need a long budget even with PTT.

## Build the configuration

```python
from dataclasses import replace
from adabmDCA import HammingBias, HammingWindow, SteeringConfig, WeightedPotential, sample_sequences

config = SteeringConfig(
    allowed_tokens="ACGU",                               # no gaps: a mask, never a gap penalty
    windows=(HammingWindow(start, 5, 10),                # inclusive, intersected when several
             HammingWindow(wild_type, 25, 30)),
    distance_biases=(HammingBias(start, weight=0.2),),   # positive weight favours shorter distances
    potentials=(WeightedPotential(score, weight=0.25),), # score(list_of_aligned_strings) -> one float each, lower favoured
    chains=128,                                          # per rung
    block_steps=L,                                       # one callback call per chain per sweep
    warmup_seconds=3600,                                 # optional wall-clock limit
)
easy = sample_sequences(model="model/params.dat.gz", steering=config, n_sequences=1024, n_sweeps=5000,
                        alphabet="rna", seed=0)
hard = sample_sequences(model="model/ptt.h5", ptt=True, steering=replace(config, entropy=True),  # entropy: PTT only
                        beta=1 / 0.3, n_sequences=1024, ptt_max_rounds=20000, seed=0)
```

- **Hard constraints go in the kernel, not in a potential.** Windows and `allowed_tokens` mask each site update with running mismatch counts, so every update is accepted and no chain leaves the allowed set. A wall written as a callback cannot do this: a strength ladder needed ~3,400 rungs to reach a distance shell that the windows impose directly. With PTT, the bottom rung is the archive's anchor restricted to the windows, drawn exactly, so no rungs sit below it.
- **References** are aligned, model-length strings in the model alphabet. Check that the start sequences satisfy every window before running.
- **Callbacks** receive aligned strings (or integer indices with `input="categorical"`), must be deterministic and return finite values, unweighted (the weight is in `WeightedPotential`). Values are cached (`cache_size`). Expensive callbacks dominate the cost: ViennaRNA took 92–97% of the wall time in the runs above, so consider `block_steps` equal to the length for full-sweep proposals and parallelizing inside the callback. Choose weights for the intended target: changing them changes the distribution, rather than merely tuning performance.
- **Temperature**: `beta` is a `sample_sequences` argument; it cools the Potts energy only, the biases keep their weights. `model="ptt.h5", ptt=True, beta=...` without `steering` uses a temperature ladder.
- **Seeding cold PTT runs**: `initial_sequences=` (at least `chains` aligned sequences that satisfy the constraints, e.g. gap-free samples at `beta = 1`) seeds every rung above the exact bottom. On the cold global run it cooled the target rung faster and lowered R-hat (1.31 vs 1.49 at equal time) but slowed renewal, because seeds count as old configurations. Use it to speed the approach, not as a shortcut to convergence.

## Check before trusting the draws

`result.steering` holds the diagnostics:

```python
s = result.steering
s["status"]                        # "converged", "warmup_budget_exhausted" or "warmup_time_exhausted"
s["observables"]                   # per observable: split R-hat, bulk and tail ESS on group-mean traces
s["old_configurations_per_chain"]  # PTT: lineages older than warmup left on the ladder (must be <= 0.01)
s["swap_acceptance"], s["block_acceptance"], s["potential_seconds"]
s["constraint_distances"]          # per draw, one column per window
s["potential_values"]              # per draw, one column per callback (unweighted)
s["entropy"]                       # PTT with entropy=True: "log_z", "entropy_nats", errors, "bar_overlap", "provisional"
s["history"]["warmup"]             # rows with "round" and "observables" for plotting traces
```

- `status == "converged"` means the configured diagnostic thresholds passed: R-hat below `rhat` (1.01) and bulk and tail ESS above `min_ess` (40) for every monitored observable, plus renewal for PTT. These checks support convergence for the monitored observables; they do not prove global equilibrium. Other statuses are **provisional**: report R-hat, ESS and old configurations. Within the authorized budget, consider more warmup or PTT; changing a callback weight changes the target and should follow the user’s design objective.
- An entropy with `provisional: true` excludes warmup or overlap bias from its error bar.
- For distributional comparisons with natural data, consider `compare_pca(train, steered_fasta)` from the `adabmdca` skill. Report energy, distances or callback distributions as relevant to the requested design assessment; generation alone does not require this whole analysis.
- `production_spacing=0` returns a single snapshot at the end of warmup, with at most `chains` draws.

Full guide: https://spqb.github.io/adabmDCApy/usage/sample/#steered-sampling-python, how the anchor is restricted in https://spqb.github.io/adabmDCApy/algorithms/anchor/, and the API reference of `adabmDCA.steering.sampling`.
