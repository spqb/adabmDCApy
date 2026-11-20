# Steer sequence sampling in Python

Use structured steering to sample from a DCA model with a hard sequence
constraint, a soft preference, or both. Pass a `SteeringConfig` to
`sample_sequences`. Plain MCMC runs
chains on the fixed model. PTT runs a replica ladder and can estimate the target
entropy.

The target energy on the allowed sequences is
\(u(x) = \beta H(x) + \sum_k b_k d_H(x,r_k) + \sum_j \lambda_j V_j(x)\).
The sampler favors lower reduced energy. The gap mask and any inclusive
Hamming windows define the allowed set on every rung.

## Steering objects

- `SteeringConfig` collects hard constraints, soft biases and sampling
  settings. Set `allowed_tokens="ACGU"` to exclude gaps from an RNA model.
- `HammingWindow(reference, minimum, maximum)` enforces an inclusive
  substitution budget against an aligned model-length reference. Put more
  than one window in `SteeringConfig.windows` to require their intersection.
- `HammingBias(reference, weight)` adds a linear mismatch cost to the Potts
  fields. A positive weight favors shorter distances. Put these objects in
  `SteeringConfig.distance_biases`.
- `WeightedPotential(callback, weight)` adds a deterministic batch cost.
  Return one finite unweighted value per sequence. The default input is
  aligned strings. Set `input="categorical"` for integer token indices.
  Put more than one `WeightedPotential` in `SteeringConfig.potentials` to add
  their weighted costs.
- `beta` is a separate `sample_sequences` argument. Values above 1 cool
  the Potts contribution while the explicit biases keep their target weights.

## Sample within a hard window

This small independent-site example needs only PyTorch and adabmDCA. Replace
the model with a trained `DCAModel` or parameter path in an experiment.

```python
import torch
from adabmDCA import DCAModel, HammingWindow, SteeringConfig, sample_sequences

model = DCAModel(
    {"bias": torch.zeros(6, 5), "coupling_matrix": torch.zeros(6, 5, 6, 5)},
    alphabet="rna",
)
shell = SteeringConfig(
    allowed_tokens="ACGU",
    windows=(HammingWindow("ACGUAC", 1, 2),),
    chains=32,
)
draws = sample_sequences(model=model, steering=shell, n_sequences=128, n_sweeps=256)
print(draws.sequences[:3], draws.steering["status"])
```

The kernel masks each site update using running mismatch counts. An output
sequence has one or two substitutions from the reference and contains only
A, C, G and U.

## Combine a field bias with a callback

Reuse the model and shell above. The callback measures a dinucleotide
feature, so it depends on more than one site.

```python
from dataclasses import replace
from adabmDCA import HammingBias, WeightedPotential

def gc_steps(sequences):
    return [-sequence.count("GC") for sequence in sequences]

biased = replace(
    shell,
    windows=(*shell.windows, HammingWindow("ACGUAA", 0, 2)),
    distance_biases=(HammingBias("ACGUAC", weight=0.2),),
    potentials=(
        WeightedPotential(gc_steps, weight=0.75),
        WeightedPotential(lambda sequences: [s.count("UU") for s in sequences], weight=0.25),
    ),
    entropy=True,
)
draws = sample_sequences(
    model=model, ptt=True, steering=biased, beta=1.25,
    n_sequences=128, ptt_max_rounds=256,
)
print(draws.steering["status"], draws.steering["entropy"])
```

A reversible block of constrained Gibbs updates handles the Potts and field
terms. A Metropolis correction applies the change in callback costs. The
sampler batches changed sequences and caches callback values across weights.

## Use a trained ladder

With an archive, pass `model="model/ptt.h5"` and `ptt=True`. The ladder follows
the stored checkpoints, then ramps the callback weights, linear biases and
inverse temperature above the trained endpoint. The pilot can insert
intermediate rungs before warmup.

An archive trained from an anchor other than the profile keeps its exact anchor at the bottom.
With hard distance windows the bottom rung is the anchor restricted to the
windows, drawn independently and normalized exactly, so no profile or bridge
rungs sit below it. Only windows whose joint count table would be too large
fall back to an exact constrained profile at the bottom, with the restricted
archive anchor as a coupled rung. A standalone model is also supported. Calling
`sample_sequences(model="model/ptt.h5", ptt=True, beta=1.25, n_sequences=128)`
uses a temperature ladder with every explicit bias weight set to zero.

Pass `initial_sequences` to structured PTT to seed every rung above the exact
bottom rung. Provide at least `SteeringConfig.chains` aligned sequences that
satisfy the hard constraints, for example a gap-free FASTA drawn at a warmer
temperature. The sampler shuffles seeds within each diagnostic group at each
upper rung. The bottom rung continues to supply exact draws for renewal.
The seeds count as old configurations during the renewal check.

## Check warmup and plot the result

Both strategies monitor rank-normalized split R-hat and bulk and tail ESS
on independent group-mean traces. PTT also tracks how many old configurations
remain anywhere on the ladder. A run that exhausts its warmup budget returns
`warmup_budget_exhausted` and provisional draws. With `sample_sequences`,
`n_sweeps` sets the structured MCMC budget and `ptt_max_rounds` sets the
structured PTT budget. Set `SteeringConfig.warmup_seconds` to add a
wall-clock limit checked between rounds. An endpoint returned at that limit
has status `warmup_time_exhausted`. Set `production_spacing=0` to take a
single snapshot at the end of warmup, with no more draws than chains.

```python
import matplotlib.pyplot as plt

history = draws.steering["history"]
warmup = history["warmup"]
plt.plot([row["round"] for row in warmup],
         [row["observables"]["reduced_energy"] for row in warmup])
plt.xlabel("Round")
plt.ylabel("Mean reduced energy")
draws.to_json("steered.json")
```

The result keeps warmup observations, retained production batches,
diagnostic checkpoints, and PTT exchange and renewal records. With
`entropy=True`, it also keeps running log Z and entropy estimates from the
production batches. See the [steering API reference](../api/adabmDCA.steering.sampling.md)
for the configuration and result fields.

The earlier `steering_potential(batch, strength)` interface is still
available for soft importance sampling. Its importance weights apply when
the source and target distributions have sufficient overlap.
