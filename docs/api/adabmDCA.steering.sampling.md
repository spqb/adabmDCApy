# Steering API

Pass `steering=SteeringConfig(...)` to
[sample_sequences](adabmDCA.api.sampling.md#function-sample_sequences).
It returns a [SamplingResult](adabmDCA.api.results.md#class-samplingresult).
See [Steer sequence sampling in Python](../usage/steering.md) for runnable examples.

## `SteeringConfig`

| Fields | Defaults and behavior |
| --- | --- |
| `allowed_tokens`, `windows`, `distance_biases`, `potentials` | Use the model alphabet and empty constraint and bias tuples by default. See [steering constraints](adabmDCA.steering.support.md). |
| `chains=128`, `groups=4`, `block_steps=None` | Allocate the chains per rung across independently seeded groups. A default block visits each site once. |
| `min_rounds=32`, `check_every=8`, `rhat=1.01`, `min_ess=40` | Set the warmup check schedule, the R-hat upper bound, and the lower bound for both bulk and tail ESS. |
| `production_spacing=1` | Advance this many rounds per retained endpoint batch. |
| `ladder_steps=4`, `bridge_steps=2` | Set the initial number of increments for the soft bias and for each base model bridge. |
| `pilot_rounds=8`, `max_rungs=24`, `min_swap=0.25` | Configure pilot blocks and midpoint insertion for weak PTT edges. Set `pilot_rounds=0` to skip the pilot. |
| `cache_size=8192`, `entropy=False` | Set the per-callback cache capacity and optionally enable absolute target entropy for PTT. |

With PTT, the old population count across the whole ladder divided by the
number of chains per rung must also be at most the `ptt_renewal_tolerance` passed to `sample_sequences`.

## `WeightedPotential`

`WeightedPotential(evaluate, weight=1.0, input="sequences")` describes a
deterministic batch function. `evaluate` takes a list of aligned strings or,
with `input="categorical"`, a NumPy integer array of shape `(batch, length)`.
It returns one finite scalar cost per row. The sampler multiplies each cost by
its rung's weight and reuses raw values across the ladder.

## Steering history

Access the following records through `result.steering["history"]`.
They are also available in the serialized result.

| Key | Records |
| --- | --- |
| `warmup`, `production` | One row per warmup round or retained production batch, with absolute `round`, `elapsed_seconds`, endpoint `observables`, and `group_means`. |
| `checks` | Each warmup stopping check with R-hat, bulk and tail ESS, the renewal ratio, and the combined pass status. |
| `pilot` | Pilot block boundaries, the rung labels, edgewise swap acceptance, and newly inserted rungs. |
| `entropy` | Running PTT log Z, entropy and BAR overlap, when requested. The final estimate also has Monte Carlo errors. |
| `rungs`, `windows`, `potential_indices` | Metadata that maps rung, distance, and active callback columns to their configuration. |

Warmup and production rows include local and exchange acceptances,
cumulative callback evaluations, and, for PTT, per-rung Potts energy,
available callback means, and old configuration counts. Production rows
report the number of draws retained and aggregate acceptances over the
preceding interval. Endpoint observables summarize the full target population.
