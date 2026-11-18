<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/sampling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.steering.sampling`
Sequence generation with compiled fields, hard support and weighted callbacks.



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/sampling.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `WeightedPotential`
Deterministic batch callback returning one finite unweighted cost per sequence.

Input is a list of strings by default or an integer NumPy array with shape (batch, sites) when input is 'categorical'.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    evaluate: 'Callable',
    weight: 'float' = 1.0,
    input: 'str' = 'sequences'
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/sampling.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SteeringConfig`
Define a structured steering target and its sampling diagnostics.



**Args:**

 - <b>`allowed_tokens`</b>:  Subset of model tokens allowed at every site.
 - <b>`windows`</b>:  Inclusive Hamming constraints, applied jointly on every rung.
 - <b>`distance_biases`</b>:  Linear Hamming costs folded into the Potts fields.
 - <b>`potentials`</b>:  Weighted deterministic batch costs applied to Gibbs proposals.
 - <b>`chains`</b>:  Number of parallel chains per rung.
 - <b>`groups`</b>:  Number of independently seeded groups for convergence checks.
 - <b>`block_steps`</b>:  Site updates per proposal. The default visits each site once.
 - <b>`check_every`</b>:  Warmup rounds between convergence checks.
 - <b>`min_rounds`</b>:  Minimum warmup rounds.
 - <b>`min_ess`</b>:  Minimum bulk and tail ESS for each monitored group-mean trace.
 - <b>`rhat`</b>:  Upper bound on rank-normalized split R-hat.
 - <b>`production_spacing`</b>:  Rounds between retained batches.
 - <b>`ladder_steps`</b>:  Initial number of increments from the trained to steered target.
 - <b>`bridge_steps`</b>:  Initial number of increments between consecutive base models.
 - <b>`pilot_rounds`</b>:  Rounds per PTT pilot block. Zero skips the pilot.
 - <b>`max_rungs`</b>:  Upper bound on the rung count for pilot insertions.
 - <b>`min_swap`</b>:  Pilot swap acceptance below which an intermediate rung is considered.
 - <b>`cache_size`</b>:  Maximum number of cached sequences per callback.
 - <b>`entropy`</b>:  Estimate target entropy using PTT and the exact bottom normalizer.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    allowed_tokens: 'str | None' = None,
    windows: 'tuple[HammingWindow, ]' = (),
    distance_biases: 'tuple[HammingBias, ]' = (),
    potentials: 'tuple[WeightedPotential, ]' = (),
    chains: 'int' = 128,
    groups: 'int' = 4,
    block_steps: 'int | None' = None,
    check_every: 'int' = 8,
    min_rounds: 'int' = 32,
    min_ess: 'float' = 40,
    rhat: 'float' = 1.01,
    production_spacing: 'int' = 1,
    ladder_steps: 'int' = 4,
    bridge_steps: 'int' = 2,
    pilot_rounds: 'int' = 8,
    max_rungs: 'int' = 24,
    min_swap: 'float' = 0.25,
    cache_size: 'int' = 8192,
    entropy: 'bool' = False
) → None
```











---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
