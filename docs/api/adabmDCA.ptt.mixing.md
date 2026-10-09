<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/mixing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.ptt.mixing`
Replica-index autocorrelation analysis used by PTT's TRWA experiment.

This follows ``ptt_paper.implement._process_experiment``: center replica labels at their known uniform mean, compute their autocorrelation with an FFT, use a self-consistent window for tau_int, and fit the exponential tail for tau_exp. Times are in exchange rounds, not individual local sweeps.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/mixing.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `renewal_forecast`

```python
renewal_forecast(ladder_old, threshold, min_points=20)
```

Fit ``log G`` against rounds over the recent half of the decay and predict the renewal round.

Only rounds with ``0 < G < 0.5`` enter the fit: the first rounds, before fresh configurations reach every replica, and the counting floor are excluded. The fit uses the most recent half of those rounds, so it follows the slowest remaining configurations as the faster ones leave; an earlier fit tends to predict renewal too early. Returns ``None`` until ``min_points`` rounds are usable. The forecast only informs; renewal is decided by counting old configurations.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/mixing.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exponential_decay`

```python
exponential_decay(t, amplitude, tau_exp)
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/mixing.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `replica_autocorrelation`

```python
replica_autocorrelation(indices: 'Tensor') → Tensor
```

Normalized C(t) for (rounds, replicas, chains) origin labels.

Labels must be replica numbers, not globally unique chain identifiers. Use zero padding to at least twice the trajectory length to avoid circular correlations. The reference's biased (unadjusted for lag) FFT estimator is retained; no empirical per-chain mean is subtracted.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/mixing.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `integrated_autocorrelation_time`

```python
integrated_autocorrelation_time(correlation: 'Tensor') → float
```

Self-consistent window t >= 6 tau_int(t), with C(0)/2.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/mixing.py#L208"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exponential_autocorrelation_time`

```python
exponential_autocorrelation_time(correlation: 'Tensor') → float
```

Fit the reference tail window: first zero / 3 through twice that zero.

For a trajectory without a zero crossing, use half the available lags as the window endpoint, as in the reference. Positive fit bounds and explicit fit failure avoid accepting a negative or non-finite relaxation time.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/mixing.py#L251"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `process_replica_experiment`

```python
process_replica_experiment(
    indices: 'Tensor',
    n_thermalization=0,
    reference=False
)
```

Return (tau_int, tau_exp, C) without changing the recorded labels.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/mixing.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MixingEstimate`
Outcome of one PTT mixing check.



**Attributes:**

 - <b>`tau_int`</b>:  Integrated autocorrelation time of the replica index, in rounds  (autocorrelation method), else ``None``.
 - <b>`tau_exp`</b>:  Exponential autocorrelation time, in rounds, else ``None``.
 - <b>`rounds`</b>:  Rounds measured.
 - <b>`required_rounds`</b>:  Rounds the check needed to pass.
 - <b>`status`</b>:  ``"converged"``, or why it did not converge (e.g. ``"budget_exceeded"``).
 - <b>`acceptance`</b>:  Swap acceptance of each adjacent replica pair.
 - <b>`local_sweeps`</b>:  Local sweeps spent on the check.
 - <b>`model_version`</b>:  Training update of the endpoint when checked.
 - <b>`ladder_version`</b>:  Version of the ladder checked.
 - <b>`replicas`</b>:  Number of replicas checked.
 - <b>`full_ladder`</b>:  Whether the full historical ladder was checked.
 - <b>`method`</b>:  ``"autocorrelation"`` or ``"renewal"``.
 - <b>`warmup_rounds`</b>:  Renewal method: rounds until the initial populations were  replaced, or ``None`` if the budget ran out first.
 - <b>`renewal_rounds`</b>:  Renewal method: rounds until the check passed, or ``None``.
 - <b>`trapped_fraction`</b>:  Renewal method: fraction of endpoint configurations still  older than the reference when the check ended (large when it failed).
 - <b>`immobile`</b>:  Renewal method, when the check failed: for each replica above  the bottom that still holds old configurations, a dict with  ``replica`` (counting from the bottom of the active ladder),  ``chains``, ``old`` (configurations born before the reference),  ``immobile`` (old configurations whose swap acceptance toward the  replica below is under ``health.IMMOBILE_THRESHOLD``, so that only  local moves can free them)  and ``blocking`` (whether the immobile ones alone exceed the renewal  tolerance).

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    tau_int: 'float | None',
    tau_exp: 'float | None',
    rounds: 'int',
    required_rounds: 'int',
    status: 'str',
    acceptance: 'tuple[float, ]' = (),
    local_sweeps: 'int' = 0,
    model_version: 'int' = 0,
    ladder_version: 'int' = 0,
    replicas: 'int' = 0,
    full_ladder: 'bool' = False,
    method: 'str' = 'autocorrelation',
    warmup_rounds: 'int | None' = None,
    renewal_rounds: 'int | None' = None,
    trapped_fraction: 'float | None' = None,
    immobile: 'tuple[dict, ]' = ()
) → None
```






---

#### <kbd>property</kbd> converged

Whether the ladder passed the check.




---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/mixing.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RenewalEstimate`
Population renewal measured with configuration birth rounds.

Rung 0 is redrawn exactly every round, so a configuration's birth round is the round in which it was drawn there. Relative to a reference round, ``ladder_old`` is the fraction of all configurations born before it (it can only decrease) and ``endpoint_fresh`` the fraction of endpoint configurations born after it. A phase ends once at most ``tolerance * chains_per_model`` old configurations remain anywhere in the ladder, which bounds the endpoint old fraction by ``tolerance`` from then on.

The warmup phase measures renewal of the initial state. The stationary phase restarts the reference at the end of the warmup and advances in fixed blocks of ``chunk_rounds`` (a fraction of ``warmup_rounds``) until the ladder is renewed again, so the retained population descends entirely from draws made after the ladder had forgotten its initialization. ``warmup_decay_rounds`` and ``stationary_decay_rounds`` are the decay times of the old fraction fitted at the end of each phase (see ``renewal_forecast``). Times are exchange rounds.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    status: 'str',
    tolerance: 'float',
    warmup_rounds: 'int | None',
    renewal_rounds: 'int | None',
    stationary_rounds: 'int',
    chunk_rounds: 'int',
    trapped_fraction: 'float',
    warmup_ladder_old: 'tuple[float, ]' = (),
    warmup_endpoint_fresh: 'tuple[float, ]' = (),
    stationary_ladder_old: 'tuple[float, ]' = (),
    stationary_endpoint_fresh: 'tuple[float, ]' = (),
    old_fraction_by_model: 'tuple[float, ]' = (),
    acceptance: 'tuple[float, ]' = (),
    local_sweeps: 'int' = 0,
    replicas: 'int' = 0,
    warmup_decay_rounds: 'float | None' = None,
    stationary_decay_rounds: 'float | None' = None,
    immobile: 'tuple[dict, ]' = ()
) → None
```






---

#### <kbd>property</kbd> converged

Whether the ladder passed the check.

---

#### <kbd>property</kbd> rounds








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/mixing.py#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RenewalForecast`
Exponential fit of the old fraction G(t) of a renewal phase.

``decay_rounds`` is the fitted decay time (``None`` if G is not decaying), ``predicted_round`` the round of the phase at which the fit reaches ``threshold``, and the fit is ``log G = intercept - t / decay`` over rounds ``fit_start``..``fit_end``.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    decay_rounds: 'float | None',
    predicted_round: 'float | None',
    intercept: 'float',
    fit_start: 'int',
    fit_end: 'int'
) → None
```











---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
