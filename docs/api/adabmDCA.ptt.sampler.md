<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.ptt.sampler`
Parallel Trajectory Tempering with snapshot retention and mixing recovery.

All replicas target beta=1. The active ladder can be shortened using a reservoir; optional full sampler mode retains the historical models. The initial profile normalizer is analytic and subsequent ones use bridges.

**Global Variables**
---------------
- **DEFAULT_PTT_N_CHAINS**
- **DEFAULT_PTT_SAMPLER**
- **IMMOBILE_THRESHOLD**
- **LOCAL_KERNELS**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bridge_increment`

```python
bridge_increment(lower, upper, samples)
```

Return log(Z_upper/Z_lower) from samples of the lower Hamiltonian.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PartitionEstimate`
Estimate of the endpoint's log partition function, with its provenance.



**Attributes:**

 - <b>`log_z`</b>:  Natural log of the partition function.
 - <b>`model_version`</b>:  Training update of the endpoint.
 - <b>`ladder_version`</b>:  Version of the ladder the bridges ran along.
 - <b>`model_id`</b>:  Hash of the endpoint parameters.
 - <b>`sample_round`</b>:  Exchange round of the chains used.
 - <b>`method`</b>:  ``"ptt_bridge"`` (forward bridges) or ``"ptt_bar"`` (BAR).
 - <b>`status`</b>:  ``"estimated"``.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    log_z: 'float',
    model_version: 'int',
    ladder_version: 'int',
    model_id: 'str',
    sample_round: 'int',
    method: 'str' = 'ptt_bridge',
    status: 'str' = 'estimated'
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PTTSampler`
A Parallel Trajectory Tempering ladder: models, their chains and its own RNG.

During training (see :func:`train_model` with ``ptt=``) the ladder runs from an exactly normalized profile model to the model being trained, and snapshots of the training trajectory are kept as intermediate replicas. Each exchange round swaps configurations between adjacent replicas and then runs local Monte Carlo sweeps. The same ladder is saved in the PTT archive and reused to sample the trained model and to estimate its log Z.

Most users only load an archive and sample or measure it:


- :meth:`from_archive`, :meth:`save_archive`, :meth:`fork`;
- :meth:`draw`, :meth:`endpoint_samples`, :meth:`endpoint_params`;
- :meth:`partition_estimate`, :meth:`entropy`, :meth:`ladder_statistics`,  :meth:`ladder_health`, :meth:`measure_renewal`.

The remaining public methods (``transition_target``, ``update_replica_chain``, ``refresh_reservoir``, ...) are the training machinery used by the PTT trainer. Mutating the ladder directly is unsupported. Archives contain arrays and JSON only, never pickled objects.



**Example:**
 ``` sampler = PTTSampler.from_archive("model/ptt.h5", device="cuda")```
     >>> sampler.prepare_sampling_ladder(2000)
     >>> sampler.measure_renewal(local_sweeps=10).status
     'converged'
     >>> sequences = sampler.draw(1000, local_sweeps=10)   # one-hot, (1000, L, q)
     >>> sampler.entropy()["entropy"]


<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    params: 'Mapping[str, Tensor]',
    tokens: 'str',
    n_chains: 'int' = 2000,
    sampler: 'str' = 'metropolized_gibbs',
    config: 'PTTConfig | None' = None,
    seed: 'int' = 0
) → None
```

Start a two-replica ladder at an uncoupled (profile) model.

Both replicas start at ``params``; training then moves the endpoint. To continue from a saved state use :meth:`from_archive`.



**Args:**

 - <b>`params`</b>:  ``bias`` of shape ``(L, q)`` and an all-zero ``coupling_matrix``  of shape ``(L, q, L, q)``, float32 or float64, on CPU, CUDA or MPS (float32).
 - <b>`tokens`</b>:  Ordered alphabet of length ``q``.
 - <b>`n_chains`</b>:  Chains per replica.
 - <b>`sampler`</b>:  Local kernel: ``"metropolized_gibbs"``, ``"gibbs"`` or ``"metropolis"``.
 - <b>`config`</b>:  PTT settings; defaults to ``PTTConfig()``.
 - <b>`seed`</b>:  Random seed of the sampler's own generator.



**Raises:**

 - <b>`InputValidationError`</b>:  If the parameters are coupled, invalid or  underflow, or a setting is invalid.


---

#### <kbd>property</kbd> n_active

Number of replicas in the active ladder (above the reservoir, if any).



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler/advance#L465"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `advance`

```python
advance(
    rounds: 'int' = 1,
    local_sweeps: 'int' = 1,
    is_cancelled: 'Callable[[], bool] | None' = None,
    on_round: 'Callable[[int, int], None] | None' = None,
    until: 'Callable[[], bool] | None' = None,
    return_samples: 'bool' = True
) → Tensor | None
```

Run exchange rounds at fixed parameters.

Each round swaps configurations between adjacent replicas, permutes chains within replicas, redraws the bottom replica, and runs local sweeps.



**Args:**

 - <b>`rounds`</b>:  Largest number of rounds.
 - <b>`local_sweeps`</b>:  Local sweeps per round.
 - <b>`is_cancelled`</b>:  Optional callable; returning ``True`` stops the run.
 - <b>`on_round`</b>:  Optional callback ``on_round(completed, rounds)`` after each round.
 - <b>`until`</b>:  Optional callable; returning ``True`` stops after the current round.
 - <b>`return_samples`</b>:  Return one-hot endpoint chains; ``False`` skips  their construction when only advancing the sampler state.



**Returns:**
 A copy of the endpoint chains, one-hot, shape ``(n_chains, L, q)``, or ``None`` when ``return_samples=False``.



**Raises:**

 - <b>`OperationCancelledError`</b>:  If ``is_cancelled`` returns ``True``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L443"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `capture_state`

```python
capture_state() → dict[str, Any]
```

Return a deep copy of the numerical state, for a recovery checkpoint.

Training history and recovery points are excluded; see :meth:`restore_state`.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L1717"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `draw`

```python
draw(
    n_samples: 'int',
    spacing: 'int' = 1,
    local_sweeps: 'int' = 1,
    is_cancelled: 'Callable[[], bool] | None' = None
) → Tensor
```

Collect endpoint samples, taking a batch of chains every ``spacing`` rounds.

Samples are only as good as the ladder's equilibration: run :meth:`measure_renewal` (or use :func:`sample_sequences`) first.



**Args:**

 - <b>`n_samples`</b>:  Number of samples.
 - <b>`spacing`</b>:  Exchange rounds between successive batches of ``n_chains``.
 - <b>`local_sweeps`</b>:  Local sweeps per round.
 - <b>`is_cancelled`</b>:  Optional callable; returning ``True`` stops the run.



**Returns:**
 One-hot samples of shape ``(n_samples, L, q)``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L299"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `endpoint_params`

```python
endpoint_params() → dict[str, Tensor]
```

Return a copy of the endpoint (last replica) parameters: the trained model.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L303"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `endpoint_samples`

```python
endpoint_samples(copy: 'bool' = True) → Tensor
```

Return the endpoint chains, one-hot, shape ``(n_chains, L, q)``.



**Args:**

 - <b>`copy`</b>:  Return a copy; ``False`` may return internal storage, which must  not be modified.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L1750"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `entropy`

```python
entropy(estimator: 'str' = 'bar') → dict[str, Any]
```

Estimate the endpoint entropy as ``<E> + log Z`` from the current chains.



**Args:**

 - <b>`estimator`</b>:  log Z estimator, ``"bar"`` or ``"forward"`` (see :meth:`partition_estimate`).



**Returns:**
 ``entropy``, ``mean_energy`` and ``free_energy`` (``-log Z``), in nats,
 - <b>`with the fields of the `</b>: class:`PartitionEstimate`.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L709"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `equilibrate`

```python
equilibrate(
    rounds: 'int | None' = None,
    local_sweeps: 'int' = 1,
    is_cancelled: 'Callable[[], bool] | None' = None,
    on_round: 'Callable[[int, int], None] | None' = None
) → Tensor
```

Run a fixed warmup of :meth:`advance`; this does not certify equilibrium.



**Args:**

 - <b>`rounds`</b>:  Rounds to run; defaults to ``config.equilibration_rounds``.
 - <b>`local_sweeps`</b>:  Local sweeps per round.
 - <b>`is_cancelled`</b>:  Optional callable; returning ``True`` stops the run.
 - <b>`on_round`</b>:  Optional callback ``on_round(completed, rounds)``.



**Returns:**
 A copy of the endpoint chains, one-hot.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L1487"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `equilibrate_for_lag`

```python
equilibrate_for_lag(
    relaxed: 'Callable[[PTTSampler], bool]',
    max_rounds: 'int',
    chunk: 'int' = 10,
    local_sweeps: 'int' = 1,
    is_cancelled: 'Callable[[], bool] | None' = None,
    on_progress: 'Callable[, None] | None' = None
) → tuple[bool, int, int]
```

Advance the ladder, then evolve at fixed parameters until ``relaxed(sampler)`` holds.

The response of the ``lag`` optimizer when the endpoint chains trail the model: parameters stay fixed while ``force_ladder_update`` gives the endpoint a recent neighbour and the ladder evolves in chunks of ``chunk`` rounds, up to ``max_rounds``. Runs on a fork and commits only a healthy result. Returns ``(healthy, rounds, work)``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L843"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `estimate_mixing_time`

```python
estimate_mixing_time(
    local_sweeps: 'int' = 1,
    is_cancelled: 'Callable[[], bool] | None' = None,
    full_ladder: 'bool' = False,
    on_progress: 'Callable[, None] | None' = None,
    bounded: 'bool' = True,
    max_rounds: 'int | None' = None,
    in_place: 'bool' = False,
    reference: 'bool' = False,
    method: 'str | None' = None
) → MixingEstimate
```

Run reference TRWA on a separate population and return its diagnostics.

Extend the experiment until it contains at least 20 times both estimated correlation times (configurable), or reaches its budget. Training populations and RNG are preserved; diagnostic work counts. ``full_ladder=True`` also checks retained history in full sampler mode. ``bounded=False`` continues until convergence or cancellation, ignoring the training mixing-round budget. ``in_place=True`` measures and equilibrates the current populations, as in reference TRWA. ``reference=True`` uses its FFT and fitting rules. The replica index of each uniquely tagged configuration (its lineage) is the observable. ``method`` defaults to ``config.mixing_method``. ``'renewal'`` replaces the replica-index experiment with a population-renewal check on the same kind of separate population (see ``_renewal_mixing_experiment``).



**Args:**

 - <b>`local_sweeps`</b>:  Local sweeps per exchange round.
 - <b>`is_cancelled`</b>:  Optional callable; returning ``True`` stops the run.
 - <b>`full_ladder`</b>:  Also check the retained history (full sampler mode).
 - <b>`on_progress`</b>:  Optional callback ``on_progress(stage, done, total, **details)``.
 - <b>`bounded`</b>:  Stop at ``config.mixing_max_rounds``; ``False`` runs until converged.
 - <b>`max_rounds`</b>:  Round budget overriding the configured one.
 - <b>`in_place`</b>:  Measure and equilibrate the current populations.
 - <b>`reference`</b>:  Use the reference TRWA fitting rules.
 - <b>`method`</b>:  ``"autocorrelation"`` or ``"renewal"``; defaults to  ``config.mixing_method``.



**Returns:**

 - <b>`A `</b>: class:`MixingEstimate`; ``converged`` tells whether the ladder mixes.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L1453"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `force_ladder_update`

```python
force_ladder_update(
    local_sweeps: 'int' = 1,
    is_cancelled: 'Callable[[], bool] | None' = None,
    on_progress: 'Callable[, None] | None' = None
) → bool
```

Advance the snapshot procedure now, as if acceptance had just dropped below its target.

With a held snapshot, it replaces the replica below the endpoint (and the reservoir is refreshed when the active ladder grows too long); otherwise the stored snapshot, or a snapshot of the current endpoint when none is stored, is inserted below the endpoint and the endpoint is flagged as a checkpoint. Nothing is inserted twice at the same model version. Returns False when a reservoir refresh fails.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L380"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fork`

```python
fork(seed: 'int | None' = None) → PTTSampler
```

Return an independent copy of the sampler, without its recovery points.



**Args:**

 - <b>`seed`</b>:  New random seed, or ``None`` to continue the same random stream.



**Returns:**

 - <b>`A new `</b>: class:`PTTSampler`; changes to it do not affect this one.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L2040"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_archive`

```python
from_archive(
    path: 'str | Path',
    device: 'str' = 'cpu',
    mode: 'str' = 'generate',
    seed: 'int | None' = None
) → PTTSampler
```

Load a sampler from an HDF5 archive written by training or :meth:`save_archive`.

Every array is validated on load. Generation never writes the source archive. Loading on a different device than the one that saved the archive requires a new ``seed`` (and then a warmup): the random stream cannot be continued exactly across devices.



**Args:**

 - <b>`path`</b>:  Archive path.
 - <b>`device`</b>:  ``"cpu"``, a CUDA device such as ``"cuda"``, or  ``"mps"`` (float32 archives).
 - <b>`mode`</b>:  ``"generate"`` (sampling; training state dropped), ``"resume"``  (continue training exactly) or ``"inspect"`` (read only).
 - <b>`seed`</b>:  New random seed, or ``None`` to continue the archived stream.



**Returns:**

 - <b>`The loaded `</b>: class:`PTTSampler`.



**Raises:**

 - <b>`InputValidationError`</b>:  If the archive is invalid, or the device changes  without a new seed.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler/ladder_health#L791"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ladder_health`

```python
ladder_health(bootstrap: 'int' = 200, seed: 'int' = 0) → dict[str, Any]
```

Bidirectional reweighting diagnostics for every adjacent active pair.

See ``adabmDCA.ptt.health``. Per pair: forward, reverse and BAR free energies with bootstrap errors, hysteresis, effective sample sizes and top-1% weight shares in both directions, the Crooks slope (-1 at equilibrium), mean swap acceptance and the fraction of configurations whose average swap probability is below ``IMMOBILE_THRESHOLD``, and quantiles of the per-configuration acceptance in both directions. With birth tracking, the mean age of immobile and mobile upper configurations is included, and ``replicas`` gives, per replica, the median and 99th percentile of the configuration ages (rounds since birth) and the flow: the fraction of configurations that reached the top replica since birth. Also returns forward and BAR endpoint log Z.



**Args:**

 - <b>`bootstrap`</b>:  Bootstrap resamples for the error bars.
 - <b>`seed`</b>:  Seed of the bootstrap.



**Returns:**
 ``pairs`` (one dictionary per adjacent pair), ``replicas`` (one per replica, empty without birth tracking), ``log_z_forward``, ``log_z_bar``, ``log_z_bar_error`` and ``immobile_threshold``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler/ladder_statistics#L1770"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ladder_statistics`

```python
ladder_statistics(
    reference: 'Tensor',
    weights: 'Tensor',
    estimator: 'str' = 'bar'
) → list[dict[str, Any]]
```

Estimate every active model's entropy and weighted data likelihood.

Energies, entropy and logZ are in nats per sequence; likelihood also includes a per-residue column, matching the training convention. logZ uses BAR bridges by default (see ``partition_estimate``); ``log_z_forward`` always reports the forward bridges for comparison.



**Args:**

 - <b>`reference`</b>:  Data sequences, categorical ``(M, L)`` or one-hot ``(M, L, q)``.
 - <b>`weights`</b>:  Weight of each reference sequence.
 - <b>`estimator`</b>:  ``"bar"`` or ``"forward"``.



**Returns:**

 - <b>`One dictionary per active replica`</b>:  ``log_z``, ``mean_energy``, ``entropy``, ``log_likelihood``, ``log_likelihood_per_residue``, ...

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L1035"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `measure_renewal`

```python
measure_renewal(
    local_sweeps: 'int' = 1,
    max_rounds: 'int' = 20000,
    tolerance: 'float' = 0.01,
    blocks_per_warmup: 'int' = 1,
    stationary: 'bool' = True,
    is_cancelled: 'Callable[[], bool] | None' = None,
    on_progress: 'Callable[, None] | None' = None,
    stages: 'Sequence[str]' = ('renewal_warmup', 'renewal_stationary')
) → RenewalEstimate
```

Equilibrate the active ladder in place until its population is renewed twice (or once).

Requires birth tracking (``prepare_sampling_ladder`` or ``start_birth_tracking``). A configuration is born when it enters the active ladder: an exact profile draw at rung 0 of a full ladder, or a reservoir emission otherwise. The warmup ends at the first round where at most ``tolerance * n_chains`` configurations born before the start remain in the active ladder. The stationary phase then restarts the reference and advances in fixed blocks of the warmup length divided by ``blocks_per_warmup`` until the same holds at a block end. Stopping only at block ends keeps the stopping round independent of which configurations are old; shorter blocks waste fewer rounds after the renewal. With ``stationary=False`` only the warmup runs, and the measurement converges at the warmup renewal. No autocorrelation fit decides anything; see ``RenewalEstimate`` for the reported quantities. ``max_rounds`` bounds both phases together; exhausting it reports the old endpoint fraction as ``trapped_fraction``.

Progress events carry the old fraction ``ladder_old`` of the whole ladder, the ``threshold`` it must reach, and every 10 rounds a ``renewal_forecast`` of the phase: ``decay_rounds``, ``predicted_round`` and the first decay time of the phase, ``first_decay_rounds``, whose growth signals configurations that leave more and more slowly.



**Args:**

 - <b>`local_sweeps`</b>:  Local sweeps per exchange round.
 - <b>`max_rounds`</b>:  Round budget of both phases together.
 - <b>`tolerance`</b>:  Largest remaining fraction of old configurations, per replica.
 - <b>`blocks_per_warmup`</b>:  Stationary phase: blocks per warmup length.
 - <b>`stationary`</b>:  Run the stationary phase after the warmup.
 - <b>`is_cancelled`</b>:  Optional callable; returning ``True`` stops the run.
 - <b>`on_progress`</b>:  Optional callback ``on_progress(stage, done, total, **details)``.
 - <b>`stages`</b>:  Stage names reported for the two phases.



**Returns:**

 - <b>`A `</b>: class:`RenewalEstimate`; ``status`` is ``"converged"`` when renewed in time.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L733"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `partition_estimate`

```python
partition_estimate(estimator: 'str' = 'forward') → PartitionEstimate
```

Estimate log Z of the endpoint from the exact anchor and one bridge per replica pair.

``estimator='forward'`` reweights each lower population to the model above (the historical bridge used during training and stored in archives). ``'bar'`` combines both populations of every pair with Bennett's acceptance ratio, which stays reliable when the forward weights are heavy-tailed.



**Args:**

 - <b>`estimator`</b>:  ``"forward"`` or ``"bar"``.



**Returns:**

 - <b>`A `</b>: class:`PartitionEstimate` with ``log_z`` and its provenance.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L328"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare_sampling_ladder`

```python
prepare_sampling_ladder(n_chains: 'int') → None
```

Rebuild the full ladder for generation, with fresh chains drawn from the profile.

The generation ladder is the training ladder: the exact profile, every update flagged ``ptt`` during training, and the final endpoint. This switches the sampler to generation mode and starts birth tracking, as :meth:`measure_renewal` requires.



**Args:**

 - <b>`n_chains`</b>:  Chains per replica.



**Raises:**

 - <b>`InputValidationError`</b>:  If the archive lacks the flagged models.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L1910"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare_steering`

```python
prepare_steering(
    steering: 'Steering',
    strength: 'float',
    proposal_steps: 'int | None' = None,
    target_acceptance: 'float' = 0.3,
    local_sweeps: 'int' = 1,
    trial_rounds: 'int' = 20,
    max_rungs: 'int' = 64,
    is_cancelled: 'Callable[[], bool] | None' = None,
    on_progress: 'Callable[, None] | None' = None
) → list[float]
```

Extend the generation ladder with steered rungs up to ``strength``.

Every new rung has the endpoint's DCA parameters plus the steering potential at a strength between 0 and ``strength``; the last one is at ``strength`` and becomes the sampled endpoint. Rungs are placed one at a time: a candidate strength is kept when, after ``trial_rounds`` rounds of warmup and ``trial_rounds`` of measurement, the swap acceptance with the rung below is at least ``target_acceptance``; otherwise the step is halved. The bottom of the ladder, the exact profile, is unchanged, so population renewal still certifies mixing. Proposal blocks of the steered rungs adapt during placement and are then frozen.

Call after :meth:`prepare_sampling_ladder`. A steered ladder cannot be saved with :meth:`save_archive`.



**Args:**

 - <b>`steering`</b>:  The validated potential.
 - <b>`strength`</b>:  Final steering strength.
 - <b>`proposal_steps`</b>:  Site updates per steered proposal, or ``None`` to adapt.
 - <b>`target_acceptance`</b>:  Smallest swap acceptance between adjacent steered rungs.
 - <b>`local_sweeps`</b>:  Local sweeps per exchange round.
 - <b>`trial_rounds`</b>:  Rounds of warmup, then of measurement, per candidate rung.
 - <b>`max_rungs`</b>:  Largest number of steered rungs.
 - <b>`is_cancelled`</b>:  Optional callable; returning ``True`` stops the run.
 - <b>`on_progress`</b>:  Optional callback ``on_progress(stage, done, total, **details)``.



**Returns:**
 The strengths of the steered rungs, increasing to ``strength``.



**Raises:**

 - <b>`InputValidationError`</b>:  If the ladder is not a generation ladder, the  potential is invalid, or ``strength`` is 0.
 - <b>`ConvergenceError`</b>:  If ``max_rungs`` rungs do not reach ``strength``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L1526"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `refresh_reservoir`

```python
refresh_reservoir(
    local_sweeps: 'int' = 1,
    is_cancelled: 'Callable[[], bool] | None' = None,
    on_progress: 'Callable[, None] | None' = None
) → bool
```

Training: collect a new reservoir from the ladder and shorten the active ladder.



**Returns:**
  ``False`` if the ladder failed the mixing check needed to collect it.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L456"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `restore_state`

```python
restore_state(state: 'Mapping[str, Any]') → None
```

Restore a state returned by :meth:`capture_state`, clearing the training state.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L2022"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_archive`

```python
save_archive(path: 'str | Path') → Path
```

Write the complete sampler to an HDF5 archive, replacing ``path`` atomically.

The archive is validated before it replaces an existing file, so an interrupted save never leaves a broken archive.



**Args:**

 - <b>`path`</b>:  Archive path, usually ``ptt.h5``.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L416"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_generation_kernel`

```python
set_generation_kernel(name: 'str | None' = None) → None
```

Select the local kernel used during generation; ``None`` restores the archived one.

Every kernel leaves each replica's distribution invariant, so a model trained with one kernel can be sampled with another. Archives keep the training kernel as ``local_sampler``.



**Args:**

 - <b>`name`</b>:  ``"metropolized_gibbs"``, ``"gibbs"``, ``"metropolis"`` or ``None``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L993"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `start_birth_tracking`

```python
start_birth_tracking() → None
```

Mark every current configuration as born before the next round.

From then on ``advance`` stamps configurations entering the active ladder with the current round: exact profile draws at rung 0 in a full ladder, or configurations emitted by the reservoir otherwise. Exchanges and permutations carry birth rounds with configurations.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L2003"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `steering_log_z_ratio`

```python
steering_log_z_ratio(estimator: 'str' = 'bar') → float
```

Estimate ``log Z_s - log Z_0``: steered top rung against the unsteered endpoint.

``-log_z_ratio`` is the free-energy cost of the steering, and ``exp(log_z_ratio)`` equals ``<exp(-V(x, s))>`` under the unsteered model.



**Args:**

 - <b>`estimator`</b>:  ``"bar"`` or ``"forward"`` bridges.



**Returns:**
 The log ratio of partition functions.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L1664"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transition_target`

```python
transition_target(
    params: 'Mapping[str, Tensor]',
    local_sweeps: 'int' = 1,
    is_cancelled: 'Callable[[], bool] | None' = None,
    on_progress: 'Callable[, None] | None' = None
) → tuple[bool, int]
```

Training: move the endpoint to ``params``, then run the snapshot procedure.

The transition runs on a fork and is committed only if the ladder stays healthy; otherwise ``last_failure`` describes the mixing failure.



**Args:**

 - <b>`params`</b>:  New endpoint parameters, same shape, precision and device.
 - <b>`local_sweeps`</b>:  Local sweeps per exchange round.
 - <b>`is_cancelled`</b>:  Optional callable; returning ``True`` stops the run.
 - <b>`on_progress`</b>:  Optional progress callback.



**Returns:**

 - <b>```(accepted, work)```</b>:  whether the transition was committed and the local sweeps it used.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/sampler.py#L1367"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_replica_chain`

```python
update_replica_chain(
    local_sweeps: 'int' = 1,
    is_cancelled: 'Callable[[], bool] | None' = None,
    on_progress: 'Callable[, None] | None' = None
) → bool
```

Training: run the snapshot procedure after an endpoint update.

When the endpoint's swap acceptance falls below ``2 * target_acceptance`` a snapshot of it is stored; below ``target_acceptance`` the stored snapshot is inserted into the ladder (or a held one replaces the replica below).



**Returns:**
  ``False`` if a required reservoir refresh failed its mixing check.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
