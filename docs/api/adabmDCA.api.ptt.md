<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/ptt.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.ptt`
Archive-based generation and direct entropy using the shared PTT sampler.

**Global Variables**
---------------
- **STATIONARY_BLOCKS_PER_WARMUP**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/ptt.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_ptt_backend`

```python
load_ptt_backend(
    source: 'PTTSampler | str | Path',
    device: 'str' = 'cpu',
    seed: 'int | None' = None,
    alphabet: 'str | None' = None
) → PTTSampler
```

Return a PTT sampler ready for generation, loaded or forked from ``source``.

An in-memory :class:`PTTSampler` is forked, so the original is not modified. An archive keeps its precision and alphabet.



**Args:**

 - <b>`source`</b>:  A :class:`PTTSampler` or a path to a PTT archive.
 - <b>`device`</b>:  Device to load the archive on; ``"auto"`` picks CUDA when available.  Changing device requires a new ``seed``.
 - <b>`seed`</b>:  New random seed, or ``None`` to continue the archived random stream.
 - <b>`alphabet`</b>:  Optional alphabet; must match the archive's tokens.



**Returns:**

 - <b>`A `</b>: class:`PTTSampler` in generation mode.



**Raises:**

 - <b>`InputValidationError`</b>:  If the archive is invalid or ``alphabet`` conflicts with it.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/ptt.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sample_ptt_sequences`

```python
sample_ptt_sequences(
    source: 'PTTSampler | str | Path',
    n_sequences: 'int',
    local_sweeps: 'int' = 10,
    reference_fasta: 'AlignmentInput | None' = None,
    weights_path: 'WeightInput | None' = None,
    test_fasta: 'AlignmentInput | None' = None,
    privet_window: 'tuple[float, float]' = (0.01, 0.5),
    clustering_seqid: 'float' = 0.8,
    no_reweighting: 'bool' = False,
    seed: 'int' = 0,
    device: 'str' = 'cpu',
    alphabet: 'str | None' = None,
    sampler: 'str | None' = None,
    dtype: 'str | None' = None,
    n_measure: 'int' = 10000,
    pseudocount: 'float | None' = None,
    max_rounds: 'int' = 20000,
    mixing_method: 'str' = 'renewal',
    renewal_tolerance: 'float' = 0.01,
    stationary: 'bool' = False,
    local_kernel: 'str | None' = 'metropolized_gibbs',
    steering_potential: 'SteeringPotential | None' = None,
    steering_strength: 'float' = 1.0,
    steering_input: 'SteeringInput' = 'sequences',
    steering_proposal_steps: 'int | None' = None,
    steering_acceptance: 'float' = 0.3,
    collect_diagnostics: 'bool' = False,
    progress: 'Callable[[SamplingProgress], None] | None' = None,
    is_cancelled: 'Callable[[], bool] | None' = None
) → SamplingResult
```

Equilibrate the ladder in place, then collect endpoint samples.

``mixing_method='renewal'`` (default) tracks the birth round of every configuration and waits until the ladder has twice been repopulated by exact rung-0 draws (see ``PTTSampler.measure_renewal``); extra batches are spaced by the measured renewal time. By default the renewal runs only the warmup, which replaces the initial populations once; ``stationary=True`` adds the stationary renewal, which certifies that the returned population descends from draws made after the ladder had forgotten its start. ``'autocorrelation'`` keeps the replica-index TRWA estimate of tau_int and tau_exp. Each round runs one pass of adjacent exchanges and population permutations, then the local sweeps, as in training. ``local_kernel`` selects the local update ('metropolized_gibbs' by default, 'metropolis' or 'gibbs'); ``None`` keeps the archived training kernel. All of them leave every replica's distribution invariant. The ladder is the training ladder (see ``PTTSampler.prepare_sampling_ladder``). With ``steering_potential``, steered rungs are added on top of it (``PTTSampler.prepare_steering``) and the samples come from the top one; see :func:`adabmDCA.api.sampling.sample_sequences` for the steering arguments.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/ptt.py#L440"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `estimate_ptt_entropy`

```python
estimate_ptt_entropy(
    model: 'PTTSampler | str | Path',
    n_sweeps: 'int' = 1,
    seed: 'int' = 0,
    device: 'str' = 'cpu',
    alphabet: 'str | None' = None,
    output_dir: 'str | Path | None' = None,
    label: 'str' = 'entropy',
    is_cancelled: 'Callable[[], bool] | None' = None
) → PTTEntropyResult
```

Estimate the entropy of a PTT-trained model as ``<E> + log Z``.

A PTT archive already carries an estimate of log Z (bridges along its replica ladder), so no thermodynamic integration is needed: the ladder is advanced for the archive's ``equilibration_rounds`` exchange rounds, then ``log Z`` is re-estimated with Bennett's acceptance ratio and ``<E>`` is averaged over the endpoint chains.



**Args:**

 - <b>`model`</b>:  A PTT archive path, or a :class:`PTTSampler` (it is forked, not modified).
 - <b>`n_sweeps`</b>:  Local sweeps per exchange round of the warmup.
 - <b>`seed`</b>:  Random seed of the warmup.
 - <b>`device`</b>:  ``"cpu"``, ``"cuda"`` or ``"auto"``.
 - <b>`alphabet`</b>:  Optional alphabet; must match the archive.
 - <b>`output_dir`</b>:  If given, the result is written to ``<output_dir>/<label>.json``.
 - <b>`label`</b>:  File-name stem of the written JSON.
 - <b>`is_cancelled`</b>:  Optional callable; returning ``True`` stops the warmup.



**Returns:**

 - <b>`A `</b>: class:`PTTEntropyResult`; ``entropy`` is in nats per sequence.



**Example:**
 ``` estimate_ptt_entropy(model="output/ptt.h5", device="cuda").entropy```
    131.2



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/ptt.py#L400"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PTTEntropyResult`
Entropy of a PTT model, returned by :func:`estimate_ptt_entropy`.



**Attributes:**

 - <b>`entropy`</b>:  Entropy of the endpoint model, in nats per sequence  (``mean_energy + log_z``).
 - <b>`free_energy`</b>:  ``-log_z``.
 - <b>`mean_energy`</b>:  Mean endpoint energy over the endpoint chains.
 - <b>`log_z`</b>:  Log partition function from the PTT bridges.
 - <b>`method`</b>:  Estimator of ``log_z``, e.g. ``"ptt_bridge"``.
 - <b>`status`</b>:  Status of the ``log_z`` estimate.
 - <b>`model_id`</b>:  Hash identifying the endpoint parameters.
 - <b>`model_version`</b>:  Number of training updates of the endpoint.
 - <b>`ladder_version`</b>:  Version of the replica ladder used.
 - <b>`sample_round`</b>:  Exchange round at which the chains were measured.
 - <b>`artifacts`</b>:  Written files by kind, when ``output_dir`` was given.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    entropy: 'float',
    free_energy: 'float',
    mean_energy: 'float',
    log_z: 'float',
    method: 'str',
    status: 'str',
    model_id: 'str',
    model_version: 'int',
    ladder_version: 'int',
    sample_round: 'int',
    artifacts: 'dict[str, Path]' = <factory>
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/ptt.py#L431"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-compatible document with the result type and schema version.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/ptt.py#L435"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path') → Path
```

Write :meth:`to_dict` as JSON and return the written path.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
