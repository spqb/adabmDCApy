<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/sampling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.sampling`
High-level sequence-generation operations.

**Global Variables**
---------------
- **DEFAULT_PRIVET_WINDOW**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/sampling.py#L348"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sample_sequences`

```python
sample_sequences(
    model: 'DCAModel | str | Path',
    n_sequences: 'int',
    ptt: 'bool' = False,
    ptt_local_sweeps: 'int' = 10,
    ptt_max_rounds: 'int' = 20000,
    ptt_mixing_method: 'str' = 'renewal',
    ptt_renewal_tolerance: 'float' = 0.01,
    ptt_stationary: 'bool' = False,
    ptt_local_kernel: 'str | None' = 'metropolized_gibbs',
    n_sweeps: 'int' = 1000,
    sampler: 'str | None' = 'metropolized_gibbs',
    beta: 'float' = 1.0,
    seed: 'int' = 0,
    reference_fasta: 'AlignmentInput | None' = None,
    weights_path: 'WeightInput | None' = None,
    test_fasta: 'AlignmentInput | None' = None,
    privet_window: 'tuple[float, float]' = (0.01, 0.5),
    n_measure: 'int' = 10000,
    mixing_multiplier: 'int' = 2,
    pseudocount: 'float | None' = None,
    clustering_seqid: 'float' = 0.8,
    no_reweighting: 'bool' = False,
    alphabet: 'str | None' = None,
    device: 'str' = 'auto',
    dtype: 'str | None' = None,
    steering_potential: 'SteeringPotential | None' = None,
    steering_strength: 'float' = 1.0,
    steering_input: 'SteeringInput' = 'sequences',
    steering_proposal_steps: 'int | None' = None,
    ptt_steering_acceptance: 'float' = 0.3,
    collect_diagnostics: 'bool' = False,
    progress: 'ProgressCallback | None' = None,
    is_cancelled: 'CancellationHook | None' = None
) → SamplingResult
```

Generate sequences from a DCA model, with an energy for each.

Ordinary sampling runs ``n_sequences`` independent Markov chains. With ``reference_fasta``, their mixing time is first measured (up to ``n_sweeps`` sweeps) and generation runs for ``mixing_multiplier`` mixing times; otherwise it runs for exactly ``n_sweeps`` sweeps.

PTT sampling (``ptt=True``, a PTT archive, or a :class:`PTTSampler` as ``model``) instead uses the replica ladder saved during training: the exact profile, the models flagged during training and the endpoint. It runs at ``beta=1``, ignores ``n_sweeps``, and equilibrates until its mixing check passes. A :class:`PTTSampler` passed as ``model`` is forked, not modified. Every result includes the summed context-dependent entropy (CDE) of each sequence and, when identifiable, a least-squares fit of energy against it.

**Steered (importance) sampling.** With ``steering_potential``, sequences are drawn from ``p_s(x) ∝ exp(-beta * H(x) - V(x, s))`` instead, where ``V(x, s) = steering_potential(x, s)`` and ``s = steering_strength``: low ``V`` is favoured, so return ``-s * score`` to favour high scores. ``V`` is called on batches: a list of aligned sequence strings (``steering_input="sequences"``) or a one-hot tensor ``(n, L, q)`` on the model's device and precision (``"onehot"``), and must return one number per sequence, and ``0`` when ``s == 0``. Each Monte Carlo move proposes ``steering_proposal_steps`` Gibbs updates under the DCA model, at random sites visited in palindromic order (which makes the proposal reversible), and accepts them with probability ``min(1, exp(-ΔV))``. This samples ``p_s`` exactly while calling ``V`` once per chain and move; ``sampler`` and ``ptt_local_kernel`` do not apply to steered moves.

With PTT, steered rungs of increasing strength are added above the trained endpoint until ``steering_strength`` is reached, keeping the swap acceptance between them at least ``ptt_steering_acceptance``; the ladder's bottom is unchanged, so the mixing check still applies, and PTT also estimates ``log Z_s - log Z_0``. The result's ``log_importance_weights`` turn averages over the steered sequences into averages under the original model.



**Args:**

 - <b>`model`</b>:  A :class:`DCAModel`, a path to a parameter file or PTT archive, or
 - <b>`a `</b>: class:`PTTSampler`.
 - <b>`n_sequences`</b>:  Number of sequences to generate.
 - <b>`ptt`</b>:  Sample with PTT from the archive given as ``model``.
 - <b>`ptt_local_sweeps`</b>:  PTT: local sweeps per exchange round.
 - <b>`ptt_max_rounds`</b>:  PTT: largest number of exchange rounds of the mixing check.
 - <b>`ptt_mixing_method`</b>:  PTT: ``"renewal"`` waits until the ladder has twice been  repopulated by exact profile draws, leaving at most  ``ptt_renewal_tolerance`` of older configurations at the endpoint;  ``"autocorrelation"`` measures the replica-index autocorrelation times  and spaces output batches by twice the integrated time.
 - <b>`ptt_renewal_tolerance`</b>:  PTT renewal: largest remaining fraction of old configurations.
 - <b>`ptt_stationary`</b>:  PTT renewal: add a stationary renewal after the warmup.
 - <b>`ptt_local_kernel`</b>:  PTT: ``"metropolized_gibbs"``, ``"metropolis"`` or  ``"gibbs"``; ``None`` keeps the kernel used in training.
 - <b>`n_sweeps`</b>:  Sweeps of ordinary sampling, or the mixing-time budget when  ``reference_fasta`` is given.
 - <b>`sampler`</b>:  ``"metropolized_gibbs"``, ``"gibbs"`` or ``"metropolis"``. For PTT,  select the local update with ``ptt_local_kernel`` instead.
 - <b>`beta`</b>:  Inverse temperature (ordinary sampling only).
 - <b>`seed`</b>:  Random seed.
 - <b>`reference_fasta`</b>:  Natural alignment used to measure mixing and to compare  the generated statistics; required for PTT and for ``collect_diagnostics``.
 - <b>`weights_path`</b>:  Optional weights of the reference sequences.
 - <b>`test_fasta`</b>:  Optional held-out alignment (e.g. the validation split). With  ``collect_diagnostics``, nearest-neighbour distances from it to the  reference give the yardstick for the generated sequences' distances to  the training set, and PRIVET compares each generated sequence's  distances to both sets (overfitting check).
 - <b>`privet_window`</b>:  Quantiles ``(q1, q2)`` of the training nearest-neighbour  distances fitted by PRIVET's extreme-value law.
 - <b>`n_measure`</b>:  Largest number of reference sequences used for the comparison.
 - <b>`mixing_multiplier`</b>:  Ordinary sampling: mixing times to run after measuring one.
 - <b>`pseudocount`</b>:  Pseudocount of the reference statistics; ``1/Meff`` if ``None``.
 - <b>`clustering_seqid`</b>:  Identity threshold for reweighting the reference.
 - <b>`no_reweighting`</b>:  Give every reference sequence the same weight.
 - <b>`alphabet`</b>:  Alphabet of a text parameter file; ``None`` reads it from a PTT  archive and assumes ``"protein"`` for text files.
 - <b>`device`</b>:  ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.
 - <b>`dtype`</b>:  ``"float32"``, ``"float64"`` or ``"bfloat16"`` (CUDA/MPS, ordinary  unsteered sampling only); ``None`` means ``"float32"``, or the archive's precision.
 - <b>`steering_potential`</b>:  Optional ``potential(batch, strength)`` added to the  DCA energy, returning one value per sequence (list, NumPy array or  tensor) and ``0`` at strength 0. ``None`` samples the model itself.
 - <b>`steering_strength`</b>:  Non-zero strength ``s`` passed to the potential.
 - <b>`steering_input`</b>:  What the potential receives: ``"sequences"`` (a list of  aligned strings, convenient for external tools) or ``"onehot"`` (a  tensor ``(n, L, q)``, fastest for potentials written in PyTorch).
 - <b>`steering_proposal_steps`</b>:  Site updates per steered move; ``None`` adapts it  during warmup (doubling above 50% acceptance, halving below 20%, at most  16 sweeps' worth) and then keeps it fixed. Small values suit strong or  rugged potentials; larger ones need fewer potential evaluations per sweep.
 - <b>`ptt_steering_acceptance`</b>:  PTT steering: smallest swap acceptance between  adjacent steered rungs; higher values place more rungs.
 - <b>`collect_diagnostics`</b>:  Keep the correlations and PCA projections needed by
 - <b>`:meth`</b>: `SamplingResult.save_diagnostic_plots`.
 - <b>`progress`</b>:  Optional callback receiving :class:`SamplingProgress` events.
 - <b>`is_cancelled`</b>:  Optional callable; returning ``True`` stops sampling.



**Returns:**

 - <b>`A `</b>: class:`SamplingResult` with the sequences, energies and diagnostics.



**Raises:**

 - <b>`InputValidationError`</b>:  If a setting is invalid or incompatible with PTT.
 - <b>`OperationCancelledError`</b>:  If ``is_cancelled`` returns ``True``.



**Example:**
 ``` result = sample_sequences(model="params.dat.gz", n_sequences=1000,```
    ...                           reference_fasta="family.fasta")
    >>> result.to_fasta("samples.fasta")

    Steering towards sequences with a high GC content (RNA), as strings:

    >>> def gc_potential(sequences, strength):
    ...     return [-strength * sum(c in "GC" for c in s) / len(s) for s in sequences]
    >>> steered = sample_sequences(model="params.dat.gz", alphabet="rna", n_sequences=500,
    ...                            steering_potential=gc_potential, steering_strength=20.0)
    >>> steered.steering["effective_sample_size"], steered.log_importance_weights[:3]



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/sampling.py#L755"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `generate_sequences`

```python
generate_sequences(**kwargs: 'Any') → tuple[str, ]
```

Return only the sequences of :func:`sample_sequences`.



**Args:**

 - <b>`**kwargs`</b>:  Keyword arguments of :func:`sample_sequences`; ``model`` and  ``n_sequences`` are required.



**Returns:**
 The generated sequences.



**Example:**
 ``` generate_sequences(model="params.dat.gz", n_sequences=10, n_sweeps=500)```





---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
