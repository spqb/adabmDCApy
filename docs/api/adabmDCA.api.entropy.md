<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/entropy.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.entropy`
High-level thermodynamic-integration entropy estimation.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/entropy.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `estimate_entropy`

```python
estimate_entropy(
    model: 'str | Path',
    natural_alignment: 'AlignmentInput',
    target_alignment: 'AlignmentInput',
    initial_chains_path: 'str | Path | None' = None,
    n_chains: 'int' = 10000,
    n_sweeps: 'int' = 100,
    n_steps: 'int' = 100,
    theta_max: 'float' = 5.0,
    theta_sweeps: 'int' = 100,
    zero_sweeps: 'int' = 100,
    target_fraction: 'float' = 0.1,
    max_theta_iterations: 'int' = 10000,
    sampler: 'str' = 'metropolized_gibbs',
    alphabet: 'str | None' = None,
    seed: 'int' = 0,
    device: 'str' = 'auto',
    dtype: 'str' = 'float32',
    output_dir: 'str | Path | None' = None,
    label: 'str' = 'entropy',
    progress: 'EntropyProgress | None' = None,
    is_cancelled: 'Callable[[], bool] | None' = None
) → ThermodynamicIntegrationResult
```

Estimate the entropy of a DCA model by thermodynamic integration.

A field ``theta * x_target`` biases the model towards a target sequence. ``theta_max`` is first increased (by 1% per 100 sweeps) until more than ``target_fraction`` of the chains coincide with the target, which fixes the free energy at ``theta_max``. The free energy at ``theta = 0`` then follows by integrating the mean sequence identity over ``n_steps`` values of ``theta`` (trapezoidal rule), and the entropy is ``<E> - F``. For PTT archives, :func:`adabmDCA.api.ptt.estimate_ptt_entropy` is cheaper and needs no integration.



**Args:**

 - <b>`model`</b>:  Path to a parameter file or PTT archive.
 - <b>`natural_alignment`</b>:  Natural alignment, checked for compatibility with the model.
 - <b>`target_alignment`</b>:  Alignment whose first valid sequence is the target; a  warning is emitted if it contains more.
 - <b>`initial_chains_path`</b>:  Optional FASTA of starting chains, resampled to  ``n_chains`` if needed; random chains otherwise.
 - <b>`n_chains`</b>:  Number of Markov chains.
 - <b>`n_sweeps`</b>:  Sweeps at each integration step.
 - <b>`n_steps`</b>:  Integration points between 0 and ``theta_max`` (at least 2).
 - <b>`theta_max`</b>:  Initial largest bias strength; increased if needed.
 - <b>`theta_sweeps`</b>:  Sweeps to equilibrate the chains at the initial ``theta_max``.
 - <b>`zero_sweeps`</b>:  Sweeps to estimate the unbiased mean energy ``<E>``.
 - <b>`target_fraction`</b>:  Fraction of chains, in ``(0, 1)``, that must reach the  target at ``theta_max``.
 - <b>`max_theta_iterations`</b>:  Largest number of ``theta_max`` increases.
 - <b>`sampler`</b>:  ``"metropolized_gibbs"``, ``"gibbs"`` or ``"metropolis"``.
 - <b>`alphabet`</b>:  Alphabet of a text parameter file; ``None`` reads it from a  PTT archive and assumes ``"protein"`` for text files.
 - <b>`seed`</b>:  Random seed.
 - <b>`device`</b>:  ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.
 - <b>`dtype`</b>:  ``"float32"`` or ``"float64"``.
 - <b>`output_dir`</b>:  If given, the result files are written there.
 - <b>`label`</b>:  Prefix of the files written to ``output_dir``.
 - <b>`progress`</b>:  Optional callback receiving
 - <b>`:class`</b>: `ThermodynamicIntegrationProgress` updates.
 - <b>`is_cancelled`</b>:  Optional callable; returning ``True`` stops the run.



**Returns:**

 - <b>`A `</b>: class:`ThermodynamicIntegrationResult` with the entropy (nats), the free energy and the integration history.



**Raises:**

 - <b>`InputValidationError`</b>:  If a numerical setting is out of range.
 - <b>`ModelCompatibilityError`</b>:  If the initial chains do not match the model.
 - <b>`ConvergenceError`</b>:  If ``target_fraction`` is not reached within  ``max_theta_iterations``.
 - <b>`OperationCancelledError`</b>:  If ``is_cancelled`` returns ``True``.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
