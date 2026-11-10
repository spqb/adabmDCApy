<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/config.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.ptt.config`
Configuration for PTT snapshot retention, reservoirs and mixing analysis.

**Global Variables**
---------------
- **DEFAULT_PTT_N_SWEEPS**
- **DEFAULT_PTT_N_CHAINS**
- **DEFAULT_PTT_SAMPLER**
- **DEFAULT_PTT_MAX_EPOCHS**
- **PTT_OPTIMIZERS**
- **DEFAULT_TRUST_RADIUS**
- **NGD_TRUST_RADIUS**
- **NGD_DAMPING**
- **NGD_CENTRED_DAMPING**
- **NGD_VALIDATION_ARC_WINDOW**
- **PTT_ACTIVATIONS**


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/ptt/config.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PTTConfig`
Settings of Parallel Trajectory Tempering, for training and sampling.

Pass an instance as `ptt=` to `train_model` (or through `TrainingConfig(ptt=...)`) to train with PTT. The defaults suit most families; see `docs/ptt.md` for the method. Round counts are exchange rounds: one pass of replica exchanges followed by the local sweeps.



**Attributes:**

 - <b>`swaps`</b>:  Exchange-round factor: each update runs  `int(swaps * sqrt(active replicas))` rounds.
 - <b>`target_acceptance`</b>:  Swap acceptance below which the current endpoint is  kept as a new snapshot in the ladder.
 - <b>`min_acceptance`</b>:  Swap acceptance below which a mixing check is run.
 - <b>`max_replicas`</b>:  Active ladder size above which the reservoir is refreshed  and the ladder compressed.
 - <b>`target_replicas`</b>:  Active replicas kept after compression.
 - <b>`reservoir_size`</b>:  Size of the reservoir population; `None` means  `10 * n_chains`. Must be at least `n_chains`.
 - <b>`full_sampler`</b>:  Use the full historical ladder when collecting a new reservoir.
 - <b>`equilibration_rounds`</b>:  Extra rounds after a snapshot is inserted or replaced.
 - <b>`initialization_rounds`</b>:  Warmup rounds before the first update.
 - <b>`mixing_chains`</b>:  Chains per replica in the autocorrelation mixing experiment.
 - <b>`mixing_initial_rounds`</b>:  Initial trajectory length of that experiment.
 - <b>`mixing_thermalization_rounds`</b>:  Warmup before a mixing measurement and after a recovery.
 - <b>`mixing_max_rounds`</b>:  Round budget of one mixing check; exceeding it is a  mixing failure, which triggers recovery.
 - <b>`mixing_window_factor`</b>:  Required trajectory length relative to the longest  correlation time (autocorrelation method).
 - <b>`mixing_method`</b>:  Training mixing check: `"renewal"` (the ladder is  repopulated by fresh configurations) or `"autocorrelation"`.
 - <b>`renewal_tolerance`</b>:  Fraction of old configurations, per replica, below which  the ladder counts as renewed.
 - <b>`max_recoveries`</b>:  Largest number of learning-rate halvings after mixing failures.
 - <b>`min_learning_rate`</b>:  Learning-rate floor of recovery.
 - <b>`optimizer`</b>:  `"adaptive"` (steps bounded by a KL trust region, training  paused while the endpoint chains lag behind the model) or `"sgd"`  (fixed learning rate).
 - <b>`trust_radius`</b>:  Adaptive optimizer: largest predicted KL divergence of one update;  `None` means 0.01, or 0.001 with `natural_gradient` (whose steps fill the radius).
 - <b>`centering`</b>:  Update with the centering trick: connected-correlation coupling  gradient, field step corrected for the centring offset. Removes the one-site  part of the coupling step, which otherwise repeats the field step and makes  the trust region stop the fields.
 - <b>`natural_gradient`</b>:  Step along the damped natural gradient (F + damping)^-1 g, with F the  Fisher matrix of the model estimated on the endpoint chains and solved by conjugate  gradient. The undamped natural gradient does not depend on the parametrization; with  `centering` the system is solved in the centred coordinates, so that large damping  tends to the centred gradient (without it, to the raw gradient). The learning rate is
 - <b>`then a fraction of the Newton step`</b>:  `--lr 1` lets the trust region alone limit the steps. The system is solved with the Fisher matrix of half of the endpoint chains and the KL of
 - <b>`the trust region measured on the other half`</b>:  F has at most as many directions as there are chains, and the solve puts the step where those chains vary least. Measured on the same chains, the KL of the step was 0.2 % of its KL on independent chains (LBD).
 - <b>`ngd_damping`</b>:  Natural gradient: damping, in units of the mean eigenvalue of F; `None`  means 50 with `centering`, 0.5 without.
 - <b>`ngd_cg_steps`</b>:  Natural gradient: largest number of conjugate-gradient iterations.
 - <b>`ngd_cg_tolerance`</b>:  Natural gradient: relative residual at which conjugate gradient stops.
 - <b>`lag_tolerance`</b>:  Adaptive optimizer: largest lag of the endpoint chains, in  population standard deviations, before training pauses.
 - <b>`lag_horizon`</b>:  Adaptive optimizer: memory of past updates, in updates.
 - <b>`lag_pause_rounds`</b>:  Adaptive optimizer: longest pause, in rounds.
 - <b>`validation_stop`</b>:  Stop when the validation log-likelihood plateaus instead  of at the target Pearson; requires a validation alignment.
 - <b>`validation_window`</b>:  Updates per window of the plateau test, which compares  the medians of the last two windows, when the windows are counted in updates  (without arc-length windows).
 - <b>`validation_min_gain`</b>:  Stop when the median gain between the two windows  is below this value (log-likelihood per site).
 - <b>`validation_arc_window`</b>:  Windows of the plateau test in Fisher arc length: each spans this  length of the path, the sum of sqrt(2 KL) over its updates (KL the predicted KL  divergence of the update), so the test sees the same change of the model whatever  the step size. `None` means 1 with `natural_gradient` and windows of  `validation_window` updates otherwise, as 0 always does; the `sgd` optimizer always  counts updates (it predicts no KL).
 - <b>`validation_arc_min_updates`</b>:  Arc-length windows: fewest updates per window, against the noise  of the log Z estimate; a window grows to this many updates when the steps are large.
 - <b>`activation`</b>:  eaDCA: `"fixed"` activates `activation_fraction` of the  inactive couplings per block; `"adaptive"` activates at most that  fraction, only significant candidates within a KL budget.
 - <b>`activation_kl_share`</b>:  Adaptive activation: share of `trust_radius` for the  first update of the new couplings; halved by each recovery.
 - <b>`activation_significance`</b>:  Adaptive activation: standard errors by which a  candidate's pair frequency must differ from the data; 0 disables the test.



**Example:**
```python
from adabmDCA import PTTConfig, train_model
train_model("family.fasta", alphabet="rna", ptt=PTTConfig(validation_stop=True),
            validation_path="validation.fasta", output_dir="model")
```


<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    swaps: 'int' = 1,
    target_acceptance: 'float' = 0.25,
    min_acceptance: 'float' = 0.1,
    max_replicas: 'int' = 2,
    target_replicas: 'int' = 2,
    reservoir_size: 'int | None' = None,
    full_sampler: 'bool' = False,
    equilibration_rounds: 'int' = 10,
    initialization_rounds: 'int' = 100,
    mixing_chains: 'int' = 100,
    mixing_initial_rounds: 'int' = 100,
    mixing_thermalization_rounds: 'int' = 1000,
    mixing_max_rounds: 'int' = 20000,
    mixing_window_factor: 'float' = 20.0,
    mixing_method: 'str' = 'renewal',
    renewal_tolerance: 'float' = 0.01,
    max_recoveries: 'int' = 3,
    min_learning_rate: 'float' = 1e-08,
    optimizer: 'str' = 'adaptive',
    trust_radius: 'float | None' = None,
    centering: 'bool' = True,
    natural_gradient: 'bool' = False,
    ngd_damping: 'float | None' = None,
    ngd_cg_steps: 'int' = 20,
    ngd_cg_tolerance: 'float' = 0.001,
    lag_tolerance: 'float' = 0.25,
    lag_horizon: 'int' = 200,
    lag_pause_rounds: 'int' = 100,
    validation_stop: 'bool' = False,
    validation_window: 'int' = 200,
    validation_min_gain: 'float' = 0.0,
    validation_arc_window: 'float | None' = None,
    validation_arc_min_updates: 'int' = 50,
    activation: 'str' = 'fixed',
    activation_kl_share: 'float' = 0.5,
    activation_significance: 'float' = 3.0
) → None
```






---

#### <kbd>property</kbd> effective_ngd_damping

The natural-gradient damping in use: `ngd_damping`, or its default for the coordinates.

---

#### <kbd>property</kbd> effective_trust_radius

The trust radius in use: `trust_radius`, or its default for the chosen update.

---

#### <kbd>property</kbd> effective_validation_arc_window

The arc length of the plateau-test windows, or `None` when they are counted in updates.






---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
