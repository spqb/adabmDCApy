---
name: adabmdca-local-free-energy
description: Rank or compare sequences under an adabmDCA Potts model with the local free energy F = E - lambda * CDE, which corrects the energy with the context-dependent (local) entropy, and fit lambda on equilibrium samples of the model. Use when asked to score, rank, filter or compare natural, designed, mutant or generated sequences with a DCA model beyond the plain energy, or when asked what lambda / --local-lambda / CDE / local free energy are and how to get them.
---

# Local free energy with adabmDCA

The energy E(x) rewards sequences at the bottom of narrow minima. The **local free energy**

    F(x) = E(x) - lambda * CDE_sum(x),    CDE_sum(x) = sum_i H[ p(a_i | x_-i) ]

also rewards sequences with a broad neighbourhood of acceptable variants: CDE_i is the entropy (nats) of the model's conditional distribution at site i given the rest of the sequence. Lower F is better, as for E. For training, sampling and the CLI basics, use the `adabmdca` skill.

## lambda is fitted, not chosen

lambda is the slope of E against CDE_sum **on equilibrium samples of the same model**:

    E = c + lambda * CDE_sum    (ordinary least squares)

It is a property of the model at the sampled temperature, not a universal constant (about 1.4 on RF00379, R² 0.55). Never reuse a lambda from another model, and never fit it on natural sequences: they are not at the model's equilibrium (training sequences sit lower in energy than samples).

**Every sampling run fits it.** It is printed at the end of `adabmDCA sample` and stored in `sampling.json`:

```bash
adabmDCA sample -p model/ptt.h5 -d train.fasta -o samples          # PTT whenever ptt.h5 exists
python -c "import json; print(json.load(open('samples/sampling.json'))['data']['local_lambda_fit'])"
# {'lambda': 1.397, 'intercept': -50.03, 'r_squared': 0.546, 'n_samples': 2000}
```

`energy_vs_cde.png` (with `--plot-sampling-logs`) shows the fit. From Python: `sample_sequences(...).local_lambda_fit`.

**From samples you already have** (`samples.csv` has `energy` and `cde_sum`):

```python
import pandas as pd
from adabmDCA.api.scoring import fit_local_lambda

s = pd.read_csv("samples/samples.csv")
fit = fit_local_lambda(s.energy.values, s.cde_sum.values)   # lambda, intercept, r_squared, n_samples
```

For samples without `cde_sum` (an older FASTA), score them first with `adabmDCA energies -d samples.fasta -p model/params.dat.gz --local-lambda 0 -o tmp`, which writes `energy` and `cde_sum`.

Before using the fit, check:

- **Sampling diagnostics support convergence**: `sampling.json` → `ptt_diagnostics.converged` true (PTT), or a mixing time found (plain MCMC). Passing these checks is evidence, not proof, of equilibrium. If they fail, treat the fit as provisional and do not present it as an equilibrium correction.
- **Enough samples**: the default number (as many as training chains) is fine; below a few hundred the slope is noisy.
- **R²**: it says how much of the energy variation the CDE explains. A low R² means CDE explains little of the sampled energy variation; report it next to lambda. The heuristic ~0.3 is not a validated cutoff for ranking utility, which requires evaluation on the relevant task.
- **Temperature**: samples at `beta ≠ 1` or from a steered run give the lambda of that distribution, not of the model at beta = 1.

## Score sequences

```bash
adabmDCA energies -d candidates.fasta -p model/params.dat.gz --local-lambda 1.397 -o scores
# scores/candidates_energies.csv: name, sequence, energy, cde_sum, local_free_energy
```

```python
from adabmDCA import score_sequences

result = score_sequences(fasta_path="candidates.fasta", model="model/params.dat.gz", alphabet="rna",
                         local_lambda=fit["lambda"])
df = result.to_dataframe().sort_values("local_free_energy")   # best first
```

- **Pass the fitted lambda.** Without `local_lambda` (Python) or `--local-lambda` (CLI) only the energy is computed, with no CDE and no F. Versions before this change defaulted to `local_lambda=1.0` in Python; with them, always pass it.
- Sequences must be aligned to the model (same length and alphabet; pass `alphabet=` for RNA without U, or custom tokens).
- Compare F only **within one model**, like E. Dividing by L does not make scores from different models, lengths or families comparable: model energies can have arbitrary additive offsets. Cross-model comparisons need a separately justified calibration.
- Report E, CDE_sum and F together: a sequence can improve F by gaining local entropy while its energy gets worse, and the user should see which.

## What the correction does in practice

On RF00379 (lambda 1.40 fitted on 2,000 converged PTT samples), scoring the 3,060 training sequences and 4,000 random mutants of four wild types:

- CDE correlates with the energy of training sequences (Pearson 0.78), so F reorders sequences rather than adding noise.
- 84% of 5-substitution mutants and 99.9% of 10-substitution mutants scored above the 95th percentile of the training F, against 61% and 97% with an exact radius-one neighbourhood correction.

These are separations from random perturbations, not fitness predictions: Hamming distance is not an experimental label. When experimental data exist, compare E and F against them (Spearman correlation) before recommending one.

Definitions: https://spqb.github.io/adabmDCApy/quantities/derived/#context-dependent-entropy and https://spqb.github.io/adabmDCApy/usage/energies/#context-dependent-entropy-and-local-free-energy
