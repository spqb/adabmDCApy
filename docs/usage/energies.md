# Score sequences · `energies`

Compute the statistical energy of aligned sequences under a trained model:

```bash
adabmDCA energies -d sequences.fasta -p model/params.dat.gz -o energies
```

The energy \(E(x)\) of a sequence is minus the sum of its fields and pairwise couplings. The model assigns probability \(p(x) \propto e^{-E(x)}\), so **lower energy means more probable**: a difference of 1 in energy is a factor *e* ≈ 2.7 in probability.

## Inputs

| Option | Meaning |
| --- | --- |
| `-d`, `--data` | Sequences to score, aligned to the training alignment (same length, same alphabet) |
| `-p`, `--path_params` | `params.dat(.gz)` or `ptt.h5` (its final model is used) |
| `-o`, `--output` | Output folder |
| `--local-lambda` | Optional. Also compute a *local free energy* \(E - \lambda\,\mathrm{CDE}\) with this coefficient |
| `--alphabet`, `--device`, `--dtype` | As usual |

## Outputs

Files named after the input, for example `sequences_energies.fasta`, `.csv` and `.json`. The FASTA headers carry the energy; the CSV has one row per sequence with `energy`, `cde_sum` (summed context-dependent entropy) and, with `--local-lambda`, `local_free_energy`.

## Interpreting energies

- **Compare sequences under the same model.** Energy differences are log-probability ratios within one model. Absolute energies of different models are not comparable: each model has its own normalization \(\log Z\). To compare models, use log-likelihoods, which include \(\log Z\) (PTT reports them; see [Entropy](entropy.md)).
- **Training sequences score lower than samples.** A fitted model gives its training sequences lower energy than its own typical samples: on RF00379, generated sequences have mean energy about 15 above the training data; on bkace, about 80. This is expected and is not a sign of a bad model. Held-out natural sequences usually sit in between.
- **Energies scale with length.** For comparisons across families, divide by the sequence length.
- **Sequences with many gaps** are scored like any other; whether gaps are favoured depends on how gappy the training alignment was.

## Context-dependent entropy and local free energy

For each position, the model defines the probability of every residue given the rest of the sequence. The entropy of that conditional distribution, the *context-dependent entropy* (CDE), measures how many residues the model would accept at that position in this context. `cde_sum` adds it over positions: sequences in flexible regions of sequence space have a high CDE sum, sequences whose every position is tightly constrained have a low one.

Energy alone favours sequences at the bottom of narrow minima. The local free energy \(E - \lambda\,\mathrm{CDE}_{\rm sum}\) also rewards a broad neighbourhood of acceptable variants. Every sampling run fits \(E = a + \lambda\,\mathrm{CDE}_{\rm sum}\) on its generated sequences and reports \(\lambda\) and \(R^2\) (`energy_vs_cde.png`, `sampling.json` → `local_lambda_fit`); that fitted value is a sensible choice for `--local-lambda`. On RF00379 it was about 1.4, with \(R^2 \approx 0.55\). It is a property of the model and the temperature, not a universal constant.

## In Python

```python
from adabmDCA import load_model

model = load_model("model/params.dat.gz", alphabet="protein")
energies = model.compute_energies(["ACDEFGHIK", "ACDEYGHIK"])        # NumPy array
result = model.score_sequences(["ACDEFGHIK", "ACDEYGHIK"], local_lambda=1.4)
result.to_dataframe()                                                 # energy, cde_sum, local_free_energy
```

Definitions: [energy](../quantities/model.md#potts-energy-and-probability) and [CDE](../quantities/derived.md#context-dependent-entropy).
