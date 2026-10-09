# Estimate entropy · `entropy`

The entropy of a Potts model measures how many sequences it effectively generates: \(e^{S}\) is the size of the sequence space the model spreads its probability over. Computing it requires the normalization \(\log Z\) of the model, a sum over all \(q^L\) sequences that cannot be done directly. adabmDCA estimates it in two ways.

## From a PTT archive (default)

```bash
adabmDCA entropy -p model/ptt.h5 -o entropy
```

PTT's ladder starts at an independent-site profile whose \(\log Z\) is known exactly, and every pair of neighbouring models overlaps well enough to measure their free-energy difference. Adding these differences gives \(\log Z\) of the final model. The command runs the archived ladder for a few exchange rounds (`--nsweeps` local sweeps per round, default 100), re-estimates every difference with Bennett's acceptance ratio, and returns

\[
S = \langle E \rangle + \log Z ,
\]

with \(\langle E\rangle\) the mean energy of the endpoint chains. It needs no alignment and takes seconds to minutes.

Output: `entropy.json` (or `<label>.json`) with `entropy`, `log_z`, `mean_energy`, `free_energy` (\(= -\log Z\)) and the provenance of the archive.

## From any model, by thermodynamic integration

```bash
adabmDCA entropy --strategy pcd -p model/params.dat.gz \
    -d train.fasta -t target.fasta -o entropy_ti
```

For models without an archive (PCD-trained, sparse PCD, reintegrated), the command adds a field that rewards similarity to a target sequence and raises it until the chains sit on the target about 10% of the time. There, \(\log Z\) follows from the observed target probability. The field is then lowered back to zero in `--nsteps` steps while the mean identity to the target is measured, and the integral of that curve gives the change of \(\log Z\). Any sequence of the family can serve as target (`-t`, first valid sequence used).

| Option | Default | Meaning |
| --- | --- | --- |
| `-d` | required | Natural alignment, for the alphabet and initial chains |
| `-t`, `--path_targetseq` | required | FASTA whose first sequence is the target |
| `--nchains` | 10000 | Chains used at every step |
| `--nsteps` | 100 | Integration points |
| `--nsweeps` | 100 | Sweeps per integration point |
| `--theta_max` | 5 | Initial maximum field, raised automatically if the target is too rare |
| `--nsweeps_theta`, `--nsweeps_zero` | 100 | Equilibration at the maximum field and at zero field |

Outputs: `entropy.log`, `entropy.csv` (the integration path: step, field, running free energy and entropy, time) and `entropy.json` (final entropy, free energy, chosen `theta_max`, observed target fraction).

## How accurate are the estimates?

On six benchmark families, \(\log Z\) from the PTT bridge and from thermodynamic integration agreed within 0.015 nats per site, the size of a typical difference between two good models. Two caveats:

- **Thermodynamic integration requires chains that mix.** For models whose chains do not equilibrate, it returns meaningless values: on sparse models that failed to mix, it gave \(\log Z\) of \(10^2\)–\(10^4\) and absurd likelihoods. Check mixing with [`sample`](sample.md) before trusting an integration result.
- **The PTT estimate is only as good as the ladder.** Poor overlap between neighbouring models makes the bridge noisy. `sample` with `--plot` reports the forward, reverse and BAR estimates of every link with bootstrap errors in `ptt_ladder_health.log`; large disagreements flag a weak link.

Both values are in nats **per sequence**. Training logs report log-likelihoods **per site** (divided by \(L\)).

## Related quantities

- PTT training records the entropy and \(\log Z\) after every update (`history.csv`: `entropy`, `logz`), using the faster one-sided bridge. Use them to follow trends, and the `entropy` command for a final value.
- PTT sampling with a reference alignment writes `logs/ptt.log`, with \(\log Z\), mean energy, entropy and data log-likelihood for **every model of the ladder**, from the profile to the final model.

## In Python

```python
from adabmDCA import estimate_ptt_entropy, estimate_entropy

ptt = estimate_ptt_entropy(model="model/ptt.h5", n_sweeps=10, device="cuda")
print(ptt.entropy, ptt.log_z)

ti = estimate_entropy(model="model_pcd/params.dat.gz", natural_alignment="train.fasta",
                      target_alignment="target.fasta", n_steps=100, alphabet="rna")
```

Mathematics: [normalization and entropy](../quantities/derived.md#normalization-and-entropy).
