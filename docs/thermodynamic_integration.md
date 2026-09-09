# <span id="thermodynamic-integration">Entropy from Thermodynamic Integration</span>

The `entropy` command estimates the entropy of a trained DCA model without enumerating all possible sequences. It uses a thermodynamic integration path that progressively biases the model toward one target sequence.

```bash
adabmDCA entropy \
  -p <params.dat> \
  -d <training_msa.fasta> \
  -t <target_sequence.fasta> \
  -o <output_folder>
```

The target file (`-t` / `--path_targetseq`) may contain multiple sequences.
Entropy estimation uses the first valid sequence in input order and emits a
warning if additional valid sequences are present, including duplicates.
Sequences with invalid alphabet symbols are dropped; an input with no valid
sequences still raises an error.

## Biased Model

Starting from the trained model with energy \(E_0(x)\), the script defines a one-parameter family of models

\[
E_\theta(x) = E_0(x) - \theta \, I(x, x^\star),
\]

where \(x^\star\) is the target sequence, typically chosen from the training set, and \(I(x, x^\star)\) is the number of matching residues between \(x\) and \(x^\star\), from \(0\) to \(L\). In the implementation this is done by adding `theta * targetseq` to the local bias parameters.

Increasing \(\theta\) makes the target sequence more likely. The script starts from the user-provided `--theta_max`, then increases it until the target sequence is sampled with measurable probability, approximately 10% of the chains in the implementation.

## Free Energy Estimate

The script works with

\[
F(\theta) = -\log Z_\theta.
\]

At large \(\theta_{\max}\), the target sequence has measurable probability \(p_\star\). Since

\[
p_\star = \frac{\exp[-E_{\theta_{\max}}(x^\star)]}{Z_{\theta_{\max}}},
\]

the script estimates

\[
F(\theta_{\max})
= \log p_\star + E_{\theta_{\max}}(x^\star).
\]

Then it integrates back to \(\theta = 0\). Because

\[
\frac{dF}{d\theta} = -\langle I(x, x^\star) \rangle_\theta,
\]

we have

\[
F(0) = F(\theta_{\max}) +
\int_0^{\theta_{\max}}
\langle I(x, x^\star) \rangle_\theta \, d\theta.
\]

The integral is approximated with the trapezoidal rule over `--nsteps` values of \(\theta\). At each value, chains are sampled for `--nsweeps` sweeps and the mean identity to the target sequence is measured.

## Entropy

After estimating \(F(0) = -\log Z_0\), the entropy is computed as

\[
S = \langle E_0(x) \rangle_0 - F(0)
  = \langle E_0(x) \rangle_0 + \log Z_0.
\]

The average energy \(\langle E_0 \rangle_0\) is estimated by thermalizing chains at \(\theta = 0\) for `--nsweeps_zero` sweeps.

Equivalently, the final estimator is

\[
S =
\langle E_0 \rangle_0 -
\left[
\log p_\star + E_{\theta_{\max}}(x^\star)
+ \int_0^{\theta_{\max}}
\langle I(x, x^\star) \rangle_\theta \, d\theta
\right].
\]

## Output

The command writes a log file named `<label>.log` in the output folder. By default, `label` is `entropy`, so the file is:

```text
<output_folder>/entropy.log
```

Each row reports:

- `Epoch`: integration step index
- `Theta`: current field strength
- `Free Energy`: running estimate of \(F(\theta_{\max}) + \int_0^\theta \langle I(x, x^\star) \rangle_{\theta'} d\theta'\); the final row gives \(F(0)\)
- `Entropy`: current entropy estimate
- `Time`: elapsed time in seconds

## Main Parameters

- `--theta_max`: initial maximum field strength. It is automatically increased until the target sequence appears in at least 10% of chains.
- `--nsteps`: number of integration points.
- `--nsweeps`: MCMC sweeps per integration point.
- `--nsweeps_zero`: sweeps used to estimate \(\langle E_0 \rangle_0\).
- `--nsweeps_theta`: sweeps used to equilibrate at the initial \(\theta_{\max}\).
- `--nchains`: number of chains used in the estimates.

More chains, more integration steps, and more sweeps reduce Monte Carlo noise, at the cost of longer runtime.
