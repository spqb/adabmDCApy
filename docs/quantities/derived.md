# Normalization, scoring and importance weights

## Normalization and entropy

The log-partition function \(\log Z_\theta = \log\sum_x e^{-E_\theta(x)}\) turns energies into probabilities, \(\log p_\theta(x) = -E_\theta(x) - \log Z_\theta\). Summing over \(q^L\) sequences is impossible, so it is estimated, by the PTT ladder or by thermodynamic integration. With \(\log Z\), the **entropy** in nats per sequence is

\[
S[p_\theta] = -\sum_x p_\theta(x)\log p_\theta(x) = \langle E_\theta\rangle_{p_\theta} + \log Z_\theta ,
\]

with the mean energy estimated from equilibrium samples. The free energy reported alongside is \(F = -\log Z\).

### PTT ladder

The bottom rung is the independent-site profile, with fields \(h^{(0)}_i(a)\) and no couplings. Its partition function factorizes and is exact:

\[
\log Z_0 = \sum_i \log\sum_a e^{h^{(0)}_i(a)} .
\]

For neighbouring rungs with \(W_k(x) = E_{k+1}(x) - E_k(x)\) and \(\Delta F_k = -\log(Z_{k+1}/Z_k)\), the identity \(Z_{k+1}/Z_k = \langle e^{-W_k}\rangle_{p_k}\) gives the **forward** estimate from \(N\) samples \(x_n\) of the lower rung,

\[
\widehat{\Delta F}_{k,\rm fwd} = -\log\Big(\frac1N\sum_{n=1}^N e^{-W_k(x_n)}\Big).
\]

**Bennett's acceptance ratio** (BAR) uses both populations, lower samples \(x_n\) and upper samples \(y_n\) (equal numbers). With \(\sigma(z) = 1/(1+e^{-z})\), \(\widehat{\Delta F}_{k,\rm BAR}\) solves

\[
\sum_{n=1}^{N}\sigma\big(\widehat{\Delta F}_{k,\rm BAR} - W_k(x_n)\big) = \sum_{n=1}^{N}\sigma\big(W_k(y_n) - \widehat{\Delta F}_{k,\rm BAR}\big),
\]

found by bisection. It is the minimum-variance estimator given both populations, and it stays reliable when the forward weights are heavy-tailed: on a 307-residue protein ladder, one pair gave \(0.34 \pm 0.68\) nats forward against \(1.36 \pm 0.03\) with BAR. Summing along the ladder,

\[
\widehat{\log Z}_K = \log Z_0 - \sum_{k=0}^{K-1}\widehat{\Delta F}_k .
\]

Where each estimator is used:

| Where | Ladder | Estimator |
| --- | --- | --- |
| Training history (`logz`, `ll_*`, `entropy`) | Active training ladder, anchored on the profile or, after compression, on the accumulated value of the discarded links | Forward |
| `entropy -p ptt.h5` | Active training ladder from the archive, after a short re-equilibration | BAR |
| `sample` (`logs/ptt.log`, ladder health, plot title) | Full sampling ladder from the exact profile | BAR (forward also reported as `log_z_forward`) |

The anchor is exact, but the bridges are estimates: poor overlap or incomplete mixing biases \(\log Z\). The forward, reverse and BAR values of every link, with bootstrap errors, are in `logs/ptt_ladder_health.log`.

### Thermodynamic integration

For a model without a ladder, add a field toward a target sequence \(x^\star\):

\[
E_\vartheta(x) = E_0(x) - \vartheta\, I(x, x^\star),
\qquad
I(x, x^\star) = \sum_i \mathbf 1[x_i = x^\star_i],
\qquad
F_\vartheta = -\log Z_\vartheta .
\]

At a large \(\vartheta_{\max}\) the target is sampled with measurable probability \(P(x^\star)\) (the field is raised until about 10% of the chains sit on it), and

\[
F_{\vartheta_{\max}} = E_{\vartheta_{\max}}(x^\star) + \log P_{\vartheta_{\max}}(x^\star).
\]

Since \(dF_\vartheta/d\vartheta = -\langle I(x, x^\star)\rangle_\vartheta\),

\[
F_0 = F_{\vartheta_{\max}} + \int_0^{\vartheta_{\max}} \langle I(x,x^\star)\rangle_\vartheta\, d\vartheta ,
\qquad
\log Z_0 = -F_0 ,
\]

with the integral computed by the trapezoidal rule over `--nsteps` values of \(\vartheta\) and MCMC averages at each. The estimate is only as good as the equilibration at each \(\vartheta\): for models whose chains do not mix it can be wrong by orders of magnitude. On six families where both were available, it agreed with the PTT bridge within 0.015 nats per site.

## Context-dependent entropy

For a sequence \(x\) and position \(i\), with conditional probabilities \(p(a \mid x_{-i})\) of the same model,

\[
\mathrm{CDE}_i(x) = -\sum_a p(a\mid x_{-i})\log p(a\mid x_{-i}),
\qquad
\mathrm{CDE}_{\rm sum}(x) = \sum_i \mathrm{CDE}_i(x),
\]

in nats. It measures how many residues the model accepts at each position in the context of the rest of the sequence. It is a local descriptor, not the entropy of the model. With a coefficient \(\lambda\), the **local free energy** is \(E(x) - \lambda\,\mathrm{CDE}_{\rm sum}(x)\). Each sampling run fits \(E = c + \lambda\,\mathrm{CDE}_{\rm sum}\) by least squares on its generated sequences and reports \(\lambda\), \(c\) and \(R^2\) (`local_lambda_fit`): a property of the model at the sampled temperature, not a universal constant.

## Contact scores

The couplings are first transformed to the zero-sum gauge, then for each pair of positions

\[
F_{ij} = \sqrt{\sum_{a,b \neq \text{gap}} J_{ij}(a,b)^2},
\qquad
F^{\rm APC}_{ij} = F_{ij} - \frac{\sum_k F_{ik}\,\sum_k F_{kj}}{\sum_{k,l} F_{kl}},
\]

with the diagonal set to zero. The average product correction (Dunn et al., 2008; Ekeberg et al., 2013) removes the part of \(F_{ij}\) explained by the overall coupling strength of \(i\) and of \(j\). Scores rank pairs; they are not probabilities. Without a model, a mean-field estimate of the couplings is used: \(J \approx -C^{-1}\), the negative inverse of the connected-correlation matrix of the pseudocounted data with the gap as reference state (pseudocount 0.5 by default).

## Mutation scores

For a wild type \(x\) and its single mutant \(x'\) at position \(i\) (state \(a \to b\)),

\[
\Delta E = E(x') - E(x) = -\big[h_i(b) - h_i(a)\big] - \sum_{j\neq i}\big[J_{ij}(b, x_j) - J_{ij}(a, x_j)\big],
\]

and \(\log[p(x')/p(x)] = -\Delta E\): the partition function cancels. The sum over \(j\) makes the score depend on the background sequence. Interpreting \(\Delta E\) as fitness requires calibration on experimental data.

## Steering and importance weights

A steering potential \(V(x, s)\) with \(V(x, 0) = 0\) defines the target

\[
p_s(x) = \frac{e^{-\beta E(x) - V(x,s)}}{Z_s}.
\]

Each move proposes a block of Gibbs updates under the unsteered model, which leave \(p_0\) invariant, and accepts it with probability \(\min\{1, e^{-\Delta V}\}\); the result satisfies detailed balance with respect to \(p_s\). With PTT, rungs of increasing \(s\) are added above the endpoint.

Averages under the original model follow from importance weights \(w(x) = p_0(x)/p_s(x) = e^{V(x,s)}\,Z_s/Z_0\):

\[
\langle g\rangle_{p_0} \approx \frac{\sum_n e^{V(x_n,s)}\,g(x_n)}{\sum_n e^{V(x_n,s)}} .
\]

`log_importance_weights` holds \(V(x_n, s) + \log(Z_s/Z_0)\) when PTT estimates the ratio (reported as `steering["log_z_ratio"]`), and otherwise \(V\) shifted so that the weights average to 1; the self-normalized average is the same. `steering["effective_sample_size"]` is \((\sum_n w_n)^2/\sum_n w_n^2\), in number of sequences. It measures the overlap between \(p_s\) and \(p_0\), not mixing or model quality; when it falls to a few sequences, the corrected averages rest on those few.
