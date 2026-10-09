# Training quantities

The columns of `history.csv` and the curves of `plot-training-log`. [Training](../algorithms/training.md#reading-the-training-plots) shows them on a real run.

## Pearson and slope

Flatten the connected correlations of the data and of the endpoint chains, over all entries \((a,b)\) of all blocks \(i > j\) (active couplings only for sparse models), into vectors \(d\) and \(m\). Then

\[
r = \frac{\sum_k (d_k-\bar d)(m_k-\bar m)}{\sqrt{\sum_k (d_k-\bar d)^2\,\sum_k (m_k-\bar m)^2}},
\qquad
s = \left|\frac{\sum_k (d_k-\bar d)(m_k-\bar m)}{\sum_k (d_k-\bar d)^2}\right| .
\]

`pearson` is \(r\): how well the pattern of correlations is reproduced, 1 at best. `slope` is \(s\), the least-squares slope of model on data: whether the amplitude is right, 1 at best. A high Pearson with a slope well below 1 means the model reproduces the pattern of correlations but too weakly. `pearson_val` and `slope_val` compare the same chains with the validation data.

Both are computed from a finite population of chains, so they fluctuate from update to update (standard deviation about 0.002 at 10 sweeps on bkace) and are biased low for small populations. Compare Pearson values only at equal sample size.

## Likelihood and entropy

With an estimate of \(\log Z\) (see [Normalization](derived.md#normalization-and-entropy)), the log-likelihood per site of an alignment is

\[
\ell = \frac{1}{L}\left(-\sum_m \tilde w_m E_\theta\big(x^{(m)}\big) - \log Z_\theta\right),
\]

computed on the training data (`ll_train`) and on the validation data (`ll_val`) with their weights. The entropy is

\[
S = \langle E_\theta\rangle_{p_\theta} + \log Z_\theta ,
\]

with \(\langle E\rangle\) the mean energy of the endpoint chains, in nats per sequence (`entropy`; `logz` is \(\log Z\)). During training \(\log Z\) comes from the one-sided forward bridge, which is cheap but noisy: single-update spikes of up to 0.01 per site at ladder changes and slower fluctuations of about 10⁻³ per site. The validation stop uses medians to ignore them. PCD has no bridge and records neither quantity.

## Adaptive optimizer and predicted KL

For bmDCA and eaDCA, the gradient directions \(g_h = f^\alpha_i - p_i\) and \(g_J = f^\alpha_{ij} - p_{ij} - \lambda_2 J\) (masked) are scaled by rates \(\eta_h = a_h\eta\) and \(\eta_J = a_J\eta\), with \(a_h, a_J \in (0, 1]\) and \(\eta\) = `--lr`. A parameter step changes the energy of a sequence by \(-\eta_h u_h(x) - \eta_J u_J(x)\), where \(u_h(x) = \sum_i g_h(i, x_i)\) and \(u_J(x) = \sum_{i<j} g_J(i,j,x_i,x_j)\) are the per-sequence scores of the two directions. To second order, the KL divergence between the model before and after the step is

\[
K_{\rm pred} = \tfrac12\,(\eta_h, \eta_J)\;F\;(\eta_h, \eta_J)^{\mathsf T},
\qquad
F = \begin{pmatrix}\operatorname{Var}u_h & \operatorname{Cov}(u_h,u_J)\\ \operatorname{Cov}(u_h,u_J) & \operatorname{Var}u_J\end{pmatrix},
\]

with variances over the endpoint chains: \(F\) is the Fisher information restricted to the two directions. The predicted likelihood gain is linear in the rates, \(\eta_h\|g_h\|^2 + \eta_J\|g_J\|^2\). The optimizer maximizes it subject to \(K_{\rm pred} \le\) `--ptt-trust-radius` (0.01) and \(a_h, a_J \le 1\). Because \(F\) keeps the off-diagonal term, correlated field and coupling steps are treated jointly; when the directions are strongly correlated and the constraint binds, the optimum puts the whole budget on the couplings and the field rate at its floor.

`lr_bias` and `lr_coupling` are \(\eta_h\) and \(\eta_J\); `kl` is \(K_{\rm pred}\) of the accepted step. It is a prediction of how much the model moves, not a measured error.

## Chain lag

The lag rests on **linear response**, the statistical-physics relation between how a system at equilibrium responds to a small perturbation and how it fluctuates without it. Perturb a Boltzmann distribution by a small energy term \(\lambda\, s(x)\):

\[
p_\lambda(x) = \frac{e^{-E(x) - \lambda s(x)}}{Z_\lambda}.
\]

Differentiating \(\langle A\rangle_\lambda = \sum_x A(x)\,p_\lambda(x)\) gives, for any observable \(A\),

\[
\frac{\partial \langle A\rangle_\lambda}{\partial \lambda}\bigg|_{\lambda=0} = -\big(\langle A\, s\rangle - \langle A\rangle\langle s\rangle\big) = -\operatorname{Cov}(A, s).
\]

The response of the mean, the susceptibility, equals a covariance measured in the unperturbed system: this is the static form of the fluctuation–dissipation theorem. A parameter update is such a perturbation, with \(s(x) = E_{t+1}(x) - E_t(x)\) and \(\lambda = 1\). Chains still distributed according to the old model therefore over-represent, for any observable \(A\), the new mean by approximately

\[
\langle A\rangle_{\rm old} - \langle A\rangle_{\rm new} \approx \operatorname{Cov}_{\rm old}(A, s).
\]

The approximation holds as long as the update is small compared with the fluctuations of \(s\), which the trust region enforces.

Local moves and swaps relax this excess. To measure what remains of the excess accumulated over many updates, each endpoint configuration \(n\) carries a **lag memory**

\[
m_n \leftarrow \Big(1-\frac1H\Big)\,m_n + s(x_n) - \overline{s},
\]

updated at every accepted update, with \(\overline{s}\) the mean over the endpoint population and \(H\) = `--ptt-lag-horizon` (200). The memory travels with its configuration through swaps and permutations; configurations entering the endpoint from below start from zero. For a monitored observable \(A\), standardized over the endpoint population to \(z_A\), the lag is the covariance

\[
\ell_A = \frac1N\sum_{n=1}^{N} z_A(x_n)\, m_n ,
\]

in population standard deviations. Two groups of observables are monitored: the **drift** \(A(x) = E_{\rm endpoint}(x) - E_{\rm below}(x)\), the direction the model has moved since the last snapshot (`lag_drift`), and the indicators of the drift's lower and upper 2σ tails, where a forming mode first appears (`lag_drift_tail`). The largest absolute value is compared with `--ptt-lag-tolerance` (0.25). Above it, training pauses: a snapshot is inserted below the endpoint, and the ladder evolves at fixed parameters in blocks of 10 rounds until the lag falls below half the tolerance or `--ptt-lag-pause-rounds` (100) have run; another pause cannot start for 10 updates.

The lag is a linear-response quantity. A mode that the chains reach only by rare transitions can be far from equilibrium while its lag stays small.

## Sparse-model selection rules

**eaDCA activation.** For each inactive coupling entry, the discrepancy between data and model is the KL divergence between two Bernoulli distributions,

\[
D_{ij}(a,b) = f_{ij}\log\frac{f_{ij}}{p_{ij}} + (1-f_{ij})\log\frac{1-f_{ij}}{1-p_{ij}},
\]

with pseudocounted frequencies. Each graph update activates the fraction `--factivate` of the inactive entries (counted once per symmetric pair, at least one) with the largest \(D\). With adaptive activation (PTT), only candidates with

\[
|f - p| \ge s\,\sqrt{\frac{f(1-f)}{M_{\rm eff}} + \frac{p(1-p)}{N}}
\]

are eligible (\(s\) = `--ptt-activation-significance`, 3; \(N\) the number of chains), and the block takes the longest prefix of the ranked list whose first update stays within `--ptt-activation-kl-share` (0.5) of the trust radius. Each mixing recovery halves that share.

**edDCA decimation.** Removing one active coupling \(J = J_{ij}(a,b)\) from a model with pair marginal \(p = p_{ij}(a,b)\) changes the distribution by

\[
D = p\,J + \log\!\big(p\,e^{-J} + 1 - p\big)
\]

in KL divergence. Each decimation step removes the fraction `--drate` of the active couplings with the smallest \(D\), then re-fits the others.

**edgeDCA.** The discrepancy of a position pair is the KL divergence of its pair distributions, \(D_{ij} = \sum_{a,b} f^\alpha_{ij}(a,b)\log\big[f^\alpha_{ij}(a,b)/p^\alpha_{ij}(a,b)\big]\), with both frequencies pseudocounted. The pair with the largest \(D_{ij}\), active or not, is updated by

\[
J_{ij}(a,b) \leftarrow J_{ij}(a,b) + \log f^\alpha_{ij}(a,b) - \log p^\alpha_{ij}(a,b).
\]

As \(\alpha \to 1\) both frequencies tend to \(1/q^2\) and the step vanishes; the fixed point \(f^\alpha = p^\alpha\) is equivalent to \(f = p\). History records the pair (`edge_i`, `edge_j`), whether it was new (`edge_new`) and the pseudocount used (`edge_pseudocount`, raised by the adaptive optimizer when the predicted KL would exceed the trust radius).

**Density** (`density`) is the fraction of coupling entries that are active; 1 for bmDCA. edgeDCA activates whole blocks, so for it this is also the fraction of active position pairs.

## Stopping rules and counters

**Pearson target.** Stop when `pearson` ≥ `--target` (0.95).

**Validation plateau.** With validation log-likelihoods \(\ell_t\) at accepted updates and a window \(w\) (`--ptt-validation-window`, 100), after at least \(2w\) updates compute

\[
g_t = \operatorname{median}(\ell_{t-w+1},\dots,\ell_t) - \operatorname{median}(\ell_{t-2w+1},\dots,\ell_{t-w}),
\]

and stop as soon as \(g_t <\) `--ptt-validation-min-gain` (0). The Pearson target is then reported but does not stop the run.

**Counters.**

| Counter | Counts |
| --- | --- |
| `gradient_steps` | Accepted parameter updates |
| `structure_steps` | Committed graph changes (activation blocks, decimation steps) |
| `steps_on_graph` | Position of an eaDCA update within its block |
| `sweeps` | All local sweeps per chain, including mixing checks, reservoir collection, pauses, failed updates and recovery. Cumulative across rollbacks, so it can rise while `step` stands still |

`--nepochs` counts gradient steps for bmDCA and PTT edgeDCA, structure steps for eaDCA and edDCA. `--max-gradient-steps` and `--max-structure-steps` are unambiguous.
