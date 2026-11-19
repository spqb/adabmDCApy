# Sampling quantities

Definitions behind [Sampling](../algorithms/sampling.md) and [Diagnose a run](../algorithms/diagnostics.md).

## Local moves

At a position \(i\) with current state \(a\), let \(\pi_b = p(x_i = b \mid x_{-i})\) be the conditional probabilities of the model (or of the rung's model, in PTT).

- **Gibbs**: draw the new state \(b\) with probability \(\pi_b\).
- **Metropolis**: propose \(b\) uniformly among the \(q\) states and accept with probability \(\min\{1, \pi_b/\pi_a\} = \min\{1, e^{-\Delta E}\}\).
- **Metropolized Gibbs** (Liu, 1996): propose \(b \neq a\) with probability \(\pi_b/(1-\pi_a)\) and accept with probability

\[
A(a\to b) = \min\!\left\{1, \frac{1-\pi_a}{1-\pi_b}\right\}.
\]

All three satisfy detailed balance with respect to the model. Metropolized Gibbs never proposes the current state, so it changes the sequence more often than Gibbs. A **sweep** is one update attempt at every position of every chain.

## Swap acceptance

Rung \(k\) samples \(p_k(x) = e^{-E_k(x)}/Z_k\). For neighbouring rungs write \(W_k(x) = E_{k+1}(x) - E_k(x)\). Exchanging configuration \(y\) of rung \(k\) with configuration \(x\) of rung \(k+1\) is accepted with probability

\[
A(x, y) = \min\big\{1, \exp[W_k(x) - W_k(y)]\big\},
\]

which keeps both rungs at equilibrium. `acceptance_min` in the training history is the smallest observed acceptance fraction over the neighbouring pairs. The snapshot rules use the top link: store a temporary snapshot below \(2a_*\), insert it below \(a_*\) (`--ptt-target-acceptance`, \(a_* = 0.25\)). An observed acceptance below `--ptt-min-acceptance` (0.1) on any link requests a mixing check.

`replicas` counts the active rungs; `models` counts the models flagged for the sampling ladder.

## Population renewal

Every configuration has a **birth round** \(b\): the round in which it was drawn exactly at the bottom of the ladder (in a compressed training ladder, the round it was emitted by the reservoir). Swaps and permutations move configurations with their birth labels; nothing copies a configuration. For a reference round \(r_0\), \(R\) rungs and \(N\) chains per rung,

\[
G(t) = \frac{\#\{\text{configurations in the ladder with } b < r_0\}}{RN},
\qquad
F(t) = \frac{\#\{\text{endpoint configurations with } b \ge r_0\}}{N}.
\]

`ladder_old` is \(G\) and `endpoint_fresh` is \(F\). Old configurations are destroyed at the bottom and never created, so \(G\) is non-increasing; \(1-F\) is not. Renewal is reached at the first round where

\[
RN\,G(t) \le \varepsilon N \quad\Longleftrightarrow\quad G(t) \le \frac{\varepsilon}{R},
\]

with \(\varepsilon\) = `--ptt-renewal-tolerance` (0.01). At that point the endpoint's old fraction is at most \(\varepsilon\) and can never exceed it again.

**Sampling phases.** The warmup uses \(r_0 = 0\) and lasts \(W\) rounds (`warmup_rounds`). With `--ptt-stationary`, a second phase uses \(r_0 = W\). Its renewal is checked only at block ends, every \(\lceil W/4\rceil\) rounds (`block_rounds`), and sampling stops at the first block end after renewal. Stopping at the exact round of renewal would let the population choose when sampling stops: the stop would fall right after the last old configuration happened to be redrawn at the bottom, so the returned population would be one selected by that event. Block ends are fixed in advance and do not depend on which configurations are old, so the returned population is not selected by the stopping rule. The price is at most a quarter of the warmup in extra rounds. `renewal_rounds` is the round at which the second renewal happened. Further batches are separated by one renewal time (`spacing_rounds`). `trapped_fraction` is \(1-F\) at the end; `old_fraction_by_model` gives the old fraction per rung. All are in `logs/mix.log` and `sampling.json`.

**Decay forecast.** Every 10 rounds, a line is fitted to \(\log G\) over the more recent half of the phase, starting from the first round with \(G < 0.5\). Its slope gives the decay time (`warmup_decay_rounds`, `stationary_decay_rounds`), and its crossing with \(\varepsilon/R\) the predicted renewal. The forecast drives the progress display and the warnings; it never decides the stop.

**Training checks.** During training, renewal is used in three ways, shown in the figure.

[![Mixing check, reservoir collection and failure diagnosis during PTT training](../figures/training_checks_scheme_2col.png){ width="900" }](../figures/training_checks_scheme_2col.png)

1. **Mixing check.** It runs on a separate population of `--ptt-mixing-chains` (100) chains per rung; the production chains are not touched. It passes when the ladder is renewed twice in a row: after the first renewal, every configuration present becomes old again and must be replaced once more. If this does not happen within `--ptt-mixing-max-rounds`, the check fails and training [recovers](../algorithms/training.md#mixing-checks-and-recovery).
2. **Reservoir collection.** It works on the production chains. The ladder first runs until the rung that will become the new bottom is renewed (warm-up), then measures that rung's renewal time \(\tau\). Its population is then copied into the reservoir in batches spaced so that at most \(\varepsilon N\) configurations of a batch predate the previous one (the spacing is checked every quarter of the warm-up), until the reservoir holds \(10N\) configurations. The lower rungs keep supplying fresh configurations throughout. Only at the end are they dropped, and the new bottom is fed from the reservoir, with \(\log Z\) carried forward.
3. **Failure diagnosis.** When a check or a collection fails, every old configuration of rung \(k\) is tested against 8 random partners in rung \(k-1\). If its mean swap probability is below \(10^{-4}\), it is **immobile**: only local moves can free it. If the immobile old configurations alone exceed the tolerance \(\varepsilon N\), they are **blocking**, and the error message recommends restarting with more local sweeps. Otherwise it reports the acceptance of each link: a link below the target calls for a smaller `--lr`, while links all above it call for more sweeps.

## Replica-index autocorrelation

The alternative `--ptt-mixing-method autocorrelation` follows the rung index \(r_t\) of each tagged configuration. With the uniform mean \((R-1)/2\) subtracted,

\[
C(t) = \frac{\langle (r_s - \bar r)(r_{s+t} - \bar r)\rangle}{\langle (r_s - \bar r)^2\rangle},
\qquad
\tau_{\rm int} = \frac12 + \sum_{t=1}^{T} C(t),
\]

with the window \(T\) chosen self-consistently (\(T \ge 6\,\tau_{\rm int}\)), and \(\tau_{\rm exp}\) from an exponential fit \(A e^{-t/\tau_{\rm exp}}\) to the tail of \(C\). A check passes when the trajectory is at least `--ptt-mixing-window-factor` (20) times \(\max(\tau_{\rm int}, \tau_{\rm exp})\) long. Generation batches are separated by \(2\lfloor\tau_{\rm int}\rfloor\) rounds. Both times are in exchange rounds.

## Plain-MCMC mixing time

Chains started from (resampled) reference sequences are run, and at each even time \(t\) the package compares the fractional identity between each chain and itself at \(t/2\),

\[
I_{t,t/2} = \frac1N\sum_{n}\mathrm{id}\big(x_n(t), x_n(t/2)\big),
\]

with the identity between distinct chains at time \(t\), \(I_t = \frac1N\sum_n \mathrm{id}(x_n(t), x_{\sigma(n)}(t))\) for a random permutation \(\sigma\) without fixed points, where \(\mathrm{id}(x,y) = \frac1L\sum_i \mathbf 1[x_i = y_i]\). The mixing time is the first \(t/2\) at which

\[
\frac{|I_{t,t/2} - I_t|}{\sqrt{\sigma_{t,t/2}^2 + \sigma_t^2}} < 0.1 ,
\]

where \(\sigma\) are the standard deviations of the identities over chains. Generation then runs fresh random chains for `--nmix` (2) mixing times. The criterion detects loss of memory of the starting point within the region the chains explore; it cannot detect a mode they never reach.

## Ladder health

For each neighbouring pair \((k, k+1)\), with \(W = E_{k+1} - E_k\) evaluated on both final populations and \(\Delta F = -\log(Z_{k+1}/Z_k)\):

- **Free-energy differences**: forward \(\widehat{\Delta F}_{\rm fwd} = -\log\langle e^{-W}\rangle_k\), reverse \(\widehat{\Delta F}_{\rm rev} = \log\langle e^{W}\rangle_{k+1}\), and Bennett's acceptance ratio `dF_bar` ([definition](derived.md#ptt-ladder)), each with a bootstrap error. `hysteresis` is forward minus reverse; beyond the errors it signals poor overlap.
- **Effective sample size** of one reweighting step, for weights \(u_n\): \((\sum u_n)^2 / \sum u_n^2\), divided by the number of samples (`ess_forward` with \(u = e^{-W}\) on rung \(k\), `ess_reverse` with \(u = e^{W}\) on rung \(k+1\)). Near 1: even weights; near 0: a few configurations dominate. `top1_share_*` is the weight carried by the heaviest 1%. Finite samples overestimate the ESS when overlap is poor, because they miss the extreme tail.
- **Crooks slope**: at equilibrium, the histograms of \(W\) in the two rungs satisfy \(\log[h_{k+1}(W)/h_k(W)] = \Delta F - W\). The slope fitted on quantile bins populated by both should be \(-1\); a deviation means the overlapping parts of the populations are not mutually consistent.
- **Mobility**: for each upper configuration, its swap probability averaged over all lower partners (downward mobility), and conversely. Quantiles of the mobilities are plotted (`acceptance_down_q50/q10/q01`, `acceptance_up_*`), and configurations with mobility below \(10^{-4}\) are **immobile** (`immobile_down_fraction`, `immobile_up_fraction`). `mean_acceptance` is the average mobility.
- **Ages**: a configuration born in round \(b\) has age \(t - b\) at round \(t\). The ratio \(R = \bar a_{\rm immobile} / \bar a_{\rm mobile}\) (`immobile_mean_age` / `mobile_mean_age`) is plotted in the trapped-configurations panel; about 1 means immobility is transient, well above 1 that immobile configurations stay stuck.
- **Age test**: with \(n\) configurations in the upper rung, \(m\) of them immobile, the null hypothesis is that the immobile ones are a uniformly random subset of size \(m\). Since the total age is fixed, \(R\) increases with the summed age of the subset, so the one-sided p-value is the fraction of \(B = 10^4\) random subsets with a ratio at least the observed one, `immobile_age_p` \(= (1 + \#\{R^* \ge R\}) / (1 + B)\), never below \(10^{-4}\). `immobile_age_null_q95` is the 95% quantile of \(R^*\). The test needs no model of the age distribution, which is far from any simple law in the lower rungs.
- **Old immobile configurations**: `immobile_old` counts immobile configurations older than the renewal time, or, when renewal failed, born before it started.
- **Trapping class** (`trapping`): `none` without immobile configurations; `noise` when `immobile_age_p` is at least \(\alpha / L\), with \(\alpha = 0.01\) and \(L\) the number of tested links (Bonferroni); otherwise `blocking` when `immobile_old` exceeds the renewal tolerance times the number of chains, and `slow` when it does not. Old immobile configurations count only on significant links: when the budget is too short for the whole ladder, most configurations are old, and immobile ones are old at the same rate.

The age test is calibrated under its null; the immobility threshold of \(10^{-4}\) and the tolerance are conventions.

Only the mobility and age measures, with the age test, are plotted (`ptt_ladder_health.png`). The free-energy differences, effective sample sizes and Crooks slopes are written to `logs/ptt_ladder_health.log` and `sampling.json` but not plotted. On the families studied they mostly reflected the spacing of the snapshots and took similar values on runs that worked and runs that failed, so they did not help to detect failures.

## Three-point correlations

For states \(u = (i, a)\), \(v = (j, b)\), \(w = (k, c)\) at three different sites, with centred one-hot indicators \(\delta x_u = x_u - f_u\), the connected three-point correlation is the weighted mean \(C_{uvw} = \langle \delta x_u\, \delta x_v\, \delta x_w \rangle\). It is not a three-body interaction: a pairwise model produces it too.

- **No strong pair.** All three pair correlation coefficients \(r_{uv} = C_{uv} / \sqrt{f_u(1-f_u) f_v(1-f_v)}\), in the data, are below 0.1 in absolute value. For indicators \(x^2 = x\), so two states that always occur together give \(C_{uvw} = (1 - 2f_u)\, C_{uw}\): a two-point correlation in disguise. Gap states are excluded, as are states with data frequency outside \([0.05, 0.95]\).
- **Resolved.** The standard error of \(C_{uvw}\) is the weighted standard deviation of \(\delta x_u \delta x_v \delta x_w\) divided by \(\sqrt{N}\), with \(N\) the sum of the data weights for the data (all sequences count as independent when training used no reweighting, so the error is then optimistic) and the number of samples for the samples. A triplet is resolved when \(z = \max(|C^{\rm data}| / \sigma^{\rm data}, |C^{\rm samples}| / \sigma^{\rm samples}) \ge 5\). The larger of the two ratios does not favour triplets where data and samples agree. Among about \(10^7\) pure-noise candidates, about five reach 5.
- **Search.** 3,000 random seed pairs \((u, v)\) that pass the selection are combined with every third state in one matrix product per block of sequences, which gives the candidates. Of the resolved triplets, 2,000 are drawn at random for the plot: the largest alone would leave out every correlation of moderate size.
- **Agreement.** The Pearson correlation, the slope of the least-squares line, and \(R^2 = 1 - \sum (y - x)^2 / \sum (x - \bar x)^2\) about the identity, with \(x\) the data and \(y\) the samples: 1 only on the diagonal, lower when the slope moves away from 1. With a gap around zero the Pearson correlation also rewards sign agreement alone; \(R^2\) much less so. The single-site and two-site panels give the same three numbers.
- **Noise ceiling.** The Pearson correlation and \(R^2\) of the plotted triplets between two random halves of the data. They are slightly optimistic, since the selection used all the data.

`sampling.json` → `three_point_comparison` has the counts (`n_candidates`, `n_resolved`), `pearson`, `r2`, `slope`, `noise_ceiling` and `noise_ceiling_r2`; `logs/three_point.log` lists the plotted triplets (sites, states, data and sample values, \(z\)).

## Distances

The fractional Hamming distance between aligned sequences is \(d(x,y) = \frac1L\sum_i \mathbf 1[x_i \neq y_i]\), and the nearest-neighbour distance of \(x\) to a set \(T\) is \(d(x,T) = \min_{y\in T} d(x,y)\), excluding \(x\) itself when \(x \in T\). `distances.png` histograms all-pair and nearest-neighbour distances; natural sequences carry their weights, and at most `--nmeasure` sequences per set are drawn without replacement. Medians, quantiles and the fraction of generated sequences identical to a training sequence are in `sampling.json` → `distance_comparison.summary`.
