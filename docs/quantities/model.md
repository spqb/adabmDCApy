# Potts model and statistics

## Potts energy and probability

A sequence \(x = (x_1, \dots, x_L)\) has one-hot encoding \(x_i(a) = \mathbf 1[x_i = a]\). With fields \(h_i(a)\) and couplings \(J_{ij}(a,b) = J_{ji}(b,a)\), the energy and probability are

\[
E_\theta(x) = -\sum_{i}h_i(x_i) - \sum_{i<j} J_{ij}(x_i, x_j),
\qquad
p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta},
\qquad
Z_\theta = \sum_{x} e^{-E_\theta(x)},
\]

where \(\theta = (h, J)\) and the sum defining the partition function \(Z\) runs over all \(q^L\) sequences. Internally the couplings are stored as a symmetric \(Lq \times Lq\) tensor and the pair term is written \(-\tfrac12\sum_{i,j,a,b} J_{ij}(a,b)\,x_i(a)\,x_j(b)\) with zero diagonal blocks.

Lower energy means higher probability, and only energy differences matter within one model: \(\log p(x) - \log p(y) = E(y) - E(x)\). Plain MCMC can also sample \(p_\beta(x) \propto e^{-\beta E(x)}\) at inverse temperature \(\beta\) (`--beta`); \(\beta > 1\) concentrates on low-energy sequences. PTT always uses \(\beta = 1\).

**Conditional probabilities.** The probability of residue \(a\) at position \(i\) given the rest of the sequence is

\[
p(x_i = a \mid x_{-i}) = \frac{\exp\big[h_i(a) + \sum_{j\neq i} J_{ij}(a, x_j)\big]}{\sum_{b}\exp\big[h_i(b) + \sum_{j\neq i} J_{ij}(b, x_j)\big]}.
\]

All local moves and the context-dependent entropy are built from it.

**Gauge.** Adding a constant to \(h_i(\cdot)\), or moving a function of \(a\) from \(J_{ij}(a,\cdot)\) to \(h_i(a)\), leaves \(p_\theta\) unchanged. Parameters are therefore not unique. Quantities that compare parameter values, such as contact scores, first move the couplings to the **zero-sum gauge**, where every row and column of each block \(J_{ij}\) sums to zero.

## Sequence weights

For retained sequence \(m\) of \(M\), count the sequences whose fractional identity with it exceeds the threshold \(t\) (default 0.8), itself included:

\[
n_m = \sum_{n=1}^{M}\mathbf 1\!\left[\frac1L\sum_{i=1}^{L}\mathbf 1\big[x^{(m)}_i = x^{(n)}_i\big] > t\right],
\qquad
w_m = \frac{1}{n_m},
\qquad
M_{\rm eff} = \sum_m w_m .
\]

Identity is computed position by position on the aligned sequences, gaps included, with a strict inequality. A sequence in a dense cluster gets a small weight; an isolated one gets 1. This corrects for uneven sampling of sequence space; it does not reconstruct phylogeny, and \(M_{\rm eff}\) is not a count of independent observations. Supplied weights replace the calculation; `--no_reweighting` sets \(w_m = 1\). [Reintegration](../algorithms/analysis.md#experimental-reintegration) adds signed weights for tested sequences.

## Frequencies and pseudocount

With normalized weights \(\tilde w_m = w_m / \sum_n w_n\), the empirical one- and two-site frequencies are

\[
f_i(a) = \sum_m \tilde w_m\, x^{(m)}_i(a),
\qquad
f_{ij}(a,b) = \sum_m \tilde w_m\, x^{(m)}_i(a)\, x^{(m)}_j(b).
\]

The pseudocount \(\alpha\) mixes them with uniform frequencies:

\[
f^\alpha_i(a) = (1-\alpha) f_i(a) + \frac{\alpha}{q},
\qquad
f^\alpha_{ij}(a,b) = (1-\alpha) f_{ij}(a,b) + \frac{\alpha}{q^2}\quad(i\neq j),
\]

with \(f^\alpha_{ii}(a,b) = \delta_{ab} f^\alpha_i(a)\). The default is \(\alpha = 1/M_{\rm eff}\), or 0.1 for edgeDCA. The model frequencies \(p_i(a)\), \(p_{ij}(a,b)\) are the same averages over the sampled chains, with equal weights.

## Connected correlations

\[
C_{ij}(a,b) = f_{ij}(a,b) - f_i(a) f_j(b), \qquad i \neq j .
\]

Raw pair frequencies are dominated by the product of single-site frequencies; the connected correlation keeps only the part that a model with fields alone cannot explain. Comparisons between data and model (training Pearson, `cij_scatter.png`) use all entries \((a, b)\) of all blocks with \(i > j\), \(\binom{L}{2} q^2\) numbers, restricted to the active couplings for sparse models. Agreement on these entries does not exclude disagreement on higher-order statistics or on the weights of separated modes.

## Likelihood and its gradient

The average log-likelihood of the weighted data is

\[
\mathcal L(\theta) = \sum_m \tilde w_m \log p_\theta\big(x^{(m)}\big) = -\big\langle E_\theta \big\rangle_{f} - \log Z_\theta ,
\]

and since \(\partial \log Z/\partial h_i(a) = -p_i(a)\) and \(\partial \log Z/\partial J_{ij}(a,b) = -p_{ij}(a,b)\),

\[
\frac{\partial \mathcal L}{\partial h_i(a)} = f_i(a) - p_i(a),
\qquad
\frac{\partial \mathcal L}{\partial J_{ij}(a,b)} = f_{ij}(a,b) - p_{ij}(a,b).
\]

The maximum is where the model reproduces all one- and two-site frequencies. Training uses the pseudocounted \(f^\alpha\), and an optional L2 penalty \(\lambda_2\) (`--l2_reg`) subtracts \(\lambda_2 J_{ij}(a,b)\) from the coupling gradient. Plain gradient ascent (PCD, and PTT with `--ptt-optimizer sgd`) updates

\[
h \leftarrow h + \eta\,(f^\alpha_i - p_i),
\qquad
J \leftarrow J + \eta\,(f^\alpha_{ij} - p_{ij} - \lambda_2 J),
\]

with \(\eta\) = `--lr`, and with sparse models masking inactive couplings. The adaptive PTT optimizer chooses separate rates for fields and couplings ([Training quantities](training.md#adaptive-optimizer-and-predicted-kl)).

The model frequencies \(p\) are estimated from 2,000 chains, so the gradient has sampling noise of order \(1/\sqrt{2000}\) on each frequency, and, more importantly, a bias whenever the chains are not at equilibrium.
