# Low-rank Potts models

A full Potts model couples every pair of positions with its own \(q \times q\) matrix. Sampling it needs Markov chains, and its normalization \(Z\) has no closed form. A **low-rank** Potts model keeps only a few collective modes of the couplings. In exchange it can be fitted by exact maximum likelihood, sampled without Markov chains, and normalized by a low-dimensional integral.

The package uses such a model as an optional starting point of bmDCA training with PTT (`--ptt-anchor-rank K`). The model is fitted to the alignment before training and replaces the profile at the bottom of the PTT ladder. This page builds the model step by step, from the Hopfield–Potts model to its exact sampler. The implementation is in `adabmDCA/lowrank.py`.

## From the Potts model to the Hopfield–Potts model

The Potts model of the package (see [Potts model and statistics](model.md)) has the energy

\[
E(x) = -\sum_i h_i(x_i) - \frac12 \sum_{i \neq j} J_{ij}(x_i, x_j),
\]

with \(L q\) fields and about \(L^2 q^2 / 2\) couplings. The Hopfield–Potts model (Cocco, Monasson and Weigt, *PLoS Comput. Biol.* 2013) replaces the couplings with \(K\) **patterns**. Pattern \(k\) assigns a number \(W_{ik}(a)\) to every state \(a\) of every position \(i\). The patterns enter the energy through the overlaps of the sequence with them:

\[
m_k(x) = \sum_i W_{ik}(x_i), \qquad
-E(x) = \sum_i g_i(x_i) + \frac12 \sum_{k=1}^{K} m_k(x)^2 .
\]

Each state of each position therefore carries a vector \(W_i(a) \in \mathbb{R}^K\), and a sequence is mapped to the point \(m(x) = \sum_i W_i(x_i)\) of \(\mathbb{R}^K\). The model favours sequences that lie far from the origin along the patterns. Expanding the square shows that this is a Potts model:

\[
\frac12 \lvert m(x) \rvert^2 = \frac12 \sum_{i \neq j} W_i(x_i) \cdot W_j(x_j) + \frac12 \sum_i \lvert W_i(x_i) \rvert^2 .
\]

The first term gives couplings of rank \(K\), and the second depends on one position only, so it is part of the fields:

\[
J_{ij}(a, b) = W_i(a) \cdot W_j(b) \quad (i \neq j), \qquad
h_i(a) = g_i(a) + \tfrac12 \lvert W_i(a) \rvert^2 .
\]

These are the Potts parameters handed to the PTT ladder. Cocco and coauthors write the same model with patterns \(\xi = \sqrt{L}\, W\).

**What the model can represent.** The coupling matrix \(J = W W^{\top}\) is positive semi-definite: the patterns describe **attractive** collective modes, such as groups of positions whose states change together, or a split of the family into subfamilies. Couplings with negative eigenvalues (the "repulsive patterns" of Cocco and coauthors) cannot be written this way. They also do not survive the construction of the next section, which needs a real Gaussian.

**How the patterns relate to principal components.** In the mean-field approximation, the best patterns are the top eigenvectors \(v_k\) of the correlation matrix of the one-hot encoded alignment, normalized by the single-site frequencies, \(D^{-1/2} C D^{-1/2}\), rescaled by their eigenvalues \(\lambda_k\):

\[
W_k \propto \sqrt{1 - 1/\lambda_k}\; D^{-1/2} v_k .
\]

The patterns are thus close relatives of the principal components of the alignment. Only modes with \(\lambda_k > 1\) are attractive. The package uses this mean-field solution only as a starting point, then maximizes the exact likelihood.

**Gauge.** As in the full Potts model, several parameter sets describe the same distribution. Adding a vector \(u_i\) to all the weights of position \(i\), \(W_i(a) \to W_i(a) + u_i\), shifts \(m(x)\) by the constant \(u = \sum_i u_i\). This is compensated by \(g_j(a) \to g_j(a) - u \cdot W_j(a)\), up to a constant. A rotation of the \(K\) latent directions also leaves the model unchanged. The fitted model is returned in the gauge where the weights of every position sum to zero over its \(q\) states.

## The equivalent RBM with Gaussian hidden units

The square \(\frac12 \lvert m \rvert^2\) can be traded for a Gaussian integral. For a standard normal vector \(z \in \mathbb{R}^K\), \(\mathbb{E}[e^{z \cdot m}] = e^{\lvert m \rvert^2/2}\), so

\[
e^{-E(x)} = \int \frac{dz}{(2\pi)^{K/2}}\; e^{-\lvert z \rvert^2/2} \prod_i e^{\,g_i(x_i) + z \cdot W_i(x_i)} .
\]

This identity (the Hubbard–Stratonovich transform) is exact for every \(L\) and \(q\). It says that the Hopfield–Potts model is the marginal distribution of the sequences in a joint model of sequences \(x\) and \(K\) continuous **hidden units** \(z\):

\[
p(x, z) \propto \exp\Big( -\frac{\lvert z \rvert^2}{2} + \sum_i \big[ g_i(x_i) + z \cdot W_i(x_i) \big] \Big).
\]

This is a restricted Boltzmann machine (RBM) with Gaussian hidden units, visible fields \(g\) and weights \(W\). Categorical positions need no special treatment: each position interacts with \(z\) through its own vector \(W_i(a)\) per state. Its two conditional distributions are simple:

- **Given \(z\), the positions are independent.** Each follows a profile-like distribution, tilted by the hidden units:
\[
p(x_i = a \mid z) = \frac{e^{\,g_i(a) + z \cdot W_i(a)}}{\sum_b e^{\,g_i(b) + z \cdot W_i(b)}} .
\]
- **Given \(x\), \(z\) is Gaussian**, centred at the sequence's overlaps with the patterns: \(p(z \mid x) = \mathcal{N}\big(m(x), I_K\big)\).

Summing out the sequences instead leaves a distribution of the hidden units alone, the **latent density**:

\[
p(z) \propto e^{F(z)}, \qquad
F(z) = -\frac{\lvert z \rvert^2}{2} + G(z), \qquad
G(z) = \sum_i \log \sum_a e^{\,g_i(a) + z \cdot W_i(a)} .
\]

\(G\) is a sum over positions of log-sum-exps, so \(F\) costs \(O(L q K)\) to evaluate. Integrating over \(z\) gives the partition function of the Potts model:

\[
\log Z = \log \int_{\mathbb{R}^K} e^{F(z)}\, dz - \frac{K}{2} \log 2\pi .
\]

This is the key simplification. A sum over \(q^L\) sequences has become an integral over \(K \le 3\) dimensions.

A second view of the latent density helps everywhere below. Since \(p(z \mid x)\) is a unit Gaussian centred at \(m(x)\), \(e^{F}\) is a **mixture of unit Gaussians**, one per sequence:

\[
e^{F(z)} = \sum_x e^{\,s(x)}\, e^{-\lvert z - m(x) \rvert^2/2}, \qquad s(x) = -E(x) = \sum_i g_i(x_i) + \tfrac12 \lvert m(x) \rvert^2 .
\]

\(p(z)\) is therefore a smoothed picture of the distribution of the overlaps \(m(x)\). If the family splits into two subfamilies along a pattern, \(p(z)\) has two bumps along that direction. This is the kind of structure that makes the first transitions of a PTT ladder hard.

## Fitting the parameters

The parameters \((g, W)\) maximize the log-likelihood of the weighted alignment. The pseudocount \(\lambda\) of the full model is applied as a mixture of the data with the uniform distribution, and the weights carry a small \(L_2\) penalty:

\[
\mathcal{L}(g, W) = \big\langle s(x) \big\rangle_{\text{target}} - \log Z - \frac{\gamma}{2} \sum_{i, a} \big\lvert W_i(a) - \bar W_i \big\rvert^2 ,
\qquad
\langle \cdot \rangle_{\text{target}} = (1 - \lambda) \sum_n w_n\, (\cdot)(x^{(n)}) + \lambda\, \mathbb{E}_{\text{uniform}}[\,\cdot\,] ,
\]

with normalized sequence weights \(w_n\), \(\gamma = 10^{-4}\), and \(\bar W_i\) the mean of the weights of position \(i\) over its \(q\) states. The penalty acts on the centred weights so that the gauge does not change it.

**The data enter only through their frequencies.** \(s(x)\) is linear in \(g\) and quadratic in \(W\), so its average needs only the single-site frequencies \(\hat f_i(a)\) and the pair frequencies \(\hat f_{ij}(a, b)\), with \(\hat f_{ii}(a, b) = \delta_{ab} \hat f_i(a)\):

\[
\big\langle s(x) \big\rangle_{\text{target}} = \sum_{i, a} \hat f_i(a)\, g_i(a) + \frac12 \sum_k \sum_{i, a} \sum_{j, b} W_{ik}(a)\, \hat f_{ij}(a, b)\, W_{jk}(b) .
\]

The pseudocount acts on the frequencies as usual: \(\hat f_i(a) = (1 - \lambda) f_i(a) + \lambda/q\) and \(\hat f_{ij}(a, b) = (1 - \lambda) f_{ij}(a, b) + \lambda/q^2\) for \(i \neq j\).

The pair frequencies form an \(Lq \times Lq\) matrix \(\hat S\), but the fit only ever needs its product with the weights, \(\hat S W\), a matrix of \(K\) columns. That product can be computed in two ways.

- **From the matrix.** Form \(\hat S\) once, then multiply. Each product costs \(O(K (Lq)^2)\) and is independent of the number of sequences \(N\), but the matrix takes \(8 (Lq)^2\) bytes: 160 MB for BLAT (\(Lq = 4515\)), 3.5 GB for a protein of 1,000 positions.
- **From the sequences.** \(\hat S W = (1 - \lambda) \sum_n w_n\, x_n\, m(x_n)^{\top} + (\text{uniform part})\), where \(x_n\) is the one-hot sequence and \(m(x_n)\) its overlaps with the patterns. Each product costs \(O(N L K)\), and no matrix is stored.

The matrix is formed when it takes at most 1 GiB (\(Lq \le 11{,}585\)); above that, products are computed from the sequences.

**The gradient matches moments.** Differentiating \(\log Z\) produces averages under the model, which the latent density gives through \(p(x_i \mid z)\):

\[
\frac{\partial \mathcal{L}}{\partial g_i(a)} = \hat f_i(a) - \mathbb{E}_z\big[p(x_i = a \mid z)\big], \qquad
\frac{\partial \mathcal{L}}{\partial W_{ik}(a)} = \sum_{j, b} \hat f_{ij}(a, b)\, W_{jk}(b) - \mathbb{E}_z\big[z_k\, p(x_i = a \mid z)\big] - \gamma \big(W_{ik}(a) - \bar W_{ik}\big),
\]

where \(\mathbb{E}_z\) averages over \(p(z)\). Since \(\mathbb{E}[z \mid x] = m(x)\), the first two terms of the second gradient are the same quantity, \(\langle \delta_{x_i, a}\, m_k(x) \rangle\), in the data and in the model. At the optimum the fields reproduce the single-site frequencies, as in the profile. The weights reproduce the correlation between each state and the overlaps with the patterns.

**Integrals over \(z\) become sums over a lattice.** \(\log Z\) and the averages \(\mathbb{E}_z\) are integrals over \(K\) dimensions. Because \(e^F\) is a mixture of unit Gaussians, it is very smooth, and its sum over a regular lattice of spacing 1 equals its integral to a relative accuracy of about \(10^{-8}\). The sum runs only over the lattice points where \(e^F\) is not negligible. That is a few dozen points for \(K = 1\), a few thousand for \(K = 2\) and about \(10^5\) for \(K = 3\) on a protein family. The [technical details](#the-lattice-sum) explain why spacing 1 is enough and how the points are found.

**The fit:**

1. Compute the frequencies \(\hat f_i\), and the pair matrix \(\hat S\) if it is formed.
2. Start from the mean-field Hopfield–Potts solution of the previous section, with \(g_i(a) = \log \hat f_i(a) - \frac12 \lvert W_i(a) \rvert^2\). Only the \(K\) top modes of the normalized correlation matrix are needed. They are found by subspace iteration, which only multiplies the correlation matrix with a block of \(K + 8\) vectors and never decomposes it. \(W = 0\) is a stationary point of the likelihood, so modes with \(\lambda_k \le 1\) start small instead of at zero. The mean-field patterns divide by \(\sqrt{\hat f_i(a)}\), which amplifies noise on rare states and can make the starting model far too strong. States absent from the alignment therefore start without a pattern, and the patterns are scaled by the factor \(\alpha \in \{1, \frac12, \dots, \frac1{32}\}\) with the highest exact likelihood. On BLAT the best factor is \(\frac12\) at rank 2; on RF00379 it is \(\frac14\), which turns a starting log-likelihood of −22.9 per site into −0.92.
3. Maximize \(\mathcal{L}\) with L-BFGS. Every evaluation computes \(\log Z\) and the gradient on the lattice points that carry the mass of \(p(z)\) for the current parameters. The points of the previous evaluation serve as the starting set.
4. Stop when 100 iterations raise the log-likelihood per sequence by less than 0.01 nats.

The stopping rule reflects nearly flat directions of the likelihood. For a state seen in only one bump of \(p(z)\), a change of its weight can be compensated by its field, so the last stretch of the fit gains very little. The remaining gap is far below the difference between training and validation sequences.

On BLAT (\(L = 215\), \(q = 21\), 128,000 sequences), the fit takes 9 s for \(K = 1\), 30 s for \(K = 2\) and about 12 minutes for \(K = 3\) on a GPU. The log-likelihood per site goes from −2.163 for the profile to −1.980, −1.874 and −1.812 for ranks 1, 2 and 3. Training prints the progress of the fit and its result before PTT starts.

## Sampling without Markov chains

The RBM view gives a two-step recipe for drawing a sequence from the Potts model:

1. draw the hidden units \(z\) from the latent density \(p(z)\);
2. draw every position independently from \(p(x_i \mid z)\).

The pair \((z, x)\) is then a draw of the joint distribution, so \(x\) is a draw of the low-rank Potts model. Step 2 is as easy as sampling the profile. All the difficulty is in step 1: \(p(z)\) lives in only \(K\) dimensions, but it is not Gaussian, and it can have several separated bumps.

**Rejection sampling.** Suppose we have a function \(\hat f(z) \ge e^{F(z)}\) everywhere, the **envelope**, which we know how to sample from. Draw \(z\) with density proportional to \(\hat f\), and accept it with probability \(e^{F(z)} / \hat f(z)\); otherwise draw again. The accepted points follow \(p(z)\) **exactly**, whatever the shape of \(F\). The only price is the rejected proposals: the fraction accepted is \(\int e^F / \int \hat f\).

**An envelope made of boxes.** The latent space is covered by boxes (cells). On a box \(B\) with centre \(c\), \(F\) is bounded by its tangent plane plus a constant:

\[
F(c + d) \le F(c) + s \cdot d + r_B, \qquad s = \nabla F(c),
\]

for every \(d\) in the box. The constant \(r_B\) comes from a bound on the curvature of \(F\) over the box (see [Bounds of \(F\) on a box](#bounds-of-f-on-a-box)). An exponential of a linear function is easy to handle. Its integral over the box is a product of one-dimensional integrals,

\[
M_B = e^{F(c) + r_B} \prod_k \frac{2 \sinh(s_k h_k)}{s_k},
\]

where \(h_k\) are the half widths of the box. Each coordinate can be drawn independently by inverting its one-dimensional distribution function. Small boxes make \(r_B\) small and the envelope tight. Boxes are refined only where \(p(z)\) has mass, and those that cannot hold a significant share of it are dropped (see [Building the cells](#building-the-cells)).

**The sampler:**

1. Pick a cell \(B\) with probability proportional to its envelope mass \(M_B\).
2. Draw \(z = c + d\) in the cell: each coordinate \(d_k\) independently, with density proportional to \(e^{s_k d_k}\) on \([-h_k, h_k]\).
3. Accept \(z\) with probability \(e^{F(z) - F(c) - s \cdot d - r_B}\). If rejected, return to step 1.
4. Draw every position from \(p(x_i \mid z)\).

The draws are independent, and exact up to the dropped cells, each of which holds less than \(e^{-50}\) of the mass. On BLAT about 95% of the proposals are accepted, and 100,000 sequences take less than a second on a GPU at every rank up to 3. The cells are built once per model, in 0.2 s for rank 2 and about 9 s for rank 3.

## At the bottom of the PTT ladder

The profile serves at the bottom of the ladder because its samples are exact and its \(\log Z\) is known (see [Normalization, scoring and importance weights](derived.md)). The low-rank model has both properties, so it can take its place:

- training starts with the low-rank model as both rungs, and its Potts parameters \(h\) and \(J\) are the first endpoint;
- in every round, rung 0 is redrawn exactly from the low-rank model, which is also where new chains start;
- \(\log Z\) of the ladder starts from the low-rank \(\log Z\) instead of the profile's, and the bridges above it are unchanged;
- the archive stores \(g\) and \(W\), so resumed training and generation use the same model.

The benefit is largest when the first transitions of training are the hard ones. This happens when the family splits along a few directions into well-separated groups, which a rank-\(K\) model captures exactly from the start. A low-rank model cannot shorten the later part of the ladder, which grows as the full model learns couplings beyond a few positive modes.

## Technical details

### Bounds of \(F\) on a box

The sampler needs, for every box, an upper bound of \(F\) that it can sample from. A matching lower bound decides when a box is refined. Both come from bounds on the curvature of \(G\).

1. **\(G\) is convex.** It is a sum of log-sum-exp functions of affine functions of \(z\). Its derivatives are averages under the conditionals \(p(x_i \mid z)\):
\[
\nabla G(z) = \sum_i \mathbb{E}_z[W_i], \qquad \nabla^2 G(z) = \sum_i \operatorname{Cov}_z[W_i].
\]
2. **Global curvature bound.** A random vector confined to a set of diameter \(D\) has variance at most \(D^2/4\) in every direction (Popoviciu's inequality). With \(D_i = \max_{a, b} \lvert W_i(a) - W_i(b) \rvert\), \(\operatorname{Cov}_z[W_i] \preceq \frac{D_i^2}{4} I\) for every \(z\).
3. **Local curvature bound.** Let the box have centre \(c\) and half widths \(h_k\), and write \(z = c + d\). The conditionals at \(z\) are those at \(c\) tilted by \(e^{d \cdot W_i(a)}\), so their ratio is bounded state by state. With \(\mu_i = \mathbb{E}_c[W_i]\),
\[
\delta_{ia} = \sum_k h_k \lvert W_{ik}(a) - \mu_{ik} \rvert, \qquad
\rho_i = \sum_k h_k \big(\max_a W_{ik}(a) - \min_a W_{ik}(a)\big),
\]
Jensen's inequality and Hoeffding's lemma, applied to \(\mathbb{E}_c[e^{d \cdot W_i}]\), give
\[
e^{-\delta_{ia} - \rho_i^2/8} \;\le\; \frac{p(x_i = a \mid z)}{p(x_i = a \mid c)} \;\le\; e^{\delta_{ia}} .
\]
A covariance is at most the second moment about any point, and at least the weighted covariance under any smaller weights. Hence, for every \(z\) in the box,
\[
\sum_a w^-_{ia} \big(W_i(a) - \bar\mu_i\big)\big(W_i(a) - \bar\mu_i\big)^{\top}
\;\preceq\; \operatorname{Cov}_z[W_i] \;\preceq\;
\sum_a w^+_{ia} \big(W_i(a) - \mu_i\big)\big(W_i(a) - \mu_i\big)^{\top},
\]
with \(w^+_{ia} = p(a \mid c)\, e^{\delta_{ia}}\), \(w^-_{ia} = p(a \mid c)\, e^{-\delta_{ia} - \rho_i^2/8}\), and \(\bar\mu_i\) the \(w^-\)-weighted mean of \(W_i\). A position whose upper matrix has a larger trace than the global bound, as happens on large boxes, uses the global bound instead.

Summing over positions gives matrices \(\Lambda^-_B \preceq \nabla^2 G \preceq \Lambda^+_B\) valid on the whole box. With \(s = \nabla F(c)\), Taylor's theorem gives

\[
\tfrac12\, d^{\top} (\Lambda^-_B - I)\, d \;\le\; F(c + d) - F(c) - s \cdot d \;\le\; \tfrac12\, d^{\top} (\Lambda^+_B - I)\, d .
\]

A quadratic form is at most its positive part, \(d^{\top} A d \le d^{\top} A_+ d\), and a convex quadratic reaches its maximum over a box at a vertex. With the vertices \(v \in \{\pm h_1\} \times \dots \times \{\pm h_K\}\) this gives

\[
r^+_B = \tfrac12 \max_{v} v^{\top} (\Lambda^+_B - I)_+ v, \qquad
r^-_B = \tfrac12 \max_{v} v^{\top} (I - \Lambda^-_B)_+ v ,
\]

and the box has the following bounds.

- **Tilted upper bound:** \(F(c + d) \le F(c) + s \cdot d + r^+_B\), with mass \(M_B\) as above.
- **Lower bound:** \(F(c + d) \ge F(c) + s \cdot d - r^-_B\), with mass \(m_B = e^{-r^+_B - r^-_B} M_B \le \int_B e^F\).
- **Flat upper bound**, for large boxes only. \(G\) reaches its maximum over a box at a vertex, and \(-\lvert z \rvert^2/2\) is largest at the point of the box closest to the origin, so
\[
F(z) \le U_B = \max_{v \in \text{vertices}} G(c + v) - \frac12 \sum_k \big(\max(\lvert c_k \rvert - h_k, 0)\big)^2 .
\]
A box whose flat mass \(\operatorname{vol}(B)\, e^{U_B}\) is smaller than its tilted mass proposes uniformly and accepts with probability \(e^{F(z) - U_B}\).

Near the mass of \(p(z)\), \(\nabla^2 G\) nearly cancels the \(-I\) of the Gaussian in the directions that the data select, so \(F\) is almost flat there. The local matrices capture this, while the global bound does not: the global bound is about 15 on BLAT at rank 3, against at most 1.5 for the true curvature. With the local bounds, boxes are several times wider for the same tightness. That is what makes rank 3 affordable.

### Building the cells

Since \(G(z) \le A + \sum_k S_k \lvert z_k \rvert\), with \(A = \sum_i (\max_a g_i(a) + \log q)\) and \(S_k = \sum_i \max_a \lvert W_{ik}(a) \rvert\), \(F\) falls below \(F(0) - 50\) outside the box of half widths

\[
R_k = S_k + \sqrt{S_k^2 + 2\Big(A - F(0) + 50 + \tfrac12 \textstyle\sum_{k' \neq k} S_{k'}^2\Big)} + 1 .
\]

This box is split into \(8^K\) cells, which are then refined. At every step, with the lower bound \(\log Z_- = \log \sum_B m_B\) of the integral:

- a cell is **dropped** if \(\log M_B < \log Z_- - 50\), because it holds at most \(e^{-50}\) of the mass;
- a cell is **halved** if \(\log M_B - \log m_B > 0.3\) and \(\log M_B > \log Z_- - 20\). It is cut across the direction \(k\) with the largest share of the gap, \(h_k^2 \big[(\Lambda^+_B - I)_+ + (I - \Lambda^-_B)_+\big]_{kk}\).

Only the new cells are bounded at each step, and refinement stops when no cell qualifies. Every refined cell accepts at least \(e^{-0.3} \approx 74\%\) of its proposals. Cells below \(e^{-20}\) of the mass keep looser bounds; they lower the acceptance only negligibly and never affect correctness.

| BLAT (\(L = 215\), \(q = 21\)) | rank 1 | rank 2 | rank 3 |
|---|---|---|---|
| cells | 45 | 1,800 | 77,000 |
| construction on the GPU | 0.3 s | 0.2 s | 9 s |
| acceptance of the sampler | 0.98 | 0.93 | 0.94 |
| 100,000 exact samples | 0.5 s | 0.5 s | 0.6 s |

The sums of \(m_B\) and \(M_B\) over the cells also give rigorous lower and upper bounds of \(\log Z\), a few hundredths of a nat apart. The value of \(\log Z\) itself comes from the lattice sum.

### The lattice sum

The Fourier transform of a unit Gaussian decays as \(e^{-\lvert k \rvert^2/2}\), and so does that of \(e^F\), a mixture of unit Gaussians, whatever the parameters. By the Poisson summation formula, the sum over a lattice of spacing \(\delta\) misses the integral by the Fourier transform at the points of the reciprocal lattice:

\[
\Big\lvert\, \delta^K \sum_{n \in \mathbb{Z}^K} e^{F(\delta n)} - \int e^{F}\, \Big\rvert
\;\le\; \int e^{F} \sum_{n \neq 0} e^{-2\pi^2 \lvert n \rvert^2/\delta^2}
\;\approx\; 2K\, e^{-2\pi^2/\delta^2} \int e^{F} .
\]

With \(\delta = 1\) the relative error is below \(2 \times 10^{-8}\) for \(K = 3\). The averages in the gradient are lattice sums too, with comparable accuracy:

\[
\mathbb{E}_z\big[p(x_i = a \mid z)\big] \approx \sum_n \omega_n\, p(x_i = a \mid z_n), \qquad
\mathbb{E}_z\big[z_k\, p(x_i = a \mid z)\big] \approx \sum_n \omega_n\, z_{n,k}\; p(x_i = a \mid z_n), \qquad
\omega_n = \frac{e^{F(z_n)}}{\sum_{n'} e^{F(z_{n'})}} .
\]

Only the lattice points with \(F\) within 32 of its maximum are summed. They are found in two steps.

1. **Seeds.** Every overlap \(m(x)\) lies in the box \(\big[\sum_i \min_a W_{ik}(a),\, \sum_i \max_a W_{ik}(a)\big]\). \(F\) is evaluated on the coarse lattice of spacing 3 that covers this box. The centre of any mixture component is at most \(3\sqrt K/2\) from a coarse node, where the component has lost at most a factor \(e^{-9K/8}\). Every bump of \(p(z)\) therefore shows on the coarse lattice. The coarse nodes within 30 of the largest are the seeds.
2. **Growth.** The set is extended by the neighbours of every point within 32 of the maximum, until no point is missing.

During a fit the points of the previous evaluation are added to the seeds, so the growth usually stops at once. \(\log Z\) is the log of the lattice sum plus \(K \log \delta - \frac K2 \log 2\pi\).

### Relation to the PCA entropy-grid model

A low-rank model can also be written as a density on the projections \(m(x) = U^{\top}(x - c)/\sqrt L\) of the sequences on the first principal components \(U\): \(p(m) \propto e^{L[S(m) + b \cdot m + \frac12 m^{\top} K m]}\), with \(S(m)\) the entropy of the projections computed by a Legendre transform. For two components that model has five parameters: \(b\) is a field confined to the plane of the principal components, and the base measure \(S\) carries no single-site fields. It is a large-\(L\) approximation, and its sequences are drawn from the tilted product measure at a grid cell.

The Gaussian construction differs on three points:

- the weights \(W\) are fitted freely rather than fixed to the principal components;
- the fields are a full table, so the single-site frequencies are matched;
- nothing is approximated. The integral over \(z\) is the finite-\(L\) object whose saddle point gives the Legendre transform, and \(z\) plays the role of the conjugate field.
