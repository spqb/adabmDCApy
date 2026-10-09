# Analysis

What a trained model can tell you, and the limits of each reading. The commands are documented in [Use the package](../usage/index.md); the exact definitions are in [Normalization, scoring and importance weights](../quantities/derived.md).

## Energies and probabilities

The energy of a sequence is the sum of its fields and couplings with a minus sign, and its probability is \(e^{-E}/Z\). Within one model, the difference of two energies is a log-probability ratio, which needs no normalization. This is what [`energies`](../usage/energies.md) and [`dms`](../usage/dms.md) report.

Comparisons **across** models need \(\log Z\): the log-likelihood \(-E(x) - \log Z\) is comparable, the energy is not. Two models trained on the same family with different settings can differ in energy scale by orders of magnitude while assigning similar probabilities. In the benchmarks, the mean energy of the training data ranged from +50 to +300 under well-behaved models, but from −100 to −9,200 under sparse PCD models that did not mix, whose energies are therefore meaningless as absolute numbers.

Typical sequences of a model have higher energy than its training sequences, because the model is fitted to them; held-out natural sequences lie in between. A designed sequence with an energy in the range of the training sequences is strongly favoured by the model; one in the range of generated sequences is typical.

## Mutation scores

A single substitution changes the energy by the field difference at that position plus the coupling differences with every other position, evaluated in the background of the wild type. The same substitution can therefore score differently in different homologues: this is the epistasis captured by the couplings and absent from profile models. \(\Delta E < 0\) means the model prefers the mutant.

## Contacts

The couplings of a pair of positions form a \(q\times q\) block. Its Frobenius norm, after moving the parameters to the zero-sum gauge (so that the result does not depend on how parameters are split between fields and couplings) and excluding gaps, measures how strongly the two positions are directly coupled. The average product correction removes the part of each norm explained by the overall coupling level of the two positions, a background due mostly to conservation and to phylogenetic noise. Top-ranked pairs are enriched in structural contacts; for RNAs, base-paired helices appear as anti-diagonal stripes in the contact map.

## Entropy

The entropy \(S = \langle E\rangle + \log Z\) of the model measures the size of the sequence space it spreads probability over: about \(e^{S}\) sequences. Comparing it with the entropy of the independent-site profile (the bottom of the PTT ladder, `logs/ptt.log` row 0) shows how much the couplings restrict the family beyond conservation. On RF00379, `ptt.log` gives one row per model of the ladder:

| Model (training update) | 0 (profile) | 110 | 539 | 1098 | 2163 | 2475 (final) |
| --- | --- | --- | --- | --- | --- | --- |
| Entropy (nats) | 125.9 | 122.6 | 109.7 | 102.6 | 95.5 | 94.0 |
| Training-data log-likelihood per site | −0.924 | −0.821 | −0.701 | −0.644 | −0.591 | −0.581 |

The couplings lower the entropy by 32 nats: the final model concentrates on a fraction \(e^{-32} \approx 10^{-14}\) of the sequence space that the profile spreads over, while raising the likelihood of the natural sequences.

Both estimates of \(\log Z\) are described in [Estimate entropy](../usage/entropy.md). The PTT estimate sums free-energy differences along the training ladder; thermodynamic integration builds a path toward one target sequence. They agreed within 0.015 nats per site on the six benchmark families, but thermodynamic integration fails silently when the chains do not mix.

## Experimental reintegration

[Reintegration](../usage/reintegrate.md) follows [Calvanese et al. (2025)](https://arxiv.org/abs/2504.01593). Natural sequences \(x_m\) with weights \(w_m\) and tested sequences \(y_r\) with signed outcomes \(a_r\) enter one objective,

\[
\mathcal L_{\rm reint}(\theta) = \sum_m \frac{w_m}{M_{\rm eff}}\log p_\theta(x_m) + \frac{\lambda}{N_{\rm exp}}\sum_r a_r \log p_\theta(y_r),
\]

whose gradient has the usual form, data minus model frequencies, with the data frequencies computed from the combined, signed weights: \(w_m\) for natural sequences and \(\lambda M_{\rm eff} a_r / N_{\rm exp}\) for tested ones, normalized together. Sequences that passed the test add to the frequencies the model must reproduce; those that failed subtract from them, so the model lowers their probability. \(\lambda\) sets the weight of the whole experimental set relative to the natural alignment. Training uses PCD because PTT requires non-negative weights. When negative contributions make a combined frequency negative, it is clipped to zero before the pseudocount, and the model departs from the objective above; keep \(\lambda\) moderate.
