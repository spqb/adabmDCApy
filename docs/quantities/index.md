# Theory and definitions

Exact definitions of the model, of the training rule, and of every number that appears in a log, a summary file or a plot. Each entry gives the formula, how the package computes it, and what it can and cannot tell you.

| Page | Defines |
| --- | --- |
| [Potts model and statistics](model.md) | Energy and probability, gauge, sequence weights, pseudocount, frequencies, connected correlations, likelihood and its gradient |
| [Training quantities](training.md) | Pearson and slope, likelihood and entropy during training, the adaptive optimizer and the predicted KL, chain lag, sparse-model selection rules, stopping rules and counters |
| [Sampling quantities](sampling.md) | Local moves, swap acceptance, population renewal, autocorrelation times, the plain-MCMC mixing criterion, ladder health, distances |
| [Normalization, scoring and importance weights](derived.md) | \(\log Z\) from the PTT ladder and by thermodynamic integration, entropy, context-dependent entropy, contact and mutation scores, steering weights |

## Conventions

- Positions \(i, j = 1, \dots, L\) (zero-based in every output file); states \(a, b = 1, \dots, q\), with the gap as one state.
- Logarithms are natural; entropies and \(\log Z\) are in **nats per sequence**, log-likelihoods in **nats per site** (divided by \(L\)).
- \(\langle \cdot \rangle_p\) is an average under distribution \(p\); a hat marks an estimate from samples.
- Times in PTT are counted in **exchange rounds** (one round: a swap attempt between every neighbouring pair, then local sweeps in every rung). Times in plain MCMC are counted in **sweeps** (one update attempt per position of every chain). The two are not interchangeable.

All diagnostics are estimates from finite alignments and finite populations of chains. Their precision depends on sample size and on mixing, and no single number guarantees that a model is representative.
