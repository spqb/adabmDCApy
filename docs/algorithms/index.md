# How it works

These pages explain what the commands do between reading the alignment and writing their files, why they do it, and how to read what they report.

## The model in one paragraph

A Potts model assigns every aligned sequence \(x = (x_1, \dots, x_L)\) an energy made of a **field** \(h_i(x_i)\) for each position and a **coupling** \(J_{ij}(x_i, x_j)\) for each pair of positions, and a probability \(p(x) \propto e^{-E(x)}\). Training adjusts the parameters until the model reproduces the frequencies of each residue at each position and of each residue pair at each pair of positions, as measured on the (reweighted) alignment. These are the maximum-likelihood conditions. The couplings then capture the direct dependencies between positions, which is what makes them useful for contact prediction and mutation scoring, and the model can generate new sequences with the statistics of the family.

## The one thing that can go wrong

The model's frequencies cannot be computed exactly: they are averages over \(q^L\) sequences. They are estimated from a population of Markov chains (2,000 by default) that sample the current model. **Training and every quality metric it reports rest on the assumption that these chains are at equilibrium.** If they are not, for instance because the family has subfamilies the chains rarely move between, the gradient points in the wrong direction, and the metrics, computed from the same chains, cannot show it.

Most of the machinery of adabmDCA exists to protect that assumption or to check it afterwards:

- **PTT** keeps a ladder of models from an exactly solvable profile to the current model, so that independent configurations keep arriving at the top (see [Training](training.md)).
- **Mixing checks** verify that the ladder actually renews its population (see [Sampling](sampling.md#ptt-sampling)).
- **The lag pause** detects when the chains fall behind a fast-changing model and lets them catch up.
- **Independent sampling** of the finished model, with the diagnostics of [Diagnose a run](diagnostics.md), is the final check.

## Pages

| Page | Covers |
| --- | --- |
| [Input and reweighting](input.md) | Validation of the alignment, sequence weights, pseudocount, held-out splits |
| [Training](training.md) | PCD and PTT, one PTT update, ladder maintenance, the adaptive optimizer, sparse models, stopping, recovery, the training plots and output files |
| [Sampling](sampling.md) | Local samplers, plain MCMC and its mixing time, PTT generation and population renewal, ladder health, the sampling plots |
| [Diagnose a run](diagnostics.md) | Triage of a stalled or failed run, the failure catalogue, checks of a finished model, overfitting tests |
| [Analysis](analysis.md) | Contacts, energies, mutation scores, entropy, reintegration |
| [Benchmarks](benchmarks.md) | Measured behaviour on six families: what to expect and how the strategies compare |

The exact formulas behind every reported quantity are in [Theory and definitions](../quantities/index.md).

## Test families

The figures and benchmarks come from runs on these families, each split 80/20 into training and validation sets by clustering at 80% identity. RF00379 and its split are included in the repository (`example_data/RF00379`) to try the commands of this documentation.

| Family | Type | L | q | Training sequences | M_eff |
| --- | --- | --- | --- | --- | --- |
| RF00379 | RNA (ydaO riboswitch) | 136 | 5 | 2,908 | 1,109 |
| RF00023 | RNA (tmRNA) | 374 | 5 | 6,484 | 2,686 |
| cm_russ_natural | protein (chorismate mutase) | 96 | 21 | 901 | 573 |
| bkace | protein | 272 | 21 | 13,058 | 3,525 |
| LBD | protein | 279 | 21 | 13,864 | 1,149 |
| beta_lactamase | protein | 215 | 21 | 128,419 | 39,255 |
