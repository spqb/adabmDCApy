# Reading the training and sampling plots

Look at the figures before reporting on a run, and describe what they show against the healthy patterns below. Full explanations with example figures: https://spqb.github.io/adabmDCApy/algorithms/training/#reading-the-training-plots and https://spqb.github.io/adabmDCApy/algorithms/sampling/#comparing-samples-with-the-data.

## Training (`--plot-training-logs`, or `adabmDCA plot-training-log model/history.csv`)

Read them together: a rising fit curve alone does not show that the chains kept up.

| Plot | Healthy | Warning |
| --- | --- | --- |
| `pearson` | Training rises toward the target; validation lower, levels off earlier. Dips at reservoir refreshes are normal at 10 sweeps | Sudden drops after a recovery (rolled-back updates); a decline over hundreds of updates |
| `loglikelihood` (PTT) | Training keeps rising; validation flattens, then declines where the validation stop acts. Single-update spikes ≤ 0.01 at ladder changes are log Z noise | Validation falling over hundreds of updates: overfitting. Compare absolute values within one family only |
| `ladder` (PTT) | Active replicas oscillate 2–3; saved models rise in steps (7–15 for bmDCA) | Many saved models: PTT sampling will be slow (edgeDCA reached 47) |
| `acceptance` (PTT) | Saw-tooth between ~0.25 and 0.5–0.9 | Long stretches pinned at 0.25 (inserting snapshots as fast as possible, typical at 10 sweeps on large families); below 0.1 triggers a mixing check |
| `learning_rates` (PTT) | At `--lr` while the trust region does not bind | Halved after recoveries and stays so; field rate at the floor (10⁻¹⁴) with `--no-ptt-centering`. Do not raise `--lr` because the trust region does not bind |
| `kl` (PTT) | Median well below the trust radius (~1e-3) | Pinned at 0.01: steps are capped |
| `lag` (PTT) | Below ~0.05 | Peaks near the 0.25 tolerance trigger pauses; repeated pauses: finding F2 of `adabmDCA diagnose` |
| `density` (sparse) | Rises (eaDCA, edgeDCA) or falls (edDCA) to the target | Stalls far from the target: step budget exhausted |

Vertical marks show the validation stop, lag pauses and recoveries when `events.jsonl` is present.

## Sampling (`sample --plot-sampling-logs`)

In this order:

1. **Convergence.** `sampling.json` says whether renewal was reached (PTT) or a mixing time found (MCMC); if not, do not use the samples. `ptt_renewal.png` (PTT): the old share of the ladder (G) must fall to the tolerance (`--ptt-renewal-tolerance`, 0.01, per model); renewal is a diagnostic, not a proof, since a mode never reached goes unnoticed. `pearson_sampling.png` (MCMC): must rise and settle; a peak then decline means the model was trained out of equilibrium (retrain with PTT), and a monotone plateau alone proves nothing.
2. **`statistics_scatter.png`.** Single-site frequencies (a check of the sampling: Pearson ≈ 1, slope ≈ 1), connected correlations (Pearson 0.003–0.007 below the last training value with the training set as `-d` and as many samples as training chains), three-point correlations (a prediction the model was not fitted to; the Pearson is inflated by sign agreement, read the slope and R² too). R² falls when the slope departs from 1, Pearson does not.
3. **`pca_1_2.png`, `pca_3_4.png`.** Generated sequences cover every natural cluster with similar density in the scatter and the marginal histograms. A missing or over-populated cluster is more informative than any aggregate number.
4. **`data_vs_samples.png`.** Share of each k-means cluster of the data, samples/data ratio on the label: ×0.8–×1.25 is typical of well-sampled models; read neighbouring clusters together (they trade samples across their boundary). Energy histograms: samples sit higher than training sequences (expected), only the shapes matter; a second peak or a much broader distribution points to a mode the data lack.
5. **`ptt_ladder_health.png`** (PTT). Trapped-configuration bars: grey within chance, amber slow but renewed, red blocking renewal; a red title means renewal failed.
6. **`distances.png`** (with `-v`). All-pair curves should coincide (samples as diverse as the family; shifted left means collapsed). Nearest-training distance of generated (orange) should not sit left of held-out (green), with no pile at zero. Generated slightly farther than held-out is normal with reweighting.
7. **`energy_vs_cde.png`.** Energy against summed context-dependent entropy; the fitted slope λ is the `--local-lambda` for local free energies.

The cluster numbers of `data_vs_samples.png` are recomputed for every run and do not correspond between plots.
