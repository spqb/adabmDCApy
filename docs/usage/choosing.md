# Choosing settings

The defaults are a good starting point for any family. This page explains when to change them. Each recommendation comes from the comparison of six families (two RNA, four protein, 96 to 374 positions) summarized in [Benchmarks](../algorithms/benchmarks.md).

## In short

| Decision | Recommendation |
| --- | --- |
| Training strategy | **PTT** (default) for the model you will rely on. **PCD** (`--strategy pcd`) for a fast first look, or for small families without strong clusters |
| Model type | **bmDCA** (default). Use a sparse model only when sparsity is itself the goal |
| Validation set | Provide one if you can (`-v`): training then stops at the best held-out likelihood, and sampling can check for memorization. Without it, training stops at the Pearson target, and there is no way to know whether that value suits the family |
| Local sampler | **Metropolized Gibbs** (default) or Gibbs. Avoid Metropolis on proteins |
| Local sweeps (`--nsweeps`) | 10 (default). Raise to 100 when PTT reports trapped configurations |
| Learning rate (`--lr`) | 0.01 (default). Do not raise it with PTT |
| Sampling | PTT sampling from `ptt.h5`. Plain MCMC only when it reports a reached mixing time and agrees with PTT |
| Device | A GPU for protein families of a few hundred positions; a CPU with the `cpu` extra is fine for small families |

## PTT or PCD?

**PCD** keeps one population of chains and updates the model after a few sweeps of each chain. It is fast: on the benchmark families it reached the same training Pearson 9–41× sooner than PTT. Its weakness is that nothing checks whether the chains still represent the model. When the family has well-separated subfamilies, the chains stop moving between them, and both the gradient and the reported metrics become unreliable without warning.

**PTT** checks this continuously and repairs the ladder when needed, at the cost of more sampling work. It also gives normalized likelihoods during training, which makes the validation stop possible.

What the benchmarks showed:

- On the two RNA families and the small protein family, PCD and PTT models had the same validation likelihood, within 0.003 nats per site, and plain MCMC reached its mixing time for both. That does not make them equivalent. Resampled with plain MCMC, the RF00023 PCD model's Pearson first rose above its final level and then declined, the signature of a model trained out of equilibrium, and it ended 0.008 below the PTT model trained to the same target. On cm_russ, the PCD model's resampled Pearson levelled off 0.026 below its training value, against 0.015 for the PTT model.
- On the three large protein families, PTT models had a 0.025–0.035 nats per site higher validation likelihood. More importantly, plain MCMC **did not mix** on these families within 10,000 sweeps, for either model: the generated sequences missed or over-populated whole clusters, and their Pearson with the data fell to 0.70–0.93. PTT sampling of the same model reproduced the data (Pearson 0.945–0.962).

So: use PCD freely to explore, but train the final model with PTT, or at least sample the PCD model and check that it mixes (see below).

## Which model type?

| Model | What it learns | Strategies |
| --- | --- | --- |
| `bmDCA` | All fields and couplings | PTT, PCD |
| `eaDCA` | Starts with no couplings and activates individual coupling entries where the model disagrees most with the data | PTT, PCD |
| `edDCA` | Starts from a dense model and removes the least informative couplings until a target density | PCD only |
| `edgeDCA` | Starts with no couplings and activates whole position pairs, one at a time | PTT, PCD |

In the benchmarks `bmDCA` was better than `edgeDCA` on every count. With PTT, `edgeDCA` took as long or longer, stopped at a lower Pearson, and had an equal or lower validation likelihood; its archives also accumulated long ladders that made sampling slow. With PCD at bmDCA-level targets, `edgeDCA` often exhausted its 50,000 graph updates, and its models did not mix: their resampled Pearson fell to 0.5–0.7 while training reported 0.92–0.96.

Choose a sparse model when you need a sparse interaction graph (to interpret, or to keep a small parameter file), and then prefer PTT and check the model by sampling.

## How many local sweeps?

`--nsweeps` sets the local Monte Carlo work per chain between two parameter updates (with PTT, per replica and exchange round). The default 10 was as good as 100 on four of the six families, at a tenth of the cost.

Raise it to 100 when:

- PTT training fails with a message recommending more local sweeps, or logs configurations that "cannot swap toward the bottom";
- training stalls for a long time in a reservoir refresh or mixing check;
- an independent sample of the finished model disagrees with what training reported.

In all three cases the model has formed a narrow, deep basin that configurations can only enter or leave by local moves. Restart training from scratch: resuming the failed archive with more sweeps does not help, because the basin is already in the model. More chains (`--nchains`) do not help either: entering a basin is a per-chain event. Details are in [Diagnose a run](../algorithms/diagnostics.md#f1-a-narrow-basin-traps-configurations).

On bkace, 100 sweeps per round gave a smoother optimization (half the update-to-update noise, no lag pauses) but cost 6× more per update, so 10 sweeps reached a given validation likelihood sooner.

## Which local sampler?

All three samplers have the same stationary distribution; they differ in speed.

| Sampler | Per-site move | Recommended for |
| --- | --- | --- |
| `metropolized_gibbs` (default) | Draws a new state, different from the current one, from the site's conditional distribution | Everything |
| `gibbs` | Draws the state from the site's conditional distribution | Everything; equally good |
| `metropolis` | Proposes a uniformly random state | Small-alphabet data on CPU only |

Per sweep, Gibbs and Metropolized Gibbs decorrelate equally fast. Metropolis is cheaper per sweep but needs 2–3× more sweeps on RNA (q = 5) and about 10× more on proteins (q = 21), because random proposals at conserved positions are almost always rejected. Per independent sample on a GPU, Gibbs and Metropolized Gibbs were 5–7× cheaper than Metropolis on a protein (1.2–1.6× on a CPU, where Metropolis is very cheap per sweep). On an RNA family the three cost 2.9 to 4.6 ms per independent sample on a GPU: none was more than 1.6× costlier than the cheapest.

## Sampling: PTT or plain MCMC?

Sample a PTT-trained model with PTT (`adabmDCA sample -p model/ptt.h5`, the default). Plain MCMC (`--strategy pcd`) was 1.5–8× faster in the benchmarks. Use it only if two conditions hold. First, it reaches its mixing time: the run summary and `mix.log` report it, and the run does not hit `--max_nsweeps`. Second, the Pearson recorded during sampling (`pearson_sampling.png`) rises and settles on a plateau. If plain MCMC hits its sweep budget, its samples are not at equilibrium: do not use them, even when the Pearson with the data looks reasonable. Both conditions are necessary, not sufficient: the mixing-time criterion only checks that chains forget their recent past, and a steady plateau can still sit in one region of a multimodal model.

A PCD-trained model has no ladder and can only be sampled with plain MCMC. On families where MCMC does not mix, a PCD model therefore cannot be checked properly; retrain it with PTT. If its sampled Pearson rises above its final level and then declines, the model was trained out of equilibrium: it reproduces the data only after a transient of the sampler, not at its own equilibrium. Retrain it with PTT too.

## GPU or CPU?

| | GPU (RTX A5000) vs 16-thread CPU |
| --- | --- |
| One sampling sweep | GPU 15–20× faster for Gibbs and Metropolized Gibbs, 5–9× for Metropolis |
| bmDCA training (PCD and PTT) | GPU 12–16× faster |

The two devices draw different random numbers, so runs differ in detail, but they reach models of the same quality in a similar number of updates. On a CPU, a small RNA family such as RF00379 (136 positions) trains in about 5 minutes with PCD and 35 minutes with PTT; protein families of 200–300 positions take hours even on a GPU.

## Typical training times

On one RTX A5000, with default settings, bmDCA:

| Family | L | Training sequences | PCD | PTT (to validation plateau) |
| --- | --- | --- | --- | --- |
| RF00379 (RNA) | 136 | 2,908 | 20 s | 3 min |
| RF00023 (RNA) | 374 | 6,484 | 2 min | 58 min |
| cm_russ (protein) | 96 | 901 | 3 min | 31 min (100 sweeps) |
| β-lactamase (protein) | 215 | 128,419 | 10 min | 1.8 h |
| bkace (protein) | 272 | 13,058 | 5 min | 4.4 h |
| LBD (protein) | 279 | 13,864 | 12 min | >10 h (100 sweeps) |

PTT stopped at the validation plateau, and PCD was then given as target the training Pearson that PTT had reached there, so both runs end at the same training Pearson.

**Size is not the whole story.** Training time depends on \(L\), \(q\) and the number of sequences, but just as much on how clustered the data are. RF00023 is longer than bkace (374 vs 272 positions) yet trains with PTT in a quarter of the time. β-lactamase has ten times more sequences than bkace but trains faster. The difference comes from the structure of the family. When the data form well-separated subfamilies, the model develops several modes during training. PTT then works harder to keep its chains at equilibrium: it inserts more snapshots into the ladder, runs longer mixing checks and reservoir collections, pauses when the chains fall behind, and rolls back when a check fails. The extra time is the price of a gradient computed from equilibrium samples.

PCD performs none of these checks and never slows down, so it finishes in minutes whatever the structure of the data. On clustered families, that speed is partly illusory: the three families where PTT took longest (β-lactamase, bkace, LBD) are those where the PCD models could not be sampled at equilibrium afterwards (see [PTT or PCD?](#ptt-or-pcd)). A long PTT run is therefore a sign that the data are hard to model, not that something is wrong. When it stalls instead of just being slow, see [Diagnose a run](../algorithms/diagnostics.md).
