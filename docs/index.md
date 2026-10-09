<p class="home-logo"><img src="assets/images/logo.png" alt="adabmDCA logo" width="180"></p>

# adabmDCA 2.1

`adabmDCA` learns a **Potts model** (a Boltzmann machine over sequences) from a multiple sequence alignment of proteins, RNA or DNA, and then uses it to:

- **generate** new sequences that share the statistics of the family;
- **score** sequences and single mutations by their statistical energy;
- **predict contacts** between alignment positions from the learned couplings;
- **estimate the entropy** and the normalization of the model;
- **retrain** the model with experimental feedback on tested sequences.

This site documents the **Python implementation**: its command line, its Python API, and the methods behind them. The name *adabmDCA 2.1* refers to the project; the Python package has its own version number, printed at the start of every command.

## Quick start

```bash
uv tool install adabmDCA                                   # or: python -m pip install adabmDCA

adabmDCA split-data family family.fasta                    # 80/20 homology-aware split
adabmDCA train  -d family.train.fasta -v family.test.fasta -o model
adabmDCA sample -p model/ptt.h5 -d family.train.fasta -v family.test.fasta \
                --plot -o samples
```

The first command keeps close homologues on the same side of the split. The second trains a fully connected model and stops when the likelihood of the held-out sequences no longer improves. The third generates 2,000 sequences from the trained model, checks that the sampler has equilibrated, and writes a set of diagnostic plots comparing them with the natural sequences. [Use the package](usage/index.md) walks through each step.

## Training and sampling with PTT

A Potts model is only as good as the Monte Carlo chains used to train and sample it. When a family has well-separated subfamilies, ordinary chains stay inside one of them for a very long time: the training gradient is then estimated from unrepresentative samples, and generated sequences miss whole clusters, while the usual metrics still look acceptable.

**Parallel Trajectory Tempering (PTT)**, the default for `train`, `sample` and `entropy`, addresses this. It keeps a ladder of models saved along the training trajectory, from an independent-site profile that can be sampled exactly up to the current model, and lets configurations travel along it. Fresh, independent configurations keep entering at the bottom and reach the top. The same ladder gives the log-partition function, hence normalized likelihoods and the entropy, at no extra cost. The method is described in [Béreux et al. (2026)](https://arxiv.org/abs/2607.27077).

The older **persistent contrastive divergence (PCD)** strategy, selected with `--strategy pcd`, is several times faster and is sufficient for families without strong cluster structure. On six benchmark families, PCD and PTT models reached the same held-out likelihood on the three easier ones. Even there, resampling revealed that the PCD model of one of them had been trained out of equilibrium. On the three large protein families, only PTT produced samples that could be shown to be at equilibrium (see [Benchmarks](algorithms/benchmarks.md)).

## Where to go

| If you want to… | Read |
| --- | --- |
| Run a command and know which files it writes | [Use the package](usage/index.md), one page per command |
| Choose a strategy, a model type, a sampler or a device | [Choosing settings](usage/choosing.md) |
| Work in a notebook or a Python application | [Python workflows](usage/python.md) |
| Understand what happens during training and sampling, and read the plots | [How it works](algorithms/index.md) |
| Find out why a run stalled or whether a model can be trusted | [Diagnose a run](algorithms/diagnostics.md) |
| Know what to expect on real data | [Benchmarks](algorithms/benchmarks.md) |
| See the exact definition of a quantity in a log or a plot | [Theory and definitions](quantities/index.md) |
| Look up a function signature | [Python API reference](api/index.md) |

Julia and C++ implementations of adabmDCA exist ([adabmDCA.jl](https://github.com/spqb/adabmDCA.jl), [adabmDCAc](https://github.com/spqb/adabmDCAc)) with the same general command shape; PTT, the diagnostics and the Python API described here belong to the Python package only.

## Citation

If you use adabmDCA in your research, please cite:

- The original adabmDCA:
  A. P. Muntoni, A. Pagnani, M. Weigt and F. Zamponi, *adabmDCA: adaptive Boltzmann machine learning for biological sequences*, BMC Bioinformatics **22**, 528 (2021). [doi:10.1186/s12859-021-04441-9](https://doi.org/10.1186/s12859-021-04441-9)
- The reference paper of the package:
  L. Rosset, R. Netti, A. P. Muntoni, M. Weigt and F. Zamponi, *adabmDCA 2.0 — a flexible but easy-to-use package for Direct Coupling Analysis*, Methods in Molecular Biology **2979**, 83–104 (2026). [doi:10.1007/978-1-0716-4828-5_6](https://doi.org/10.1007/978-1-0716-4828-5_6); preprint [arXiv:2501.18456](https://arxiv.org/abs/2501.18456)
- Parallel Trajectory Tempering, when you train or sample with PTT:
  N. Béreux, A. Decelle, C. Furtlehner and B. Seoane, *Equilibrium training of energy-based models with Parallel Trajectory Tempering*, arXiv:2607.27077 (2026). [arXiv:2607.27077](https://arxiv.org/abs/2607.27077)
