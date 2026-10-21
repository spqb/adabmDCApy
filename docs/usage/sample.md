# Generate sequences · `sample`

Sample a PTT-trained model through the ladder saved in its archive:

```bash
adabmDCA sample -p model/ptt.h5 -d train.fasta -v validation.fasta --plot -o samples
```

Sample any model, including a PCD-trained one, with plain Markov chain Monte Carlo:

```bash
adabmDCA sample --strategy pcd -p model/params.dat.gz -d train.fasta --plot -o samples_mcmc
```

Both commands also **check the model**: they verify that the chains have equilibrated and compare the generated sequences with the natural ones. Sampling is the main way to validate a trained model, so run it with `--plot` and a held-out set at least once for every model you intend to use.

## How each strategy decides it is done

**PTT** starts every rung of the ladder from exact draws of the independent-site profile, then exchanges configurations between neighbouring models while refreshing the bottom. It stops when every configuration that was in the ladder at the start has been replaced by one born later at the bottom (*population renewal*). The returned sequences then descend from fresh profile draws and carry no memory of the initialization. This takes from a few hundred exchange rounds on small RNA families to a few thousand on large proteins. `--ptt-stationary` renews the ladder a second time for extra certainty, at about twice the cost.

**Plain MCMC** first measures the mixing time of the model: it runs chains started from natural sequences and records when the identity between a chain at time *t* and at time *t*/2 matches the identity between two independent chains. It then runs fresh chains for `--nmix` (default 2) mixing times, started from the profile (independent-site) model of `-d`, or from uniformly random sequences without `-d` or with `--init-distribution uniform`. If the mixing time is not reached within `--max_nsweeps` (default 5,000), sampling stops at the budget and **the samples are not at equilibrium**, even when they look reasonable. On three of the six benchmark families this happened at 10,000 sweeps, and the samples missed whole clusters of the data (see [Benchmarks](../algorithms/benchmarks.md#sampling-from-random-sequences)). Without `-d`, plain MCMC simply runs `--max_nsweeps` sweeps.

The mixing time is measured from natural sequences, so it cannot see states that only the chains' starting point leads into. Uniformly random sequences lie far from anything in the data, where the model is not constrained, and chains can fall from there into metastable states that single-site updates never leave, such as blocks of gaps ([details](../algorithms/diagnostics.md#plain-mcmc-and-blocks-of-gaps)). This is why the profile is the default start.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `-p`, `--path_params` | required | `ptt.h5` (PTT) or a parameter file / `ptt.h5` (plain MCMC, which then reads the final model) |
| `-o`, `--output` | required | Output folder |
| `--ngen` | chains per model in training (PTT), 2048 (plain MCMC) | Number of sequences |
| `-d`, `--data` | none (required for PTT) | Natural alignment used as reference for mixing, comparisons and likelihoods; normally the training set |
| `-v`, `--validation` | none | Held-out alignment, normally the validation set of training. With `--plot`, enables the overfitting checks (`--test` is an older alias) |
| `--plot` | off | Write the diagnostic figures. Requires `-d` |
| `--strategy` | `ptt` | `ptt` or `pcd` |
| `--ptt-local-sweeps` | 10 | Local sweeps per replica per exchange round |
| `--ptt-local-kernel` | `metropolized_gibbs` | Local move; `archived` keeps the one used in training |
| `--ptt-stationary` | off | Renew the ladder a second time before returning samples |
| `--ptt-max-rounds` | 20000 | Budget of exchange rounds |
| `--ptt-renewal-tolerance` | 0.01 | Largest fraction of old configurations allowed at renewal |
| `--ptt-return-full-ladder` | off | Also save `ladder_samples.h5`, with samples of every model of the ladder |
| `--sampler` | `metropolized_gibbs` | Local move for plain MCMC |
| `--beta` | 1.0 | Inverse temperature (plain MCMC). Values above 1 concentrate on low-energy sequences |
| `--nmix` | 2 | Mixing times to run (plain MCMC) |
| `--max_nsweeps` | 5000 | Sweep budget (plain MCMC) |
| `--init-sequences` | none | Start plain MCMC from these sequences, e.g. the `samples.fasta` of a PTT run. `--ngen` defaults to their number; a smaller value starts from a random subset |
| `--init-distribution` | profile with `-d`, uniform without | Distribution of the starting sequences of plain MCMC without `--init-sequences`: `profile` (independent-site draws from the weighted frequencies of `-d`) or `uniform` |
| `--nsweeps` | none | Run exactly this many sweeps, without measuring the mixing time (plain MCMC; replaces `--max_nsweeps` and `--nmix`) |
| `--record-every` | 1 | Sweeps between rows of `logs/sampling.log` (plain MCMC with `-d`) |
| `--nmeasure` | 10000 | Maximum number of reference sequences used for the mixing time, the PCA and the distances |
| `--diagnostic` | off | Print renewal progress and adjacent-model acceptances (PTT) |
| `--seed`, `--device`, `--dtype`, `-l` | | As for `train` |

Reweighting options (`-w`, `--clustering_seqid`, `--no_reweighting`) apply to the reference alignment.

## Outputs

Samples and summary at the top of the output folder:

| File | Content |
| --- | --- |
| `samples.fasta` | The generated sequences |
| `samples.csv` | One row per sequence: energy and summed context-dependent entropy |
| `sampling.json` | Settings, mixing outcome, all diagnostics as numbers |
| `ladder_samples.h5` | With `--ptt-return-full-ladder`: `--ngen` samples of every ladder model, from the profile (first) to the endpoint (last), drawn at the same rounds. Datasets `sequences` (token indices, models × sequences × sites, tokens in the `tokens` attribute), `energies` (each under its own model) and `models/` (the columns of `ptt.log`) |

Figures, with `--plot`:

| Figure | Shows | Strategy |
| --- | --- | --- |
| `ptt_renewal.png` | Old and fresh fractions of the ladder during sampling | PTT |
| `ptt_ladder_health.png` | Swap acceptance, trapped configurations, ages and flow along the ladder | PTT |
| `autocorrelation.png` | The mixing-time measurement | plain |
| `pearson_sampling.png` | Pearson with the data along the sampling run | plain |
| `cij_scatter.png` | Connected correlations, data vs samples | both |
| `pca_1_2.png`, `pca_3_4.png` | Natural and generated sequences on the principal components of the data | both |
| `data_vs_samples.png` | Share of each data cluster in data and samples; energy distributions | both |
| `distances.png` | Hamming distances within and between sets; nearest-neighbour distances | both |
| `energy_vs_cde.png` | Energy against summed context-dependent entropy, with fitted slope | both |

Numerical logs, in `logs/`: `mix.log`, `sampling.log` (Pearson, slope and mean chain energy along the run; with `--init-sequences` its first row, at sweep 0, describes the starting sequences), `data_vs_samples.log`, `distances.log`, and for PTT `ptt.log` (one row per ladder model: log Z, mean energy, entropy, likelihood of the data), `ptt_renewal.log`, `ptt_ladder_health.log` and `ptt_replicas.log`. With `-l`, file names are prefixed.

## Reading the result in five minutes

1. **Did it converge?** The completion summary and `sampling.json` say whether renewal was reached (PTT) or the mixing time was found (plain MCMC). If not, do not use the samples. With plain MCMC, also look at `pearson_sampling.png`: it should rise and settle. A curve that peaks and then declines means the model was trained out of equilibrium ([details](../algorithms/benchmarks.md#out-of-equilibrium-training-seen-in-resampling)).
2. **Pearson with the data** (summary, `cij_scatter.png`). What to compare it with depends on the reference given with `-d`. With the training set, the usual choice, expect the final training Pearson: a healthy model loses only 0.003–0.007. With the validation set, expect the final *validation* Pearson of training. Either way, sample as many sequences as training chains (the default for PTT): the Pearson grows with sample size, so smaller samples score lower. The distance checks of point 5 need the training set as `-d`.
3. **Clusters** (`pca_1_2.png`, `data_vs_samples.png`). Generated sequences should cover every cluster of the data in roughly the right proportion; ratios between ×0.8 and ×1.25 are typical of well-sampled models.
4. **Ladder health** (PTT, `ptt_ladder_health.png`). A bar above 2 in the trapped-configurations panel flags a link where configurations stay stuck.
5. **Memorization** (`distances.png`, with `-v`). Generated sequences should not be closer to the training set than held-out sequences are, and none should be identical to a training sequence.

The plots are explained one by one in [Sampling](../algorithms/sampling.md#comparing-samples-with-the-data) and [Diagnose a run](../algorithms/diagnostics.md#checking-a-finished-model).

## In Python

```python
from adabmDCA import load_alignment, sample_sequences

reference = load_alignment("train.fasta")
result = sample_sequences(
    model="model/ptt.h5", ptt=True, reference_fasta=reference,
    n_sequences=2048, collect_diagnostics=True,       # required unless initial_sequences is given
)
result.save_bundle("samples")
result.save_diagnostic_plots("samples")
df = result.to_dataframe()            # sequence, energy, cde_sum, ...
```

In Python, `ptt=False` (plain MCMC) is the default; pass `ptt=True` to sample through the ladder.

Plain MCMC starts from the profile of `reference_fasta` when it is given and from uniformly random sequences otherwise; `initial_distribution="profile"` or `"uniform"` chooses explicitly. It can also start from sequences already in memory: `initial_sequences` takes a path, an `Alignment`, a list of aligned strings, a `SamplingResult` or an array of token indices or one-hot vectors. With `measure_mixing_time=False` it runs exactly `n_sweeps` sweeps. For example, to follow how a model moves equilibrium samples of another one:

```python
ptt = sample_sequences(model="model/ptt.h5", ptt=True, reference_fasta=reference, n_sequences=2048)
long_run = sample_sequences(
    model="pcd/params.dat", initial_sequences=ptt, reference_fasta=reference,
    n_sweeps=50_000, measure_mixing_time=False, record_every=500,
)
long_run.sampling_history     # nsweeps (from 0), pearson, slope, mean_energy
```

Python can also **steer** sampling toward sequences with a property of your choice and reweight them back to the model; see [Python workflows](python.md#steered-sampling).
