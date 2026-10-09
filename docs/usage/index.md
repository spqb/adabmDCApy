# Use the package

Every task is a subcommand of `adabmDCA`. `adabmDCA --help` lists them and `adabmDCA <command> --help` gives the complete, authoritative list of options of the installed version. This section has one page per command: what it is for, the options that matter, and what it writes.

## A typical workflow

```text
family.sto ──preprocess──► family.fasta ──split-data──► train.fasta + test.fasta
                                                              │
                                                    train (PTT, validation stop)
                                                              │
                                     model/params.dat.gz + model/ptt.h5 + logs
                                                              │
           ┌───────────────┬──────────────┬─────────────┬─────┴─────────┐
        sample          energies         dms         contacts        entropy
   (check the model)
```

1. **Clean the alignment** with [`preprocess`](preprocess.md) if it comes from Stockholm, contains insertion columns or unknown symbols, or has very gappy rows.
2. **Hold out sequences** with [`split-data`](split-data.md). A validation set lets training stop at the right time and makes the overfitting checks possible.
3. **Train** with [`train`](train.md). The default is a fully connected `bmDCA` model trained with PTT.
4. **Check the model** by sampling from it with [`sample`](sample.md) and reading its diagnostic plots. A clean training run is not, by itself, proof of a good model.
5. **Use the model**: [score sequences](energies.md), [scan single mutations](dms.md), [predict contacts](contacts.md), [estimate the entropy](entropy.md), or [retrain with experimental feedback](reintegrate.md).

[`plot-training-log`](plot-training-log.md) redraws the training curves of a finished or running job. [Choosing settings](choosing.md) summarizes which strategy, model type, sampler and device to use, and [Python workflows](python.md) shows the same steps from Python.

## Command quick reference

```bash
# Prepare
adabmDCA preprocess family.sto -o family.fasta --remove-insertions --remove-duplicates --report family.json
adabmDCA split-data family family.fasta                        # family.train.fasta, family.test.fasta

# Train
adabmDCA train -d family.train.fasta -v family.test.fasta -o model          # bmDCA, PTT
adabmDCA train --strategy pcd -d family.train.fasta -o model_pcd            # bmDCA, PCD (fast)
adabmDCA train -m eaDCA -d family.train.fasta -o model_ea                   # sparse, PTT
adabmDCA train --strategy pcd -m edDCA -d family.train.fasta -o model_ed    # decimated, PCD only
adabmDCA train -d family.train.fasta -o model --ptt-resume model/ptt.h5 --max-gradient-steps 20000

# Sample
adabmDCA sample -p model/ptt.h5 -d family.train.fasta -v family.test.fasta --plot -o samples
adabmDCA sample --strategy pcd -p model/params.dat.gz -d family.train.fasta --plot -o samples_mcmc

# Analyse
adabmDCA energies -d sequences.fasta -p model/params.dat.gz -o energies
adabmDCA dms      -d wildtype.fasta  -p model/params.dat.gz -o dms
adabmDCA contacts -p model/params.dat.gz -o contacts
adabmDCA entropy  -p model/ptt.h5 -o entropy

# Other
adabmDCA reintegrate -d natural.fasta --reint tested.fasta --adj adjustments.txt -o reintegrated
adabmDCA plot-training-log model/history.csv
```

## Alphabets

Every command detects the standard alphabets automatically:

| Type | Symbols |
| --- | --- |
| protein | `- A C D E F G H I K L M N P Q R S T V W Y` |
| RNA | `- A C G U` |
| DNA | `- A C G T` |

Detection tries DNA, then RNA, then protein, and picks the first that fits. An alignment that contains only `A`, `C`, `G` and gaps is therefore read as DNA; pass `--alphabet rna` to read it as RNA. For any other symbol set, pass the tokens in order, for example `--alphabet ABCD-`. The order matters: it fixes the meaning of the state indices in the parameter file, so use the same string for every command that reads that model.

## Which model file to use

| Task | File |
| --- | --- |
| Scoring, mutation scans, contacts, plain MCMC sampling | `params.dat.gz` (or `params.dat`). A PTT archive `ptt.h5` also works: its final model is read |
| PTT sampling, PTT entropy, resuming PTT training | `ptt.h5` |
| Resuming PCD training | `params.dat.gz` **and** `chains.fasta` |

`params.dat.gz` is a gzip-compressed text file with one line per non-zero parameter: `J i j a b value` for couplings and `h i a value` for fields, with zero-based positions and the states written as alphabet symbols (`J 0 1 - A -0.179`). Every command reads it compressed or not, and `zcat` turns it back into plain text for other tools.

`ptt.h5` holds everything PTT needs besides the final model:

```text
ptt.h5
├── metadata            alphabet, configuration, counters, log Z estimate
├── replicas/           active ladder: model parameters and chains of each rung
├── ptt_checkpoints/    models saved along the training trajectory (the sampling ladder)
├── algorithm/          exact profile, reservoir, optimizer and recovery state
├── lineage             identity of each configuration as it moves between rungs
└── rng                 random-number state, for an exact restart
```

You never need to open it yourself: pass it to `sample`, `entropy` or `train --ptt-resume`. Keep it as long as you may want to sample from the model with PTT; it cannot be rebuilt from `params.dat.gz`.
