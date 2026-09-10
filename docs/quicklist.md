# Command quick reference

The CLI detects standard protein, DNA, and RNA alphabets automatically. Add
`--alphabet <tokens>` when the data uses a custom alphabet or when an ambiguous
nucleotide alignment should be interpreted explicitly.

## Prepare an alignment

Convert FASTA or Stockholm input, remove insertions and duplicates, and write a
processing report:

```bash
adabmDCA preprocess family.sto \
    --output family.fasta \
    --remove-insertions \
    --remove-duplicates \
    --report family.preprocessing.json
```

## Train and resume models

Train a dense `bmDCA` model:

```bash
adabmDCA train --data family.fasta --output model
```

Resume from a parameter and chain checkpoint:

```bash
adabmDCA train --data family.fasta --output model \
    --path_params model/params.dat --path_chains model/chains.fasta
```

Train sparse models:

```bash
adabmDCA train --model eaDCA --data family.fasta --output model --nsweeps 5
adabmDCA train --model edgeDCA --data family.fasta --output model
adabmDCA train --model edDCA --data family.fasta --output model \
    --path_params dense_params.dat --path_chains dense_chains.fasta
```

Omit the `edDCA` checkpoint arguments to train the initial dense model before
decimation. See [Training](training.md) for stopping budgets and sparse-model
controls.

## Generate and analyze sequences

Generate sequences using a reference MSA to estimate the mixing time, and save
the full diagnostic plot set:

```bash
adabmDCA sample --path_params model/params.dat --data family.fasta \
    --output samples --ngen 1000 --plot
```

Score sequences, create a single-mutant library, and predict contacts:

```bash
adabmDCA energies --data sequences.fasta --path_params model/params.dat --output energies
adabmDCA dms --data wild_type.fasta --path_params model/params.dat --output mutations
adabmDCA contacts --path_params model/params.dat --output contacts
```

## Specialized workflows

Estimate model entropy by thermodynamic integration:

```bash
adabmDCA entropy --path_params model/params.dat --data family.fasta \
    --path_targetseq target.fasta --output entropy
```

Reintegrate experimental measurements:

```bash
adabmDCA reintegrate --data natural.fasta --reint tested.fasta \
    --adj adjustments.txt --output reintegrated_model
```

Create a profile-aware train/test split:

```bash
adabmDCA profmark -t1 0.5 -t2 0.5 --bestof 10 split/family family.fasta
```

Plot a structured training log:

```bash
adabmDCA plot-training-log model/adabmDCA.log
```

Run `adabmDCA <command> --help` for the complete and authoritative option list.
