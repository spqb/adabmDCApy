# Prepare an alignment · `preprocess`

Training expects an aligned FASTA file in which every sequence has the same length and uses one alphabet. `preprocess` converts Stockholm or FASTA input into such a file and, optionally, cleans it:

```bash
adabmDCA preprocess family.sto -o family.fasta \
    --remove-insertions --max-gap-fraction 0.2 --remove-duplicates \
    --report family.preprocessing.json
```

Without any transformation flag, it only converts the format. Reading an alignment never removes residues or sequences silently: every change is a flag, and the report records what each one did.

## Options

| Option | Effect |
| --- | --- |
| `input` | FASTA (plain or gzip) or Stockholm file. The format is detected from the content; force it with `--input-format fasta\|stockholm` |
| `-o`, `--output` | Output FASTA path |
| `--alignment-index N` | Which alignment to read when a Stockholm file contains several |
| `--remove-insertions` | Delete lowercase residues and `.` characters, the insertion columns of Stockholm and HMMER/Infernal output. `ACd.-E` becomes `AC-E` |
| `--max-gap-fraction F` | Drop sequences with **more than** a fraction `F` of gaps (a sequence with exactly 20% gaps survives `0.2`) |
| `--remove-duplicates` | Keep only the first copy of identical sequences |
| `--alphabet` | `auto` (default), `protein`, `rna`, `dna`, or a custom token string |
| `--unknown-tokens` | What to do with characters outside the alphabet: `gap` (default) replaces them with `-`, `remove` drops their sequences, `error` stops and lists them |
| `--gap-token` | Character counted as a gap (default `-`) |
| `--line-width` | Wrap sequences at this width; `0` writes one line per sequence |
| `--report` | Write a JSON report of the processing |

The steps run in this order: insertion removal, conversion of the remaining `.` to `-`, handling of unknown characters, gap filtering, duplicate removal, writing. Unknown characters replaced by gaps therefore count toward the gap filter. Without `--remove-insertions`, dots are kept as gaps. Alphabet detection happens after these transformations and ignores characters that belong to no standard alphabet.

!!! tip "Stockholm files from Rfam or Pfam"
    Seed and full alignments from Rfam and Pfam contain insertion columns in lowercase and dots. Use `--remove-insertions`, otherwise the insertion columns become (mostly gap) positions of the model.

## Outputs

- The cleaned FASTA at `-o`. Gaps are always written as `-`.
- With `--report`, a JSON file with the number of sequences and the alignment length before and after processing, the counts of deleted insertion characters, converted dots and replaced unknown characters, the number of sequences dropped by each filter, and the names of all dropped sequences.

## What happens at training time

`train` and the other commands apply a lighter, fixed validation to their input: sequences with symbols outside the alphabet are dropped, exact duplicates are removed, and the counts are printed and stored in the training log (`original_sequences`, `removed_invalid`, `removed_duplicates`, `retained_sequences`). Then each sequence receives a weight that reduces the influence of close homologues. With the default threshold of 80% identity, the effective number of sequences `M_eff` is typically a third to a tenth of the number of rows (RF00379: 2,908 rows, M_eff = 1,109; LBD: 13,864 rows, M_eff = 1,149). See [Input and reweighting](../algorithms/input.md).

## In Python

```python
from adabmDCA import preprocess_alignment

result = preprocess_alignment(
    "family.sto",
    output_path="family.fasta",
    remove_insertions=True,
    max_gap_fraction=0.2,
    remove_duplicates=True,
    alphabet="auto",
    unknown_tokens="gap",
)
print(result.report.to_dict())
result.report.to_json("family.preprocessing.json")
kept = result.keep_mask          # which input rows survived, in input order
```

The individual steps are also available on in-memory alignments, which are immutable:

```python
from adabmDCA import read_alignment, remove_insertions, filter_gap_fraction

raw = read_alignment("family.sto")                 # keeps lowercase and dots
cleaned = remove_insertions(raw)
filtered = filter_gap_fraction(cleaned, max_gap_fraction=0.2)
print(filtered.removed_names, filtered.gap_fractions)
alignment = filtered.alignment
```

`convert_stockholm_to_fasta("family.sto", "family.fasta")` performs the plain conversion. More in [Python workflows](python.md#alignments).
