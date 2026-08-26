# Alignment conversion and preprocessing

adabmDCA provides lightweight utilities for inspecting, converting, and
cleaning multiple-sequence alignments before model training. Transformations
are explicit: reading an alignment never silently removes residues or
sequences.

## Convert Stockholm to FASTA

```python
from adabmDCA import convert_stockholm_to_fasta

conversion = convert_stockholm_to_fasta(
    "family.sto",
    "family.fasta",
)
print(conversion.to_dict())
```

The Stockholm reader preserves lowercase letters and dots in memory so their
insertion semantics remain available during preprocessing. On conversion,
dots are written as the canonical FASTA/adabmDCA gap symbol `-`; lowercase
residues remain unchanged. Stockholm annotations that cannot be represented in
FASTA are ignored. If a file contains multiple alignments, choose one
explicitly with `alignment_index`.

## Work with alignments in memory

```python
from adabmDCA import read_alignment

alignment = read_alignment("family.sto")

print(alignment.num_sequences)
print(alignment.sequence_length)
display(alignment.to_dataframe())
```

An `Alignment` is immutable and always contains sequences of equal, nonzero
length.

## Remove insertions

Dots and lowercase letters are deleted rather than converted to gaps or
uppercase residues:

```python
from adabmDCA import remove_insertions

cleaned = remove_insertions(alignment)
```

For example, `ACd.-E` becomes `AC-E`. The resulting alignment is revalidated;
an error is raised if insertion removal produces unequal sequence lengths.

## Filter sequences by gap fraction

```python
from adabmDCA import filter_gap_fraction

filtered = filter_gap_fraction(
    cleaned,
    max_gap_fraction=0.20,
)

alignment = filtered.alignment
print(filtered.removed_names)
print(filtered.gap_fractions)
print(filtered.keep_mask)
```

Sequences with a gap fraction strictly greater than the threshold are removed.
A sequence containing exactly 20% gaps is therefore retained for a threshold
of `0.20`.

## Compose a preprocessing pipeline

```python
from adabmDCA import preprocess_alignment

result = preprocess_alignment(
    "family.sto",
    output_path="family_clean.fasta",
    remove_insertions=True,
    max_gap_fraction=0.20,
    remove_duplicates=True,
    alphabet="protein",
)

print(result.report.to_dict())
result.report.to_json("family_clean.json")
```

The processing order is insertion removal (when requested), dot-to-hyphen gap
normalization, gap filtering, duplicate removal, alphabet validation, and
optional FASTA writing. Without `remove_insertions=True`, dots are retained as
alignment gaps and represented by `-`; with it, dots and lowercase insertion
residues are deleted. `result.keep_mask` maps the retained sequences back to
the original input order.

A reusable configuration is also supported:

```python
from adabmDCA import AlignmentProcessingConfig, preprocess_alignment

config = AlignmentProcessingConfig(
    remove_insertions=True,
    max_gap_fraction=0.20,
    remove_duplicates=True,
    alphabet="rna",
)
result = preprocess_alignment("family.sto", config=config)
```

## Command line

```bash
adabmDCA preprocess family.sto \
    --output family_clean.fasta \
    --remove-insertions \
    --max-gap-fraction 0.20 \
    --remove-duplicates \
    --alphabet protein \
    --report family_clean.json
```

Omit the transformation flags to perform format conversion only. Output FASTA
always uses `-`, never `.`, for retained alignment gaps. The input format is
detected from content by default and can be set explicitly with
`--input-format fasta` or `--input-format stockholm`.
