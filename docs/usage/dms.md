# Scan single mutations · `dms`

Score every single-residue substitution of a wild-type sequence, as an in-silico deep mutational scan:

```bash
adabmDCA dms -d wildtype.fasta -p model/params.dat.gz -o dms
```

For a sequence of length \(L\) over \(q\) symbols, this produces \(L(q-1)\) mutants (gaps included as a symbol).

## Inputs

| Option | Meaning |
| --- | --- |
| `-d`, `--data` | Aligned FASTA with the wild type. If it contains several sequences, the first is used |
| `-p`, `--path_params` | `params.dat(.gz)` or `ptt.h5` |
| `-o`, `--output` | Output folder |

The wild type must be aligned like the training data: same length, same alphabet.

## Outputs

`<name>_DMS.fasta`, `.csv` and `.json`, where `<name>` is the wild type's FASTA header with non-alphanumeric characters removed. Each mutant is labelled by mutation and score, for example:

```text
>G27A | DCAscore: -0.6
```

The score is \(\Delta E = E(\text{mutant}) - E(\text{wild type})\). **Negative values mean the model prefers the mutant**; \(-\Delta E\) is the log of the probability ratio between mutant and wild type. The mutation labels (`G27A`) use **zero-based** alignment positions, like every other output of the package; the CSV also gives `position_1based`. Check the numbering against your assay before joining the two, and remember that alignment positions are not residue numbers of the full-length protein when the alignment has removed insertions.

## Interpreting the scores

\(\Delta E\) measures how compatible a substitution is with the family's sequence statistics. It often correlates with measured fitness, but the strength of that correlation depends on the protein, the assay and the selected property; scores should be validated on experimental data before being used as fitness predictions. Substitutions at strongly coupled positions depend on the background: the same mutation can be favourable in one homologue and deleterious in another.

## In Python

```python
from adabmDCA import load_model

model = load_model("model/params.dat.gz", alphabet="protein")
scan = model.scan_mutations("MKV...", name="wild_type")
df = scan.to_dataframe()
scan.to_fasta("wild_type_DMS.fasta")
```

Definition: [mutation scores](../quantities/derived.md#mutation-scores).
