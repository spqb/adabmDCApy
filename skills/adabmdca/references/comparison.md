# Comparing sequences with the data, and contacts with a structure

## Generated sequences against the natural alignment

When the user asks to compare generated (or designed, or sequenced) sequences with the natural ones, always show the **PCA with marginal histograms**, then add the statistics that PCA cannot see.

**If they come from an adabmDCA model**, sample with `--plot-sampling-logs` and a reference: `pca_1_2.png`, `pca_3_4.png`, `statistics_scatter.png`, `data_vs_samples.png` and (with `-v`) `distances.png` are all made for you. Read them with [plots.md](plots.md).

**Otherwise** (another generator, an experiment, an older run), use `compare_pca`:

```python
from adabmDCA import compare_pca

comparison = compare_pca("train.fasta", "generated.fasta")     # alphabet detected from the data
comparison.save_plots("comparison")                            # comparison/pca_1_2.png, pca_3_4.png
comparison.explained_variance_ratio                            # share of the variance of each component
fig = comparison.plot(components=(0, 1), labels=("Natural", "ProteinMPNN"))
```

What it does, and what to do the same way if you write the analysis yourself:

- **The components are fitted on the natural sequences only**; the generated ones are projected on them. Never fit the PCA on the union: generated sequences would then shape the axes they are judged on.
- **The natural sequences are drawn with their weights** (`compute_weights`, 80% identity by default; `data_weights=` to pass your own, `no_reweighting=True` to turn it off). The model is fitted to the weighted distribution, so unweighted data make the histograms differ by construction on redundant families.
- **Same number of sequences on both sides** where possible (`n_reference`, default 10,000, at most the alignment size).
- **Only the leading components are computed**, by randomized low-rank PCA (`torch.pca_lowrank`) on the one-hot sequences. In your own code use `torch.pca_lowrank(X, q=k)` or, if scikit-learn is installed (it is not a dependency), `PCA(n_components=k, svd_solver="randomized")`. Never a full SVD or an eigendecomposition of the (L·q)×(L·q) covariance.
- **Plot with `adabmDCA.plot.plot_PCA`** (scatter with marginal histograms, explained variance on the axes), natural `#31688E`, generated `#E76F51`, as in the package's own figures. Show PC1–2 and PC3–4.

PCA shows only a few directions of large variance and can miss small subfamilies. Propose these alongside it:

| Check | How | What to look for |
| --- | --- | --- |
| Single-site frequencies and connected correlations | `get_freq_single_point`, `get_freq_two_points` (natural with weights), `get_correlation_two_points` → Pearson and slope; `adabmDCA.stats.extract_Cij_from_freq` + `adabmDCA.plot.plot_cij_scatter` for the scatter | Points on the diagonal; compare at equal sample size, the Pearson grows with it |
| Energies under a model | `compute_energies(fasta_path=..., model=..., alphabet=...)` for both sets, overlaid histograms | Same shape; natural sequences sit lower because the model was fitted to them, so only the shapes are informative |
| Distance to the training set | Hamming distance of each sequence to its nearest training sequence, for the generated and for held-out sequences | Generated not closer than held-out, no identical copies (memorization) |

Statistics snippet (natural weighted, generated unweighted):

```python
import torch
from adabmDCA import (load_alignment, compute_weights, get_freq_single_point, get_freq_two_points,
                      get_correlation_two_points)

natural = load_alignment("train.fasta")
generated = load_alignment("generated.fasta", alphabet=natural.tokens)
x, y = natural.to_onehot().double(), generated.to_onehot().double()
w = compute_weights(x.argmax(-1), dtype=torch.float64)
fi, fij = get_freq_single_point(x, w), get_freq_two_points(x, w)
pi, pij = get_freq_single_point(y), get_freq_two_points(y)
pearson, slope = get_correlation_two_points(fij=fij, pij=pij, fi=fi, pi=pi)
```

## Predicted contacts against a structure

Contact scores refer to zero-based **alignment columns**. To compare them with a PDB structure, map columns to residues through the alignment row of the structure's sequence, then measure distances. Biopython is already a dependency:

```python
import numpy as np, pandas as pd
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from adabmDCA import load_alignment

residues = [r for r in PDBParser(QUIET=True).get_structure("s", "3CHY.pdb")[0]["A"] if is_aa(r, standard=True)]
pdb_seq = "".join(seq1(r.get_resname()) for r in residues)

aln = load_alignment("family.fasta")
# The row of the structure's protein: by name if the alignment has it, else the closest row (seconds for 50k rows).
local = PairwiseAligner(mode="local", match_score=1, mismatch_score=0, open_gap_score=-1, extend_gap_score=-0.5)
row = max(aln.sequences, key=lambda s: local.score(s.replace("-", ""), pdb_seq))

columns = [c for c, a in enumerate(row) if a != "-"]
glob = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1, open_gap_score=-4,
                       extend_gap_score=-0.5, end_gap_score=0)
blocks = glob.align(row.replace("-", ""), pdb_seq)[0].aligned
column_to_residue = {columns[i]: residues[j] for (a0, a1), (b0, b1) in zip(*blocks)
                     for i, j in zip(range(a0, a1), range(b0, b1))}

def distance(r1, r2):   # minimal heavy-atom distance
    x = np.array([a.coord for a in r1 if a.element != "H"]); y = np.array([a.coord for a in r2 if a.element != "H"])
    return np.sqrt(((x[:, None] - y[None]) ** 2).sum(-1)).min()

c = pd.read_csv("contacts/contact_map.csv")
c = c[((c.position_j - c.position_i).abs() > 4) & c.position_i.isin(column_to_residue) & c.position_j.isin(column_to_residue)]
top = c.sort_values("score", ascending=False).head(len(column_to_residue))
top["contact"] = [distance(column_to_residue[i], column_to_residue[j]) < 8.0 for i, j in zip(top.position_i, top.position_j)]
ppv = {k: top.contact.head(k).mean() for k in (len(top) // 2, len(top))}   # PPV at L/2 and L
```

- Report which row was used and how many columns map to residues; columns without a residue cannot be scored.
- 8 Å between heavy atoms is the usual threshold (Cβ–Cβ < 8 Å is the other common one: say which). Exclude |i−j| ≤ 4 for proteins; for RNA, base pairs are the meaningful contacts and the same distance criterion works on the nucleotides.
- Mean-field contacts (`adabmDCA contacts -d family.fasta`) give a quick baseline; on PF00072 against 3CHY they reached a PPV of 0.9 at L/2.
- For a picture, draw the score matrix with the native contacts overlaid (upper triangle predictions, lower triangle structure).
