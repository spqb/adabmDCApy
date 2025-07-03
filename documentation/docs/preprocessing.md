## Input Data & Preprocessing

### 📥 Input Format

`adabmDCA 2.0` takes as input a **multiple sequence alignment (MSA)** in [FASTA format](https://en.wikipedia.org/wiki/Multiple_sequence_alignment), typically of aligned protein or RNA/DNA sequences (see [Fig. 1](#fig-fasta)).

The tool supports three built-in alphabets and also allows **custom alphabets**, as long as they match the MSA content.

| Type     | Alphabet Symbols |
|----------|------------------|
| protein  | `-, A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y` |
| RNA      | `-, A, C, G, U` |
| DNA      | `-, A, C, G, T` |

💡 *Line breaks within sequences are supported.*

<span id="fig-fasta">___Figure 1__: Example of a fasta file containg the MSA._</span>
![Example FASTA](images/example_fasta_2.png)

---

### 🔧 Preprocessing Steps

The following steps are applied to every input MSA:

1. **Remove sequences** with invalid symbols.
2. **Remove** duplicate sequences.
3. **Reweight** sequences to correct for phylogenetic and sampling bias (optional).
4. **Compute empirical statistics** using a pseudocount.

---

### ⚖️ Sequence Reweighting

To downweight overrepresented or phylogenetically related sequences, `adabmDCA` uses a clustering threshold (default: **80% identity**). The **weight** for sequence $\mathbf{a}^{(m)}$ is:

$$
w^{(m)} = \frac{1}{N^{(m)}}
$$

where $N^{(m)}$ is the number of sequences that have sequence identity with $\mathbf{a}^{(m)}$ above the clustering threshold.

- Set the threshold with `--clustering_seqid <value>`
- Disable with `--no_reweighting`

---

### 🧮 Pseudocount Regularization

A small **pseudocount** $\alpha$ is added to frequency estimates to prevent issues with rare or unobserved symbols:

- One-site frequency:   
  $f_i(a) = (1 - \alpha) f^{\mathrm{data}}_i(a) + \frac{\alpha}{q}$

- Two-site frequency:  
  $f_{ij}(a, b) = (1 - \alpha) f^{\mathrm{data}}_{ij}(a, b) + \frac{\alpha}{q^2}$

If not set via `--pseudocount`, the default is:
$$
\alpha = \frac{1}{M_{\text{eff}}}, \quad \text{with} \quad M_{\text{eff}} = \sum_{m=1}^M w^{(m)}
$$
being the _effective number of sequences._
---

✅ *These preprocessing steps ensure robust and unbiased training for all DCA methods supported by `adabmDCA`.*

