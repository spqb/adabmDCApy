## Input data and preprocessing

### Input data
`adabmDCA 2.0` takes as input a multiple sequence alignment ([MSA](https://en.wikipedia.org/wiki/Multiple_sequence_alignment)) of aligned amino acid or nucleotide sequences, usually forming a protein or RNA family. DCA implementations require the data to be saved in FASTA format @pearson_improved_1988. 

`adabmDCA 2.0` implements the three default alphabets shown in table [](tab-alphabets), but the user can specify an ad-hoc alphabet as far as it is compatible with the input MSA.

:::{table} Default alphabets implemented in `adabmDCA 2.0`, where “-”indicates the alignment gap.
:name: tab-alphabets
:align: center

|  |  |
|:--:|:--:|
| **protein** | `-, A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y` |
| **rna** | `-, A, C, G, U` |
| **dna** | `-, A, C, G, T` |
:::

An example of a FASTA file format is shown in Figure [](fig-example_fasta). In particular, `adabmDCA 2.0` correctly handles FASTA files in which line breaks within a sequence are present.

```{figure} images/example_fasta_2.png
:name: fig-example_fasta
:alt: Fasta example
:align: center

Example of protein sequences formatted in FASTA format.
```

### Preprocessing

#### Preprocessing pipeline
The `adabmDCA 2.0` code applies the following preprocessing pipeline to
the input MSA:

-   Remove the sequences having some tokens not included in the default
    alphabet;
-   Compute the importance weights for the sequences in the MSA;
-   Apply a pseudocount to compute the MSA stastistics.

Their precise implementation is described in the following.

#### Computing the importance weights
The sequence weights are computed to mitigate as much as possible the systematic biases in the data, such as correlations due to the phylogeny or over-representation of some regions of the sequence space because of a sequencing bias.

Given an MSA of {math}`M` sequences, to compute the importance weight of each sequence $\pmb a^{(m)}$, {math}`m=1, \dots, M`, we consider {math}`N^{(m)}` as the number of sequences in the dataset having Hamming distance from {math}`\pmb a^{(m)}` smaller or equal to {math}`0.2 \cdot L` (this threshold can be tuned by the user). Then, the importance  weight of {math}`\pmb{a}^{(m)}` will be

$$
w^{(m)} = \frac{1}{N^{(m)}}.
$$(eqn:weights)

This reweighting allows us to give less importance to sequences found in very densely populated regions of the sequence space while enhancing the importance of isolated sequences.

#### Pseudo count and reweighted statistics
DCA models are trained to reproduce the one and two-site frequencies of the empirical data. To compute these, we introduce in the computation of the empirical statistics a small parameter $\alpha$, called pseudo count, that allows us to deal with unobserved (pairs of) symbols in one (or two) column(s) of the MSA. The one and two-site frequencies are given by

$$
    f_i(a) &= (1 - \alpha) f^{\mathrm{data}}_i(a) + \frac{\alpha}{q}, \\
    f_{ij}(a, b) &= (1 - \alpha) f^{\mathrm{data}}_{ij}(a, b) + \frac{\alpha}{q^2}

$$(eqn:freqs)

where {math}`f_i^{\mathrm{data}}(a)` and {math}`f_{ij}^{\mathrm{data}}(a, b)` are computed from the MSA as in Eq. [](#eqn:freqs). 