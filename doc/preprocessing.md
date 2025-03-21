## Input data and preprocessing

### Input data
`adabmDCA 2.0` takes as input a multiple sequence alignment ([MSA](https://en.wikipedia.org/wiki/Multiple_sequence_alignment)) of aligned amino acid or nucleotide sequences, usually forming a protein or RNA family. DCA implementations require the data to be saved in FASTA format {cite:p}`pearson_improved_1988`. 

`adabmDCA 2.0` implements the three default alphabets shown in table [Alphabets](#alphabets), but the user can specify an ad-hoc alphabet as far as it is compatible with the input MSA.

:::{table}
:align: center
:name: alphabets

|  |  |
|:--:|:--:|
| **protein** | `-, A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y` |
| **rna** | `-, A, C, G, U` |
| **dna** | `-, A, C, G, T` |
:::

An example of a FASTA file format is shown in Figure [Example FASTA](fig-example_fasta). In particular, `adabmDCA 2.0` correctly handles FASTA files in which line breaks within a sequence are present.

```{figure} images/example_fasta_2.png
:name: fig-example_fasta
:alt: Fasta example
:align: center
:scale: 50

Example of protein sequences formatted in FASTA format.
```

### Preprocessing

#### Preprocessing pipeline
The `adabmDCA 2.0` code applies the following preprocessing pipeline to
the input MSA:

-   Remove the sequences having some tokens not included in the default
    alphabet;
-   Remove duplicate sequences in the alignment;
-   If needed (default), compute the importance weights for the sequences in the MSA;
-   Apply a pseudocount to compute the MSA stastistics.

Their precise implementation is described in the following.

#### Computing the importance weights
The sequence weights are computed to mitigate as much as possible the systematic biases in the data, such as correlations due to the phylogeny or over-representation of some regions of the sequence space because of a sequencing bias.

Given an MSA of {math}`M` sequences, to compute the importance weight of each sequence $\pmb a^{(m)}$, $m=1, \dots, M$, we consider $N^{(m)}$ as the number of sequences in the dataset having sequence identity from $\pmb a^{(m)}$ greater or equal to $0.8 \cdot L$ (this threshold can be tuned by the user). Then, the importance  weight of $\pmb{a}^{(m)}$ will be

$$
w^{(m)} = \frac{1}{N^{(m)}}.
$$(eqn:weights)

This reweighting allows us to give less importance to sequences found in very densely populated regions of the sequence space while enhancing the importance of isolated sequences.

The user can change the sequence identity threshold through the argument `--clustering_seqid <value>`, or disable the reweighting by specifying the flag `--no_reweighting`. 

#### Pseudo count and reweighted statistics
DCA models are trained to reproduce the one and two-site frequencies of the empirical data. To compute these, we introduce in the computation of the empirical statistics a small parameter $\alpha$, called pseudo count, that allows us to deal with unobserved (pairs of) symbols in one (or two) column(s) of the MSA. The one and two-site frequencies are given by

$$
    f_i(a) &= (1 - \alpha) f^{\mathrm{data}}_i(a) + \frac{\alpha}{q}, \\
    f_{ij}(a, b) &= (1 - \alpha) f^{\mathrm{data}}_{ij}(a, b) + \frac{\alpha}{q^2}

$$(eqn:freqs_pseudocount)

where $f_i^{\mathrm{data}}(a)$ and $f_{ij}^{\mathrm{data}}(a, b)$ are computed from the MSA as in Eq. [](#eqn:freqs). If not specified by the user (`--pseudocount <alpha>`), the script will automatically set the pseudocount to the default value $\alpha = 1/\sum_{m=1}^M w^{(m)} = 1/M_{\mathrm{eff}}$, where we defined $M_{\mathrm{eff}}$ as the *effective number of sequences* in the MSA.