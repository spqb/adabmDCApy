<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/comparison.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.comparison`
Compare generated sequences with an alignment on the principal components of the alignment.

**Global Variables**
---------------
- **TYPE_CHECKING**
- **PCA_COLORS**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/comparison.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `pca_figure`

```python
pca_figure(
    reference_scores,
    generated_scores,
    explained_variance_ratio=None,
    components=(0, 1),
    labels=('Natural', 'Generated'),
    title=None,
    figure=None,
    dpi=192
) → Figure
```

Scatter of two principal components with their marginal histograms, natural and generated superimposed.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/comparison.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compare_pca`

```python
compare_pca(
    data: 'AlignmentInput',
    samples: 'AlignmentInput',
    alphabet: 'str | None' = None,
    data_weights: 'ndarray | Tensor | None' = None,
    clustering_seqid: 'float' = 0.8,
    no_reweighting: 'bool' = False,
    n_reference: 'int' = 10000,
    n_components: 'int' = 4,
    seed: 'int' = 0
) → PCAComparison
```

Project an alignment and generated sequences on the principal components of the alignment.

The natural sequences are first drawn with replacement according to their weights, as the model sees them, so that redundant clusters do not dominate the components or the histograms; `n_reference` of them are drawn (at most the size of the alignment). The components are fitted on these draws alone and the generated sequences are projected on them. Only the leading `n_components` are computed, by randomized low-rank PCA (`torch.pca_lowrank`) on the one-hot sequences.



**Args:**

 - <b>`data`</b>:  Natural alignment: path, `Alignment` or sequences.
 - <b>`samples`</b>:  Generated (or any other) sequences of the same length.
 - <b>`alphabet`</b>:  Alphabet of both; detected from `data` by default.
 - <b>`data_weights`</b>:  Weights of the retained natural sequences; computed by default.
 - <b>`clustering_seqid`</b>:  Identity above which natural sequences share weight.
 - <b>`no_reweighting`</b>:  Give every natural sequence the same weight.
 - <b>`n_reference`</b>:  Natural sequences drawn for the comparison.
 - <b>`n_components`</b>:  Leading components to compute.
 - <b>`seed`</b>:  Seed of the draw of the natural sequences.



**Returns:**
 A `PCAComparison`; `.plot()` and `.save_plots(folder)` draw it.



**Example:**
```python
from adabmDCA import compare_pca
comparison = compare_pca("train.fasta", "samples/samples.fasta")
comparison.save_plots("comparison")
```


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/comparison.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PCAComparison`
Natural and generated sequences on the principal components of the natural ones.



**Attributes:**

 - <b>`reference_scores`</b>:  Natural sequences, drawn with their weights, projected on the components; `(N, k)`.
 - <b>`generated_scores`</b>:  Generated sequences projected on the same components; `(M, k)`.
 - <b>`explained_variance_ratio`</b>:  Share of the variance of the natural sequences carried by each component.
 - <b>`tokens`</b>:  Alphabet of both alignments.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    reference_scores: 'ndarray',
    generated_scores: 'ndarray',
    explained_variance_ratio: 'ndarray',
    tokens: 'str'
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/comparison.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot`

```python
plot(
    components: 'tuple[int, int]' = (0, 1),
    labels: 'tuple[str, str]' = ('Natural', 'Generated'),
    title: 'str | None' = None,
    figure: 'Figure | None' = None
) → Figure
```

Scatter of two components (zero-based) with marginal histograms; returns the figure.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/comparison.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_plots`

```python
save_plots(
    directory: 'str | Path',
    label: 'str | None' = None,
    labels: 'tuple[str, str]' = ('Natural', 'Generated')
) → dict[str, Path]
```

Write `pca_1_2.png` and, with four components, `pca_3_4.png`; returns the paths.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
