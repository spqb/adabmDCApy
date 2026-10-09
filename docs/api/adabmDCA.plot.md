<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.plot`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_PCA`

```python
plot_PCA(
    fig: Figure,
    data1: ndarray,
    pc1: int = 0,
    pc2: int = 1,
    data2: ndarray | None = None,
    labels: list[str] | str = 'Natural',
    colors: list[str] | str = '#31688E',
    title: str | None = None,
    explained_variance_ratio: ndarray | None = None
) → Figure
```

Makes the scatter plot of the components (pc1, pc2) of the input data and shows the histograms of the components.



**Args:**

 - <b>`fig`</b> (Figure):  Figure to plot the data.
 - <b>`data1`</b> (np.ndarray):  Data to plot.
 - <b>`pc1`</b> (int, optional):  First principal direction. Defaults to 0.
 - <b>`pc2`</b> (int, optional):  Second principal direction. Defaults to 1.
 - <b>`data2`</b> (Optional[np.ndarray], optional):  Data to be superimposed to data1. Defaults to None.
 - <b>`labels`</b> (Union[List[str], str], optional):  Labels to put in the legend. Defaults to "Data".
 - <b>`colors`</b> (Union[List[str], str], optional):  Colors to be used. Defaults to "black".
 - <b>`title`</b> (Optional[str], optional):  Title of the plot. Defaults to None.



**Returns:**

 - <b>`Figure`</b>:  Updated figure.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L209"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_pearson_sampling`

```python
plot_pearson_sampling(
    ax: Axes,
    checkpoints: ndarray,
    pearsons: ndarray,
    pearson_training: float | None = None
) → Axes
```

Plots the Pearson correlation coefficient over sampling time.



**Args:**

 - <b>`ax`</b> (Axes):  Axes to plot the data.
 - <b>`checkpoints`</b> (np.ndarray):  Checkpoints of the sampling.
 - <b>`pearsons`</b> (np.ndarray):  Pearson correlation coefficients at different checkpoints.
 - <b>`pearson_training`</b> (Optional[float], optional):  Pearson correlation coefficient obtained during training. Defaults to None.



**Returns:**

 - <b>`Axes`</b>:  Updated axes.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L273"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_autocorrelation`

```python
plot_autocorrelation(
    ax: Axes,
    checkpoints: ndarray,
    autocorr: ndarray,
    gen_seqid: float | ndarray,
    data_seqid: float | None = None,
    autocorr_std: ndarray | None = None,
    independent_std: ndarray | None = None
) → Axes
```

Plots the time-autocorrelation curve of the sequence identity and the generated and data sequence identities.



**Args:**

 - <b>`ax`</b> (Axes):  Axes to plot the data.
 - <b>`checkpoints`</b> (np.ndarray):  Checkpoints of the sampling.
 - <b>`autocorr`</b> (np.ndarray):  Time-autocorrelation of the sequence identity.
 - <b>`gen_seqid`</b> (float or np.ndarray):  Independent-chain sequence identity, either as a level or a curve.
 - <b>`data_seqid`</b> (float, optional):  Reference-data sequence identity level.
 - <b>`autocorr_std`</b> (np.ndarray, optional):  Uncertainty of the autocorrelation curve.
 - <b>`independent_std`</b> (np.ndarray, optional):  Uncertainty of the independent-chain curve.



**Returns:**

 - <b>`Axes`</b>:  Updated axes.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L390"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_ptt_autocorrelation`

```python
plot_ptt_autocorrelation(
    ax: Axes,
    correlation: ndarray,
    tau_int: float | None,
    tau_exp: float | None
) → Axes
```

Plot replica-index autocorrelation and the two PTT time estimates.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L434"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_ptt_renewal`

```python
plot_ptt_renewal(
    figure: Figure,
    history: dict[str, ndarray],
    tolerance: float,
    n_models: int,
    warmup_rounds: int | None,
    renewal_rounds: int | None,
    chunk_rounds: int = 0,
    stationary: bool = True
) → Figure
```

Plot PTT population renewal for the warmup and, if it ran, the stationary phase.

``history`` holds per-round ``{phase}_ladder_old`` (G, fraction of all ladder configurations born before the phase reference) and ``{phase}_endpoint_fresh`` (F, fraction of endpoint configurations born after it) for ``phase`` in ``warmup`` and ``stationary``. The top row shows G and F; the bottom row shows the old fractions G and 1 - F on a logarithmic scale with the stopping thresholds, and the exponential fit of G over the recent half of each phase (see ``renewal_forecast``). Renewal is decided by G reaching ``tolerance / n_models``; the endpoint fraction 1 - F can rise again when old configurations climb back up the ladder.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L524"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_ptt_ladder_health`

```python
plot_ptt_ladder_health(
    figure: Figure,
    health: dict,
    mixing: dict | None = None
) → Figure
```

Plot how configurations move through a PTT ladder.

Panels:


- swap acceptance per link: the mean and the median, 10% and 1% quantiles  of the per-configuration acceptance against random partners, for upper  configurations moving down and lower ones moving up. The ladder is built  so that means sit near the 0.25 target; low quantiles reveal  configurations that rarely cross;
- trapped configurations per link: mean age of the configurations that  practically cannot move down (swap probability below the immobility  threshold) divided by that of the others. Near 1, immobility is  transient; well above 1, configurations stay stuck above that link;
- configuration ages per replica: median and 99th percentile of the rounds  since birth at the bottom; an old tail that starts at one replica marks  a bottleneck below it;
- flow: the fraction of configurations at each replica that reached the top  since birth. With free diffusion it rises smoothly from 0 at the bottom  to 1 at the top; a step marks a bottleneck.

Free energies, effective sample sizes and Crooks slopes are in the log only.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L650"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_distance_comparison`

```python
plot_distance_comparison(figure: Figure, distances: dict) → Figure
```

Hamming distances within and between natural and generated sequences.

Left: all-pair distance distributions within the training sequences, within the samples, and between the two; matching curves mean the samples reproduce the family's diversity. Right: nearest-neighbour distances. A model that copies its training set puts generated -> training below training -> training and, with a held-out set, below held-out -> training, the distance at which unseen members of the family sit; generated -> generated near zero means the samples collapse onto few sequences. Natural sequences carry their reweighting weights.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L716"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_privet`

```python
plot_privet(figure: Figure, privet: dict) → Figure
```

PRIVET: the null law, the excess of samples near the training set, and per-sample scores.

Left: cumulative distribution of the (reweighted) training sequences' distances to their nearest other training sequence (steps) and the fitted extreme-value law (line), on a log scale to show the lower tail; the band is the fitted window. Middle: the same law against the distances of the generated and held-out sequences to their nearest training sequence; a generated curve above the null at small distances is an excess of sequences close to the training set, scored in the title (most improbable group, log10 probability under the null). Right: per sample, ``log10 p_train`` against ``log10 p_test``; flagged samples lie below the dashed line ``delta_p = threshold``.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L802"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_data_comparison`

```python
plot_data_comparison(
    figure: Figure,
    comparison: dict,
    pca_reference=None,
    pca_generated=None
) → Figure
```

Compare samples with the data: clusters of the data and energy distributions.

Panels: the data on its first two principal components, coloured by cluster (k-means on the principal components of the PCA plots); the share of data and samples in each cluster; and the energy distributions of data and samples under the model. A cluster whose share differs is a region the model, or its sampling, over- or under-weights. Training sequences have lower energies than samples of a fitted model, so a shift of the energy distributions is expected; their shapes, and held-out sequences, are what to compare.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L868"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_energy_cde_scatter`

```python
plot_energy_cde_scatter(
    ax: Axes,
    cde_sum: ndarray,
    energies: ndarray,
    fit: dict[str, float | int] | None
) → Axes
```

Plot sequence energies against summed CDE and their fitted line.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L912"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_cij_scatter`

```python
plot_cij_scatter(
    ax: Axes,
    Cij_data: ndarray,
    Cij_gen: ndarray,
    pearson: float | None = None
) → Axes
```

Plot reference versus generated connected two-site correlations.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L982"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_scatter_correlations`

```python
plot_scatter_correlations(
    ax: tuple[Axes, Axes],
    Cij_data: ndarray,
    Cij_gen: ndarray,
    Cijk_data: ndarray,
    Cijk_gen: ndarray,
    pearson_Cij: float,
    pearson_Cijk: float
) → tuple[Axes, Axes]
```

Plots the scatter plot of the data and generated Cij and Cijk values.



**Args:**

 - <b>`ax`</b> (Tuple[Axes, Axes]):  Tuple of 2 Axes to plot the data.
 - <b>`Cij_data`</b> (np.ndarray):  Data Cij values.
 - <b>`Cij_gen`</b> (np.ndarray):  Generated Cij values.
 - <b>`Cijk_data`</b> (np.ndarray):  Data Cijk values.
 - <b>`Cijk_gen`</b> (np.ndarray):  Generated Cijk values.
 - <b>`pearson_Cij`</b> (float):  Pearson correlation coefficient of Cij.
 - <b>`pearson_Cijk`</b> (float):  Pearson correlation coefficient of Cijk.



**Returns:**

 - <b>`Tuple[Axes, Axes]`</b>:  Updated axes.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/plot.py#L1032"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_contact_map`

```python
plot_contact_map(ax: Axes, cm: ndarray, title: str | None = None) → Axes
```

Plots the contact map.



**Args:**

 - <b>`ax`</b> (Axes):  Axes to plot the contact map.
 - <b>`cm`</b> (np.ndarray):  Contact map to plot.
 - <b>`title`</b> (Optional[str], optional):  Title of the plot. Defaults to None.



**Returns:**

 - <b>`Axes`</b>:  Updated axes.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
