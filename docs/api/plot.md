<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/plot.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `plot`





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/plot.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_PCA`

```python
plot_PCA(
    fig: Figure,
    data1: ndarray,
    pc1: int = 0,
    pc2: int = 1,
    data2: ndarray | None = None,
    labels: Union[List[str], str] = 'Data',
    colors: Union[List[str], str] = 'black',
    title: str | None = None
) → Figure
```

Makes the scatter plot of the components (pc1, pc2) of the input data and shows the histograms of the components. 



**Args:**
 
 - <b>`fig`</b> (plt.figure):  Figure to plot the data. 
 - <b>`data1`</b> (np.ndarray):  Data to plot. 
 - <b>`pc1`</b> (int, optional):  First principal direction. Defaults to 0. 
 - <b>`pc2`</b> (int, optional):  Second principal direction. Defaults to 1. 
 - <b>`data2`</b> (np.ndarray | None, optional):  Data to be superimposed to data1. Defaults to None. 
 - <b>`labels`</b> (List[str] | str, optional):  Labels to put in the legend. Defaults to "Data". 
 - <b>`colors`</b> (List[str] | str, optional):  Colors to be used. Defaults to "black". 
 - <b>`title`</b> (str | None, optional):  Title of the plot. Defaults to None. 



**Returns:**
 
 - <b>`Figure`</b>:  Updated figure. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/plot.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_pearson_sampling`

```python
plot_pearson_sampling(
    ax: Axes,
    checkpoints: ndarray,
    pearsons: ndarray,
    pearson_training: ndarray | None = None
)
```

Plots the Pearson correlation coefficient over sampling time. 



**Args:**
 
 - <b>`ax`</b> (Axes):  Axes to plot the data. 
 - <b>`checkpoints`</b> (np.ndarray):  Checkpoints of the sampling. 
 - <b>`pearsons`</b> (np.ndarray):  Pearson correlation coefficients at different checkpoints. 
 - <b>`pearson_training`</b> (np.ndarray | None, optional):  Pearson correlation coefficient obtained during training. Defaults to None. 



**Returns:**
 
 - <b>`Axes`</b>:  Updated axes. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/plot.py#L166"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_autocorrelation`

```python
plot_autocorrelation(
    ax: Axes,
    checkpoints: ndarray,
    autocorr: ndarray,
    gen_seqid: float,
    data_seqid: float
) → Axes
```

Plots the time-autocorrelation curve of the sequence identity and the generated and data sequence identities. 



**Args:**
 
 - <b>`ax`</b> (Axes):  Axes to plot the data. 
 - <b>`checkpoints`</b> (np.ndarray):  Checkpoints of the sampling. 
 - <b>`autocorr`</b> (np.ndarray):  Time-autocorrelation of the sequence identity. 
 - <b>`gen_seqid`</b> (float):  Sequence identity of the generated data. 
 - <b>`data_seqid`</b> (float):  Sequence identity of the data. 



**Returns:**
 
 - <b>`Axes`</b>:  Updated axes. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/plot.py#L203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_scatter_correlations`

```python
plot_scatter_correlations(
    ax: Tuple[Axes, Axes],
    Cij_data: ndarray,
    Cij_gen: ndarray,
    Cijk_data: ndarray,
    Cijk_gen: ndarray,
    pearson_Cij: float,
    pearson_Cijk: float
) → Axes
```

Plots the scatter plot of the data and generated Cij and Cijk values. 



**Args:**
 
 - <b>`ax`</b> (Axes):  Axes to plot the data. Must have 2 subplots. 
 - <b>`Cij_data`</b> (np.ndarray):  Data Cij values. 
 - <b>`Cij_gen`</b> (np.ndarray):  Generated Cij values. 
 - <b>`Cijk_data`</b> (np.ndarray):  Data Cijk values. 
 - <b>`Cijk_gen`</b> (np.ndarray):  Generated Cijk values. 
 - <b>`pearson_Cij`</b> (float):  Pearson correlation coefficient of Cij. 
 - <b>`pearson_Cijk`</b> (float):  Pearson correlation coefficient of Cijk. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  Updated axes. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/adabmDCA/plot.py#L253"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_contact_map`

```python
plot_contact_map(ax: Axes, cm: ndarray, title: str | None = None) → Axes
```

Plots the contact map. 



**Args:**
 
 - <b>`ax`</b> (Axes):  Axes to plot the contact map. 
 - <b>`cm`</b> (np.ndarray):  Contact map to plot. 
 - <b>`title`</b> (str | None, optional):  Title of the plot. Defaults to None. 



**Returns:**
 
 - <b>`Axes`</b>:  Updated axes. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
