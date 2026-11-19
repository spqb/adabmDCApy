<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/stats.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.stats`




**Global Variables**
---------------
- **nan**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/stats.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_freq_single_point`

```python
get_freq_single_point(
    data: Tensor,
    weights: Optional[Tensor] = None,
    pseudo_count: float = 0.0
) → Tensor
```

Computes the single point frequencies of the input MSA.

**Args:**

 - <b>`data`</b> (torch.Tensor):  One-hot encoded data array.
 - <b>`weights`</b> (Optional[torch.Tensor], optional):  Weights of the sequences.
 - <b>`pseudo_count`</b> (float, optional):  Pseudo count to be added to the frequencies. Defaults to 0.0.



**Raises:**

 - <b>`ValueError`</b>:  If the input data is not a 3D tensor.



**Returns:**

 - <b>`torch.Tensor`</b>:  Single point frequencies.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/stats.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_freq_two_points`

```python
get_freq_two_points(
    data: Tensor,
    weights: Optional[Tensor] = None,
    pseudo_count: float = 0.0
) → Tensor
```

Computes the 2-points statistics of the input MSA.



**Args:**

 - <b>`data`</b> (torch.Tensor):  One-hot encoded data array.
 - <b>`weights`</b> (Optional[torch.Tensor], optional):  Array of weights to assign to the sequences of shape.
 - <b>`pseudo_count`</b> (float, optional):  Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.0.



**Raises:**

 - <b>`ValueError`</b>:  If the input data is not a 3D tensor.



**Returns:**

 - <b>`torch.Tensor`</b>:  Matrix of two-point frequencies of shape (L, q, L, q).


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/stats.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `generate_unique_triplets`

```python
generate_unique_triplets(
    L: int,
    ntriplets: int,
    device: device = device(type='cpu')
) → Tensor
```

Generates a set of unique triplets of positions. Used to compute the 3-points statistics.



**Args:**

 - <b>`L`</b> (int):  Length of the sequences.
 - <b>`ntriplets`</b> (int):  Number of triplets to be generated.
 - <b>`device`</b> (torch.device, optional):  Device to perform computations on. Defaults to "cpu".



**Returns:**

 - <b>`torch.Tensor`</b>:  Indices of up to ntriplets unique triplets, with shape (count, 3).


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/stats.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_freq_three_points`

```python
get_freq_three_points(
    nat: Tensor,
    gen: Tensor,
    ntriplets: int = 10000,
    triplets: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
    device: device = device(type='cpu')
) → Tuple[Tensor, Tensor]
```

Computes the 3-body connected correlation statistics of the input MSAs.



**Args:**

 - <b>`nat`</b> (torch.Tensor):  Input MSA representing natural data in one-hot encoding.
 - <b>`gen`</b> (torch.Tensor):  Input MSA representing generated data in one-hot encoding.
 - <b>`ntriplets`</b> (int):  Number of triplets to test. Defaults to 10000.
 - <b>`triplets`</b> (Optional[torch.Tensor], optional):  Predefined triplets to use for the computation. Defaults to None.
 - <b>`weights`</b> (Optional[torch.Tensor], optional):  Importance weights for the natural sequences. Defaults to None.
 - <b>`device`</b> (torch.device, optional):  Device to perform computations on. Defaults to "cpu".



**Returns:**

 - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  Natural and generated 3-points connected correlation for ntriplets randomly extracted triplets.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/stats.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_covariance_matrix`

```python
get_covariance_matrix(
    data: Tensor,
    weights: Optional[Tensor] = None,
    pseudo_count: float = 0.0
) → Tensor
```

Computes the weighted covariance matrix of the input multi sequence alignment.



**Args:**

 - <b>`data`</b> (torch.Tensor):  Input MSA in one-hot variables.
 - <b>`weights`</b> (torch.Tensor | None, optional):  Importance weights of the sequences.
 - <b>`pseudo_count`</b> (float, optional):  Pseudo count. Defaults to 0.0.



**Returns:**

 - <b>`torch.Tensor`</b>:  Covariance matrix.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/stats.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `extract_Cij_from_freq`

```python
extract_Cij_from_freq(
    fij: Tensor,
    pij: Tensor,
    fi: Tensor,
    pi: Tensor,
    mask: Optional[Tensor] = None
) → Tuple[Tensor, Tensor]
```

Extracts the lower triangular part of the covariance matrices of the natural data and generated data starting from the frequencies.



**Args:**

 - <b>`fij`</b> (torch.Tensor):  Two-point frequencies of the natural data.
 - <b>`pij`</b> (torch.Tensor):  Two-point frequencies of the generated data.
 - <b>`fi`</b> (torch.Tensor):  Single-point frequencies of the natural data.
 - <b>`pi`</b> (torch.Tensor):  Single-point frequencies of the generated data.
 - <b>`mask`</b> (Optional[torch.Tensor], optional):  Mask for comparing just a subset of the couplings. Defaults to None.



**Returns:**

 - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  Extracted covariance matrix entries of the natural data and generated data.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/stats.py#L243"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `extract_Cij_from_seqs`

```python
extract_Cij_from_seqs(
    data: Tensor,
    chains: Tensor,
    weights: Optional[Tensor] = None,
    pseudo_count: float = 0.0,
    mask: Optional[Tensor] = None
) → Tuple[Tensor, Tensor]
```

Extracts the lower triangular part of the covariance matrices of the natural data and generated data starting from the sequences.



**Args:**

 - <b>`data`</b> (torch.Tensor):  Natural data sequences.
 - <b>`chains`</b> (torch.Tensor):  Generated data sequences.
 - <b>`weights`</b> (torch.Tensor | None, optional):  Weights of the sequences. Defaults to None.
 - <b>`pseudo_count`</b> (float, optional):  Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.0.
 - <b>`mask`</b> (torch.Tensor | None, optional):  Mask for comparing just a subset of the couplings. Defaults to None.



**Returns:**

 - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  Selected connected correlations of the natural and generated data.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/stats.py#L270"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_correlation_two_points`

```python
get_correlation_two_points(
    fij: Tensor,
    pij: Tensor,
    fi: Tensor,
    pi: Tensor,
    mask: Optional[Tensor] = None
) → Tuple[float, float]
```

Compute Pearson correlation and absolute regression slope for connected correlations.



**Args:**

 - <b>`fij`</b> (torch.Tensor):  Two-point frequencies of the natural data.
 - <b>`pij`</b> (torch.Tensor):  Two-point frequencies of the generated data.
 - <b>`fi`</b> (torch.Tensor):  Single-point frequencies of the natural data.
 - <b>`pi`</b> (torch.Tensor):  Single-point frequencies of the generated data.
 - <b>`mask`</b> (Optional[torch.Tensor], optional):  Mask to select the couplings to use for the correlation coefficient. Defaults to None.



**Returns:**

 - <b>`Tuple[float, float]`</b>:  Pearson correlation and absolute least-squares slope. Undefined values are NaN.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/stats.py#L355"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compare_three_point_correlations`

```python
compare_three_point_correlations(
    data: Tensor,
    samples: Tensor,
    q: int,
    data_weights: Optional[Tensor] = None,
    gap_index: Optional[int] = 0,
    n_seeds: int = 3000,
    max_triplets: int = 2000,
    max_pair_correlation: float = 0.1,
    frequency_range: Tuple[float, float] = (0.05, 0.95),
    z_threshold: float = 5.0,
    seed: int = 0,
    chunk: int = 4096
) → dict
```

Connected three-point correlations of data and samples on resolvable, not pairwise-implied triplets.

A connected correlation `C_ijk(a, b, c)` of one-hot indicators is not a three-body interaction: pairwise models produce it too. Two selections keep the comparison informative:


- no strong pair: the correlation coefficient of each two of the three  states, in the data, is below `max_pair_correlation` in absolute value.  Near-redundant states would turn `C_ijk` into a rescaled two-point  correlation (for indicators `x^2 = x`, so `x_u = x_v` gives  `C_uvw = (1 - 2 f_u) C_uw`). States are non-gap and have data  frequencies within `frequency_range`;
- resolvable: `z = max(|C_data| / SE_data, |C_samples| / SE_samples)`  is at least `z_threshold`. The standard errors are the weighted  standard deviation of the product of the three centred indicators over  the square root of the effective number of sequences: the sum of the  data weights (as given, so without reweighting every sequence counts as  independent and the error is optimistic) and the number of samples.  Taking the larger ratio does not reward agreement between data and  samples. With about 10^7 candidates, about five pure-noise ones reach  `z = 5`.

The search cannot enumerate all triplets. It draws `n_seeds` random state pairs that pass the selection and computes their correlation with every third state in one matrix product per data chunk. Of the triplets above the threshold it keeps a random sample of `max_triplets`, so that the comparison covers moderate correlations as well as the largest. The data are also split at random into two halves: the Pearson correlation of the selected triplets between them is the `noise_ceiling`, the agreement the data allow (slightly optimistic, since the selection used all the data).



**Args:**

 - <b>`data`</b>:  Integer-encoded data `(N, L)`.
 - <b>`samples`</b>:  Integer-encoded `(M, L)` or one-hot `(M, L, q)` samples.
 - <b>`q`</b>:  Number of states.
 - <b>`data_weights`</b>:  Sequence weights of the data; `None` weighs them equally.
 - <b>`gap_index`</b>:  Index of the gap state, excluded; `None` keeps all states.
 - <b>`n_seeds`</b>:  Random seed pairs.
 - <b>`max_triplets`</b>:  Largest number of triplets returned.
 - <b>`max_pair_correlation`</b>:  Largest absolute pair correlation coefficient.
 - <b>`frequency_range`</b>:  Allowed data frequencies of the three states.
 - <b>`z_threshold`</b>:  Smallest signal-to-noise ratio.
 - <b>`seed`</b>:  Seed of the seed pairs and of the half split.
 - <b>`chunk`</b>:  Sequences per one-hot block.



**Returns:**
 `triplets` (`(K, 3, 2)` array of `(site, state)`), `data`, `samples` and `z` (one value per triplet), `pearson`, `slope`, `r2` (`r2_about_identity`), `noise_ceiling` and `noise_ceiling_r2` (the same two measures between the data halves), `n_seeds`, `n_candidates` (triplets passing the pair and frequency selection), `n_resolved` (of them, above the threshold), `n_eff_data`, `n_samples` and the selection settings.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/stats.py#L511"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `r2_about_identity`

```python
r2_about_identity(reference, generated) → float
```

`1 - sum (generated - reference)^2 / sum (reference - mean)^2`.

Unlike the Pearson correlation it penalizes a wrong scale (a slope away from 1) and an offset as well as scatter: it is 1 only when the points lie on the diagonal, and can be negative.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
