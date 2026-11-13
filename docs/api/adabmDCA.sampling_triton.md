<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.sampling_triton`
Optional Triton kernels for categorical GPU sampling.

**Global Variables**
---------------
- **SPARSE_MAX_DEGREE_FRACTION**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L743"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_triton_available`

```python
is_triton_available() → bool
```

Return whether Triton was imported successfully.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L748"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gibbs_sampling_triton`

```python
gibbs_sampling_triton(
    chains: 'Tensor',
    params: 'dict[str, Tensor]',
    nsweeps: 'int',
    beta: 'float' = 1.0,
    steps_per_launch: 'int' = 16,
    transpose_couplings: 'bool' = True,
    coupling_dtype: 'dtype | None' = None
) → Tensor
```

Run Gibbs updates, keeping each chain in registers across short chunks.

`steps_per_launch` controls sequential updates per kernel, independently of random-number chunking. `transpose_couplings` makes candidate fields contiguous at the cost of one extra coupling-sized allocation per call. Disable it to save memory or benchmark the original gather layout. `coupling_dtype` optionally converts sampling couplings without changing the caller's parameters. BF16 conversion and transposition use one copy.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L776"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gibbs_sampling_categorical_triton`

```python
gibbs_sampling_categorical_triton(
    states: 'Tensor',
    params: 'dict[str, Tensor]',
    nsweeps: 'int',
    beta: 'float' = 1.0,
    steps_per_launch: 'int' = 16,
    transpose_couplings: 'bool' = True,
    coupling_dtype: 'dtype | None' = None,
    sites: 'Tensor | None' = None
) → Tensor
```

Run Gibbs sampling directly on contiguous int32 categorical states.

Each step updates one uniformly drawn site, shared by all chains, for `nsweeps * L` steps. `sites` instead gives the site of every step (its length then sets the number of steps and `nsweeps` is ignored). Sparse graphs (see `sparse_coupling_layout`) read only the coupled pairs. Dense float32 models with at least 4096 chains use the row kernel of Metropolized Gibbs (`metropolized_gibbs_row_layout`, cached while the couplings are unchanged) unless `transpose_couplings` is disabled: the same random numbers and distributions, with fields that differ from the per-chain kernel by float32 rounding.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L855"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gibbs_step_independent_sites_triton`

```python
gibbs_step_independent_sites_triton(
    chains: 'Tensor',
    params: 'dict[str, Tensor]',
    beta: 'float' = 1.0
) → Tensor
```

Apply one fused Gibbs update at an independently drawn site per chain (CUDA, Triton).



**Args:**

 - <b>`chains`</b>:  One-hot chains, shape `(n_chains, L, q)`, on a CUDA device.
 - <b>`params`</b>:  `bias` `(L, q)` and `coupling_matrix` `(L, q, L, q)`.
 - <b>`beta`</b>:  Inverse temperature.



**Returns:**
 The updated chains.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L880"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolis_sampling_triton`

```python
metropolis_sampling_triton(
    chains: 'Tensor',
    params: 'dict[str, Tensor]',
    nsweeps: 'int',
    beta: 'float' = 1.0,
    steps_per_launch: 'int' = 16,
    coupling_dtype: 'dtype | None' = None
) → Tensor
```

Run Metropolis updates with `steps_per_launch` updates per kernel.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L898"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolis_sampling_categorical_triton`

```python
metropolis_sampling_categorical_triton(
    states: 'Tensor',
    params: 'dict[str, Tensor]',
    nsweeps: 'int',
    beta: 'float' = 1.0,
    steps_per_launch: 'int' = 16,
    coupling_dtype: 'dtype | None' = None
) → Tensor
```

Run Metropolis sampling directly on contiguous int32 categorical states.

Sparse graphs (see `sparse_coupling_layout`) read only the coupled pairs. Large dense float32 populations (see `_metropolis_columns_applies`) use the kernel with one thread per chain, on a transposed copy of the couplings that is cached while they are unchanged: the same random numbers and updates as the per-program kernel.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1004"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolis_sampling_replicas_categorical_triton`

```python
metropolis_sampling_replicas_categorical_triton(
    states: 'Tensor',
    biases: 'Tensor',
    couplings: 'Tensor',
    nsweeps: 'int',
    beta: 'float' = 1.0,
    steps_per_launch: 'int | None' = None
) → Tensor
```

Sample a stack of PTT replicas in one sequence of Triton launches.

`steps_per_launch=None` uses the tuned default for the alphabet size. Sparse graphs (see `sparse_coupling_layout`) read only the coupled pairs.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1076"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `transpose_couplings_for_metropolized_gibbs`

```python
transpose_couplings_for_metropolized_gibbs(couplings: 'Tensor') → Tensor
```

View (L, q, L, q) couplings as (site, source_site, source_state, state).

Stacking such views materializes the layout expected by `metropolized_gibbs_sampling_replicas_categorical_triton` in one copy.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1085"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolized_gibbs_sampling_replicas_categorical_triton`

```python
metropolized_gibbs_sampling_replicas_categorical_triton(
    states: 'Tensor',
    biases: 'Tensor',
    couplings: 'Tensor',
    nsweeps: 'int',
    beta: 'float' = 1.0,
    steps_per_launch: 'int' = 8
) → Tensor
```

Metropolized Gibbs sampling of a stack of replicas in one sequence of launches.

`couplings` has shape (R, L, L, q, q) in the layout returned by `transpose_couplings_for_metropolized_gibbs` for each replica. Every replica updates one random site per step, shared by all its chains.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolized_gibbs_row_layout`

```python
metropolized_gibbs_row_layout(
    couplings: 'Tensor',
    dtype: 'dtype | None' = None
) → Tensor
```

Stacked couplings `(R, L, q, L, q)` as the rows read by the dense Metropolized Gibbs kernel.

`rows[r, i, j, b, a] = J[r, i, a, j, b] - J[r, i, q - 1, j, b]` for `a < q - 1`, with shape `(R, L, L, q, stride)` and zero padding: the coupling fields that source site `j` in state `b` exerts on site `i`, relative to the last state of `i`. Shifting all the fields of a site by the same amount leaves its conditional distribution unchanged, so the last column is neither stored nor read. `dtype` (float32 or bfloat16, default: that of `couplings`) is the stored type; the differences are taken in float32.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1316"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sparse_coupling_layout`

```python
sparse_coupling_layout(
    couplings: 'Tensor',
    dtype: 'dtype | None' = None,
    source=None,
    method: 'str' = 'metropolized_gibbs'
)
```

The sparse layout of stacked couplings `(R, L, q, L, q)` if it pays off for `method`, else `None`.

Returns `(neighbours, blocks, degree)`: `neighbours` `(R, L, D)` lists each site's coupled sites (padded with -1), `blocks` `(R, L, D, q, q)` holds `blocks[r, i, k, b, a] = J[i, a, j_k, b]`. The degree and the layout are cached per `source` tensor (`couplings` by default); the layout is built only when used.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolized_gibbs_sampling_replicas_triton`

```python
metropolized_gibbs_sampling_replicas_triton(
    states: 'Tensor',
    biases: 'Tensor',
    couplings: 'Tensor',
    nsweeps: 'int',
    beta: 'float' = 1.0,
    steps_per_launch: 'int' = 8,
    coupling_dtype: 'dtype | None' = None
) → Tensor
```

Metropolized Gibbs sampling of a stack of replicas, from couplings in their original layout.

`couplings` has shape (R, L, q, L, q). Sparse graphs use the sparse kernel. Dense float32 models with at least 4096 chains in all use the row kernel (`metropolized_gibbs_row_layout`), which reads the couplings relative to the last state of each site: the same distributions, with fields that differ from the per-chain kernel by float32 rounding. Other dense models use the transposed copy of `metropolized_gibbs_sampling_replicas_categorical_triton`. Each layout is cached and reused while `couplings` is unchanged. `coupling_dtype=torch.bfloat16` stores the layout in BF16; fields and acceptance stay in the bias precision.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1412"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolized_gibbs_sampling_triton`

```python
metropolized_gibbs_sampling_triton(
    chains: 'Tensor',
    params: 'dict[str, Tensor]',
    nsweeps: 'int',
    beta: 'float' = 1.0,
    steps_per_launch: 'int' = 8,
    coupling_dtype: 'dtype | None' = None
) → Tensor
```

Metropolized Gibbs sampling of one-hot chains; see `metropolized_gibbs_sampling_categorical_triton`.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1432"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolized_gibbs_sampling_categorical_triton`

```python
metropolized_gibbs_sampling_categorical_triton(
    states: 'Tensor',
    params: 'dict[str, Tensor]',
    nsweeps: 'int',
    beta: 'float' = 1.0,
    steps_per_launch: 'int' = 8,
    coupling_dtype: 'dtype | None' = None
) → Tensor
```

Metropolized Gibbs sampling of contiguous CUDA int32 states (N, L).

Each update draws one site shared by all chains. A new state b != a is proposed from the site conditional restricted to the other states and accepted with min(1, (1 - p_a) / (1 - p_b)) (Liu 1996). Every chain reads, for every source position, a contiguous row of candidate couplings: from the row layout of `metropolized_gibbs_row_layout` for float32 models with at least 4096 chains (cached while the couplings are unchanged), otherwise from a transposed copy made once per call. Sparse graphs (see `sparse_coupling_layout`) read only each site's coupled neighbours instead. `coupling_dtype=torch.bfloat16` stores the couplings in BF16. Fields and acceptance use the bias precision, or FP32 for BF16 biases, as do the uniform random numbers. Couplings must have zero within-site blocks.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1500"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolis_step_independent_sites_triton`

```python
metropolis_step_independent_sites_triton(
    chains: 'Tensor',
    params: 'dict[str, Tensor]',
    beta: 'float' = 1.0
) → Tensor
```

Apply one fused Metropolis update at an independently drawn site per chain (CUDA, Triton).



**Args:**

 - <b>`chains`</b>:  One-hot chains, shape `(n_chains, L, q)`, on a CUDA device.
 - <b>`params`</b>:  `bias` `(L, q)` and `coupling_matrix` `(L, q, L, q)`.
 - <b>`beta`</b>:  Inverse temperature.



**Returns:**
 The updated chains.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1604"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exchange_log_acceptance_sparse_triton`

```python
exchange_log_acceptance_sparse_triton(
    lower_params: 'dict[str, Tensor]',
    upper_params: 'dict[str, Tensor]',
    lower_states: 'Tensor',
    upper_states: 'Tensor'
) → Tensor | None
```

Hamiltonian-exchange log acceptance from sparse coupling layouts, or `None` if a graph is dense.

Computes the four energies `E_m(z)` for both models and both populations from each site's coupled neighbours only.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1644"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exchange_log_acceptance_categorical_triton`

```python
exchange_log_acceptance_categorical_triton(
    lower_params: 'dict[str, Tensor]',
    upper_params: 'dict[str, Tensor]',
    lower_states: 'Tensor',
    upper_states: 'Tensor'
) → Tensor
```

Compute Hamiltonian-exchange log acceptance from categorical states.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
