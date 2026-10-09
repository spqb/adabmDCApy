<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.sampling_triton`
Optional Triton kernels for categorical GPU sampling.

**Global Variables**
---------------
- **SPARSE_MAX_DEGREE_FRACTION**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L584"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_triton_available`

```python
is_triton_available() → bool
```

Return whether Triton was imported successfully.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L589"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

``steps_per_launch`` controls sequential updates per kernel, independently of random-number chunking. ``transpose_couplings`` makes candidate fields contiguous at the cost of one extra coupling-sized allocation per call. Disable it to save memory or benchmark the original gather layout. ``coupling_dtype`` optionally converts sampling couplings without changing the caller's parameters. BF16 conversion and transposition use one copy.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L617"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Each step updates one uniformly drawn site, shared by all chains, for ``nsweeps * L`` steps. ``sites`` instead gives the site of every step (its length then sets the number of steps and ``nsweeps`` is ignored). Sparse graphs (see :func:`sparse_coupling_layout`) read only the coupled pairs.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L676"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

 - <b>`chains`</b>:  One-hot chains, shape ``(n_chains, L, q)``, on a CUDA device.
 - <b>`params`</b>:  ``bias`` ``(L, q)`` and ``coupling_matrix`` ``(L, q, L, q)``.
 - <b>`beta`</b>:  Inverse temperature.



**Returns:**
 The updated chains.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L701"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Run Metropolis updates with ``steps_per_launch`` updates per kernel.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L719"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Sparse graphs (see :func:`sparse_coupling_layout`) read only the coupled pairs.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L765"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

``steps_per_launch=None`` uses the tuned default for the alphabet size. Sparse graphs (see :func:`sparse_coupling_layout`) read only the coupled pairs.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L837"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `transpose_couplings_for_metropolized_gibbs`

```python
transpose_couplings_for_metropolized_gibbs(couplings: 'Tensor') → Tensor
```

View (L, q, L, q) couplings as (site, source_site, source_state, state).

Stacking such views materializes the layout expected by ``metropolized_gibbs_sampling_replicas_categorical_triton`` in one copy.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L846"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

``couplings`` has shape (R, L, L, q, q) in the layout returned by ``transpose_couplings_for_metropolized_gibbs`` for each replica. Every replica updates one random site per step, shared by all its chains.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L960"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sparse_coupling_layout`

```python
sparse_coupling_layout(
    couplings: 'Tensor',
    dtype: 'dtype | None' = None,
    source=None,
    method: 'str' = 'metropolized_gibbs'
)
```

The sparse layout of stacked couplings ``(R, L, q, L, q)`` if it pays off for ``method``, else ``None``.

Returns ``(neighbours, blocks, degree)``: ``neighbours`` ``(R, L, D)`` lists each site's coupled sites (padded with -1), ``blocks`` ``(R, L, D, q, q)`` holds ``blocks[r, i, k, b, a] = J[i, a, j_k, b]``. The degree and the layout are cached per ``source`` tensor (``couplings`` by default); the layout is built only when used.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1009"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolized_gibbs_sampling_replicas_triton`

```python
metropolized_gibbs_sampling_replicas_triton(
    states: 'Tensor',
    biases: 'Tensor',
    couplings: 'Tensor',
    nsweeps: 'int',
    beta: 'float' = 1.0,
    steps_per_launch: 'int' = 8
) → Tensor
```

Metropolized Gibbs sampling of a stack of replicas, from couplings in their original layout.

``couplings`` has shape (R, L, q, L, q). Sparse graphs use the sparse kernel; otherwise the transposed copy of :func:`metropolized_gibbs_sampling_replicas_categorical_triton` is made. Either layout is cached and reused while ``couplings`` is unchanged.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1040"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Metropolized Gibbs sampling of one-hot chains; see ``metropolized_gibbs_sampling_categorical_triton``.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1060"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Each update draws one site shared by all chains. A new state b != a is proposed from the site conditional restricted to the other states and accepted with min(1, (1 - p_a) / (1 - p_b)) (Liu 1996). One program owns a chain and reads, for every source position, a contiguous row of candidate couplings from a transposed copy made once per call. Sparse graphs (see :func:`sparse_coupling_layout`) read only each site's coupled neighbours instead. ``coupling_dtype=torch.bfloat16`` stores that copy in BF16. Fields and acceptance use the bias precision, or FP32 for BF16 biases, as do the uniform random numbers. Couplings must have zero within-site blocks.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

 - <b>`chains`</b>:  One-hot chains, shape ``(n_chains, L, q)``, on a CUDA device.
 - <b>`params`</b>:  ``bias`` ``(L, q)`` and ``coupling_matrix`` ``(L, q, L, q)``.
 - <b>`beta`</b>:  Inverse temperature.



**Returns:**
 The updated chains.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exchange_log_acceptance_sparse_triton`

```python
exchange_log_acceptance_sparse_triton(
    lower_params: 'dict[str, Tensor]',
    upper_params: 'dict[str, Tensor]',
    lower_states: 'Tensor',
    upper_states: 'Tensor'
) → Tensor | None
```

Hamiltonian-exchange log acceptance from sparse coupling layouts, or ``None`` if a graph is dense.

Computes the four energies ``E_m(z)`` for both models and both populations from each site's coupled neighbours only.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L1264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
