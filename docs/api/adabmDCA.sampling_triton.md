<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.sampling_triton`
Optional Triton kernels for categorical GPU sampling.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_triton_available`

```python
is_triton_available() → bool
```

Return whether Triton was imported successfully.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gibbs_sampling_triton`

```python
gibbs_sampling_triton(
    chains: 'Tensor',
    params: 'Dict[str, Tensor]',
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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gibbs_step_independent_sites_triton`

```python
gibbs_step_independent_sites_triton(
    chains: 'Tensor',
    params: 'Dict[str, Tensor]',
    beta: 'float' = 1.0
) → Tensor
```

Apply one fused Gibbs update at an independently drawn site per chain.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L290"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolis_sampling_triton`

```python
metropolis_sampling_triton(
    chains: 'Tensor',
    params: 'Dict[str, Tensor]',
    nsweeps: 'int',
    beta: 'float' = 1.0,
    steps_per_launch: 'int' = 16,
    coupling_dtype: 'dtype | None' = None
) → Tensor
```

Run Metropolis updates with ``steps_per_launch`` updates per kernel.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/sampling_triton.py#L333"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `metropolis_step_independent_sites_triton`

```python
metropolis_step_independent_sites_triton(
    chains: 'Tensor',
    params: 'Dict[str, Tensor]',
    beta: 'float' = 1.0
) → Tensor
```

Apply one fused Metropolis update at an independently drawn site per chain.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
