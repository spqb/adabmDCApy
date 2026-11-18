<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.io`
FASTA and text parameter I/O.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_chains`

```python
save_chains(
    fname: str | Path,
    chains: str | list | ndarray | Tensor,
    tokens: str
) → None
```

Save aligned strings, categorical sequences or one-hot batches to FASTA.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_gzip`

```python
is_gzip(fname: Union[str, Path]) → bool
```

Whether a file starts with the gzip signature.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `open_params`

```python
open_params(fname: Union[str, Path], mode: str = 'r') → IO[str]
```

Open a text parameter file in r/rt or w/wt mode.

Reads detect gzip by content. Writes use gzip for paths ending in .gz.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L238"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_params`

```python
load_params(
    fname: str | Path,
    tokens: str,
    device: device,
    dtype: dtype = torch.float32
) → Dict[str, Tensor]
```

Read letter-based J/h records, including gzip files regardless of their name.

Missing entries are zero. Opposite coupling orientations add for legacy compatibility, while repeated records for the same oriented entry are invalid.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_params_old`

```python
load_params_old(
    fname: str | Path,
    tokens: str,
    device: device,
    dtype: dtype = torch.float32
) → Dict[str, Tensor]
```

Deprecated alias for loading letter-based parameters.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L263"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_params_oldformat`

```python
load_params_oldformat(
    fname: str | Path,
    device: device,
    dtype: dtype = torch.float32
) → Dict[str, Tensor]
```

Load legacy J/h records that identify states by their integer indices.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L348"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_params`

```python
save_params(
    fname: str | Path,
    params: Dict[str, Tensor],
    tokens: str,
    mask: Optional[Tensor] = None
) → None
```

Atomically save letter-based J/h records, using gzip for .gz paths.

Off-site pairs use the effective symmetric couplings and are written once. Selected same-site diagonal terms are absorbed into the fields. A mask selects an off-site interaction if either orientation is set.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/io.py#L363"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_params_oldformat`

```python
save_params_oldformat(
    fname: str | Path,
    params: Dict[str, Tensor],
    mask: Optional[Tensor] = None
) → None
```

Atomically save legacy J/h records with integer state indices.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
