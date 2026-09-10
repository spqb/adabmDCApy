<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/serialization.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.serialization`
Portable and atomic serialization helpers for high-level API results.

**Global Variables**
---------------
- **RESULT_SCHEMA_VERSION**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/serialization.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_jsonable`

```python
to_jsonable(value: 'Any') → Any
```

Recursively convert scientific Python values to strict JSON values.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/serialization.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `result_document`

```python
result_document(result_type: 'str', data: 'Mapping[str, Any]') → dict[str, Any]
```

Wrap result data in a stable, versioned document envelope.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/serialization.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `write_json`

```python
write_json(
    path: 'str | Path',
    payload: 'Mapping[str, Any]',
    indent: 'int' = 2
) → Path
```

Write strict UTF-8 JSON atomically.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/serialization.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `write_dataframe`

```python
write_dataframe(path: 'str | Path', dataframe: 'Any', **kwargs: 'Any') → Path
```

Write a pandas-compatible dataframe atomically.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/serialization.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `write_text`

```python
write_text(path: 'str | Path', content: 'str') → Path
```

Write UTF-8 text atomically.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/serialization.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `write_numpy`

```python
write_numpy(path: 'str | Path', array: 'ndarray') → Path
```

Write a NumPy array atomically without altering the requested suffix.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/serialization.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `write_numpy_text`

```python
write_numpy_text(path: 'str | Path', array: 'ndarray', **kwargs: 'Any') → Path
```

Write a NumPy array as an atomic text artifact.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/serialization.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `resolve_format`

```python
resolve_format(
    path: 'str | Path',
    format: 'str | None',
    supported: 'set[str]'
) → str
```

Resolve an explicit format or infer it from a filename suffix.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
