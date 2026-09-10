<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.exceptions`
Structured exceptions exposed by the high-level adabmDCA API.



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AdabmDCAError`
Base class for recoverable errors raised by the application API.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `InputValidationError`
Raised when a high-level API input is invalid.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `InputLoadError`
Raised when an input resource cannot be read or decoded.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `WeightLoadError`
Raised when sequence weights cannot be loaded or aligned.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ChainLoadError`
Raised when an initial chain state cannot be loaded or validated.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `OutputSerializationError`
Raised when a result cannot be serialized to the requested output.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ComputationError`
Raised when a valid scientific operation cannot be completed.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConvergenceError`
Raised when an iterative operation exhausts its convergence budget.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ModelCompatibilityError`
Raised when data are incompatible with a DCA model.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ModelLoadError`
Raised when model parameters cannot be loaded.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `OperationCancelledError`
Raised when a caller-provided cancellation hook stops an operation.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentError`
Base class for alignment input and processing errors.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentLoadError`
Raised when an alignment resource cannot be opened or decoded.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentFormatError`
Raised when an alignment format cannot be detected or parsed.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlignmentLengthError`
Raised when sequences do not share a common aligned length.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(message: 'str', details: 'Mapping[str, Any] | None' = None) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/exceptions.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-serializable representation of the error.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
