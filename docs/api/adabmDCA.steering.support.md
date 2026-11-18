<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/support.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.steering.support`
Compiled Hamming costs, hard support, and an exactly sampled profile.



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/support.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HammingWindow`
Inclusive substitution-distance bounds on an aligned reference.



**Args:**

 - <b>`reference`</b>:  Model-length sequence in the model alphabet.
 - <b>`minimum`</b>:  Smallest allowed number of substitutions.
 - <b>`maximum`</b>:  Largest allowed number of substitutions.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(reference: 'str', minimum: 'int', maximum: 'int') → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/support.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HammingBias`
Reduced energy equal to weight times the distance from reference.



**Args:**

 - <b>`reference`</b>:  Model-length sequence in the model alphabet.
 - <b>`weight`</b>:  Cost per substitution. Positive values favor proximity.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(reference: 'str', weight: 'float') → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/support.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Support`




<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/support.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(tokens, length, allowed_tokens, windows)
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/support.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `contains`

```python
contains(states)
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/support.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `distances`

```python
distances(states)
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/support.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encode_references`

```python
encode_references(tokens, length, references)
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/support.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ExactProfile`
Suffix log partitions and independent samples from a constrained profile.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/support.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(fields, support)
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering/support.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `draw`

```python
draw(size, rng)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
