<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/pca.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `pca`






---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/pca.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Pca`




<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/pca.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/pca.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(M: Tensor, num_directions: int = 2) → None
```

Fit the PCA model to the data. 



**Args:**
 
 - <b>`M`</b> (torch.Tensor):  Data matrix (num_samples, num_variables). 
 - <b>`num_directions`</b> (int):  Number of principal components to compute. 

---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/pca.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_transform`

```python
fit_transform(M: Tensor, num_directions: int = 2) → Tensor
```

Fit the PCA model to the data and project the data onto the principal components. 



**Args:**
 
 - <b>`M`</b> (torch.Tensor):  Data matrix (num_samples, num_variables). 
 - <b>`num_directions`</b> (int):  Number of principal components to compute. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Projected data matrix (num_samples, num_directions). 

---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/pca.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transform`

```python
transform(M: Tensor) → Tensor
```

Projects the data onto the principal components. 



**Args:**
 
 - <b>`M`</b> (torch.Tensor):  Data matrix (num_samples, num_variables). 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Projected data matrix (num_samples, num_directions). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
