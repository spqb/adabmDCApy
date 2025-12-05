<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/cobalt.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.cobalt`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/cobalt.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `split_train_test`

```python
split_train_test(
    headers: ndarray,
    X: Tensor,
    seqid_th: float,
    rnd_gen: Optional[Generator] = None
) → Tuple[ndarray, Tensor, ndarray, Tensor]
```

Splits X into two sets, T and S, such that no sequence in S has more than 'seqid_th' fraction of its residues identical to any sequence in T. 



**Args:**
 
 - <b>`headers`</b> (np.ndarray):  Array of sequence headers. 
 - <b>`X`</b> (torch.Tensor):  Encoded input MSA, shape (batch_size, L). 
 - <b>`seqid_th`</b> (float):  Threshold sequence identity. 
 - <b>`rnd_gen`</b> (Optional[torch.Generator], optional):  Random number generator. Defaults to None. 



**Returns:**
 Training and test sets as:  (np.ndarray) Training headers,  (torch.Tensor) Training sequences,  (np.ndarray) Test headers,  (torch.Tensor) Test sequences. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/cobalt.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `prune_redundant_sequences`

```python
prune_redundant_sequences(
    headers: ndarray,
    X: Tensor,
    seqid_th: float,
    rnd_gen: Optional[Generator] = None
) → Tuple[ndarray, Tensor]
```

Prunes sequences from X such that no sequence has more than 'seqid_th' fraction of its residues identical to any other sequence in the set. 



**Args:**
 
 - <b>`headers`</b> (np.ndarray):  Array of sequence headers. 
 - <b>`X`</b> (torch.Tensor):  Encoded input MSA. 
 - <b>`seqid_th`</b> (float):  Threshold sequence identity. 
 - <b>`rnd_gen`</b> (Optional[torch.Generator], optional):  Random generator. Defaults to None. 



**Returns:**
 Tuple[np.ndarray, torch.Tensor]:  (np.ndarray) Headers of pruned sequences  (torch.Tensor) Pruned sequences. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/cobalt.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `run_cobalt`

```python
run_cobalt(
    headers: ndarray,
    X: Tensor,
    t1: float,
    t2: float,
    t3: float,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    rnd_gen: Optional[Generator] = None
) → Tuple[ndarray, Tensor, ndarray, Tensor]
```

Runs the Cobalt algorithm to split the input MSA into training and test sets. 



**Args:**
 
 - <b>`headers`</b> (np.ndarray):  Array of sequence headers. 
 - <b>`X`</b> (torch.Tensor):  Encoded input MSA. 
 - <b>`t1`</b> (float):  No sequence in S has more than this fraction of its residues identical to any sequence in T. 
 - <b>`t2`</b> (float):  No pair of test sequences has more than this value fractional identity. 
 - <b>`t3`</b> (float):  No pair of training sequences has more than this value fractional identity. 
 - <b>`max_train`</b> (Optional[int], optional):  Maximum number of sequences in the training set. Defaults to None. 
 - <b>`max_test`</b> (Optional[int], optional):  Maximum number of sequences in the test set. Defaults to None. 
 - <b>`rnd_gen`</b> (Optional[torch.Generator], optional):  Random number generator. Defaults to None. 



**Returns:**
 Training and test sets as:  (np.ndarray) Training headers,  (torch.Tensor) Training sequences,  (np.ndarray) Test headers,  (torch.Tensor) Test sequences. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
