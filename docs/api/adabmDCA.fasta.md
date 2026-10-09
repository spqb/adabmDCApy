<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/fasta.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.fasta`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/fasta.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `encode_sequence`

```python
encode_sequence(sequence: Union[str, Iterable[str]], tokens: str) → ndarray
```

Encodes a sequence or a list of sequences into a numeric format.



**Args:**

 - <b>`sequence`</b> (Union[str, Iterable[str]]):  Input sequence or iterable of sequences of size (batch_size,).
 - <b>`tokens`</b> (str):  Alphabet to be used for the encoding.



**Returns:**

 - <b>`np.ndarray`</b>:  Array of shape (L,) or (batch_size, L) with the encoded sequence or sequences.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/fasta.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `decode_sequence`

```python
decode_sequence(
    sequence: Union[ndarray, Tensor, list],
    tokens: str
) → Union[str, ndarray]
```

Takes a numeric sequence or list of seqences in input an returns the corresponding string encoding.



**Args:**

 - <b>`sequence`</b> (Union[np.ndarray, torch.Tensor, list]):  Input sequences. Can be of shape
        - (L,): single sequence in encoded format
        - (batch_size, L): multiple sequences in encoded format
        - (batch_size, L, q) multiple one-hot encoded sequences
 - <b>`tokens`</b> (str):  Alphabet to be used for the encoding.



**Returns:**

 - <b>`Union[str, np.ndarray]`</b>:  string or array of strings with the decoded input.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/fasta.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_weights`

```python
compute_weights(
    data: Union[ndarray, Tensor],
    th: float = 0.8,
    device: device = device(type='cpu'),
    dtype: dtype = torch.float32
) → Tensor
```

Computes the weight to be assigned to each sequence 's' in 'data' as 1 / n_clust, where 'n_clust' is the number of sequences that have a sequence identity with 's' > th (including 's' itself).



**Args:**

 - <b>`data`</b> (Union[np.ndarray, torch.Tensor]):  Input dataset. Must be either a (batch_size, L) or a (batch_size, L, q) (one-hot encoded) array.
 - <b>`th`</b> (float, optional):  Sequence identity threshold for the clustering. Defaults to 0.8.
 - <b>`device`</b> (torch.device, optional):  Device. Defaults to "cpu".
 - <b>`dtype`</b> (torch.dtype, optional):  Data type. Defaults to torch.float32.



**Returns:**

 - <b>`torch.Tensor`</b>:  Array with the weights of the sequences.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/fasta.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `validate_alphabet`

```python
validate_alphabet(sequences: Iterable[str], tokens: str)
```

Validates that all characters in the sequences are present in the provided alphabet.

**Args:**

 - <b>`sequences`</b> (Iterable[str]):  Iterable of sequences to be validated.
 - <b>`tokens`</b> (str):  Alphabet to be used for the validation.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
