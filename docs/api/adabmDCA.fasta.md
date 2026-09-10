<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/fasta.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.fasta`





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/fasta.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/fasta.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/fasta.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `import_from_fasta`

```python
import_from_fasta(
    fasta_name: str,
    tokens: Optional[str] = None,
    filter_sequences: bool = False,
    remove_duplicates: bool = False,
    return_mask: bool = False
)
```

Import sequences from a FASTA file using the legacy array interface. 

.. deprecated:: 0.7.8  Use :func:`adabmDCA.read_alignment` for parsing or  :func:`adabmDCA.load_alignment` for validated filtering and provenance. 

The following operations are performed: 
- If 'tokens' is provided, encodes the sequences in numeric format. 
- If 'filter_sequences' is True, removes the sequences whose tokens are not present in the alphabet. 
- If 'remove_duplicates' is True, removes the duplicated sequences. 
- If 'return_mask' is True, returns also the mask selecting the retained sequences from the original ones. 



**Args:**
 
 - <b>`fasta_name`</b> (str | Path):  Path to the fasta or compressed fasta (.fas.gz) file. 
 - <b>`tokens`</b> (str | None, optional):  Alphabet to be used for the encoding. If provided, encodes the sequences in numeric format. 
 - <b>`filter_sequences`</b> (bool, optional):  If True, removes the sequences whose tokens are not present in the alphabet. Defaults to False. 
 - <b>`remove_duplicates`</b> (bool, optional):  If True, removes the duplicated sequences. Defaults to False. 
 - <b>`return_mask`</b> (bool, optional):  If True, returns also the mask selecting the retained sequences from the original ones. Defaults to False. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  The file is not in fasta format. 



**Returns:**
 Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]: 
    - If 'return_mask' is False: Tuple of (headers, sequences) 
    - If 'return_mask' is True: Tuple of (headers, sequences, mask) 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/fasta.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `write_fasta`

```python
write_fasta(
    fname: str,
    headers: Union[Iterable[str], ndarray, Tensor],
    sequences: Union[Iterable[str], ndarray, Tensor],
    remove_gaps: bool = False,
    tokens: str = 'protein'
) → None
```

Generate a FASTA file using the legacy array interface. 

.. deprecated:: 0.7.8  Construct an :class:`adabmDCA.Alignment` and call  :func:`adabmDCA.write_alignment`, or use a high-level result object's  ``to_fasta``/``save_bundle`` method. 



**Args:**
 
 - <b>`fname`</b> (str):  Name of the output fasta file. 
 - <b>`headers`</b> (Union[Iterable[str], np.ndarray, torch.Tensor]):  Iterable with sequences' headers. 
 - <b>`sequences`</b> (Union[Iterable[str], np.ndarray, torch.Tensor]):  Iterable with sequences in string, categorical or one-hot encoded format. 
 - <b>`remove_gaps`</b> (bool, optional):  If True, removes the gap from the alignment. Defaults to False. 
 - <b>`tokens`</b> (str):  Alphabet to be used for the encoding. Defaults to 'protein'. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/fasta.py#L272"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_weights`

```python
compute_weights(
    data: Union[ndarray, Tensor],
    th: float = 0.8,
    device: device = device(type='cpu'),
    dtype: dtype = torch.float32
) → Tensor
```

Computes the weight to be assigned to each sequence 's' in 'data' as 1 / n_clust, where 'n_clust' is the number of sequences that have a sequence identity with 's' >= th. 



**Args:**
 
 - <b>`data`</b> (Union[np.ndarray, torch.Tensor]):  Input dataset. Must be either a (batch_size, L) or a (batch_size, L, q) (one-hot encoded) array. 
 - <b>`th`</b> (float, optional):  Sequence identity threshold for the clustering. Defaults to 0.8. 
 - <b>`device`</b> (torch.device, optional):  Device. Defaults to "cpu". 
 - <b>`dtype`</b> (torch.dtype, optional):  Data type. Defaults to torch.float32. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Array with the weights of the sequences. 


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/fasta.py#L304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
