<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/fasta.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `fasta`




**Global Variables**
---------------
- **TOKENS_PROTEIN**
- **TOKENS_RNA**
- **TOKENS_DNA**

---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/fasta.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_tokens`

```python
get_tokens(alphabet: str) → str
```

Converts the alphabet into the corresponding tokens. 



**Args:**
 
 - <b>`alphabet`</b> (str):  Alphabet to be used for the encoding. It can be either "protein", "rna", "dna" or a custom string of tokens. 



**Returns:**
 
 - <b>`str`</b>:  Tokens of the alphabet. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/fasta.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `encode_sequence`

```python
encode_sequence(sequence: str | ndarray | list, tokens: str) → ndarray
```

Encodes a sequence or a list of sequences into a numeric format. 



**Args:**
 
 - <b>`sequence`</b> (str | np.ndarray | list):  Input sequence. 
 - <b>`tokens`</b> (str):  Alphabet to be used for the encoding. 



**Returns:**
 
 - <b>`np.ndarray`</b>:  Encoded sequence or sequences. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/fasta.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `decode_sequence`

```python
decode_sequence(sequence: list | ndarray | Tensor, tokens: str) → str | ndarray
```

Takes a numeric sequence or list of seqences in input an returns the corresponding string encoding. 



**Args:**
 
 - <b>`sequence`</b> (np.ndarray):  Input sequences. Can be either a 1D or a 2D iterable. 
 - <b>`tokens`</b> (str):  Alphabet to be used for the encoding. 



**Returns:**
 
 - <b>`str | np.ndarray`</b>:  string or array of strings with the decoded input. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/fasta.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `import_from_fasta`

```python
import_from_fasta(
    fasta_name: str | Path,
    tokens: str | None = None,
    filter_sequences: bool = False,
    remove_duplicates: bool = True
) → Tuple[ndarray, ndarray]
```

Import sequences from a fasta file. The following operations are performed: 
- If 'tokens' is provided, encodes the sequences in numeric format. 
- If 'filter_sequences' is True, removes the sequences whose tokens are not present in the alphabet. 
- If 'remove_duplicates' is True, removes the duplicated sequences. 



**Args:**
 
 - <b>`fasta_name`</b> (str | Path):  Path to the fasta file. 
 - <b>`tokens`</b> (str | None, optional):  Alphabet to be used for the encoding. If provided, encodes the sequences in numeric format. 
 - <b>`filter_sequences`</b> (bool, optional):  If True, removes the sequences whose tokens are not present in the alphabet. Defaults to False. 
 - <b>`remove_duplicates`</b> (bool, optional):  If True, removes the duplicated sequences. Defaults to True. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  The file is not in fasta format. 



**Returns:**
 
 - <b>`Tuple[np.ndarray, np.ndarray]`</b>:  headers, sequences. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/fasta.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `write_fasta`

```python
write_fasta(
    fname: str,
    headers: ndarray | list,
    sequences: ndarray,
    numeric_input: bool = False,
    remove_gaps: bool = False,
    tokens: str = 'protein'
)
```

Generate a fasta file with the input sequences. 



**Args:**
 
 - <b>`fname`</b> (str):  Name of the output fasta file. 
 - <b>`headers`</b> (np.ndarray | list):  Array or list of sequences' headers. 
 - <b>`sequences`</b> (np.ndarray):  Array of sequences. 
 - <b>`numeric_input`</b> (bool, optional):  Whether the sequences are in numeric (encoded) format or not. Defaults to False. 
 - <b>`remove_gaps`</b> (bool, optional):  If True, removes the gap from the alignment. Defaults to False. 
 - <b>`tokens`</b> (str):  Alphabet to be used for the encoding. Defaults to protein. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/fasta.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_weights`

```python
compute_weights(
    data: ndarray | Tensor,
    th: float = 0.8,
    device: device = device(type='cpu'),
    dtype: dtype = torch.float32
) → Tensor
```

Computes the weight to be assigned to each sequence 's' in 'data' as 1 / n_clust, where 'n_clust' is the number of sequences that have a sequence identity with 's' >= th. 



**Args:**
 
 - <b>`data`</b> (np.ndarray | torch.Tensor):  Encoded input dataset. 
 - <b>`th`</b> (float, optional):  Sequence identity threshold for the clustering. Defaults to 0.8. 
 - <b>`device`</b> (toch.device, optional):  Device. Defaults to "cpu". 
 - <b>`dtype`</b> (torch.dtype, optional):  Data type. Defaults to torch.float32. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Array with the weights of the sequences. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/fasta.py#L245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `validate_alphabet`

```python
validate_alphabet(sequences: ndarray, tokens: str)
```

Check if the chosen alphabet is compatible with the input sequences. 



**Args:**
 
 - <b>`sequences`</b> (np.ndarray):  Input sequences. 
 - <b>`tokens`</b> (str):  Alphabet to be used for the encoding. 



**Raises:**
 
 - <b>`KeyError`</b>:  The chosen alphabet is incompatible with the Multi-Sequence Alignment. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
