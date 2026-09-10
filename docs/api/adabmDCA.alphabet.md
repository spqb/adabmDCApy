<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alphabet.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.alphabet`
Alphabet definitions shared by lightweight and tensor-based APIs.

**Global Variables**
---------------
- **TOKENS_PROTEIN**
- **TOKENS_RNA**
- **TOKENS_DNA**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alphabet.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_tokens`

```python
get_tokens(alphabet: str) → str
```

Return built-in tokens or an explicitly supplied custom alphabet.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/alphabet.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `detect_alphabet`

```python
detect_alphabet(sequences) → str
```

Select the smallest compatible standard alphabet (DNA wins ties).

All symbols must fit a standard alphabet; custom tokens are never inferred. Gap-only or empty input cannot identify an alphabet.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
