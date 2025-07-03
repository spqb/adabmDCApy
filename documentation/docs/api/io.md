<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/io.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `io`
The io module provides the Python interfaces to stream handling. The builtin open function is defined in this module. 

At the top of the I/O hierarchy is the abstract base class IOBase. It defines the basic interface to a stream. Note, however, that there is no separation between reading and writing to streams; implementations are allowed to raise an OSError if they do not support a given operation. 

Extending IOBase is RawIOBase which deals simply with the reading and writing of raw bytes to a stream. FileIO subclasses RawIOBase to provide an interface to OS files. 

BufferedIOBase deals with buffering on a raw byte stream (RawIOBase). Its subclasses, BufferedWriter, BufferedReader, and BufferedRWPair buffer streams that are readable, writable, and both respectively. BufferedRandom provides a buffered interface to random access streams. BytesIO is a simple stream of in-memory bytes. 

Another IOBase subclass, TextIOBase, deals with the encoding and decoding of streams into text. TextIOWrapper, which extends it, is a buffered text interface to a buffered raw stream (`BufferedIOBase`). Finally, StringIO is an in-memory stream for text. 

Argument names are not part of the specification, and only the arguments of open() are intended to be used as keyword arguments. 

data: 

DEFAULT_BUFFER_SIZE 

 An int containing the default buffer size used by the module's buffered  I/O classes. open() uses the file's blksize (as obtained by os.stat) if  possible. 

**Global Variables**
---------------
- **DEFAULT_BUFFER_SIZE**
- **SEEK_SET**
- **SEEK_CUR**
- **SEEK_END**

---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/io.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_chains`

```python
load_chains(
    fname: str,
    tokens: str,
    load_weights: bool = False
) → Union[ndarray, Tuple[ndarray, ndarray]]
```

Loads the sequences from a fasta file and returns the numeric-encoded version. If the sequences are weighted, the log-weights are also returned. If the sequences are not weighted, the log-weights are set to 0. 



**Args:**
 
 - <b>`fname`</b> (str):  Path to the file containing the sequences. 
 - <b>`tokens`</b> (str):  "protein", "dna", "rna" or another string with the alphabet to be used. 
 - <b>`load_weights`</b> (bool, optional):  If True, the log-weights are loaded and returned. Defaults to False. 

Return: 
 - <b>`np.ndarray | Tuple[np.ndarray, np.ndarray]`</b>:  Numeric-encoded sequences and log-weights if load_weights is True. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/io.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_chains`

```python
save_chains(
    fname: str,
    chains: Tensor,
    tokens: str,
    log_weights: Tensor = None
) → None
```

Saves the chains in a fasta file. 



**Args:**
 
 - <b>`fname`</b> (str):  Path to the file where to save the chains. 
 - <b>`chains`</b> (torch.Tensor):  Chains. 
 - <b>`tokens`</b> (str):  "protein", "dna", "rna" or another string with the alphabet to be used. 
 - <b>`log_weights`</b> (torch.Tensor, optional):  Log-weights of the chains. Defaults to None. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/io.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_params`

```python
load_params(
    fname: str,
    tokens: str,
    device: device,
    dtype: dtype = torch.float32
) → Dict[str, Tensor]
```

Import the parameters of the model from a file. 



**Args:**
 
 - <b>`fname`</b> (str):  Path of the file that stores the parameters. 
 - <b>`tokens`</b> (str):  "protein", "dna", "rna" or another string with the alphabet to be used. 
 - <b>`device`</b> (torch.device):  Device where to store the parameters. 
 - <b>`dtype`</b> (torch.dtype):  Data type of the parameters. Defaults to torch.float32. 



**Returns:**
 
 - <b>`Dict[str, torch.Tensor]`</b>:  Parameters of the model. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/io.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_params`

```python
save_params(
    fname: str,
    params: Dict[str, Tensor],
    tokens: str,
    mask: Tensor | None = None
) → None
```

Saves the parameters of the model in a file. 



**Args:**
 
 - <b>`fname`</b> (str):  Path to the file where to save the parameters. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`tokens`</b> (str):  "protein", "dna", "rna" or another string with the alphabet to be used. 
 - <b>`mask`</b> (torch.Tensor | None):  Mask of the coupling matrix that determines which are the non-zero entries.  If None, the lower-triangular part of the coupling matrix is masked. Defaults to None. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/io.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_params_oldformat`

```python
load_params_oldformat(
    fname: str,
    device: device,
    dtype: dtype = torch.float32
) → Dict[str, Tensor]
```

Import the parameters of the model from a file. Assumes the old DCA format. 



**Args:**
 
 - <b>`fname`</b> (str):  Path of the file that stores the parameters. 
 - <b>`device`</b> (torch.device):  Device where to store the parameters. 
 - <b>`dtype`</b> (torch.dtype):  Data type of the parameters. Defaults to torch.float32. 



**Returns:**
 
 - <b>`Dict[str, torch.Tensor]`</b>:  Parameters of the model. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/io.py#L258"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_params_oldformat`

```python
save_params_oldformat(
    fname: str,
    params: Dict[str, Tensor],
    mask: Tensor | None = None
) → None
```

Saves the parameters of the model in a file. Assumes the old DCA format. 



**Args:**
 
 - <b>`fname`</b> (str):  Path to the file where to save the parameters. 
 - <b>`params`</b> (Dict[str, torch.Tensor]):  Parameters of the model. 
 - <b>`mask`</b> (torch.Tensor):  Mask of the coupling matrix that determines which are the non-zero entries.  If None, the lower-triangular part of the coupling matrix is masked. Defaults to None. 


---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BufferedIOBase`
Base class for buffered IO objects. 

The main difference with RawIOBase is that the read() method supports omitting the size argument, and does not have a default implementation that defers to readinto(). 

In addition, read(), readinto() and write() may raise BlockingIOError if the underlying raw stream is in non-blocking mode and not ready; unlike their raw counterparts, they will never return None. 

A typical implementation should not inherit from a RawIOBase implementation, but wrap one. 





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `IOBase`
The abstract base class for all I/O classes. 

This class provides dummy implementations for many methods that derived classes can override selectively; the default implementations represent a file that cannot be read, written or seeked. 

Even though IOBase does not declare read, readinto, or write because their signatures will vary, implementations and clients should consider those methods part of the interface. Also, implementations may raise UnsupportedOperation when operations they do not support are called. 

The basic type used for binary data read from or written to a file is bytes. Other bytes-like objects are accepted as method arguments too. In some cases (such as readinto), a writable object is required. Text I/O classes work with str data. 

Note that calling any method (except additional calls to close(), which are ignored) on a closed stream should raise a ValueError. 

IOBase (and its subclasses) support the iterator protocol, meaning that an IOBase object can be iterated over yielding the lines in a stream. 

IOBase also supports the :keyword:`with` statement. In this example, fp is closed after the suite of the with statement is complete: 

with open('spam.txt', 'r') as fp:  fp.write('Spam and eggs!') 





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RawIOBase`
Base class for raw binary I/O. 





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TextIOBase`
Base class for text I/O. 

This class provides a character and line based interface to stream I/O. There is no readinto method because Python's character strings are immutable. 





---

<a href="https://github.com/spqb/adabmDCApy/tree/main/adabmDCA/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UnsupportedOperation`










---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
