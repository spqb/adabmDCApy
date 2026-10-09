<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.results`
Result objects returned by the high-level adabmDCA API.

**Global Variables**
---------------
- **TYPE_CHECKING**


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ModelMetadata`
Portable description of a loaded DCA model.



**Attributes:**

 - <b>`length`</b>:  Number of sites ``L``.
 - <b>`alphabet`</b>:  Alphabet name or custom token string given when loading.
 - <b>`tokens`</b>:  Ordered token string of length ``q``.
 - <b>`device`</b>:  Device of the parameters, e.g. ``"cpu"`` or ``"cuda:0"``.
 - <b>`dtype`</b>:  Precision of the parameters, e.g. ``"float32"``.
 - <b>`source`</b>:  Path of the file the model was loaded from, if any.
 - <b>`package_version`</b>:  adabmDCA version that loaded the model.
 - <b>`schema_version`</b>:  Version of this metadata format.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    length: 'int',
    alphabet: 'str',
    tokens: 'str',
    device: 'str',
    dtype: 'str',
    source: 'str | None' = None,
    package_version: 'str | None' = None,
    schema_version: 'str' = '1.0'
) → None
```






---

#### <kbd>property</kbd> num_states

Number of states per site, ``q``.



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-compatible document with the result type and schema version.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```

Write the metadata as a versioned JSON document.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`indent`</b>:  JSON indentation.



**Returns:**
 The written path.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EnergyResult`
DCA energies of scored sequences, returned by :func:`score_sequences`.



**Attributes:**

 - <b>`sequences`</b>:  Scored sequences, in input order.
 - <b>`energies`</b>:  Energy of each sequence (lower is more probable).
 - <b>`model`</b>:  Metadata of the scoring model.
 - <b>`names`</b>:  Sequence names when read from an alignment, else empty.
 - <b>`warnings`</b>:  Non-fatal issues found while scoring.
 - <b>`cde_sum`</b>:  Summed context-dependent entropy of each sequence, or ``None``.
 - <b>`local_free_energies`</b>:  ``energies - local_lambda * cde_sum``, or ``None``.
 - <b>`local_lambda`</b>:  Weight used for the local free energies, or ``None``.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    sequences: 'tuple[str, ]',
    energies: 'ndarray',
    model: 'ModelMetadata',
    names: 'tuple[str, ]' = (),
    warnings: 'tuple[str, ]' = (),
    cde_sum: 'ndarray | None' = None,
    local_free_energies: 'ndarray | None' = None,
    local_lambda: 'float | None' = None
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L176"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(path: 'str | Path', format: 'str | None' = None) → Path
```

Write the result in the format given by ``format`` or by the file extension.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`format`</b>:  One of ``"csv"``, ``"fasta"`` or ``"json"``; inferred from the extension when ``None``.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(directory: 'str | Path', stem: 'str' = 'energies') → dict[str, Path]
```

Write ``<stem>.fasta``, ``<stem>.csv`` and ``<stem>.json`` to ``directory``.



**Args:**

 - <b>`directory`</b>:  Output folder.
 - <b>`stem`</b>:  File-name stem.



**Returns:**

 - <b>`The written paths by kind`</b>:  ``fasta``, ``csv`` and ``summary``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_csv`

```python
to_csv(path: 'str | Path') → Path
```

Write :meth:`to_dataframe` as CSV.



**Args:**

 - <b>`path`</b>:  Output file.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dataframe`

```python
to_dataframe() → DataFrame
```

Return one row per sequence: name (if known), sequence, energy and CDE columns.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a portable representation of the result.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_fasta`

```python
to_fasta(path: 'str | Path') → Path
```

Write the sequences to FASTA, with energies in the headers.



**Args:**

 - <b>`path`</b>:  Output file.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```

Write :meth:`to_dict` as JSON.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`indent`</b>:  JSON indentation.



**Returns:**
 The written path.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ContactMapResult`
Contact scores returned by :func:`predict_contacts`.



**Attributes:**

 - <b>`scores`</b>:  Symmetric ``(L, L)`` array of APC-corrected scores, zero diagonal.
 - <b>`method`</b>:  ``"model"`` for a trained model, ``"mean_field"`` for an alignment.
 - <b>`alphabet`</b>:  Alphabet used.
 - <b>`tokens`</b>:  Ordered token string.
 - <b>`model`</b>:  Metadata of the model, for ``method="model"``.
 - <b>`warnings`</b>:  Non-fatal issues found while scoring.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    scores: 'ndarray',
    method: 'str',
    alphabet: 'str',
    tokens: 'str',
    model: 'ModelMetadata | None' = None,
    warnings: 'tuple[str, ]' = ()
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L299"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(path: 'str | Path', format: 'str | None' = None) → Path
```

Write the result in the format given by ``format`` or by the file extension.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`format`</b>:  One of ``"csv"``, ``"json"`` or ``"npy"``; inferred from the extension when ``None``.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L312"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(
    directory: 'str | Path',
    label: 'str | None' = None
) → dict[str, Path]
```

Write the matrix (``.txt``, ``.csv``, ``.npy``) and a JSON summary to ``directory``.



**Args:**

 - <b>`directory`</b>:  Output folder.
 - <b>`label`</b>:  Optional file-name prefix, giving ``<label>_contact_map.*``.



**Returns:**

 - <b>`The written paths by kind`</b>:  ``matrix``, ``csv``, ``npy`` and ``summary``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L266"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_matrix`

```python
save_matrix(path: 'str | Path') → Path
```

Write the scores as headerless ``i,j,score`` lines, the format of ``adabmDCA contacts``.



**Args:**

 - <b>`path`</b>:  Output file.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L277"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_csv`

```python
to_csv(path: 'str | Path') → Path
```

Write :meth:`to_long_dataframe` as CSV with a header.



**Args:**

 - <b>`path`</b>:  Output file.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-compatible document with the result type and schema version.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```

Write :meth:`to_dict` as JSON.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`indent`</b>:  JSON indentation.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L253"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_long_dataframe`

```python
to_long_dataframe() → DataFrame
```

Return one row per matrix entry: ``position_i``, ``position_j`` (0-based) and ``score``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L288"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_npy`

```python
to_npy(path: 'str | Path') → Path
```

Write the ``(L, L)`` score matrix as a NumPy ``.npy`` file.



**Args:**

 - <b>`path`</b>:  Output file.



**Returns:**
 The written path.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L332"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MutationRecord`
One single-site mutant from :func:`scan_mutations`.



**Attributes:**

 - <b>`position`</b>:  0-based site of the substitution.
 - <b>`wild_type`</b>:  Wild-type token at that site.
 - <b>`mutant`</b>:  Substituted token.
 - <b>`sequence`</b>:  Full mutant sequence.
 - <b>`delta_energy`</b>:  ``E(mutant) - E(wild type)``; negative values favour the mutant.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    position: 'int',
    wild_type: 'str',
    mutant: 'str',
    sequence: 'str',
    delta_energy: 'float'
) → None
```






---

#### <kbd>property</kbd> label

Historical, zero-based mutation label used by the CLI.

---

#### <kbd>property</kbd> position_1based

1-based site of the substitution.



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L360"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the record as a flat dictionary, one row of :meth:`MutationScanResult.to_dataframe`.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L373"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MutationScanResult`
All single-site mutants of a sequence, returned by :func:`scan_mutations`.



**Attributes:**

 - <b>`wild_type`</b>:  The scanned sequence.
 - <b>`wild_type_energy`</b>:  Its DCA energy.
 - <b>`mutations`</b>:  One :class:`MutationRecord` per mutant, by site then token.
 - <b>`model`</b>:  Metadata of the scoring model.
 - <b>`name`</b>:  Label of the scan, used in file names.
 - <b>`warnings`</b>:  Non-fatal issues found while scoring.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    wild_type: 'str',
    wild_type_energy: 'float',
    mutations: 'tuple[MutationRecord, ]',
    model: 'ModelMetadata',
    name: 'str' = 'wild_type',
    warnings: 'tuple[str, ]' = ()
) → None
```






---

#### <kbd>property</kbd> delta_energies

Energy differences of all mutants, in the order of ``mutations``.



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L457"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(path: 'str | Path', format: 'str | None' = None) → Path
```

Write the result in the format given by ``format`` or by the file extension.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`format`</b>:  One of ``"csv"``, ``"fasta"`` or ``"json"``; inferred from the extension when ``None``.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L470"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(
    directory: 'str | Path',
    stem: 'str | None' = None
) → dict[str, Path]
```

Write the FASTA library, CSV table and JSON summary to ``directory``.



**Args:**

 - <b>`directory`</b>:  Output folder.
 - <b>`stem`</b>:  File-name stem; defaults to ``<name>_DMS``.



**Returns:**

 - <b>`The written paths by kind`</b>:  ``fasta``, ``csv`` and ``summary``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L446"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_csv`

```python
to_csv(path: 'str | Path') → Path
```

Write :meth:`to_dataframe` as CSV.



**Args:**

 - <b>`path`</b>:  Output file.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L424"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dataframe`

```python
to_dataframe() → DataFrame
```

Return one row per mutant, with 0- and 1-based positions and ``delta_energy``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L398"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-compatible document with the result type and schema version.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L430"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_fasta`

```python
to_fasta(path: 'str | Path') → Path
```

Write the mutant sequences as FASTA, headers ``<wt><pos><mut> | DCAscore: <delta>``.

Positions in the headers are 0-based, as in ``adabmDCA dms``.



**Args:**

 - <b>`path`</b>:  Output file.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L412"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```

Write :meth:`to_dict` as JSON.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`indent`</b>:  JSON indentation.



**Returns:**
 The written path.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L489"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SamplingProgress`
Progress event passed to the ``progress`` callback of :func:`sample_sequences`.



**Attributes:**

 - <b>`stage`</b>:  ``"sampling"``, or a ``"ptt_*"`` stage when sampling with PTT.
 - <b>`completed`</b>:  Work done in this phase (sweeps or exchange rounds).
 - <b>`total`</b>:  Work planned for this phase.
 - <b>`pearson`</b>:  Current Cij Pearson correlation with the reference, if measured.
 - <b>`slope`</b>:  Current Cij regression slope, if measured.
 - <b>`details`</b>:  Extra phase-specific values (PTT renewal fractions, ...).

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    stage: 'str',
    completed: 'int',
    total: 'int',
    pearson: 'float | None' = None,
    slope: 'float | None' = None,
    details: 'dict[str, Any]' = <factory>
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L510"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ProfileSplitResult`
Training/test split of an alignment, returned by :func:`split_alignment`.



**Attributes:**

 - <b>`training`</b>:  Training alignment.
 - <b>`test`</b>:  Test alignment.
 - <b>`score`</b>:  ``len(training) * len(test)``; Cobalt keeps the attempt maximizing it.
 - <b>`attempts`</b>:  Number of attempts made.
 - <b>`tokens`</b>:  Alphabet tokens.
 - <b>`seed`</b>:  Random seed used.
 - <b>`method`</b>:  ``"cobalt"`` or ``"clustering"``.
 - <b>`identity`</b>:  Sequence-identity threshold of the clustering method.
 - <b>`train_fraction`</b>:  Requested training fraction of the clustering method.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    training: 'Alignment',
    test: 'Alignment',
    score: 'int',
    attempts: 'int',
    tokens: 'str',
    seed: 'int',
    method: 'str' = 'cobalt',
    identity: 'float | None' = None,
    train_fraction: 'float | None' = None
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L568"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(output_prefix: 'str | Path') → dict[str, Path]
```

Write ``<prefix>.train.fasta``, ``<prefix>.test.fasta`` and ``<prefix>.split.json``.



**Args:**

 - <b>`output_prefix`</b>:  Path prefix of the three files.



**Returns:**

 - <b>`The written paths by kind`</b>:  ``training``, ``test`` and ``summary``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L536"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-compatible document with the result type and schema version.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L556"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```

Write :meth:`to_dict` as JSON.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`indent`</b>:  JSON indentation.



**Returns:**
 The written path.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L588"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SamplingResult`
Generated sequences and diagnostics, returned by :func:`sample_sequences`.



**Attributes:**

 - <b>`sequences`</b>:  Generated sequences.
 - <b>`energies`</b>:  DCA energy of each sequence.
 - <b>`num_sweeps`</b>:  Monte Carlo sweeps performed (all replicas, for PTT).
 - <b>`sampler`</b>:  Local sampler used, or ``"ptt"``.
 - <b>`beta`</b>:  Inverse temperature.
 - <b>`seed`</b>:  Random seed.
 - <b>`model`</b>:  Metadata of the sampled model.
 - <b>`mixing_history`</b>:  Mixing-time measurements, by quantity.
 - <b>`sampling_history`</b>:  Pearson and slope of the Cij against the reference during  generation, when a reference was given.
 - <b>`warnings`</b>:  Non-fatal issues, e.g. an unconverged mixing estimate.
 - <b>`sampling_dtype`</b>:  Precision used for sampling.
 - <b>`ptt_diagnostics`</b>:  PTT only: mixing, renewal and ladder-health diagnostics.
 - <b>`cde_sum`</b>:  Summed context-dependent entropy of each sequence.
 - <b>`local_lambda_fit`</b>:  Least-squares fit of energy against ``cde_sum``, if identifiable.
 - <b>`cij_reference`</b>:  Reference connected correlations, when diagnostics were collected.
 - <b>`cij_generated`</b>:  Generated connected correlations, when diagnostics were collected.
 - <b>`pca_reference`</b>:  Reference sequences projected on their principal components.
 - <b>`pca_generated`</b>:  Generated sequences projected on the same components.
 - <b>`data_comparison`</b>:  With diagnostics and a reference: the share of data and  samples in each cluster of the data (k-means on the principal  components) and their energy distributions under the model.
 - <b>`distance_comparison`</b>:  With diagnostics and a reference: Hamming distances  (fraction of sites) within and between natural and generated  sequences, all pairs and nearest neighbours, plus held-out ->  reference and generated -> held-out nearest distances when a test  alignment was given; ``privet`` holds the PRIVET fit and per-sample
 - <b>`scores (also columns of `</b>: meth:`to_dataframe`).
 - <b>`pca_explained_variance_ratio`</b>:  Variance explained by each component.
 - <b>`steering_potentials`</b>:  Steered sampling only: ``V(x, steering_strength)`` of  each sequence.
 - <b>`log_importance_weights`</b>:  Steered sampling only: log weights that turn  averages over these sequences into averages under the unsteered model,  ``<g>_p = mean(exp(log_w) * g)``. PTT normalizes them with its estimate  of ``log Z_s - log Z_0``; ordinary sampling self-normalizes them to mean 1.
 - <b>`steering`</b>:  Steered sampling only: ``strength``, ``input``, proposal block  length and acceptance, ``effective_sample_size`` of the weights, and for  PTT ``log_z_ratio`` and the rung ``strengths``.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    sequences: 'tuple[str, ]',
    energies: 'ndarray',
    num_sweeps: 'int',
    sampler: 'str',
    beta: 'float',
    seed: 'int',
    model: 'ModelMetadata',
    mixing_history: 'dict[str, Sequence[float]]' = <factory>,
    sampling_history: 'dict[str, Sequence[float]]' = <factory>,
    warnings: 'tuple[str, ]' = (),
    sampling_dtype: 'str' = 'float32',
    ptt_diagnostics: 'dict[str, Any]' = <factory>,
    cde_sum: 'ndarray | None' = None,
    local_lambda_fit: 'dict[str, float | int] | None' = None,
    cij_reference: 'ndarray | None' = None,
    cij_generated: 'ndarray | None' = None,
    pca_reference: 'ndarray | None' = None,
    pca_generated: 'ndarray | None' = None,
    pca_explained_variance_ratio: 'ndarray | None' = None,
    data_comparison: 'dict[str, Any]' = <factory>,
    distance_comparison: 'dict[str, Any]' = <factory>,
    steering_potentials: 'ndarray | None' = None,
    log_importance_weights: 'ndarray | None' = None,
    steering: 'dict[str, Any]' = <factory>
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L749"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(path: 'str | Path', format: 'str | None' = None) → Path
```

Write the result in the format given by ``format`` or by the file extension.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`format`</b>:  One of ``"csv"``, ``"fasta"`` or ``"json"``; inferred from the extension when ``None``.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L762"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(
    directory: 'str | Path',
    label: 'str | None' = None
) → dict[str, Path]
```

Write samples (FASTA and CSV), a JSON summary and, in ``logs/``, the mixing/sampling logs.

PTT results also get ``logs/ptt.log`` and, when available, ladder-health and renewal logs; diagnostics add the data-comparison and distance logs.



**Args:**

 - <b>`directory`</b>:  Output folder.
 - <b>`label`</b>:  Optional file-name prefix, giving ``<label>_samples.fasta`` etc.



**Returns:**
 The written paths by kind.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L906"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_diagnostic_plots`

```python
save_diagnostic_plots(
    directory: 'str | Path',
    label: 'str | None' = None
) → dict[str, Path]
```

Save diagnostic plots as PNG files: mixing, Cij scatter, PCA and energy against CDE.

Plots whose data were not collected (see ``collect_diagnostics`` of :func:`sample_sequences`) are skipped.



**Args:**

 - <b>`directory`</b>:  Output folder.
 - <b>`label`</b>:  Optional file-name prefix.



**Returns:**
 The written paths by plot kind.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L738"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_csv`

```python
to_csv(path: 'str | Path') → Path
```

Write :meth:`to_dataframe` as CSV.



**Args:**

 - <b>`path`</b>:  Output file.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L697"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dataframe`

```python
to_dataframe() → DataFrame
```

Return one row per sequence: name, sequence, energy, and ``cde_sum``, steering and PRIVET columns when present.

PRIVET columns are ``NaN`` for sequences left out of the comparison (beyond ``n_measure``).

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L658"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-compatible document with the result type and schema version.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L723"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_fasta`

```python
to_fasta(path: 'str | Path') → Path
```

Write the sequences as FASTA, with energies in the headers.



**Args:**

 - <b>`path`</b>:  Output file.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L685"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```

Write :meth:`to_dict` as JSON.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`indent`</b>:  JSON indentation.



**Returns:**
 The written path.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingProgress`
One accepted update, passed to the ``progress`` callback of :func:`train_model`.



**Attributes:**

 - <b>`epoch`</b>:  Step number of the record (gradient steps, or graph steps for PCD eaDCA/edDCA).
 - <b>`metrics`</b>:  Numeric values of the history row: ``Pearson``, ``LL_val``, ``Density``, ...
 - <b>`stage`</b>:  Current training phase.
 - <b>`gradient_steps`</b>:  Accepted parameter updates so far.
 - <b>`structure_steps`</b>:  Graph activations or decimations so far.
 - <b>`sweeps`</b>:  Monte Carlo sweeps so far.
 - <b>`partition_estimate`</b>:  PTT only: log Z and its provenance.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    epoch: 'int',
    metrics: 'dict[str, Any]',
    stage: 'str' = 'optimization',
    gradient_steps: 'int' = 0,
    structure_steps: 'int' = 0,
    sweeps: 'int' = 0,
    partition_estimate: 'dict[str, Any] | None' = None
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingDatasetSummary`
Size and filtering of one training or validation alignment.



**Attributes:**

 - <b>`source`</b>:  Path of the alignment, if read from a file.
 - <b>`original_sequences`</b>:  Sequences in the file.
 - <b>`retained_sequences`</b>:  Sequences kept after filtering.
 - <b>`removed_invalid`</b>:  Sequences dropped for unknown tokens or wrong length.
 - <b>`removed_duplicates`</b>:  Duplicate sequences dropped.
 - <b>`sequence_length`</b>:  Number of sites ``L``.
 - <b>`num_states`</b>:  Number of states ``q``.
 - <b>`effective_sequences`</b>:  Sum of the sequence weights (``Meff``).

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    source: 'str | None',
    original_sequences: 'int',
    retained_sequences: 'int',
    removed_invalid: 'int',
    removed_duplicates: 'int',
    sequence_length: 'int',
    num_states: 'int',
    effective_sequences: 'float'
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingInitialization`
Resolved training setup, passed to ``on_initialized`` of :func:`train_model`.



**Attributes:**

 - <b>`training`</b>:  Summary of the training alignment.
 - <b>`validation`</b>:  Summary of the validation alignment, if any.
 - <b>`device`</b>:  Device used for training.
 - <b>`dtype`</b>:  Precision of the parameters.
 - <b>`n_chains`</b>:  Number of Markov chains.
 - <b>`effective_pseudocount`</b>:  Pseudocount applied to the statistics.
 - <b>`config`</b>:  The validated :class:`TrainingConfig`.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    training: 'TrainingDatasetSummary',
    validation: 'TrainingDatasetSummary | None',
    device: 'str',
    dtype: 'str',
    n_chains: 'int',
    effective_pseudocount: 'float',
    config: 'TrainingConfig'
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingResult`
Outcome of :func:`train_model`.



**Attributes:**

 - <b>`model`</b>:  The trained :class:`DCAModel`.
 - <b>`history`</b>:  One list per quantity, one entry per recorded update (see
 - <b>`:meth`</b>: `history_dataframe`).
 - <b>`chains`</b>:  Final Markov chains, one-hot, shape ``(n_chains, L, q)``.
 - <b>`pseudocount`</b>:  Pseudocount applied to the training statistics.
 - <b>`num_sequences`</b>:  Training sequences after filtering.
 - <b>`effective_sequences`</b>:  Effective number of training sequences (``Meff``).
 - <b>`artifacts`</b>:  Written files by kind (``params``, ``history``, ``ptt_archive``, ...).
 - <b>`warnings`</b>:  Non-fatal issues found during training.
 - <b>`converged`</b>:  Whether training stopped at its target (Pearson, density,  validation plateau or converged graph) rather than at a step limit.
 - <b>`stop_reason`</b>:  Why training stopped, e.g. ``"target_pearson"``.
 - <b>`gradient_steps`</b>:  Accepted parameter updates.
 - <b>`structure_steps`</b>:  Graph activations or decimations.
 - <b>`sweeps`</b>:  Monte Carlo sweeps performed.
 - <b>`config`</b>:  The validated :class:`TrainingConfig`.
 - <b>`input_report`</b>:  Filtering report of the training alignment.
 - <b>`initialization`</b>:  Resolved setup, see :class:`TrainingInitialization`.
 - <b>`partition_estimate`</b>:  PTT only: estimate of log Z for the final model.
 - <b>`ptt_sampler`</b>:  PTT only: the sampler, usable to draw more sequences.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model: 'DCAModel',
    history: 'dict[str, Sequence[float]]',
    chains: 'Tensor',
    pseudocount: 'float',
    num_sequences: 'int',
    effective_sequences: 'float',
    artifacts: 'dict[str, Path]' = <factory>,
    warnings: 'tuple[str, ]' = (),
    converged: 'bool' = False,
    stop_reason: 'str | None' = None,
    gradient_steps: 'int' = 0,
    structure_steps: 'int' = 0,
    sweeps: 'int' = 0,
    config: 'TrainingConfig | None' = None,
    input_report: 'dict[str, Any]' = <factory>,
    initialization: 'TrainingInitialization | None' = None,
    partition_estimate: 'PartitionEstimate | None' = None,
    ptt_sampler: 'PTTSampler | None' = None
) → None
```






---

#### <kbd>property</kbd> final_metrics

Last value of every history quantity, or an empty mapping before any update.



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `history_dataframe`

```python
history_dataframe() → DataFrame
```

Return the history as a DataFrame, one row per recorded update.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1296"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(
    directory: 'str | Path',
    label: 'str | None' = None
) → dict[str, Path]
```

Write a JSON summary, and the history table unless training already wrote it.



**Args:**

 - <b>`directory`</b>:  Output folder.
 - <b>`label`</b>:  Optional file-name prefix.



**Returns:**

 - <b>`The written paths by kind`</b>:  ``summary`` and possibly ``history``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1270"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_csv`

```python
to_csv(path: 'str | Path') → Path
```

Write the history with the columns and names of ``history.csv``.



**Args:**

 - <b>`path`</b>:  Output file.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1233"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a portable summary without embedding model or chain tensors.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1258"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```

Write :meth:`to_dict` as JSON.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`indent`</b>:  JSON indentation.



**Returns:**
 The written path.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1321"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ReintegrationResult`
Outcome of :func:`reintegrate_model`.



**Attributes:**

 - <b>`training`</b>:  Result of training on the combined alignment.
 - <b>`alignment`</b>:  Natural plus experimental sequences used for training.
 - <b>`weights`</b>:  Signed weight of each sequence of ``alignment``.
 - <b>`lambda_value`</b>:  Weight of the experimental data.
 - <b>`scaling_factor`</b>:  Factor applied to the experimental weights.
 - <b>`label`</b>:  File-name prefix of the written files.
 - <b>`artifacts`</b>:  Written files by kind.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    training: 'TrainingResult',
    alignment: 'Alignment',
    weights: 'ndarray',
    lambda_value: 'float',
    scaling_factor: 'float',
    label: 'str',
    artifacts: 'dict[str, Path]' = <factory>
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1371"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(directory: 'str | Path') → dict[str, Path]
```

Write the combined alignment, its weights and a JSON summary to ``directory``.

Files are named ``<label>_msa.fasta``, ``<label>_weights.dat`` and ``<label>_reintegration.json``.



**Args:**

 - <b>`directory`</b>:  Output folder.



**Returns:**

 - <b>`The written paths by kind`</b>:  ``alignment``, ``weights`` and ``summary``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1343"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-compatible document with the result type and schema version.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1359"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```

Write :meth:`to_dict` as JSON.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`indent`</b>:  JSON indentation.



**Returns:**
 The written path.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1393"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ThermodynamicIntegrationProgress`
Progress event passed to the ``progress`` callback of :func:`estimate_entropy`.



**Attributes:**

 - <b>`stage`</b>:  ``"theta_search"`` while raising ``theta_max``, then ``"integration"``.
 - <b>`completed`</b>:  Steps done in this stage.
 - <b>`total`</b>:  Largest number of steps of this stage.
 - <b>`theta`</b>:  Current bias strength.
 - <b>`entropy`</b>:  Current entropy estimate (integration stage only).
 - <b>`mean_sequence_identity`</b>:  Mean identity of the chains with the target.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    stage: 'str',
    completed: 'int',
    total: 'int',
    theta: 'float',
    entropy: 'float | None' = None,
    mean_sequence_identity: 'float | None' = None
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1414"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ThermodynamicIntegrationResult`
Outcome of :func:`estimate_entropy`.



**Attributes:**

 - <b>`entropy`</b>:  Estimated model entropy, in nats.
 - <b>`free_energy`</b>:  Free energy at ``theta = 0`` from the integration.
 - <b>`theta_max`</b>:  Largest bias strength reached.
 - <b>`target_fraction`</b>:  Fraction of chains matching the target at ``theta_max``.
 - <b>`history`</b>:  Per integration step: ``theta``, ``free_energy``, ``entropy``,  ``mean_sequence_identity`` and ``elapsed_seconds``.
 - <b>`model`</b>:  Metadata of the model.
 - <b>`artifacts`</b>:  Written files by kind, when ``output_dir`` was given.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    entropy: 'float',
    free_energy: 'float',
    theta_max: 'float',
    target_fraction: 'float',
    history: 'dict[str, Sequence[float]]',
    model: 'ModelMetadata',
    artifacts: 'dict[str, Path]' = <factory>
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1437"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `history_dataframe`

```python
history_dataframe() → DataFrame
```

Return the integration history as a DataFrame, one row per step.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1494"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(directory: 'str | Path', label: 'str' = 'entropy') → dict[str, Path]
```

Write ``<label>.log``, ``<label>.csv`` and ``<label>.json`` to ``directory``.



**Args:**

 - <b>`directory`</b>:  Output folder.
 - <b>`label`</b>:  File-name stem.



**Returns:**

 - <b>`The written paths by kind`</b>:  ``log``, ``csv`` and ``summary``.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1468"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_csv`

```python
to_csv(path: 'str | Path') → Path
```

Write the integration history as CSV.



**Args:**

 - <b>`path`</b>:  Output file.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1441"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a JSON-compatible document with the result type and schema version.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1456"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```

Write :meth:`to_dict` as JSON.



**Args:**

 - <b>`path`</b>:  Output file.
 - <b>`indent`</b>:  JSON indentation.



**Returns:**
 The written path.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L1479"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_log`

```python
to_log(path: 'str | Path') → Path
```

Write the integration history as an aligned, space-separated text table.



**Args:**

 - <b>`path`</b>:  Output file.



**Returns:**
 The written path.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
