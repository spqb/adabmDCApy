<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.api.results`
Result objects returned by the high-level adabmDCA API.



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ModelMetadata`
Portable description of a loaded DCA model.

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







---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```

Write model metadata as a versioned JSON document.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EnergyResult`
Energies and associated sequence/model metadata.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    sequences: 'tuple[str, ]',
    energies: 'ndarray',
    model: 'ModelMetadata',
    names: 'tuple[str, ]' = (),
    warnings: 'tuple[str, ]' = ()
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(path: 'str | Path', format: 'str | None' = None) → Path
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(directory: 'str | Path', stem: 'str' = 'energies') → dict[str, Path]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_csv`

```python
to_csv(path: 'str | Path') → Path
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dataframe`

```python
to_dataframe()
```

Return the result as a pandas DataFrame.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a portable representation of the result.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_fasta`

```python
to_fasta(path: 'str | Path') → Path
```

Write sequences and energies to a FASTA file.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ContactMapResult`
Contact scores computed from a model or alignment.

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(path: 'str | Path', format: 'str | None' = None) → Path
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(
    directory: 'str | Path',
    label: 'str | None' = None
) → dict[str, Path]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_matrix`

```python
save_matrix(path: 'str | Path') → Path
```

Write the contact map using the historical ``i,j,score`` format.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_csv`

```python
to_csv(path: 'str | Path') → Path
```

Write a labelled long-form contact table.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_long_dataframe`

```python
to_long_dataframe()
```

Return one row per matrix entry.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_npy`

```python
to_npy(path: 'str | Path') → Path
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MutationRecord`
One single-residue mutation and its DCA score.

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







---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L198"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L210"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MutationScanResult`
Single-mutant scores for a wild-type sequence.

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







---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(path: 'str | Path', format: 'str | None' = None) → Path
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L261"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(
    directory: 'str | Path',
    stem: 'str | None' = None
) → dict[str, Path]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L254"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_csv`

```python
to_csv(path: 'str | Path') → Path
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dataframe`

```python
to_dataframe()
```

Return one row per mutation, with zero- and one-based positions.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_fasta`

```python
to_fasta(path: 'str | Path') → Path
```

Write the mutation library using the historical CLI format.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L238"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SamplingProgress`
Progress event emitted during sequence generation.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    stage: 'str',
    completed: 'int',
    total: 'int',
    pearson: 'float | None' = None,
    slope: 'float | None' = None
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L282"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ProfileSplitResult`
Training/test alignment split produced by the Cobalt algorithm.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    training: 'Alignment',
    test: 'Alignment',
    score: 'int',
    attempts: 'int',
    tokens: 'str',
    seed: 'int'
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L312"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(output_prefix: 'str | Path') → dict[str, Path]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L293"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L324"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SamplingResult`
Generated sequences, energies, and sampling diagnostics.

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
    cij_reference: 'ndarray | None' = None,
    cij_generated: 'ndarray | None' = None,
    pca_reference: 'ndarray | None' = None,
    pca_generated: 'ndarray | None' = None,
    pca_explained_variance_ratio: 'ndarray | None' = None
) → None
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L389"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(path: 'str | Path', format: 'str | None' = None) → Path
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L393"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(
    directory: 'str | Path',
    label: 'str | None' = None
) → dict[str, Path]
```

Save samples, diagnostics, and metadata using predictable filenames.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L414"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_diagnostic_plots`

```python
save_diagnostic_plots(
    directory: 'str | Path',
    label: 'str | None' = None
) → dict[str, Path]
```

Save mixing, correlation, and PCA diagnostics as PNG files.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L386"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_csv`

```python
to_csv(path: 'str | Path') → Path
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L366"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dataframe`

```python
to_dataframe()
```

Return generated sequences and energies as a DataFrame.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L345"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L378"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_fasta`

```python
to_fasta(path: 'str | Path') → Path
```

Write generated sequences and energies to FASTA.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L363"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L494"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingProgress`
One metrics update emitted by a training routine.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    epoch: 'int',
    metrics: 'dict[str, float]',
    stage: 'str' = 'optimization',
    gradient_steps: 'int' = 0,
    structure_steps: 'int' = 0,
    sweeps: 'int' = 0
) → None
```









---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L506"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingDatasetSummary`
Dimensions, filtering outcomes, and statistical size of one MSA.

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L520"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingInitialization`
Resolved training setup emitted after inputs are loaded and weighted.

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L533"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingResult`
Trained model and the state required to inspect or resume it.

<a href="https://github.com/spqb/adabmDCApy/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model: 'Any',
    history: 'dict[str, Sequence[float]]',
    chains: 'Any',
    log_weights: 'Any',
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
    initialization: 'TrainingInitialization | None' = None
) → None
```






---

#### <kbd>property</kbd> final_metrics

Return the last emitted metric values, or an empty mapping.



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L555"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `history_dataframe`

```python
history_dataframe()
```

Return the training history as a pandas DataFrame.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L591"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(
    directory: 'str | Path',
    label: 'str | None' = None
) → dict[str, Path]
```

Save a portable training summary and normalized history table.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L588"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_csv`

```python
to_csv(path: 'str | Path') → Path
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L561"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return a portable summary without embedding model or chain tensors.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L585"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L608"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ReintegrationResult`
Prepared reintegration dataset and its completed training result.

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L638"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(directory: 'str | Path') → dict[str, Path]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L620"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L635"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```






---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L649"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ThermodynamicIntegrationProgress`
One progress event from thermodynamic integration.

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L661"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ThermodynamicIntegrationResult`
Entropy estimate and integration trajectory.

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

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L673"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `history_dataframe`

```python
history_dataframe()
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L703"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_bundle`

```python
save_bundle(directory: 'str | Path', label: 'str' = 'entropy') → dict[str, Path]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L693"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_csv`

```python
to_csv(path: 'str | Path') → Path
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L676"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L690"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_json`

```python
to_json(path: 'str | Path', indent: 'int' = 2) → Path
```





---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/api/results.py#L696"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_log`

```python
to_log(path: 'str | Path') → Path
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
