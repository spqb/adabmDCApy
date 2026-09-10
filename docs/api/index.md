# Python API reference

For notebook and application workflows, begin with the
[high-level Python API guide](../high_level_api.md). It explains the public
objects exported directly from `adabmDCA`, with complete examples for loading,
training, sampling, scoring, contact prediction, mutation scanning, entropy
estimation, reintegration, and result serialization.

The pages under **High-level workflows** document the current callable
signatures and result types. The pages under **Core data APIs** cover alignment
loading, preprocessing, configuration, and training control. **Numerical and
low-level APIs** are intended for code that needs direct access to tensors,
samplers, statistics, and parameter files.

Most application code should import the public interfaces from the package
root:

```python
from adabmDCA import load_model, sample_sequences, train_model
```

Use the [generated symbol index](symbols.md) to find an individual class,
method, or function.
