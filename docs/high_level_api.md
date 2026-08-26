# High-level Python API

The high-level API exposes the same scientific workflows as the CLI while
remaining convenient in notebooks and regular Python applications. The CLI is
implemented as an adapter over these operations.

## Load and reuse a model

```python
from adabmDCA import load_model

model = load_model(
    "DCA_model/params.dat",
    alphabet="protein",
    device="auto",
)

print(model.metadata.to_dict())
```

`device="auto"` selects an available accelerator and falls back to CPU.
Loading once avoids repeating the alphabet, device, and dtype configuration.

## Score sequences

For a compact NumPy result:

```python
energies = model.compute_energies([
    "ACDEFGHIK",
    "ACDEYGHIK",
])
```

For sequences, names, metadata, warnings, and DataFrame conversion:

```python
result = model.score_sequences(["ACDEFGHIK", "ACDEYGHIK"])
display(result.to_dataframe())
```

FASTA files can be scored without loading a model object explicitly:

```python
from adabmDCA import score_sequences

result = score_sequences(
    model="DCA_model/params.dat",
    fasta_path="sequences.fasta",
    alphabet="protein",
)
result.to_fasta("scored_sequences.fasta")
```

The historical low-level tensor operation remains available:

```python
from adabmDCA import compute_energy

energies_tensor = compute_energy(one_hot_sequences, params)
```

## Contacts and mutation scanning

```python
contact_scores = model.compute_contact_map()

scan = model.scan_mutations("ACDEFGHIK", name="wild_type")
display(scan.to_dataframe())
scan.to_fasta("wild_type_DMS.fasta")
```

Mutation results contain both the historical zero-based position and a
`position_1based` field for biological reporting.

## Generate sequences

```python
sequences = model.sample(
    100,
    n_sweeps=1_000,
    sampler="gibbs",
    seed=42,
)
```

Use the structured operation for energies, diagnostics, progress events, and
cancellation:

```python
from adabmDCA import sample_sequences

result = sample_sequences(
    model=model,
    n_sequences=100,
    n_sweeps=1_000,
    seed=42,
    progress=lambda event: print(event.completed, event.total),
)

display(result.to_dataframe())
result.to_fasta("samples.fasta")
```

## Train a model

Training can run entirely in memory:

```python
from adabmDCA import train_model

training = train_model(
    "alignment.fasta",
    model_type="bmDCA",
    alphabet="protein",
    n_chains=1_000,
    max_epochs=5_000,
    device="auto",
    seed=42,
)

model = training.model
display(training.history_dataframe())
```

Set `output_dir` to retain parameters, chains, weights, and the historical
training log:

```python
training = train_model(
    "alignment.fasta",
    output_dir="DCA_model",
    label="family_name",
)
print(training.artifacts)
```

Long-running clients can supply `progress` and `is_cancelled` callbacks. A
future MCP or web service can use these hooks to implement background jobs
without adding server concerns to the scientific package.

The `adabmDCA train` command uses the same progress callback to render an
interactive terminal bar. Pass `--no-progress` when machine-readable or quiet
terminal output is preferred.

## Structured errors

High-level operations raise exceptions derived from `AdabmDCAError`:

```python
from adabmDCA import AdabmDCAError

try:
    model.compute_energies(["INVALID"])
except AdabmDCAError as error:
    print(error.to_dict())
```

The dictionary contains a stable error code, a human-readable message, and
structured details suitable for notebooks, CLIs, and tool adapters.

## Compatibility and versioning

The existing low-level tensor APIs remain backward-compatible. High-level
model metadata includes a `schema_version` so external adapters can validate
the result contract independently of the package release number. Additive
high-level fields may be introduced in minor releases; removing or changing a
documented field requires a deprecation period and a major schema-version
change.
