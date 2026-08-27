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
print(training.stop_reason, training.converged)
print(training.gradient_steps, training.structure_steps, training.sweeps)
```

`max_epochs` remains a compatibility limit: it counts gradient steps for
`bmDCA` and graph-structure steps for sparse models. Nested `eaDCA` and
`edDCA` runs can set independent global budgets with
`max_gradient_steps=` and `max_structure_steps=`. Progress events expose the
same counters in addition to the legacy `epoch` field.

For reproducible applications, collect training values in the immutable,
validated configuration object:

```python
from adabmDCA import TrainingConfig, train_model

config = TrainingConfig(
    model_type="eaDCA",
    n_chains=2_000,
    max_structure_steps=100,
    max_gradient_steps=5_000,
    checkpoint_interval=5,
)
training = train_model("alignment.fasta", config=config)
```

When `config=` is provided it is authoritative for training values; path,
output, progress, and cancellation arguments remain on `train_model()`. The
configuration also exposes model-aware `limits`,
`resolved_checkpoint_interval`, and `resolve_pseudocount(effective_size)`.

## Input loading

High-level alignment arguments accept either a path or an in-memory
`Alignment`. FASTA, gzip-compressed FASTA, and Stockholm paths pass through
the same parser and validation policy:

```python
from adabmDCA import Alignment, TrainingConfig, train_model

alignment = Alignment(
    names=("sequence_1", "sequence_2"),
    sequences=("ACDE-", "ACD--"),
)
training = train_model(alignment, config=TrainingConfig(n_chains=1_000))
print(training.input_report)
```

`AlignmentLoadConfig` controls invalid-sequence handling, deduplication,
expected length, format selection, and gap normalization. `LoadedAlignment`
reports retained, invalid, and duplicate indices. Sequence weights can be a
path, sequence, NumPy array, or tensor and are aligned using those retained
indices.

`DatasetDCA.from_alignment()` and `DatasetDCA.from_loaded_alignment()` build
the tensor dataset without constructor-owned file parsing. The legacy
`DatasetDCA(path_data=...)` form remains available as a compatibility wrapper.

## Output serialization

High-level result objects own their serialization. JSON outputs use a stable
envelope with `schema_version`, `result_type`, and `data`; NumPy values,
tensors, paths, dataclasses, and non-finite floating-point values are converted
to strict JSON safely.

```python
scores = model.score_sequences(["ACDEFGHIK"])
scores.to_json("outputs/scores.json")
scores.to_csv("outputs/scores.csv")
scores.to_fasta("outputs/scores.fasta")

# Infer the serializer from the suffix.
scores.save("outputs/scores.json")
```

Composite results expose `save_bundle()`. Bundles use predictable filenames,
create their parent directories, write files atomically, and return a mapping
of artifact names to paths:

```python
artifacts = result.save_bundle("outputs", label="experiment_1")
print(artifacts["summary"])
```

Contact maps include labelled CSV, NumPy `.npy`, JSON metadata, and the
historical headerless matrix. Sampling bundles include FASTA, sequence CSV,
JSON metadata, and diagnostic histories. Training bundles contain a portable
summary and normalized history without duplicating model and chain tensors in
JSON. Serialization failures raise `OutputSerializationError`.

The CLI uses these same methods; output formatting is therefore identical in
notebooks, Python services, and terminal commands.

## Additional workflow APIs

Specialized front-end workflows are also callable without constructing command
line arguments:

```python
from adabmDCA import estimate_entropy, reintegrate_model, split_alignment

split = split_alignment("family.fasta", attempts=10, seed=4)
split.save_bundle("profile/family")

reintegrated = reintegrate_model(
    "natural.fasta",
    "tested.fasta",
    "adjustments.dat",
    config=config,
    output_dir="reintegrated_model",
)

entropy = estimate_entropy(
    model="params.dat",
    natural_alignment="family.fasta",
    target_alignment="target.fasta",
    n_steps=100,
    progress=lambda event: print(event.stage, event.completed, event.total),
)
```

Entropy estimation supports cancellation and bounds the adaptive `theta_max`
search with `max_theta_iterations`, avoiding an unbounded front-end loop.

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

The dictionary contains a stable error code, a process-oriented `exit_code`, a
human-readable message, and structured details suitable for notebooks, CLIs,
and tool adapters. Exit code 2 denotes invalid usage or incompatible inputs,
1 denotes loading, serialization, computation, or convergence failure, and
130 denotes cancellation.

Expected errors from external parsers and serializers retain their original
exception through Python exception chaining (`error.__cause__`). Unexpected
programming exceptions are deliberately not converted into application
errors, so bugs keep their traceback instead of being mislabeled as bad user
input.

The top-level CLI renders these structured errors consistently on standard
error and returns the exception's declared exit code. Unexpected programming
errors are not swallowed, so they retain their traceback.

## Compatibility and versioning

The existing low-level tensor APIs remain backward-compatible. High-level
model metadata includes a `schema_version` so external adapters can validate
the result contract independently of the package release number. Additive
high-level fields may be introduced in minor releases; removing or changing a
documented field requires a deprecation period and a major schema-version
change.
