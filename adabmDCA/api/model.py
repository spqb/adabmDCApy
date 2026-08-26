"""Notebook-friendly DCA model object."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import torch

from adabmDCA.api.exceptions import InputValidationError, ModelLoadError
from adabmDCA.api.results import ModelMetadata
from adabmDCA.api.runtime import resolve_runtime
from adabmDCA.fasta import get_tokens
from adabmDCA.io import load_params

if TYPE_CHECKING:
    from collections.abc import Iterable
    from adabmDCA.api.results import ContactMapResult, EnergyResult, MutationScanResult, SamplingResult


def _package_version() -> str | None:
    from adabmDCA import __version__

    return __version__


class DCAModel:
    """Loaded DCA parameters with convenient analysis methods.

    Prefer :func:`load_model` when loading a model saved by adabmDCA. Direct
    construction is useful for advanced users and tests that already have the
    parameter tensors in memory.
    """

    def __init__(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        alphabet: str = "protein",
        source: str | Path | None = None,
    ) -> None:
        if "bias" not in params or "coupling_matrix" not in params:
            raise InputValidationError(
                "Model parameters must contain 'bias' and 'coupling_matrix'."
            )
        bias = params["bias"]
        couplings = params["coupling_matrix"]
        if bias.ndim != 2 or couplings.shape != (*bias.shape, *bias.shape):
            raise InputValidationError(
                "Model parameter shapes are inconsistent.",
                details={"bias_shape": tuple(bias.shape), "coupling_shape": tuple(couplings.shape)},
            )
        tokens = get_tokens(alphabet)
        if bias.shape[1] != len(tokens):
            raise InputValidationError(
                "The model state count does not match the selected alphabet.",
                details={"model_states": bias.shape[1], "tokens": tokens},
            )
        self.params = {key: value for key, value in params.items()}
        self.alphabet = alphabet
        self.tokens = tokens
        self.source = str(source) if source is not None else None

    @property
    def metadata(self) -> ModelMetadata:
        bias = self.params["bias"]
        return ModelMetadata(
            length=bias.shape[0],
            alphabet=self.alphabet,
            tokens=self.tokens,
            device=str(bias.device),
            dtype=str(bias.dtype).removeprefix("torch."),
            source=self.source,
            package_version=_package_version(),
        )

    def compute_energies(self, sequences: str | Iterable[str]):
        """Return a NumPy vector with one energy per sequence."""
        from adabmDCA.api.scoring import compute_energies

        return compute_energies(sequences=sequences, model=self)

    def score_sequences(self, sequences: str | Iterable[str]) -> "EnergyResult":
        """Return energies plus sequence and model metadata."""
        from adabmDCA.api.scoring import score_sequences

        return score_sequences(sequences=sequences, model=self)

    def compute_contact_map(self):
        """Return the model's APC-corrected contact-score matrix."""
        from adabmDCA.api.contacts import compute_contact_map

        return compute_contact_map(model=self)

    def predict_contacts(self) -> "ContactMapResult":
        """Return contact scores plus method and model metadata."""
        from adabmDCA.api.contacts import predict_contacts

        return predict_contacts(model=self)

    def scan_mutations(self, wild_type: str, *, name: str = "wild_type") -> "MutationScanResult":
        """Score every single-residue mutant of ``wild_type``."""
        from adabmDCA.api.mutations import scan_mutations

        return scan_mutations(wild_type=wild_type, model=self, name=name)

    def sample(
        self,
        n_sequences: int,
        *,
        n_sweeps: int = 1000,
        sampler: str = "gibbs",
        beta: float = 1.0,
        seed: int = 0,
    ) -> tuple[str, ...]:
        """Generate sequences and return them as ordinary strings."""
        from adabmDCA.api.sampling import generate_sequences

        return generate_sequences(
            model=self,
            n_sequences=n_sequences,
            n_sweeps=n_sweeps,
            sampler=sampler,
            beta=beta,
            seed=seed,
        )

    def sample_sequences(self, n_sequences: int, **kwargs) -> "SamplingResult":
        """Generate sequences and return structured diagnostics."""
        from adabmDCA.api.sampling import sample_sequences

        return sample_sequences(model=self, n_sequences=n_sequences, **kwargs)


def load_model(
    path: str | Path,
    *,
    alphabet: str = "protein",
    device: str = "auto",
    dtype: str = "float32",
) -> DCAModel:
    """Load DCA parameters into a reusable :class:`DCAModel`.

    ``device='auto'`` selects CUDA when available and otherwise uses CPU.
    Explicit ``'cpu'``, ``'cuda'``, and ``'mps'`` values remain supported.
    """
    model_path = Path(path)
    if not model_path.is_file():
        raise ModelLoadError(
            f"Model parameter file '{model_path}' was not found.",
            details={"path": str(model_path)},
        )
    resolved_device, resolved_dtype = resolve_runtime(device, dtype)
    tokens = get_tokens(alphabet)
    try:
        params = load_params(
            str(model_path),
            tokens=tokens,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        return DCAModel(params, alphabet=alphabet, source=model_path)
    except (InputValidationError, ModelLoadError):
        raise
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load model parameters from '{model_path}': {exc}",
            details={"path": str(model_path)},
        ) from exc


def inspect_model(
    model: DCAModel | str | Path,
    *,
    alphabet: str = "protein",
    device: str = "auto",
    dtype: str = "float32",
) -> ModelMetadata:
    """Return portable metadata for an in-memory or saved model."""
    loaded = model if isinstance(model, DCAModel) else load_model(
        model, alphabet=alphabet, device=device, dtype=dtype
    )
    return loaded.metadata
