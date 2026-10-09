"""Notebook-friendly DCA model object."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from adabmDCA._validation import validate_finite_parameters
from adabmDCA.api.results import ModelMetadata
from adabmDCA.api.runtime import resolve_runtime
from adabmDCA.exceptions import AdabmDCAError, InputValidationError, ModelLoadError
from adabmDCA.fasta import get_tokens
from adabmDCA.io import load_params

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

    from adabmDCA.api.results import (
        ContactMapResult,
        EnergyResult,
        MutationScanResult,
        SamplingResult,
    )


def _package_version() -> str | None:
    from adabmDCA import __version__

    return __version__


class DCAModel:
    """DCA parameters and convenience methods for scoring and sampling.

    Use :func:`load_model` for a saved model. Construct ``DCAModel`` directly
    when parameter tensors are already in memory. The bias tensor determines
    the model length ``L``, number of states ``q``, device, and dtype. The
    ordered ``tokens`` string maps state indices to residue symbols.

    Attributes:
        params: Parameter tensors, including ``bias`` with shape ``(L, q)``
            and ``coupling_matrix`` with shape ``(L, q, L, q)``.
        alphabet: Standard alphabet name or custom alphabet passed at creation.
        tokens: Resolved, ordered token string of length ``q``.
        source: Source path as a string, or ``None`` for an in-memory model.
    """

    def __init__(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        alphabet: str = "protein",
        source: str | Path | None = None,
    ) -> None:
        """Validate and retain a set of DCA parameter tensors.

        Args:
            params: Mapping containing finite ``bias`` and
                ``coupling_matrix`` tensors with compatible shapes.
            alphabet: ``"protein"``, ``"dna"``, ``"rna"``, or an ordered
                custom token string whose length equals ``q``.
            source: Optional path to the file from which the parameters came.

        Raises:
            InputValidationError: If required tensors are missing, nonfinite,
                incompatible in shape, or inconsistent with ``alphabet``.
        """
        if "bias" not in params or "coupling_matrix" not in params:
            raise InputValidationError("Model parameters must contain 'bias' and 'coupling_matrix'.")
        bias = params["bias"]
        couplings = params["coupling_matrix"]
        if not isinstance(bias, torch.Tensor) or not isinstance(couplings, torch.Tensor):
            raise InputValidationError("Model parameters must be tensors.")
        if bias.ndim != 2 or couplings.shape != (*bias.shape, *bias.shape):
            raise InputValidationError(
                "Model parameter shapes are inconsistent.",
                details={"bias_shape": tuple(bias.shape), "coupling_shape": tuple(couplings.shape)},
            )
        try:
            validate_finite_parameters(params)
        except ValueError as exc:
            raise InputValidationError(str(exc)) from exc
        try:
            tokens = get_tokens(alphabet)
        except (TypeError, ValueError) as exc:
            raise InputValidationError("alphabet must be a valid non-empty string.") from exc
        if bias.shape[1] != len(tokens):
            raise InputValidationError(
                "The model state count does not match the selected alphabet.",
                details={"model_states": bias.shape[1], "tokens": tokens},
            )
        self.params = {key: value for key, value in params.items()}
        self.alphabet = alphabet
        self.tokens = tokens
        self.source = str(source) if source is not None else None

    def __repr__(self) -> str:
        """Show ``L``, ``q``, tokens, source, device, and dtype of the model."""
        bias = self.params["bias"]
        return (
            f"DCAModel(L={bias.shape[0]}, q={bias.shape[1]}, tokens={self.tokens!r}, "
            f"source={self.source!r}, device={str(bias.device)!r}, "
            f"dtype={str(bias.dtype).removeprefix('torch.')!r})"
        )

    @property
    def metadata(self) -> ModelMetadata:
        """Portable model description derived from the current bias tensor.

        Includes length, alphabet, tokens, source, package version, device,
        and dtype. ``num_states`` is available on the returned metadata.
        """
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

    def compute_energies(self, sequences: str | Iterable[str]) -> np.ndarray:
        """Compute model energies for one or more aligned sequences.

        Args:
            sequences: One sequence string or an iterable of strings, each of
                length ``L`` and containing only the model's tokens.

        Returns:
            A one-dimensional NumPy array with one energy per input sequence,
            including when a single string is supplied.
        """
        from adabmDCA.api.scoring import compute_energies

        return compute_energies(sequences=sequences, model=self)

    def score_sequences(self, sequences: str | Iterable[str], *, local_lambda: float = 1.0) -> EnergyResult:
        """Compute energy and CDE-based local free energy with sequence metadata.

        Args:
            sequences: One aligned sequence or an iterable of aligned
                sequences compatible with this model.
            local_lambda: Coefficient of summed CDE; defaults to 1.

        Returns:
            An :class:`EnergyResult` containing the sequences, energy and
            local-free-energy vectors, and model metadata.
        """
        from adabmDCA.api.scoring import score_sequences

        return score_sequences(sequences=sequences, model=self, local_lambda=local_lambda)

    def compute_contact_map(self) -> np.ndarray:
        """Return the model's APC-corrected contact scores.

        Returns:
            A NumPy array of shape ``(L, L)``. Contact prediction requires
            the ``"-"`` gap token in the model alphabet.
        """
        from adabmDCA.api.contacts import compute_contact_map

        return compute_contact_map(model=self)

    def predict_contacts(self) -> ContactMapResult:
        """Return contact scores together with method and model metadata.

        Returns:
            A :class:`ContactMapResult` whose score matrix has shape
            ``(L, L)``. The model alphabet must include ``"-"``.
        """
        from adabmDCA.api.contacts import predict_contacts

        return predict_contacts(model=self)

    def scan_mutations(self, wild_type: str, *, name: str = "wild_type") -> MutationScanResult:
        """Score every single-token substitution of a wild-type sequence.

        Args:
            wild_type: Aligned sequence of length ``L`` using model tokens.
            name: Label attached to the resulting mutation scan.

        Returns:
            A :class:`MutationScanResult` with wild-type energy and each
            mutant's energy difference relative to the wild type. Gap
            substitutions are included when ``"-"`` is a model token.
        """
        from adabmDCA.api.mutations import scan_mutations

        return scan_mutations(wild_type=wild_type, model=self, name=name)

    def sample(
        self,
        n_sequences: int,
        *,
        n_sweeps: int = 1000,
        sampler: str = "metropolized_gibbs",
        beta: float = 1.0,
        seed: int = 0,
    ) -> tuple[str, ...]:
        """Generate sequences and return only their decoded strings.

        Args:
            n_sequences: Number of sequences to generate.
            n_sweeps: Sampling sweeps applied to the initial chains.
            sampler: ``"metropolis"``, ``"gibbs"`` or ``"metropolized_gibbs"``.
            beta: Positive inverse temperature used for sampling.
            seed: Random seed for reproducible initialization and sampling.

        Returns:
            A tuple of ``n_sequences`` strings, each of length ``L``.
        """
        from adabmDCA.api.sampling import generate_sequences

        return generate_sequences(
            model=self,
            n_sequences=n_sequences,
            n_sweeps=n_sweeps,
            sampler=sampler,
            beta=beta,
            seed=seed,
        )

    def sample_sequences(self, n_sequences: int, **kwargs) -> SamplingResult:
        """Generate sequences with energies and optional diagnostics.

        Args:
            n_sequences: Number of sequences to generate.
            **kwargs: Additional options accepted by
                :func:`adabmDCA.api.sampling.sample_sequences`, such as
                ``n_sweeps``, ``sampler``, ``seed``, or ``reference_fasta``.

        Returns:
            A :class:`SamplingResult` containing generated sequences,
            energies, model metadata, and any requested diagnostics.
        """
        from adabmDCA.api.sampling import sample_sequences

        return sample_sequences(model=self, n_sequences=n_sequences, **kwargs)


def load_model(
    path: str | Path,
    *,
    alphabet: str | None = None,
    device: str = "auto",
    dtype: str = "float32",
) -> DCAModel:
    """Load DCA parameters from a text parameter file or a PTT archive.

    The file type is detected from its content. For a PTT archive (``.h5``),
    the final model of the training run is loaded together with its alphabet.

    Args:
        path: Parameter file written by ``adabmDCA train`` (``params.dat``,
            optionally gzipped) or a PTT archive (``ptt.h5``).
        alphabet: ``"protein"``, ``"rna"``, ``"dna"`` or an ordered custom token
            string. ``None`` reads it from a PTT archive and assumes
            ``"protein"`` for text files. An explicit value must match the archive.
        device: ``"auto"`` (CUDA when available, else CPU), ``"cpu"``, ``"cuda"``
            or ``"mps"``.
        dtype: ``"float32"`` or ``"float64"``. PTT archives keep their precision.

    Returns:
        A :class:`DCAModel` on the requested device.

    Raises:
        ModelLoadError: If the file is missing or cannot be parsed.
        InputValidationError: If the alphabet is invalid or conflicts with the file.

    Example:
        >>> model = load_model("output/params.dat.gz", alphabet="rna")
        >>> model
        DCAModel(L=136, q=5, tokens='-ACGU', ...)
    """
    model_path = Path(path)
    if not model_path.is_file():
        raise ModelLoadError(
            f"Model parameter file '{model_path}' was not found.",
            details={"path": str(model_path)},
        )
    resolved_device, resolved_dtype = resolve_runtime(device, dtype)
    with model_path.open("rb") as handle:
        is_archive = handle.read(8) == b"\x89HDF\r\n\x1a\n"
    if is_archive:
        from adabmDCA.ptt.archive import load_ptt_endpoint
        params, tokens = load_ptt_endpoint(model_path, device=resolved_device, alphabet=alphabet)
        return DCAModel(params, alphabet=tokens, source=model_path)
    alphabet = "protein" if alphabet is None else alphabet
    try:
        tokens = get_tokens(alphabet)
    except (TypeError, ValueError) as exc:
        raise InputValidationError("alphabet must be a valid non-empty string.") from exc
    try:
        params = load_params(
            str(model_path),
            tokens=tokens,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        return DCAModel(params, alphabet=alphabet, source=model_path)
    except AdabmDCAError:
        raise
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ModelLoadError(
            f"Could not load model parameters from '{model_path}': {exc}",
            details={"path": str(model_path)},
        ) from exc


def inspect_model(
    model: DCAModel | str | Path,
    *,
    alphabet: str | None = None,
    device: str = "auto",
    dtype: str = "float32",
) -> ModelMetadata:
    """Describe a model without using it: length, alphabet, device and precision.

    Args:
        model: A :class:`DCAModel`, or a path to a parameter file or PTT archive.
        alphabet: Alphabet of a text parameter file: ``"protein"``, ``"rna"``,
            ``"dna"`` or an ordered custom token string. ``None`` reads it from a
            PTT archive and assumes ``"protein"`` for text files. Ignored when
            ``model`` is already a :class:`DCAModel`.
        device: ``"auto"`` (CUDA when available, else CPU), ``"cpu"``,
            ``"cuda"`` or ``"mps"``. Ignored for an in-memory model.
        dtype: ``"float32"`` or ``"float64"`` for loaded parameters. Ignored for an
            in-memory model.

    Returns:
        The model's :class:`ModelMetadata`.
    """
    loaded = model if isinstance(model, DCAModel) else load_model(model, alphabet=alphabet, device=device, dtype=dtype)
    return loaded.metadata
