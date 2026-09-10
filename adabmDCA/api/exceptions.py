"""Structured exceptions exposed by the high-level adabmDCA API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class AdabmDCAError(Exception):
    """Base class for recoverable errors raised by the application API."""

    code = "adabmdca_error"
    exit_code = 1

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the error."""
        return {
            "code": self.code,
            "exit_code": self.exit_code,
            "message": self.message,
            "details": self.details,
        }


class InputValidationError(AdabmDCAError, ValueError):
    """Raised when a high-level API input is invalid."""

    code = "input_validation_error"
    exit_code = 2


class InputLoadError(AdabmDCAError, RuntimeError):
    """Raised when an input resource cannot be read or decoded."""

    code = "input_load_error"


class WeightLoadError(InputLoadError):
    """Raised when sequence weights cannot be loaded or aligned."""

    code = "weight_load_error"


class ChainLoadError(InputLoadError):
    """Raised when an initial chain state cannot be loaded or validated."""

    code = "chain_load_error"


class OutputSerializationError(AdabmDCAError, RuntimeError):
    """Raised when a result cannot be serialized to the requested output."""

    code = "output_serialization_error"


class ComputationError(AdabmDCAError, RuntimeError):
    """Raised when a valid scientific operation cannot be completed."""

    code = "computation_error"


class ConvergenceError(ComputationError):
    """Raised when an iterative operation exhausts its convergence budget."""

    code = "convergence_error"


class ModelCompatibilityError(AdabmDCAError, ValueError):
    """Raised when data are incompatible with a DCA model."""

    code = "model_compatibility_error"
    exit_code = 2


class ModelLoadError(AdabmDCAError, RuntimeError):
    """Raised when model parameters cannot be loaded."""

    code = "model_load_error"


class OperationCancelledError(AdabmDCAError, RuntimeError):
    """Raised when a caller-provided cancellation hook stops an operation."""

    code = "operation_cancelled"
    exit_code = 130


class AlignmentError(AdabmDCAError):
    """Base class for alignment input and processing errors."""

    code = "alignment_error"
    exit_code = 2


class AlignmentLoadError(AlignmentError, RuntimeError):
    """Raised when an alignment resource cannot be opened or decoded."""

    code = "alignment_load_error"
    exit_code = 1


class AlignmentFormatError(AlignmentError, ValueError):
    """Raised when an alignment format cannot be detected or parsed."""

    code = "alignment_format_error"


class AlignmentLengthError(AlignmentError, ValueError):
    """Raised when sequences do not share a common aligned length."""

    code = "alignment_length_error"
