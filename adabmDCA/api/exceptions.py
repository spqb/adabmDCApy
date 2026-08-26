"""Structured exceptions exposed by the high-level adabmDCA API."""

from __future__ import annotations

from typing import Any, Mapping


class AdabmDCAError(Exception):
    """Base class for recoverable errors raised by the application API."""

    code = "adabmdca_error"

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
            "message": self.message,
            "details": self.details,
        }


class InputValidationError(AdabmDCAError, ValueError):
    """Raised when a high-level API input is invalid."""

    code = "input_validation_error"


class ModelCompatibilityError(AdabmDCAError, ValueError):
    """Raised when data are incompatible with a DCA model."""

    code = "model_compatibility_error"


class ModelLoadError(AdabmDCAError, RuntimeError):
    """Raised when model parameters cannot be loaded."""

    code = "model_load_error"


class OperationCancelledError(AdabmDCAError, RuntimeError):
    """Raised when a caller-provided cancellation hook stops an operation."""

    code = "operation_cancelled"


class AlignmentError(AdabmDCAError):
    """Base class for alignment input and processing errors."""

    code = "alignment_error"


class AlignmentFormatError(AlignmentError, ValueError):
    """Raised when an alignment format cannot be detected or parsed."""

    code = "alignment_format_error"


class AlignmentLengthError(AlignmentError, ValueError):
    """Raised when sequences do not share a common aligned length."""

    code = "alignment_length_error"
