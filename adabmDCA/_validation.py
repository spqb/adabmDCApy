"""Validation shared by public workflow boundaries."""

from numbers import Integral

from adabmDCA.api.exceptions import InputValidationError


def validate_integer(
    name: str,
    value: int,
    *,
    minimum: int = 1,
    maximum: int | None = None,
    error_type: type[InputValidationError] = InputValidationError,
) -> None:
    """Reject coercible numbers and booleans before allocating or iterating."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise error_type(f"{name} must be an integer.", details={"name": name, "value": value})
    if value < minimum:
        requirement = "positive" if minimum == 1 else f"at least {minimum}"
        raise error_type(f"{name} must be {requirement}.", details={"name": name, "value": value})
    if maximum is not None and value > maximum:
        raise error_type(f"{name} must be at most {maximum}.", details={"name": name, "value": value})


def validate_seed(seed: int, *, error_type: type[InputValidationError] = InputValidationError) -> None:
    """Use the seed range accepted by PyTorch without silently truncating floats."""
    validate_integer("seed", seed, minimum=-(2**63), maximum=2**64 - 1, error_type=error_type)


def validate_finite_parameters(params) -> None:
    """Check tensors in bounded slices, including overflow after dtype conversion."""
    for name in ("bias", "coupling_matrix"):
        tensor = params[name]
        if tensor.numel() == 0:
            continue
        # Avoid a boolean allocation as large as the dense coupling tensor.
        rows = max(1, 1_048_576 // max(1, tensor[0].numel()))
        for chunk in tensor.split(rows):
            if not chunk.isfinite().all().item():
                raise ValueError(f"Model parameter '{name}' must contain only finite values.")
