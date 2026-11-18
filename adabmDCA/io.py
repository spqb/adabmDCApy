"""FASTA and text parameter I/O."""

import gzip
import math
import warnings
from io import TextIOWrapper
from pathlib import Path
from typing import IO, Dict, Optional, Union

import numpy as np
import torch

from adabmDCA._validation import validate_finite_parameters
from adabmDCA.alignment import Alignment, write_alignment
from adabmDCA.alphabet import get_tokens
from adabmDCA.fasta import decode_sequence, validate_alphabet
from adabmDCA.serialization import _atomic_write

_GZIP_MAGIC = b"\x1f\x8b"
# Checkpoints favor the fastest gzip level.
_PARAMS_COMPRESSLEVEL = 1
_CHUNK_SIZE = 65_536


def _text_tokens(alphabet: str) -> str:
    tokens = get_tokens(alphabet)
    if not tokens or len(set(tokens)) != len(tokens) or any(token.isspace() for token in tokens):
        raise ValueError("The alphabet must contain distinct, non-whitespace tokens.")
    return tokens


def save_chains(
    fname: str | Path,
    chains: str | list | np.ndarray | torch.Tensor,
    tokens: str,
) -> None:
    """Save aligned strings, categorical sequences or one-hot batches to FASTA."""
    tokens = _text_tokens(tokens)
    if isinstance(chains, str):
        sequences = (chains,)
        validate_alphabet(sequences, tokens)
    else:
        encoded = chains if isinstance(chains, torch.Tensor) else np.asarray(chains)
        if isinstance(encoded, np.ndarray) and encoded.ndim == 1 and encoded.dtype.kind == "U":
            sequences = tuple(encoded.tolist())
            validate_alphabet(sequences, tokens)
        else:
            decoded = decode_sequence(encoded, tokens)
            sequences = (decoded,) if isinstance(decoded, str) else tuple(decoded.tolist())

    alignment = Alignment(
        names=tuple(f"chain_{i}" for i in range(len(sequences))),
        sequences=sequences,
    )
    _atomic_write(fname, lambda temporary: write_alignment(alignment, temporary))


def is_gzip(fname: Union[str, Path]) -> bool:
    """Whether a file starts with the gzip signature."""
    with open(fname, "rb") as handle:
        return handle.read(len(_GZIP_MAGIC)) == _GZIP_MAGIC


class _GzipInput(gzip.GzipFile):
    """Close the underlying stream supplied when detecting gzip."""

    def close(self) -> None:
        raw = self.fileobj
        try:
            super().close()
        finally:
            if raw is not None:
                raw.close()


def _open_params_output(fname: str | Path, *, compressed: bool) -> IO[str]:
    if compressed:
        return gzip.open(fname, "wt", encoding="utf-8", compresslevel=_PARAMS_COMPRESSLEVEL)
    return open(fname, "w", encoding="utf-8", buffering=1024 * 1024)


def open_params(fname: Union[str, Path], mode: str = "r") -> IO[str]:
    """Open a text parameter file in r/rt or w/wt mode.

    Reads detect gzip by content. Writes use gzip for paths ending in .gz.
    """
    if mode in ("w", "wt"):
        return _open_params_output(fname, compressed=Path(fname).suffix == ".gz")
    if mode not in ("r", "rt"):
        raise ValueError("Parameter file mode must be 'r', 'rt', 'w', or 'wt'.")

    raw = open(fname, "rb")
    try:
        if raw.peek(len(_GZIP_MAGIC)).startswith(_GZIP_MAGIC):
            return TextIOWrapper(_GzipInput(fileobj=raw, mode="rb"), encoding="utf-8")
        return TextIOWrapper(raw, encoding="utf-8")
    except BaseException:
        raw.close()
        raise


def _parameter_records(handle: IO[str], tokens: str | None):
    """Parse record coordinates without converting parameter values."""
    state = int if tokens is None else {token: idx for idx, token in enumerate(tokens)}.__getitem__
    for line_number, raw_line in enumerate(handle, 1):
        parts = raw_line.split()
        if not parts or parts[0].startswith("#"):
            continue
        kind = parts[0]
        if kind not in ("J", "h"):
            raise ValueError(f"Unknown parameter record {kind!r} on line {line_number}.")
        if len(parts) != (6 if kind == "J" else 4):
            label = "coupling" if kind == "J" else "bias"
            raise ValueError(f"Malformed {label} record on line {line_number}.")
        try:
            i = int(parts[1])
            if kind == "J":
                j, a, b = int(parts[2]), state(parts[3]), state(parts[4])
                if min(i, a, j, b) < 0:
                    raise ValueError
            else:
                a = state(parts[2])
                if min(i, a) < 0:
                    raise ValueError
                j = b = None
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid position or state on parameter line {line_number}.") from exc
        yield line_number, i, a, j, b, parts[-1]


def _parameter_dimensions(handle: IO[str], tokens: str | None) -> tuple[int, int]:
    max_bias_position = max_coupling_position = max_bias_state = -1
    for line_number, raw_line in enumerate(handle, 1):
        parts = raw_line.split(maxsplit=3)
        if not parts or parts[0].startswith("#"):
            continue
        try:
            if parts[0] == "J":
                max_coupling_position = max(max_coupling_position, int(parts[1]), int(parts[2]))
            elif parts[0] == "h":
                max_bias_position = max(max_bias_position, int(parts[1]))
                if tokens is None:
                    max_bias_state = max(max_bias_state, int(parts[2]))
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid position or state on parameter line {line_number}.") from exc
    if max_bias_position < 0:
        raise ValueError("The parameter file contains no bias records.")
    if tokens is None:
        if max_coupling_position > max_bias_position:
            raise ValueError("Numeric coupling positions exceed the length declared by the bias records.")
        return max_bias_position + 1, max_bias_state + 1
    return max(max_bias_position, max_coupling_position) + 1, len(tokens)


def _symmetrize_couplings(couplings: np.ndarray) -> None:
    """Combine both stored triangles using at most one site's row of temporary values."""
    for row in couplings:
        row[np.isnan(row)] = 0
    for i in range(len(couplings)):
        diagonal = couplings[i, :, i, :]
        diagonal[:] = diagonal + diagonal.T
        upper = couplings[i, :, i + 1:, :]
        lower = couplings[i + 1:, :, i, :]
        block = upper + lower.transpose(2, 0, 1)
        upper[:] = block
        lower[:] = block.transpose(1, 2, 0)


def _load_text_params(fname, tokens, device, dtype) -> Dict[str, torch.Tensor]:
    numpy_dtype = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
    }.get(dtype, np.float32)

    with open_params(fname) as handle:
        length, states = _parameter_dimensions(handle, tokens)

        # NaN marks a missing entry so duplicate records can be detected without a second dense tensor.
        bias = np.full((length, states), np.nan, dtype=numpy_dtype)
        coupling = np.full((length, states, length, states), np.nan, dtype=numpy_dtype)
        flattened = coupling.reshape(-1)
        indices: list[int] = []
        values: list[float] = []
        lines: list[int] = []

        def flush_couplings() -> None:
            if not indices:
                return
            flat_indices = np.asarray(indices, dtype=np.intp)
            already_seen = np.flatnonzero(~np.isnan(flattened[flat_indices]))
            if already_seen.size:
                raise ValueError(f"Duplicate coupling record on line {lines[already_seen[0]]}.")
            unique, first = np.unique(flat_indices, return_index=True)
            if len(unique) != len(flat_indices):
                duplicate = np.ones(len(flat_indices), dtype=bool)
                duplicate[first] = False
                raise ValueError(f"Duplicate coupling record on line {lines[np.flatnonzero(duplicate)[0]]}.")
            flattened[flat_indices] = np.asarray(values, dtype=numpy_dtype)
            indices.clear()
            values.clear()
            lines.clear()

        # Keep both passes on the same stream even if a new checkpoint replaces the path.
        handle.seek(0)
        for line_number, i, a, j, b, raw_value in _parameter_records(handle, tokens):
            try:
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError
            except ValueError as exc:
                label = "bias" if j is None else "coupling"
                raise ValueError(f"Invalid {label} value on line {line_number}.") from exc
            if j is None:
                if not np.isnan(bias[i, a]):
                    raise ValueError(f"Duplicate bias record on line {line_number}.")
                bias[i, a] = value
            else:
                if tokens is None and (a >= states or b >= states):
                    raise ValueError(f"Numeric coupling state exceeds the bias state count on line {line_number}.")
                indices.append(((i * states + a) * length + j) * states + b)
                values.append(value)
                lines.append(line_number)
                if len(indices) >= _CHUNK_SIZE:
                    flush_couplings()
        flush_couplings()

    bias[np.isnan(bias)] = 0
    _symmetrize_couplings(coupling)
    params = {
        "bias": torch.from_numpy(bias).to(device=device, dtype=dtype),
        "coupling_matrix": torch.from_numpy(coupling).to(device=device, dtype=dtype),
    }
    validate_finite_parameters(params)
    return params


def load_params(
    fname: str | Path,
    tokens: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Read letter-based J/h records, including gzip files regardless of their name.

    Missing entries are zero. Opposite coupling orientations add for legacy
    compatibility, while repeated records for the same oriented entry are invalid.
    """
    return _load_text_params(fname, _text_tokens(tokens), device, dtype)


def load_params_old(
    fname: str | Path,
    tokens: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Deprecated alias for loading letter-based parameters."""
    warnings.warn("load_params_old is deprecated. Please use load_params instead.", DeprecationWarning, stacklevel=2)
    return load_params(fname, tokens, device, dtype)


def load_params_oldformat(
    fname: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Load legacy J/h records that identify states by their integer indices."""
    return _load_text_params(fname, None, device, dtype)


def _parameter_array(tensor: torch.Tensor) -> np.ndarray:
    if tensor.is_complex():
        raise ValueError("Text model parameters must contain real values.")
    cpu = tensor.detach().cpu()
    if not cpu.is_floating_point():
        return cpu.double().numpy()
    try:
        return cpu.numpy()
    except TypeError:
        # NumPy does not expose all Torch dtypes, including bfloat16.
        return cpu.float().numpy()


def _save_text_params(fname, params, tokens, mask) -> None:
    if "bias" not in params or "coupling_matrix" not in params:
        raise ValueError("params must contain 'bias' and 'coupling_matrix'.")
    bias_tensor, coupling_tensor = params["bias"], params["coupling_matrix"]
    if bias_tensor.ndim != 2 or min(bias_tensor.shape) == 0:
        raise ValueError("params['bias'] must have a nonempty shape (L, q).")
    length, states = bias_tensor.shape
    expected = (length, states, length, states)
    if tokens is not None and len(tokens) != states:
        raise ValueError(f"The alphabet has {len(tokens)} states but the parameters have {states} states.")
    if coupling_tensor.shape != expected:
        raise ValueError("params['coupling_matrix'] must have shape (L, q, L, q).")
    if mask is not None and mask.shape != expected:
        raise ValueError("mask must have shape (L, q, L, q).")

    bias, coupling = _parameter_array(bias_tensor), _parameter_array(coupling_tensor)
    validate_finite_parameters({"bias": torch.from_numpy(bias), "coupling_matrix": torch.from_numpy(coupling)})
    mask_array = None if mask is None else mask.detach().to(device="cpu", dtype=torch.bool).numpy()
    sites = np.arange(length)
    diagonal = coupling[sites, :, sites, :].diagonal(axis1=1, axis2=2)
    if mask_array is not None:
        diagonal = np.where(mask_array[sites, :, sites, :].diagonal(axis1=1, axis2=2), diagonal, 0)
    if np.any(diagonal):
        with np.errstate(over="ignore"):
            bias = bias + 0.5 * diagonal
        if not np.isfinite(bias).all():
            raise ValueError("Bias values must remain finite after absorbing same-site couplings.")

    labels = tuple(map(str, range(states))) if tokens is None else tuple(tokens)
    state0_all = np.repeat(np.arange(states), states)
    state1_all = np.tile(np.arange(states), states)
    compressed = Path(fname).suffix == ".gz"

    def write(temporary: Path) -> None:
        with _open_params_output(temporary, compressed=compressed) as handle:
            buffer: list[str] = []
            for i in range(length - 1):
                upper = coupling[i, :, i + 1:, :]
                lower = coupling[i + 1:, :, i, :].transpose(2, 0, 1)
                effective = np.where(upper == lower, upper, 0.5 * upper + 0.5 * lower)
                for j in range(i + 1, length):
                    if mask_array is None:
                        state0, state1 = state0_all, state1_all
                    else:
                        selected = mask_array[i, :, j, :] | mask_array[j, :, i, :].T
                        state0, state1 = np.nonzero(selected)
                    values = effective[state0, j - i - 1, state1]
                    buffer.extend(f"J {i} {j} {labels[a]} {labels[b]} {value!s}\n"
                                  for a, b, value in zip(state0, state1, values))
                    if len(buffer) >= _CHUNK_SIZE:
                        handle.write("".join(buffer))
                        buffer.clear()
            for i in range(length):
                buffer.extend(f"h {i} {label} {value!s}\n" for label, value in zip(labels, bias[i]))
                if len(buffer) >= _CHUNK_SIZE:
                    handle.write("".join(buffer))
                    buffer.clear()
            if buffer:
                handle.write("".join(buffer))

    _atomic_write(fname, write)


def save_params(
    fname: str | Path,
    params: Dict[str, torch.Tensor],
    tokens: str,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """Atomically save letter-based J/h records, using gzip for .gz paths.

    Off-site pairs use the effective symmetric couplings and are written once.
    Selected same-site diagonal terms are absorbed into the fields. A mask
    selects an off-site interaction if either orientation is set.
    """
    _save_text_params(fname, params, _text_tokens(tokens), mask)


def save_params_oldformat(
    fname: str | Path,
    params: Dict[str, torch.Tensor],
    mask: Optional[torch.Tensor] = None,
) -> None:
    """Atomically save legacy J/h records with integer state indices."""
    _save_text_params(fname, params, None, mask)
