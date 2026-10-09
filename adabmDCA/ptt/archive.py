"""HDF5 archives of a PTT sampler: arrays and JSON only, never pickled objects."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, fields
from pathlib import Path

import torch

from adabmDCA._validation import validate_integer, validate_seed
from adabmDCA.exceptions import InputValidationError
from adabmDCA.ptt.config import PTTConfig
from adabmDCA.ptt.kernels import (
    _prepare_categorical_sampler,
    _prepare_exchange_kernel,
    _prepare_replica_sampler,
)

# Format of the archives written and read by this version. Archives of the
# unreleased development formats 1-7 are no longer read.
ARCHIVE_SCHEMA_VERSION = 8
_ENERGY_CONVENTION = "potts_minus_h_minus_half_J"
_ALGORITHM_KEYS = frozenset({
    "active_start", "anchor_log_z", "reservoir", "temporary", "held", "last_mixing", "mixing_correlation",
    "events", "recovery_points", "profile_params", "total_models", "optimizer_state", "timings",
    "last_advance_timing", "lag_memory",
})


def _check_schema(meta):
    if meta["schema_version"] != ARCHIVE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported PTT archive format {meta['schema_version']}; this version reads format "
            f"{ARCHIVE_SCHEMA_VERSION}. Formats 1-7 were unreleased development formats and are no longer read."
        )
    if meta["energy_convention"] != _ENERGY_CONVENTION:
        raise ValueError("Unsupported PTT energy convention.")


def model_id(params):
    digest = hashlib.sha256()
    for key in ("bias", "coupling_matrix"):
        value = params[key].detach().cpu().contiguous()
        digest.update(str((key, value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()



def _write_tree(group, value):
    """Store recovery state as typed HDF5 arrays and JSON, never pickle."""
    if isinstance(value, torch.Tensor):
        group.attrs["kind"] = "tensor"
        group.create_dataset("value", data=value.detach().cpu().numpy(), fletcher32=value.ndim > 0)
    elif isinstance(value, (dict, list, tuple)):
        group.attrs["kind"] = "dict" if isinstance(value, dict) else "list"
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in items:
            _write_tree(group.create_group(str(key)), item)
    else:
        group.attrs["kind"] = "scalar"
        group.attrs["value"] = json.dumps(value, allow_nan=False)


def _read_tree(group):
    kind = group.attrs["kind"]
    if kind == "tensor":
        return torch.from_numpy(group["value"][...])
    if kind == "dict":
        return {key: _read_tree(group[key]) for key in group}
    if kind == "list":
        return [_read_tree(group[str(k)]) for k in range(len(group))]
    if kind == "scalar":
        return json.loads(group.attrs["value"])
    raise ValueError("Invalid PTT state encoding.")


def _tree_to_device(value, device, key=None):
    if isinstance(value, torch.Tensor):
        return value.to("cpu" if key == "_rng" else device)
    if isinstance(value, dict):
        return {k: _tree_to_device(v, device, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_tree_to_device(v, device) for v in value]
    return value



def save_archive(sampler, path):
    """Publish a complete HDF5 generation by same-filesystem atomic replace."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as archive:
            metadata = {
                "schema_version": ARCHIVE_SCHEMA_VERSION,
                "mode": sampler.mode,
                "energy_convention": _ENERGY_CONVENTION,
                "torch_version": torch.__version__,
                "tokens": sampler.tokens,
                "local_sampler": sampler.local_sampler,
                "config": asdict(sampler.config),
                "seed": sampler.seed,
                "model_version": sampler.model_version,
                "ladder_version": sampler.ladder_version,
                "rounds": sampler.rounds,
                "local_sweeps": sampler.local_sweeps,
                "acceptance": sampler.acceptance,
                "training_state": sampler.training_state,
                "partition": asdict(sampler.partition_estimate()),
                "rng_device": str(sampler.device),
            }
            from adabmDCA.serialization import to_jsonable

            archive.attrs["metadata"] = json.dumps(to_jsonable(metadata), allow_nan=False)
            for k, (params, chains) in enumerate(zip(sampler.models, sampler.chains)):
                group = archive.create_group(f"replicas/{k}")
                group.attrs["model_id"] = model_id(params)
                for key, tensor in params.items():
                    group.create_dataset(key, data=tensor.detach().cpu().numpy())
                group.create_dataset("chains", data=chains.cpu().numpy())
            checkpoints = archive.create_group("ptt_checkpoints")
            checkpoints.attrs["complete"] = sampler.ptt_checkpoints_complete
            for point in sampler.ptt_checkpoints:
                group = checkpoints.create_group(str(point["step"]))
                group.attrs["flag"] = "ptt"
                group.attrs["model_id"] = model_id(point["params"])
                for key, tensor in point["params"].items():
                    group.create_dataset(key, data=tensor.detach().cpu().numpy())
            archive.create_dataset("lineage", data=sampler.lineage.cpu().numpy())
            archive.create_dataset("rng", data=sampler._rng.cpu().numpy())
            _write_tree(archive.create_group("algorithm"),
                        {key: getattr(sampler, key) for key in sorted(_ALGORITHM_KEYS)})
            archive.flush()
        # Validate before publication; a failed write preserves the old file.
        type(sampler).from_archive(temporary, device="cpu", mode="inspect")
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path



def load_archive(cls, path, *, device="cpu", mode="generate", seed=None):
    """Load validated arrays on CPU first; resume requires the saved backend.

    Generation never writes the source archive. Changing backend requires
    an explicit new seed and subsequent warmup, rather than exact resume.
    """
    import h5py

    if mode not in ("generate", "resume", "inspect"):
        raise InputValidationError("PTT archive mode must be generate, resume or inspect.")
    if mode == "resume" and seed is not None:
        raise InputValidationError("An exact PTT resume cannot change the RNG seed.")
    try:
        with h5py.File(path, "r") as archive:
            meta = json.loads(archive.attrs["metadata"])
            _check_schema(meta)
            # Settings added to PTTConfig after the archive was written take
            # their defaults; settings since removed are dropped.
            valid_config_fields = {field.name for field in fields(PTTConfig)}
            meta["config"] = {
                key: value for key, value in meta["config"].items()
                if key in valid_config_fields
            }
            resolved_config = asdict(PTTConfig(**meta["config"]))
            if meta["training_state"]:
                meta["training_state"]["settings"]["ptt"] = resolved_config
            meta["config"] = resolved_config
            models, populations = [], []
            for k in range(len(archive["replicas"])):
                group = archive[f"replicas/{k}"]
                params = {key: torch.from_numpy(group[key][...]) for key in ("bias", "coupling_matrix")}
                if group.attrs["model_id"] != model_id(params):
                    raise ValueError("PTT model hash mismatch.")
                models.append(params)
                categorical = torch.from_numpy(group["chains"][...])
                if (
                    categorical.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
                    or categorical.ndim != 2
                ):
                    raise ValueError("PTT chains must be a two-dimensional integer array.")
                categorical = categorical.long()
                if categorical.numel() == 0 or categorical.min() < 0 or categorical.max() >= len(meta["tokens"]):
                    raise ValueError("Invalid PTT chain categories.")
                populations.append(categorical.to(torch.int32))
            algorithm = _read_tree(archive["algorithm"])
            result = cls(
                algorithm["profile_params"],
                tokens=meta["tokens"],
                n_chains=len(populations[0]),
                sampler=meta["local_sampler"],
                config=PTTConfig(**meta["config"]),
                seed=meta["seed"],
            )
            if len(models) < 2:
                raise ValueError("Invalid PTT ladder size.")
            for params, chains in zip(models, populations):
                result._validate_params(params)
                if chains.shape != populations[0].shape or chains.shape[1] != models[0]["bias"].shape[0]:
                    raise ValueError("Invalid PTT chain dimensions.")
            result.models, result.chains = models, populations
            result.total_models = len(models)
            if set(algorithm) != _ALGORITHM_KEYS:
                raise ValueError("Invalid PTT algorithm state.")
            result.__dict__.update(algorithm)
            checkpoints = archive["ptt_checkpoints"]
            result.ptt_checkpoints_complete = bool(checkpoints.attrs["complete"])
            result.ptt_checkpoints = []
            for step in sorted(checkpoints, key=int):
                group = checkpoints[step]
                params = {key: torch.from_numpy(group[key][...]) for key in ("bias", "coupling_matrix")}
                result._validate_params(params)
                if group.attrs["flag"] != "ptt" or group.attrs["model_id"] != model_id(params):
                    raise ValueError("Invalid flagged PTT checkpoint or model hash.")
                if not 0 <= int(step) <= meta["model_version"]:
                    raise ValueError("Invalid flagged PTT checkpoint step.")
                result.ptt_checkpoints.append({"step": int(step), "params": params})
            expected_steps = [0] + [e["model_version"] for e in result.events if e["kind"] == "snapshot_inserted"]
            if result.ptt_checkpoints_complete and [p["step"] for p in result.ptt_checkpoints] != expected_steps:
                raise ValueError("Flagged PTT checkpoints do not match training events.")
            if (not result.ptt_checkpoints or result.ptt_checkpoints[0]["step"] != 0
                    or model_id(result.ptt_checkpoints[0]["params"]) != model_id(result.profile_params)):
                raise ValueError("Flagged PTT checkpoints lack the initial profile.")
            result.lineage = torch.from_numpy(archive["lineage"][...])
            if result.lineage.shape != (len(models), len(populations[0])):
                raise ValueError("Invalid PTT lineage dimensions.")
            for key in (
                "model_version",
                "ladder_version",
                "rounds",
                "local_sweeps",
                "acceptance",
                "training_state",
            ):
                setattr(result, key, meta[key])
            for key in ("model_version", "ladder_version", "rounds", "local_sweeps"):
                validate_integer(key, getattr(result, key), minimum=0)
            if len(result.acceptance) != result.n_active - 1 or not all(0 <= a <= 1 for a in result.acceptance):
                raise ValueError("Invalid PTT acceptance diagnostics.")
            if sorted(result.lineage.flatten().tolist()) != list(range(result.lineage.numel())):
                raise ValueError("Invalid PTT lineage identifiers.")
            result._validate_algorithm_state()
            estimate = result.partition_estimate()
            saved = meta["partition"]
            cross_device_float32 = (
                meta["rng_device"].startswith("cuda")
                and models[-1]["bias"].dtype == torch.float32
            )
            if cross_device_float32:
                reduction_scale = torch.finfo(torch.float32).eps * models[-1]["bias"].shape[0]
                log_z_atol = max(1e-5, 64 * reduction_scale)
            else:
                log_z_atol = 1e-5
            if (
                saved["method"] != "ptt_bridge"
                or saved["model_id"] != estimate.model_id
                or saved["model_version"] != estimate.model_version
                or saved["ladder_version"] != estimate.ladder_version
                or saved["sample_round"] != estimate.sample_round
                or not math.isclose(saved["log_z"], estimate.log_z, abs_tol=log_z_atol, rel_tol=1e-6)
            ):
                raise ValueError(
                    "PTT partition provenance mismatch "
                    f"(saved log_z={saved['log_z']:.10g}, recomputed={estimate.log_z:.10g})."
                )
            if result.training_state:
                state = result.training_state
                history = state["history"]
                steps = state["gradient_steps"]
                validate_integer("gradient_steps", steps, minimum=0)
                validate_integer("sweeps", state["sweeps"], minimum=0)
                if (
                    steps != result.model_version
                    or state["sweeps"] < result.local_sweeps
                    or not math.isfinite(state["learning_rate"])
                    or state["learning_rate"] <= 0
                    or not math.isfinite(state["elapsed"])
                    or state["elapsed"] < 0
                    or history["Epochs"] != list(range(steps + 1))
                    or any(len(values) != steps + 1 for values in history.values())
                    or history["model_version"][-1] != result.model_version
                    or history["logZ_method"][-1] != "ptt_bridge"
                    or not math.isclose(
                        history["logZ"][-1], estimate.log_z, rel_tol=1e-6, abs_tol=log_z_atol
                    )
                    or state["settings"]["ptt"] != asdict(result.config)
                    or state["settings"]["sampler"] != result.local_sampler
                    or state["settings"]["n_chains"] != len(result.chains[0])
                    or state["settings"]["dtype"] != str(result.models[0]["bias"].dtype).removeprefix("torch.")
                ):
                    raise ValueError("PTT training checkpoint does not match the sampler state.")
            target = torch.device(device)
            if target.type == "cuda" and target.index is None:
                target = torch.device("cuda", torch.cuda.current_device())
            if str(target) != meta["rng_device"] and mode != "inspect" and (seed is None or mode == "resume"):
                raise ValueError(
                    "Changing PTT backend requires a new seed and warmup; exact resume is unavailable."
                )
            result.mode = "generate" if mode == "generate" else meta["mode"]
            if result.mode not in ("train", "generate"):
                raise ValueError("Invalid PTT archive mode.")
            if mode == "generate":
                result.training_state = {}
            result.device = target
            result.models = [{k: v.to(target) for k, v in p.items()} for p in models]
            result.chains = [c.to(target) for c in populations]
            result.lineage = result.lineage.to(target)
            for key in ("reservoir", "temporary", "held", "recovery_points", "profile_params", "ptt_checkpoints",
                        "lag_memory"):
                setattr(result, key, _tree_to_device(getattr(result, key), target))
            result._rng = torch.from_numpy(archive["rng"][...])
            if mode != "inspect" and seed is None:
                torch.Generator(device=target).set_state(result._rng)
            if seed is not None:
                validate_seed(seed)
                result.seed = seed
                result._rng = torch.Generator(device=target).manual_seed(seed).get_state()
            result._local_kernel = _prepare_categorical_sampler(result.local_sampler, target)
            result._replica_kernel = _prepare_replica_sampler(result.local_sampler, target)
            result._replica_params_cache = None
            result._exchange_kernel = _prepare_exchange_kernel(target)
            return result
    except (OSError, KeyError, IndexError, ValueError, RuntimeError, TypeError) as exc:
        raise InputValidationError(f"Invalid PTT archive '{path}': {exc}") from exc


def load_ptt_endpoint(path, *, device="cpu", alphabet=None):
    """Read only the final parameter arrays, without loading chain populations."""
    import h5py

    from adabmDCA.fasta import get_tokens

    try:
        with h5py.File(path, "r") as archive:
            meta = json.loads(archive.attrs["metadata"])
            _check_schema(meta)
            tokens = meta["tokens"]
            if alphabet not in (None, "auto") and get_tokens(alphabet) != tokens:
                raise ValueError("Explicit alphabet conflicts with the PTT archive.")
            group = archive[f"replicas/{len(archive['replicas']) - 1}"]
            params = {key: torch.from_numpy(group[key][...]) for key in ("bias", "coupling_matrix")}
            if model_id(params) != group.attrs["model_id"] or model_id(params) != meta["partition"]["model_id"]:
                raise ValueError("PTT endpoint hash mismatch.")
            from adabmDCA.ptt.sampler import PTTSampler

            validator = PTTSampler.__new__(PTTSampler)
            validator.tokens = tokens
            validator._validate_params(params)
            return {key: value.to(device) for key, value in params.items()}, tokens
    except (OSError, KeyError, IndexError, ValueError, RuntimeError, TypeError) as exc:
        raise InputValidationError(f"Invalid PTT endpoint archive '{path}': {exc}") from exc
