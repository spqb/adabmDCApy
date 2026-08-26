"""High-level DCA training workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adabmDCA.api.exceptions import InputValidationError, OperationCancelledError
from adabmDCA.api.model import DCAModel
from adabmDCA.api.results import TrainingProgress, TrainingResult
from adabmDCA.api.runtime import resolve_runtime
from adabmDCA.checkpoint import Checkpoint
from adabmDCA.dataset import DatasetDCA
from adabmDCA.fasta import get_tokens
from adabmDCA.io import load_chains, load_params
from adabmDCA.sampling import get_sampler
from adabmDCA.training import train_eaDCA, train_edDCA, train_edgeDCA, train_graph
from adabmDCA.utils import init_chains, init_parameters


ProgressCallback = Callable[[TrainingProgress], None]
CancellationHook = Callable[[], bool]


class _TrainingObserver:
    """Adapt checkpoints, callbacks, and cancellation to the training API."""

    def __init__(
        self,
        checkpoint: Checkpoint | None,
        progress: ProgressCallback | None,
        is_cancelled: CancellationHook | None,
        max_epochs: int,
    ) -> None:
        self._checkpoint = checkpoint
        self._progress = progress
        self._is_cancelled = is_cancelled
        self.max_epochs = max_epochs
        self.checkpt_interval = 50
        self.file_paths = (
            checkpoint.file_paths
            if checkpoint is not None
            else {"log": "training.log", "params": "params.dat", "chains": "chains.fasta"}
        )

    def log(self, record: dict[str, Any]) -> None:
        if self._is_cancelled is not None and self._is_cancelled():
            raise OperationCancelledError("Model training was cancelled by the caller.")
        if self._checkpoint is not None:
            self._checkpoint.log(record)
        if self._progress is not None:
            metrics = {
                key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
                for key, value in record.items()
                if isinstance(value, (int, float, torch.Tensor))
            }
            self._progress(TrainingProgress(epoch=int(record.get("Epochs", 0)), metrics=metrics))

    def check(self, updates: int) -> bool:
        return updates % self.checkpt_interval == 0 or updates == self.max_epochs

    def save(self, **kwargs) -> None:
        if self._checkpoint is not None:
            self._checkpoint.file_paths = self.file_paths
            self._checkpoint.save(**kwargs)


def train_model(
    data_path: str | Path,
    *,
    model_type: str = "bmDCA",
    validation_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    label: str | None = None,
    initial_params_path: str | Path | None = None,
    initial_chains_path: str | Path | None = None,
    alphabet: str = "protein",
    learning_rate: float = 0.01,
    n_sweeps: int = 10,
    sampler: str = "gibbs",
    n_chains: int = 10_000,
    target_pearson: float = 0.95,
    max_epochs: int = 50_000,
    pseudocount: float | None = None,
    l2_regularization: float = 0.0,
    seed: int = 0,
    clustering_seqid: float = 0.8,
    no_reweighting: bool = False,
    activation_steps: int = 10,
    activation_fraction: float = 0.001,
    target_density: float = 0.02,
    decimation_rate: float = 0.01,
    device: str = "auto",
    dtype: str = "float32",
    use_wandb: bool = False,
    progress: ProgressCallback | None = None,
    is_cancelled: CancellationHook | None = None,
) -> TrainingResult:
    """Train a DCA model from a FASTA alignment.

    Persistence is optional. Set ``output_dir`` to retain the historical
    parameter, chain, weight, and log artifacts; omit it for an in-memory
    notebook workflow.
    """
    if model_type not in {"bmDCA", "eaDCA", "edDCA", "edgeDCA"}:
        raise InputValidationError(f"Unsupported model_type '{model_type}'.")
    if sampler not in {"gibbs", "metropolis"}:
        raise InputValidationError("sampler must be either 'gibbs' or 'metropolis'.")
    if n_chains < 1 or n_sweeps < 1 or max_epochs < 1:
        raise InputValidationError("n_chains, n_sweeps, and max_epochs must be positive.")
    if not 0.0 <= target_pearson <= 1.0:
        raise InputValidationError("target_pearson must be between 0 and 1.")
    if is_cancelled is not None and is_cancelled():
        raise OperationCancelledError("Model training was cancelled by the caller.")

    data_path = Path(data_path)
    if not data_path.is_file():
        raise InputValidationError(f"Training FASTA file '{data_path}' was not found.")
    if validation_path is not None and not Path(validation_path).is_file():
        raise InputValidationError(f"Validation FASTA file '{validation_path}' was not found.")
    if weights_path is not None and not Path(weights_path).is_file():
        raise InputValidationError(f"Weights file '{weights_path}' was not found.")
    if initial_params_path is not None and not Path(initial_params_path).is_file():
        raise InputValidationError(
            f"Initial parameter file '{initial_params_path}' was not found."
        )
    if initial_chains_path is not None and not Path(initial_chains_path).is_file():
        raise InputValidationError(
            f"Initial chain file '{initial_chains_path}' was not found."
        )

    resolved_device, resolved_dtype = resolve_runtime(device, dtype)
    tokens = get_tokens(alphabet)
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    dataset = DatasetDCA(
        path_data=str(data_path),
        path_weights=None if weights_path is None else str(weights_path),
        alphabet=alphabet,
        clustering_th=clustering_seqid,
        no_reweighting=no_reweighting,
        device=resolved_device,
        dtype=resolved_dtype,
        message=False,
        filter_sequences=True,
        remove_duplicates=True,
    )
    if validation_path is not None:
        validation = DatasetDCA(
            path_data=str(validation_path),
            path_weights=None,
            alphabet=alphabet,
            clustering_th=clustering_seqid,
            no_reweighting=no_reweighting,
            device=resolved_device,
            dtype=resolved_dtype,
            message=False,
            filter_sequences=True,
            remove_duplicates=True,
        )
        validation_pseudocount = 1.0 / validation.get_effective_size()
        fi_val, fij_val = validation.get_frequencies(
            pseudocount=1e-6 if model_type == "edgeDCA" else validation_pseudocount
        )
    else:
        fi_val = fij_val = None

    effective_size = float(dataset.get_effective_size())
    effective_pseudocount = pseudocount
    if effective_pseudocount is None:
        effective_pseudocount = 0.1 if model_type == "edgeDCA" else 1.0 / effective_size

    dataset.shuffle()
    length = dataset.get_num_residues()
    num_states = dataset.get_num_states()
    if model_type == "edgeDCA":
        fi_target, fij_target = dataset.get_frequencies(pseudocount=1e-6)
        fi_pseudocounted, fij_pseudocounted = dataset.get_frequencies(
            pseudocount=effective_pseudocount
        )
    else:
        fi_target, fij_target = dataset.get_frequencies(pseudocount=effective_pseudocount)
        fi_pseudocounted, fij_pseudocounted = fi_target, fij_target

    if initial_params_path is not None:
        params = load_params(
            str(initial_params_path), tokens=tokens, device=resolved_device, dtype=resolved_dtype
        )
        mask = ~torch.isclose(params["coupling_matrix"], torch.zeros_like(params["coupling_matrix"]))
    else:
        params = init_parameters(fi=fi_target)
        if model_type in {"bmDCA", "edDCA"}:
            mask = torch.ones(
                (length, num_states, length, num_states),
                dtype=torch.bool,
                device=resolved_device,
            )
            mask[torch.arange(length), :, torch.arange(length), :] = 0
        else:
            mask = torch.zeros(
                (length, num_states, length, num_states),
                dtype=torch.bool,
                device=resolved_device,
            )

    if initial_chains_path is not None:
        chains, log_weights = load_chains(
            str(initial_chains_path),
            tokens=dataset.tokens,
            load_weights=True,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        n_chains = chains.shape[0]
    else:
        chains = init_chains(
            num_chains=n_chains,
            L=length,
            q=num_states,
            fi=fi_target,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        log_weights = torch.zeros(n_chains, device=resolved_device, dtype=resolved_dtype)

    artifacts: dict[str, Path] = {}
    checkpoint = None
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        stem = label or "adabmDCA"
        file_paths = {
            "log": str(folder / f"{stem}.log"),
            "params": str(folder / (f"{label}_params.dat" if label else "params.dat")),
            "chains": str(folder / (f"{label}_chains.fasta" if label else "chains.fasta")),
        }
        config = {
            "label": label,
            "model": model_type,
            "data": str(data_path),
            "val": None if validation_path is None else str(validation_path),
            "alphabet": alphabet,
            "sampler": sampler,
            "nchains": n_chains,
            "nsweeps": n_sweeps,
            "lr": learning_rate,
            "pseudocount": effective_pseudocount,
            "dtype": dtype,
            "target": target_pearson,
            "gsteps": activation_steps,
            "factivate": activation_fraction,
            "seed": seed,
            "nepochs": max_epochs,
        }
        checkpoint = Checkpoint(file_paths, tokens, config, use_wandb=use_wandb)
        artifacts = {key: Path(value) for key, value in file_paths.items()}
        if weights_path is None:
            weights_output = folder / (f"{label}_weights.dat" if label else "weights.dat")
            np.savetxt(weights_output, dataset.weights.detach().cpu().numpy())
            artifacts["weights"] = weights_output

    observer = _TrainingObserver(checkpoint, progress, is_cancelled, max_epochs)
    sampling_function = torch.jit.script(get_sampler(sampler))

    if model_type == "bmDCA":
        chains, params, log_weights, history = train_graph(
            sampler=sampling_function,
            chains=chains,
            mask=mask,
            fi_target=fi_target,
            fij_target=fij_target,
            params=params,
            nsweeps=n_sweeps,
            lr=learning_rate,
            max_epochs=max_epochs,
            target_pearson=target_pearson,
            fi_val=fi_val,
            fij_val=fij_val,
            checkpoint=observer,
            log_weights=log_weights,
            l2_reg=l2_regularization,
            progress_bar=False,
        )
    elif model_type == "eaDCA":
        chains, params, log_weights, history = train_eaDCA(
            sampler=sampling_function,
            fi_target=fi_target,
            fij_target=fij_target,
            params=params,
            mask=mask,
            chains=chains,
            log_weights=log_weights,
            target_pearson=target_pearson,
            nsweeps=n_sweeps,
            max_epochs=max_epochs,
            pseudo_count=effective_pseudocount,
            lr=learning_rate,
            factivate=activation_fraction,
            gsteps=activation_steps,
            fi_val=fi_val,
            fij_val=fij_val,
            checkpoint=observer,
            l2_reg=l2_regularization,
            progress_bar=False,
        )
    elif model_type == "edDCA":
        chains, params, log_weights, history = train_edDCA(
            sampler=sampling_function,
            chains=chains,
            log_weights=log_weights,
            fi_target=fi_target,
            fij_target=fij_target,
            params=params,
            mask=mask,
            lr=learning_rate,
            nsweeps=n_sweeps,
            target_pearson=target_pearson,
            target_density=target_density,
            drate=decimation_rate,
            checkpoint=observer,
            fi_val=fi_val,
            fij_val=fij_val,
            l2_reg=l2_regularization,
            progress_bar=False,
        )
    else:
        chains, params, log_weights, history = train_edgeDCA(
            sampler=sampling_function,
            fi_target=fi_target,
            fij_target=fij_target,
            fi_pseudocounted=fi_pseudocounted,
            fij_pseudocounted=fij_pseudocounted,
            params=params,
            mask=mask,
            chains=chains,
            target_pearson=target_pearson,
            nsweeps=n_sweeps,
            max_epochs=max_epochs,
            pseudo_count=effective_pseudocount,
            fi_val=fi_val,
            fij_val=fij_val,
            checkpoint=observer,
            progress_bar=False,
        )

    trained = DCAModel(params, alphabet=alphabet, source=artifacts.get("params"))
    return TrainingResult(
        model=trained,
        history=history,
        chains=chains,
        log_weights=log_weights,
        pseudocount=float(effective_pseudocount),
        num_sequences=len(dataset),
        effective_sequences=effective_size,
        artifacts=artifacts,
    )
