"""Command-line adapter for DCA model training."""

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING, Any

from adabmDCA.parser import add_args_train

if TYPE_CHECKING:
    from adabmDCA.api.results import TrainingProgress


def create_parser() -> argparse.ArgumentParser:
    return add_args_train(argparse.ArgumentParser(description="Train a DCA model."))


class _TrainingProgressRenderer:
    """Render structured training events for an interactive terminal."""

    def __init__(self, *, model_type: str, target_pearson: float, max_steps: int) -> None:
        from tqdm import tqdm

        self._tracks_gradients = model_type == "bmDCA"
        self._target_pearson = target_pearson
        self._max_steps = max_steps
        self._bar = tqdm(
            total=target_pearson,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
            bar_format="  {desc} [{bar}] Pearson {n:.4f}/{total:.4f} [{elapsed}]",
        )

    def __call__(self, event: TrainingProgress) -> None:
        completed = event.gradient_steps if self._tracks_gradients else event.structure_steps
        pearson = event.metrics.get("Pearson", 0.0)
        stage = event.stage.replace("_", " ").title()
        description = f"{stage} | Step {completed}/{self._max_steps}"
        if math.isfinite(pearson):
            self._bar.n = min(max(0.0, pearson), self._target_pearson)
        likelihood = event.metrics.get("LL_train")
        if likelihood is not None and math.isfinite(likelihood):
            description += f" | LL/L {likelihood:.3f}"
        density = event.metrics.get("Density")
        if density is not None and math.isfinite(density):
            description += f" | density {density:.2f}"
        self._bar.set_description(description)
        self._bar.refresh()

    def close(self) -> None:
        self._bar.close()


def run(args: argparse.Namespace, *, progress: Any = None):
    """Execute training from parsed CLI arguments and return its result."""
    from adabmDCA.api.training import train_model

    return train_model(
        args.data,
        model_type=args.model,
        validation_path=args.val,
        weights_path=args.weights,
        output_dir=args.output,
        label=args.label,
        initial_params_path=args.path_params,
        initial_chains_path=args.path_chains,
        alphabet=args.alphabet,
        learning_rate=args.lr,
        n_sweeps=args.nsweeps,
        sampler=args.sampler,
        n_chains=args.nchains,
        target_pearson=args.target,
        max_epochs=args.nepochs,
        max_gradient_steps=args.max_gradient_steps,
        max_structure_steps=args.max_structure_steps,
        checkpoint_interval=args.checkpoint_interval,
        pseudocount=args.pseudocount,
        l2_regularization=args.l2_reg,
        seed=args.seed,
        clustering_seqid=args.clustering_seqid,
        no_reweighting=args.no_reweighting,
        activation_steps=args.gsteps,
        activation_fraction=args.factivate,
        target_density=args.density,
        decimation_rate=args.drate,
        device=args.device,
        dtype=args.dtype,
        use_wandb=args.wandb,
        progress=progress,
    )


def main(args: argparse.Namespace | None = None) -> int:
    args = create_parser().parse_args() if args is None else args

    from adabmDCA.scripts._frontend import print_completion, print_configuration, print_header

    print_header(f"Training {args.model} model")
    print_configuration(
        {
            "input": args.data,
            "validation": args.val,
            "output": args.output,
            "model": args.model,
            "sampler": args.sampler,
            "alphabet": args.alphabet,
            "max epochs": args.nepochs,
            "checkpoint interval": args.checkpoint_interval,
            "target Pearson": args.target,
            "number of chains": args.nchains,
            "device": args.device,
            "dtype": args.dtype,
        }
    )
    renderer = (
        None
        if args.no_progress
        else _TrainingProgressRenderer(
            model_type=args.model,
            target_pearson=args.target,
            max_steps=(
                args.max_gradient_steps or args.nepochs
                if args.model == "bmDCA"
                else args.max_structure_steps or args.nepochs
            ),
        )
    )
    try:
        result = run(args, progress=renderer)
    finally:
        if renderer is not None:
            renderer.close()

    serialized = result.save_bundle(args.output, label=args.label)
    artifacts = {**result.artifacts, **serialized}
    print_completion(
        "Training completed successfully.",
        metrics={
            "sequence length": result.model.metadata.length,
            "sequences": result.num_sequences,
            "effective sequences": result.effective_sequences,
            "stop reason": result.stop_reason,
            "gradient steps": result.gradient_steps,
            "structure steps": result.structure_steps,
            "sweeps": result.sweeps,
            **result.final_metrics,
        },
        artifacts=artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
