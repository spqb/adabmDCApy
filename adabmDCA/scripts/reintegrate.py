"""Command-line adapter for experimentally informed DCA training."""

from __future__ import annotations

import argparse

from adabmDCA.parser import add_args_reintegration, add_args_train


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DCA model with experimental reintegration.")
    return add_args_reintegration(add_args_train(parser))


def _config_from_args(args):
    from adabmDCA.training_config import TrainingConfig

    return TrainingConfig(
        model_type=args.model,
        alphabet=args.alphabet,
        learning_rate=args.lr,
        n_sweeps=args.nsweeps,
        sampler=args.sampler,
        n_chains=args.nchains,
        target_pearson=args.target,
        max_epochs=args.nepochs,
        max_gradient_steps=args.max_gradient_steps,
        max_structure_steps=args.max_structure_steps,
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
    )


def run(args, *, progress=None):
    """Execute reintegration from parsed CLI arguments and return its result."""
    from adabmDCA.api.reintegration import reintegrate_model

    return reintegrate_model(
        args.data,
        args.reint,
        args.adj,
        config=_config_from_args(args),
        natural_weights=args.weights,
        lambda_value=args.lambda_,
        output_dir=args.output,
        label=args.label,
        initial_params_path=args.path_params,
        initial_chains_path=args.path_chains,
        progress=progress,
    )


def main(args=None) -> int:
    args = create_parser().parse_args() if args is None else args

    from adabmDCA.scripts._frontend import print_completion, print_configuration, print_header
    from adabmDCA.scripts.train import _TrainingProgressRenderer

    print_header("Reintegrated DCA model training")
    print_configuration(
        {
            "natural alignment": args.data,
            "experimental alignment": args.reint,
            "adjustments": args.adj,
            "lambda": args.lambda_,
            "output": args.output,
            "model": args.model,
            "alphabet": args.alphabet,
            "device": args.device,
        }
    )
    renderer = (
        None
        if args.no_progress
        else _TrainingProgressRenderer(
            target_pearson=args.target,
            max_epochs=args.nepochs,
        )
    )
    try:
        result = run(args, progress=renderer)
    finally:
        if renderer is not None:
            renderer.close()
    artifacts = {**result.training.artifacts, **result.artifacts}
    print_completion(
        "Reintegrated training completed successfully.",
        metrics={
            "lambda": f"{result.lambda_value:.6g}",
            "scaling factor": f"{result.scaling_factor:.6g}",
            "sequences": result.alignment.num_sequences,
            "stop reason": result.training.stop_reason,
        },
        artifacts=artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
