"""Command-line adapter for experimentally informed DCA training."""

from __future__ import annotations

import argparse

from adabmDCA.parser import add_args_reintegration, add_args_train


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a DCA model with experimental reintegration, using PCD (Persistent Contrastive Divergence, "
            "default) or PTT (Parallel Trajectory Tempering, --strategy ptt) for bmDCA, eaDCA and edgeDCA."
        ),
        allow_abbrev=False,
    )
    parser = add_args_reintegration(add_args_train(parser))
    parser.set_defaults(strategy="pcd")
    for action in parser._actions:
        if action.dest == "strategy":
            action.help = "Training strategy (default: pcd)."
    return parser


def _config_from_args(args):
    from adabmDCA.scripts.train import ptt_config_from_args
    from adabmDCA.training_config import TrainingConfig

    return TrainingConfig(
        model_type=args.model,
        ptt=ptt_config_from_args(args),
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
    )


def run(args, *, progress=None, stage_progress=None, on_initialized=None):
    """Execute reintegration from parsed CLI arguments and return its result."""
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)
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
        ptt_resume=args.ptt_resume,
        progress=progress,
        stage_progress=stage_progress,
        on_initialized=on_initialized,
    )


def main(args=None) -> int:
    args = create_parser().parse_args() if args is None else args
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)

    from adabmDCA.scripts._frontend import print_completion, print_configuration, print_header
    from adabmDCA.scripts.train import initialization_callback, make_renderer, stage_callback

    print_header("Reintegrated DCA model training")
    print_configuration(
        {
            "natural alignment": args.data,
            "experimental alignment": args.reint,
            "adjustments": args.adj,
            "lambda": args.lambda_,
            "strategy": args.strategy,
        }
    )
    renderer = make_renderer(args)
    try:
        result = run(args, progress=renderer, stage_progress=stage_callback(renderer),
                     on_initialized=initialization_callback(args, renderer))
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
