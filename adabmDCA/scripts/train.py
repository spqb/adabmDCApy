import argparse
import math

from adabmDCA.parser import add_args_train


def create_parser():
    parser = argparse.ArgumentParser(description="Train a DCA model.")
    return add_args_train(parser)


class _TrainingProgressRenderer:
    """Render structured training events for an interactive terminal."""

    def __init__(self, *, target_pearson: float, max_epochs: int) -> None:
        from tqdm import tqdm

        self._target = target_pearson
        self._max_epochs = max_epochs
        self._bar = tqdm(
            total=target_pearson,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
            bar_format="  {desc}: [{bar}] Pearson {n:.4f}/{total:.4f} [{elapsed}]",
        )

    def __call__(self, event) -> None:
        pearson = event.metrics.get("Pearson", 0.0)
        if math.isfinite(pearson):
            self._bar.n = min(max(0.0, pearson), self._target)

        description = f"Epoch {event.epoch}/{self._max_epochs}"
        likelihood = event.metrics.get("LL_train")
        if likelihood is not None and math.isfinite(likelihood):
            description += f" | LL/L {likelihood:.3f}"
        density = event.metrics.get("Density")
        if density is not None and math.isfinite(density):
            description += f" | density {density:.4f}"
        self._bar.set_description(description)
        self._bar.refresh()

    def close(self) -> None:
        self._bar.close()


def main():
    args = create_parser().parse_args()

    from adabmDCA.api.training import train_model
    from adabmDCA.utils import get_device, get_dtype

    print("\n" + "=" * 80)
    print(f"  TRAINING {args.model.upper()} MODEL")
    print("=" * 80 + "\n")

    device = get_device(args.device)
    get_dtype(args.dtype)

    print("[CONFIGURATION]")
    print("-" * 80)
    template = "  {0:<28} {1:<50}"
    print(template.format("Input MSA:", str(args.data)))
    if args.val is not None:
        print(template.format("Validation MSA:", str(args.val)))
    print(template.format("Output folder:", str(args.output)))
    print(template.format("Model type:", args.model))
    print(template.format("Alphabet:", args.alphabet))
    print(template.format("Learning rate:", args.lr))
    print(template.format("Number of sweeps:", args.nsweeps))
    print(template.format("Sampler:", args.sampler))
    print(template.format("Target Pearson Cij:", args.target))
    if args.pseudocount is not None:
        print(template.format("Pseudocount:", args.pseudocount))
    if args.l2_reg > 0.0 and args.model in {"bmDCA", "edDCA", "eaDCA"}:
        print(template.format("L2 regularization:", args.l2_reg))
    print(template.format("Random seed:", args.seed))
    print(template.format("Device:", str(device)))
    print(template.format("Data type:", args.dtype))
    print("-" * 80 + "\n")

    progress = None
    if not args.no_progress:
        progress = _TrainingProgressRenderer(
            target_pearson=args.target,
            max_epochs=args.nepochs,
        )
    try:
        result = train_model(
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
            pseudocount=args.pseudocount,
            l2_regularization=args.l2_reg,
            seed=args.seed,
            clustering_seqid=args.clustering_seqid,
            no_reweighting=args.no_reweighting,
            activation_steps=args.gsteps,
            activation_fraction=args.factivate,
            target_density=args.density,
            decimation_rate=args.drate,
            device=str(device),
            dtype=args.dtype,
            use_wandb=args.wandb,
            progress=progress,
        )
    finally:
        if progress is not None:
            progress.close()

    history = result.history
    print("\n" + "=" * 80)
    print("  TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\n" + "-" * 80)
    print(f"  Sequence length: {result.model.metadata.length}")
    print(f"  Number of sequences: {result.num_sequences}")
    print(f"  Effective sequences: {result.effective_sequences}")
    print(f"  Pseudocount: {result.pseudocount:.6f}")
    print(f"  Final graph density: {history['Density'][-1]:.4f}")
    print(f"  Final Pearson: {history['Pearson'][-1]:.4f}")
    print(f"  Final log-likelihood per residue: {history['LL_train'][-1]:.3f}")
    print(f"  Total steps: {history['Epochs'][-1]}")
    print(f"\n  Results saved in: {args.output}")
    for name, path in result.artifacts.items():
        print(f"    ✓ {name.capitalize()}: {path}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
