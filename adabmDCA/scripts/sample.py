"""Command-line adapter for model-based sequence generation."""

from __future__ import annotations

import argparse

from adabmDCA.parser import add_args_sample


def create_parser() -> argparse.ArgumentParser:
    return add_args_sample(argparse.ArgumentParser(description="Sample sequences from a DCA model."))


class _SamplingProgressRenderer:
    def __init__(self) -> None:
        self._bar = None

    def __call__(self, event) -> None:
        if self._bar is None:
            from tqdm import tqdm

            self._bar = tqdm(
                total=event.total,
                colour="red",
                dynamic_ncols=True,
                leave=False,
                ascii="-#",
                bar_format="  {desc}: [{bar}] {n}/{total} sweeps [{elapsed}]",
            )
            self._bar.set_description("Generating sequences")
        self._bar.update(event.completed - self._bar.n)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def run(args, *, progress=None):
    """Execute sampling from parsed CLI arguments and return its result."""
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)
    from adabmDCA.api.sampling import sample_sequences

    return sample_sequences(
        model=args.path_params,
        n_sequences=args.ngen,
        n_sweeps=args.max_nsweeps,
        sampler=args.sampler,
        beta=args.beta,
        seed=args.seed,
        reference_fasta=args.data,
        weights_path=args.weights,
        n_measure=args.nmeasure,
        mixing_multiplier=args.nmix,
        pseudocount=args.pseudocount,
        clustering_seqid=args.clustering_seqid,
        no_reweighting=args.no_reweighting,
        alphabet=args.alphabet,
        device=args.device,
        dtype=args.dtype,
        collect_diagnostics=args.plot,
        progress=progress,
    )


def main(args=None) -> int:
    args = create_parser().parse_args() if args is None else args
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)

    from adabmDCA.scripts._frontend import print_completion, print_configuration, print_header

    print_header("Sampling from a DCA model")
    print_configuration(
        {
            "model": args.path_params,
            "reference": args.data,
            "output": args.output,
            "label": args.label,
            "sequences": args.ngen,
            "sampler": args.sampler,
            "alphabet": args.alphabet,
            "beta": args.beta,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "plots": args.plot,
        }
    )
    renderer = _SamplingProgressRenderer()
    try:
        result = run(args, progress=renderer)
    finally:
        renderer.close()
    artifacts = result.save_bundle(args.output, label=args.label)
    if args.plot:
        artifacts.update(result.save_diagnostic_plots(args.output, label=args.label))
    print_completion(
        "Sampling completed successfully.",
        metrics={
            "sequences": len(result.sequences),
            "sweeps": result.num_sweeps,
            "mean energy": f"{result.energies.mean():.3f}",
            "standard deviation": f"{result.energies.std():.3f}",
        },
        artifacts=artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
