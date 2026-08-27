"""Command-line adapter for DCA sequence scoring."""

from __future__ import annotations

import argparse

from adabmDCA.parser import add_args_energies


def create_parser() -> argparse.ArgumentParser:
    return add_args_energies(argparse.ArgumentParser(description="Compute DCA sequence energies."))


def run(args):
    """Execute scoring from parsed CLI arguments and return its result."""
    from adabmDCA.api.scoring import score_sequences

    return score_sequences(
        model=args.path_params,
        fasta_path=args.data,
        alphabet=args.alphabet,
        device=args.device,
        dtype=args.dtype,
        remove_duplicates=True,
    )


def main(args=None) -> int:
    args = create_parser().parse_args() if args is None else args

    from adabmDCA.scripts._frontend import input_stem, print_completion, print_configuration, print_header

    print_header("DCA energy computation")
    print_configuration(
        {
            "input": args.data,
            "model": args.path_params,
            "output": args.output,
            "alphabet": args.alphabet,
            "device": args.device,
            "dtype": args.dtype,
        }
    )
    result = run(args)
    artifacts = result.save_bundle(args.output, stem=f"{input_stem(args.data)}_energies")
    print_completion(
        "Energy computation completed successfully.",
        metrics={
            "sequences": len(result.sequences),
            "length": result.model.length,
            "mean energy": f"{result.energies.mean():.3f}",
            "standard deviation": f"{result.energies.std():.3f}",
        },
        artifacts=artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
