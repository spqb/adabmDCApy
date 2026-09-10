"""Command-line adapter for Cobalt alignment splitting."""

from __future__ import annotations

import argparse

from adabmDCA.parser import add_args_profmark


def create_parser() -> argparse.ArgumentParser:
    return add_args_profmark(
        argparse.ArgumentParser(description="Split an alignment into profile-model training and test sets.")
    )


def run(args):
    """Execute Cobalt splitting from parsed CLI arguments and return its result."""
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)
    from adabmDCA.api.splitting import split_alignment

    return split_alignment(
        args.input_msa,
        t1=args.t1,
        t2=args.t2,
        t3=args.t3,
        max_train=args.maxtrain,
        max_test=args.maxtest,
        attempts=args.bestof,
        alphabet=args.alphabet,
        seed=args.seed,
        device=args.device,
    )


def main(args=None) -> int:
    args = create_parser().parse_args() if args is None else args
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)

    from adabmDCA.scripts._frontend import print_completion, print_configuration, print_header

    print_header("Cobalt alignment splitting")
    print_configuration(
        {
            "input": args.input_msa,
            "output prefix": args.output_prefix,
            "thresholds": f"{args.t1}, {args.t2}, {args.t3}",
            "attempts": args.bestof,
            "alphabet": args.alphabet,
            "seed": args.seed,
            "device": args.device,
        }
    )
    result = run(args)
    artifacts = result.save_bundle(args.output_prefix)
    print_completion(
        "Alignment splitting completed successfully.",
        metrics={
            "training sequences": result.training.num_sequences,
            "test sequences": result.test.num_sequences,
            "score": result.score,
        },
        artifacts=artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
