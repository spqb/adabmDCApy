"""Command-line adapter for homology-aware alignment splitting."""

from __future__ import annotations

import argparse

from adabmDCA.parser import add_args_split_data


def create_parser() -> argparse.ArgumentParser:
    return add_args_split_data(
        argparse.ArgumentParser(description="Split an alignment into training and test sets with reduced homology correlation.")
    )


def run(args):
    """Split an alignment with the selected method and return its result."""
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)
    from adabmDCA.api.splitting import split_alignment

    return split_alignment(
        args.input_msa,
        method=args.method,
        identity=args.identity,
        train_fraction=args.train_fraction,
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

    print_header("Homology-aware alignment splitting")
    configuration = {
        "input": args.input_msa,
        "output prefix": args.output_prefix,
        "method": args.method,
        "alphabet": args.alphabet,
        "seed": args.seed,
        "device": args.device,
    }
    if args.method == "clustering":
        configuration.update({"minimum identity": args.identity, "train fraction": args.train_fraction})
    else:
        configuration.update({"thresholds": f"{args.t1}, {args.t2}, {args.t3}", "attempts": args.bestof})
    print_configuration(configuration)
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
