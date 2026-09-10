"""Command-line adapter for deep mutational scanning."""

from __future__ import annotations

import argparse

from adabmDCA.parser import add_args_dms


def create_parser() -> argparse.ArgumentParser:
    return add_args_dms(argparse.ArgumentParser(description="Generate a single-mutant DCA scan."))


def run(args):
    """Execute a mutation scan from parsed CLI arguments and return its result."""
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)
    from adabmDCA.api.mutations import scan_mutations
    from adabmDCA.input_loading import AlignmentLoadConfig, load_alignment

    loaded = load_alignment(
        args.data,
        config=AlignmentLoadConfig(alphabet=args.alphabet, invalid_sequences="error"),
    )
    name = "".join(character for character in loaded.alignment.names[0] if character.isalnum())
    return scan_mutations(
        loaded.alignment.sequences[0],
        model=args.path_params,
        name=name or "wild_type",
        alphabet=args.alphabet,
        device=args.device,
        dtype=args.dtype,
    )


def main(args=None) -> int:
    args = create_parser().parse_args() if args is None else args
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)

    from adabmDCA.scripts._frontend import print_completion, print_configuration, print_header

    print_header("Deep mutational scanning")
    print_configuration(
        {
            "wild type": args.data,
            "model": args.path_params,
            "output": args.output,
            "alphabet": args.alphabet,
            "device": args.device,
            "dtype": args.dtype,
        }
    )
    result = run(args)
    artifacts = result.save_bundle(args.output)
    print_completion(
        "Mutation scan completed successfully.",
        metrics={
            "wild type": result.name,
            "mutations": len(result.mutations),
            "sites": result.model.length,
            "wild-type energy": f"{result.wild_type_energy:.3f}",
        },
        artifacts=artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
