"""Command-line adapter for contact prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

from adabmDCA.parser import add_args_contacts


def create_parser() -> argparse.ArgumentParser:
    return add_args_contacts(argparse.ArgumentParser(description="Compute a DCA contact map."))


def run(args):
    """Execute contact prediction from parsed CLI arguments and return its result."""
    from adabmDCA.api.contacts import predict_contacts

    options = {"alphabet": args.alphabet, "device": args.device, "dtype": args.dtype}
    if args.path_params is not None:
        return predict_contacts(model=args.path_params, **options)
    return predict_contacts(fasta_path=args.data, pseudocount=args.pseudocount, **options)


def _save_plot(result, path: Path) -> Path:
    import matplotlib.pyplot as plt

    from adabmDCA.plot import plot_contact_map

    plt.rcParams.update({"font.size": 12})
    figure, axis = plt.subplots(dpi=150, figsize=(6, 5))
    plot_contact_map(axis, result.scores)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def main(args=None) -> int:
    args = create_parser().parse_args() if args is None else args

    from adabmDCA.scripts._frontend import print_completion, print_configuration, print_header

    print_header("Contact-map prediction")
    print_configuration(
        {
            "model": args.path_params,
            "alignment": args.data if args.path_params is None else None,
            "output": args.output,
            "label": args.label,
            "alphabet": args.alphabet,
            "device": args.device,
            "dtype": args.dtype,
        }
    )
    result = run(args)
    artifacts = result.save_bundle(args.output, label=args.label)
    stem = f"{args.label}_contact_map" if args.label else "contact_map"
    artifacts["plot"] = _save_plot(result, Path(args.output) / f"{stem}.png")
    print_completion(
        "Contact-map prediction completed successfully.",
        metrics={
            "method": result.method,
            "matrix size": f"{result.scores.shape[0]} x {result.scores.shape[1]}",
            "score range": f"[{result.scores.min():.4f}, {result.scores.max():.4f}]",
        },
        artifacts=artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
