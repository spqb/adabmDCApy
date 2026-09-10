"""Command-line adapter for thermodynamic-integration entropy estimation."""

from __future__ import annotations

import argparse

from adabmDCA.parser import add_args_tdint


def create_parser() -> argparse.ArgumentParser:
    return add_args_tdint(argparse.ArgumentParser(description="Estimate DCA entropy by thermodynamic integration."))


class _EntropyProgressRenderer:
    def __init__(self) -> None:
        self._bar = None
        self._stage = None

    def __call__(self, event) -> None:
        if self._bar is None or event.stage != self._stage:
            from tqdm import tqdm

            self.close()
            self._stage = event.stage
            self._bar = tqdm(total=event.total, dynamic_ncols=True, leave=False, ascii="-#")
        description = f"{event.stage} | theta {event.theta:.3f}"
        if event.entropy is not None:
            description += f" | entropy {event.entropy:.3f}"
        self._bar.set_description(description)
        self._bar.update(event.completed - self._bar.n)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def run(args, *, progress=None):
    """Execute entropy estimation from parsed CLI arguments and return its result."""
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)
    from adabmDCA.api.entropy import estimate_entropy

    return estimate_entropy(
        model=args.path_params,
        natural_alignment=args.data,
        target_alignment=args.path_targetseq,
        initial_chains_path=args.path_chains,
        n_chains=args.nchains,
        n_sweeps=args.nsweeps,
        n_steps=args.nsteps,
        theta_max=args.theta_max,
        theta_sweeps=args.nsweeps_theta,
        zero_sweeps=args.nsweeps_zero,
        sampler=args.sampler,
        alphabet=args.alphabet,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        output_dir=args.output,
        label=args.label or "entropy",
        progress=progress,
    )


def main(args=None) -> int:
    args = create_parser().parse_args() if args is None else args
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)

    from adabmDCA.scripts._frontend import print_completion, print_configuration, print_header

    print_header("Thermodynamic-integration entropy")
    print_configuration(
        {
            "alignment": args.data,
            "model": args.path_params,
            "target": args.path_targetseq,
            "output": args.output,
            "chains": args.nchains,
            "integration steps": args.nsteps,
            "sweeps per step": args.nsweeps,
            "device": args.device,
            "dtype": args.dtype,
        }
    )
    renderer = _EntropyProgressRenderer()
    try:
        result = run(args, progress=renderer)
    finally:
        renderer.close()
    print_completion(
        "Thermodynamic integration completed successfully.",
        metrics={
            "entropy": f"{result.entropy:.6g}",
            "free energy": f"{result.free_energy:.6g}",
            "theta max": f"{result.theta_max:.6g}",
            "target fraction": f"{result.target_fraction:.3%}",
        },
        artifacts=result.artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
