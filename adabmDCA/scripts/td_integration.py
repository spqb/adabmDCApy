"""Command-line adapter for thermodynamic-integration entropy estimation."""

from __future__ import annotations

import argparse

from adabmDCA.parser import add_args_tdint


def create_parser() -> argparse.ArgumentParser:
    from adabmDCA.scripts._frontend import ExplicitOptionParser

    class EntropyParser(ExplicitOptionParser):
        def parse_known_args(self, args=None, namespace=None):
            result, unknown = super().parse_known_args(args, namespace)
            if result.strategy != "ptt" and (result.data is None or result.path_targetseq is None):
                self.error("--data and --path_targetseq are required with --strategy pcd")
            return result, unknown
    return add_args_tdint(EntropyParser(description=("Estimate DCA entropy. PTT uses the PTT (Parallel Trajectory Tempering) sampling strategy; "
                      "PCD uses normal MCMC sampling for thermodynamic integration.")))



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
    if args.strategy == "ptt":
        from adabmDCA.exceptions import InputValidationError
        from adabmDCA.api.ptt import estimate_ptt_entropy
        if args.data is not None or args.path_targetseq is not None or args.path_chains is not None or args.dtype == "bfloat16":
            raise InputValidationError("PTT entropy requires only an archive; target/data/chains and BF16 are unsupported.")
        incompatible = {"theta_max", "nsteps", "nsweeps_theta", "nsweeps_zero", "sampler", "dtype"}
        if incompatible.intersection(getattr(args, "_explicit_options", ())):
            raise InputValidationError("Integration/local-kernel options cannot be combined with --strategy ptt; the archive owns the kernel.")
        return estimate_ptt_entropy(model=args.path_params, n_sweeps=args.nsweeps, device=args.device,
                                    alphabet=args.alphabet, seed=args.seed, output_dir=args.output, label=args.label)
    from adabmDCA.exceptions import InputValidationError
    if args.data is None or args.path_targetseq is None:
        raise InputValidationError("Entropy integration requires --data and --path_targetseq (or select --strategy ptt).")
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

    print_header("Direct PTT entropy" if args.strategy == "ptt" else "Thermodynamic-integration entropy")
    print_configuration(
        {
            "alignment": args.data,
            "model": args.path_params,
            "target": args.path_targetseq,
            "output": args.output,
            "chains": args.nchains if args.strategy != "ptt" else None,
            "integration steps": args.nsteps if args.strategy != "ptt" else None,
            "sweeps per step": args.nsweeps if args.strategy != "ptt" else None,
            "PTT local sweeps": args.nsweeps if args.strategy == "ptt" else None,
            "--nchains": "ignored; the archive owns the chains" if args.strategy == "ptt" and
                "nchains" in getattr(args, "_explicit_options", ()) else None,
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
        "PTT entropy estimation completed successfully." if args.strategy == "ptt" else "Thermodynamic integration completed successfully.",
        metrics={
            "entropy": f"{result.entropy:.6g}",
            "free energy": f"{result.free_energy:.6g}",
            **({"logZ": result.log_z, "status": result.status} if args.strategy == "ptt" else {
                "theta max": f"{result.theta_max:.6g}", "target fraction": f"{result.target_fraction:.3%}"}),
        },
        artifacts=result.artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
