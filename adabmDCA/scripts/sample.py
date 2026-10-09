"""Command-line adapter for model-based sequence generation."""

from __future__ import annotations

import argparse

from adabmDCA.parser import add_args_sample


def create_parser() -> argparse.ArgumentParser:
    from adabmDCA.scripts._frontend import ExplicitOptionParser
    return add_args_sample(ExplicitOptionParser(description=("Sample sequences from a DCA model. PTT uses the PTT (Parallel Trajectory Tempering) "
                      "sampling strategy; PCD uses normal MCMC sampling.")))


DEFAULT_PCD_SEQUENCES = 2000


def resolve_ngen(args) -> None:
    """Default --ngen: the training chains per model of a PTT archive, or 2000 for ordinary sampling."""
    if args.ngen is not None:
        return
    if args.strategy != "ptt":
        args.ngen = DEFAULT_PCD_SEQUENCES
        return
    import h5py

    from adabmDCA.exceptions import InputValidationError
    try:
        with h5py.File(args.path_params, "r") as handle:
            args.ngen = int(handle["replicas/0/chains"].shape[0])
    except (OSError, KeyError) as exc:
        raise InputValidationError("PTT requires a structured HDF5 ladder archive.") from exc


def run(args, *, progress=None):
    """Execute sampling from parsed CLI arguments and return its result."""
    from adabmDCA.scripts._frontend import resolve_alphabet

    resolve_alphabet(args)
    resolve_ngen(args)
    from adabmDCA.api.sampling import sample_sequences

    explicit = getattr(args, "_explicit_options", ())
    return sample_sequences(
        model=args.path_params,
        n_sequences=args.ngen,
        seed=args.seed,
        alphabet=args.alphabet,
        device=args.device,
        # PTT archives keep their training sampler and precision unless the options are given explicitly.
        dtype=args.dtype if args.strategy != "ptt" or "dtype" in explicit else None,
        sampler=args.sampler if args.strategy != "ptt" or "sampler" in explicit else None,
        beta=args.beta,
        n_sweeps=args.max_nsweeps,
        mixing_multiplier=args.nmix,
        ptt=(args.strategy == "ptt"),
        ptt_local_sweeps=args.ptt_local_sweeps,
        ptt_local_kernel=None if args.ptt_local_kernel == "archived" else args.ptt_local_kernel,
        ptt_mixing_method=args.ptt_mixing_method,
        ptt_renewal_tolerance=args.ptt_renewal_tolerance,
        ptt_stationary=args.ptt_stationary,
        ptt_max_rounds=args.ptt_max_rounds,
        reference_fasta=args.data,
        test_fasta=args.test,
        privet_window=tuple(args.privet_window),
        pseudocount=args.pseudocount,
        n_measure=args.nmeasure,
        weights_path=args.weights,
        clustering_seqid=args.clustering_seqid,
        no_reweighting=args.no_reweighting,
        collect_diagnostics=args.plot,
        progress=progress,
    )


def main(args=None) -> int:
    args = create_parser().parse_args() if args is None else args
    from adabmDCA.scripts._frontend import print_completion, print_configuration, print_header, resolve_alphabet

    resolve_alphabet(args)
    resolve_ngen(args)
    print_header("Sampling from a DCA model")
    print_configuration(_configuration(args))
    renderer = _SamplingProgressRenderer(diagnostic=args.diagnostic)
    try:
        result = run(args, progress=renderer)
    finally:
        renderer.close()
    artifacts = result.save_bundle(args.output, label=args.label)
    if args.plot:
        artifacts.update(result.save_diagnostic_plots(args.output, label=args.label))
    converged = not result.ptt_diagnostics or result.ptt_diagnostics["mixing"]["status"] == "converged"
    print_completion(
        "Sampling completed successfully." if converged else "PTT round budget exhausted; endpoint samples retained.",
        metrics={
            "sequences": len(result.sequences),
            "sweeps": result.num_sweeps,
            "mean energy": f"{result.energies.mean():.3f}",
            "standard deviation": f"{result.energies.std():.3f}",
            **({"fitted lambda (E vs summed CDE)": f"{result.local_lambda_fit['lambda']:.6g}",
                "lambda fit intercept": f"{result.local_lambda_fit['intercept']:.6g}",
                "lambda fit R²": f"{result.local_lambda_fit['r_squared']:.4f}"}
               if result.local_lambda_fit else
               {"lambda fit": f"unavailable: {result.warnings[-1]}"}),
            **(_ptt_summary(result.ptt_diagnostics) if result.ptt_diagnostics else {}),
            **(_distance_summary(result.distance_comparison) if result.distance_comparison else {}),
        },
        artifacts=artifacts,
    )
    return 0


def _configuration(args) -> dict:
    """The options of this run, as printed before sampling starts."""
    ptt, renewal = args.strategy == "ptt", args.strategy == "ptt" and args.ptt_mixing_method == "renewal"
    if ptt:
        sampler = ("PTT (archived local sampler)" if args.ptt_local_kernel == "archived"
                   else f"PTT ({args.ptt_local_kernel} local kernel)")
    else:
        sampler = args.sampler
    return {
        "model": args.path_params,
        "reference": args.data,
        "held-out": args.test,
        "output": args.output,
        "label": args.label,
        "sequences": args.ngen,
        "sampler": sampler,
        "PTT mixing method": args.ptt_mixing_method if ptt else None,
        "PTT renewal tolerance": args.ptt_renewal_tolerance if renewal else None,
        "PTT renewal phases": ("warmup and stationary" if args.ptt_stationary else "warmup only") if renewal else None,
        "PTT round budget": args.ptt_max_rounds if ptt else None,
        "alphabet": args.alphabet,
        "beta": args.beta,
        "seed": args.seed,
        "device": args.device,
        "dtype": "archive precision" if ptt else args.dtype,
        "plots": args.plot,
    }


def _distance_summary(distances: dict) -> dict:
    """Median distances to the nearest training sequence: generated, and held-out when given."""
    summary = distances["summary"]
    metrics = {"nearest training (generated, median)": f"{summary['generated_to_natural']['median']:.3f}"}
    if "test_to_natural" in summary:
        metrics["nearest training (held-out, median)"] = f"{summary['test_to_natural']['median']:.3f}"
    identical = summary["generated_to_natural"]["identical"]
    if identical > 0:
        metrics["generated identical to a training sequence"] = f"{identical:.1%}"
    privet = distances.get("privet")
    if privet:
        excess = privet["excess_train"]
        metrics["PRIVET excess near training"] = (
            "none" if excess["log10_p"] > -1 else
            f"{excess['count']} samples within {excess['distance']:.3f} "
            f"(null {excess['expected']:.1f}, log10 p {excess['log10_p']:.1f})"
        )
        if "delta_p" in privet:
            metrics["PRIVET flagged (delta_p < -3)"] = f"{privet['n_flagged']} of {privet['n_generated']}"
            metrics["PRIVET N_pleaks"] = privet["n_pleaks"]
        else:
            metrics["PRIVET flagged (log10 p_train < -3)"] = f"{privet['n_memorized']} of {privet['n_generated']}"
    return metrics


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


def _rounds_text(rounds) -> str:
    return "unresolved" if rounds is None else f"{rounds} rounds"


def _ptt_summary(diagnostics: dict) -> dict:
    """End-of-run PTT figures: how the ladder mixed, and where it is weakest."""
    mixing = diagnostics["mixing"]
    if mixing.get("method") == "renewal":
        def decay(key):
            value = mixing.get(key)
            return "" if value is None else f" (G decay time {value:.0f} rounds)"

        summary = {"mixing status": mixing["status"],
                   "renewal warmup": _rounds_text(mixing["warmup_rounds"]) + decay("warmup_decay_rounds")}
        if mixing.get("stationary", True):
            summary["stationary renewal"] = _rounds_text(mixing["renewal_rounds"]) + decay("stationary_decay_rounds")
            summary["stationary phase"] = (f"{mixing['stationary_rounds']} rounds, ending at the first "
                                           f"{mixing['chunk_rounds']}-round block end after renewal")
        else:
            summary["stationary renewal"] = "not run (add --ptt-stationary to renew the ladder a second time)"
        summary["trapped endpoint fraction"] = f"{mixing['trapped_fraction']:.4f}"
    else:
        summary = {"mixing status": mixing["status"],
                   "tau_int (rounds)": "unresolved" if mixing["tau_int"] is None else f"{mixing['tau_int']:.3f}",
                   "tau_exp (rounds)": "unresolved" if mixing["tau_exp"] is None else f"{mixing['tau_exp']:.3f}"}
    summary["final Pearson"] = f"{diagnostics['final_pearson']:.6f}"
    health = diagnostics.get("ladder_health")
    if health and health["pairs"]:
        pairs = health["pairs"]
        summary["endpoint log Z (BAR)"] = f"{health['log_z_bar']:.3f} +- {health['log_z_bar_error']:.3f}"
        aged = [pair for pair in pairs if pair.get("immobile_mean_age") is not None]
        if aged:
            # Long-lived immobile configurations are trapped; brief immobility clears by local moves.
            worst = max(aged, key=lambda pair: pair["immobile_mean_age"] / max(pair["mobile_mean_age"], 1.0))
            summary["most trapped pair"] = (
                f"{worst['lower']}->{worst['upper']} ({worst['immobile_down_fraction']:.4f} immobile, "
                f"mean age {worst['immobile_mean_age']:.0f} vs {worst['mobile_mean_age']:.0f} rounds)"
            )
        else:
            worst = max(pairs, key=lambda pair: pair["immobile_down_fraction"])
            summary["most immobile pair"] = (
                f"{worst['lower']}->{worst['upper']} ({worst['immobile_down_fraction']:.4f} of the upper replica)"
            )
        weakest = min(pairs, key=lambda pair: pair["ess_forward"])
        summary["lowest forward ESS"] = f"{weakest['lower']}->{weakest['upper']} ({weakest['ess_forward']:.4f})"
    return summary


class _SamplingProgressRenderer:
    """Progress bars for sampling stages, plus PTT diagnostics with --diagnostic."""

    def __init__(self, *, diagnostic: bool = False, stream=None) -> None:
        import sys

        self._bar = None
        self._stage = None
        self._diagnostic = diagnostic
        self._stream = sys.stderr if stream is None else stream
        self._phase_start = {}
        self._warned = set()

    def __call__(self, event) -> None:
        details = event.details
        if event.stage == "ptt_ladder":
            print(f"PTT | selected updates: {details['selected_updates']} | "
                  f"{details['chains_per_model']} chains per model", file=self._stream)
            return
        if event.stage == "ptt_ladder_health":
            if self._diagnostic:
                self._ladder_health(details)
            return
        if event.stage == "ptt_renewal":
            if self._diagnostic:
                rates = ", ".join(f"{rate:.4f}" for rate in details.get("acceptance", ()))
                self._write(
                    f"PTT renewal | {details['status']} | warmup {_rounds_text(details['warmup_rounds'])} | "
                    f"stationary renewal {_rounds_text(details['renewal_rounds'])} | "
                    f"trapped endpoint fraction {details['trapped_fraction']:.4f} | acceptance [{rates}]"
                )
            return
        if event.stage in ("ptt_renewal_warmup", "ptt_renewal_stationary") and "ladder_old" in details:
            self._renewal_progress(event)
            return
        self._advance_bar(event)
        if self._diagnostic and event.stage == "ptt_mixing_measure" and "required_rounds" in details:
            rates = ", ".join(f"{rate:.4f}" for rate in details.get("acceptance", ()))
            tau_int, tau_exp = details.get("tau_int"), details.get("tau_exp")
            self._write(
                f"PTT diagnostic | rounds {event.completed} -> {event.total} | "
                f"tau_int {'unresolved' if tau_int is None else f'{tau_int:.3f}'} | "
                f"tau_exp {'unresolved' if tau_exp is None else f'{tau_exp:.3f}'} | acceptance [{rates}]"
            )

    def _renewal_progress(self, event) -> None:
        """Show the old fraction G against its threshold, and when the fit predicts renewal."""
        import math
        import time

        details = event.details
        phase = event.stage.removeprefix("ptt_renewal_")
        phase_round = details["phase_round"]
        start_time, start_round = self._phase_start.setdefault(event.stage, (time.monotonic(), phase_round))
        text = f"PTT {phase} | G {details['ladder_old']:.1e} -> {details['threshold']:.1e}"
        predicted, stop = details.get("predicted_round"), None
        if "decay_rounds" in details:
            if details["decay_rounds"] is None:
                text += " | G not decreasing"
                self._warn(event.stage + "flat", f"PTT {phase} | the old fraction G is not decreasing: "
                           "some configurations may be trapped.")
            else:
                block = details.get("block_rounds")
                stop = math.ceil(predicted / block) * block if block else math.ceil(predicted)
                text += f" | decay ~{details['decay_rounds']:.0f} rounds | "
                text += f"stop ~ round {stop}" if block else f"renewal ~ round {math.ceil(predicted)}"
                if phase_round > start_round:
                    seconds = (time.monotonic() - start_time) / (phase_round - start_round) * max(stop - phase_round, 0)
                    text += f" (~{_duration(seconds)})"
                first = details.get("first_decay_rounds")
                if first and details["decay_rounds"] > 2 * first:
                    self._warn(event.stage + "slow", f"PTT {phase} | G decays more and more slowly (decay time "
                               f"{first:.0f} -> {details['decay_rounds']:.0f} rounds): some configurations may be "
                               "trapped.")
                if event.completed - phase_round + stop > event.total:
                    self._warn(event.stage + "budget", f"PTT {phase} | renewal is predicted at phase round {stop}, "
                               f"beyond the budget of {event.total} rounds.")
        total = event.total if stop is None else min(event.total, event.completed - phase_round + stop)
        self._advance_bar(event, total=max(total, event.completed))
        self._bar.set_description(text)
        if self._diagnostic and phase_round % 100 == 0:
            self._write(f"PTT diagnostic | {phase} | round {phase_round} | ladder old {details['ladder_old']:.4f} | "
                        f"endpoint fresh {details['endpoint_fresh']:.4f}"
                        + (f" | predicted renewal round {predicted:.0f}" if predicted is not None else ""))

    def _warn(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self._write(message)

    def _advance_bar(self, event, total=None) -> None:
        if self._bar is not None and self._stage != event.stage:
            self.close()
        if self._bar is None:
            from tqdm import tqdm

            unit = "sweeps"
            if event.stage.startswith("ptt_"):
                unit = {"ptt_generation": "samples", "ptt_ladder_statistics": "models"}.get(event.stage, "rounds")
            self._bar = tqdm(
                total=event.total, colour="red", dynamic_ncols=True, leave=False, ascii="-#", file=self._stream,
                bar_format="  {desc}: [{bar}] {n}/{total} " + unit + " [{elapsed}]",
            )
            self._stage = event.stage
            self._bar.set_description(
                event.stage.replace("_", " ") if event.stage.startswith("ptt_") else "Generating sequences"
            )
        self._bar.total = event.total if total is None else total
        self._bar.update(event.completed - self._bar.n)

    def _ladder_health(self, details) -> None:
        self._write(
            f"PTT ladder health | log Z forward {details['log_z_forward']:.3f} | "
            f"BAR {details['log_z_bar']:.3f} +- {details['log_z_bar_error']:.3f}"
        )
        for pair in details["pairs"]:
            slope = ("n/a" if pair["crooks_slope"] is None
                     else f"{pair['crooks_slope']:+.2f} +- {pair['crooks_slope_error']:.2f}")
            self._write(
                f"  models {pair['lower']}->{pair['upper']} | dF fwd {pair['dF_forward']:.3f} "
                f"+- {pair['dF_forward_error']:.3f} rev {pair['dF_reverse']:.3f} "
                f"+- {pair['dF_reverse_error']:.3f} BAR {pair['dF_bar']:.3f} +- {pair['dF_bar_error']:.3f} | "
                f"ESS fwd/rev {pair['ess_forward']:.3f}/{pair['ess_reverse']:.3f} | "
                f"Crooks slope {slope} | acceptance {pair['mean_acceptance']:.3f} | "
                f"immobile down/up {pair['immobile_down_fraction']:.4f}/{pair['immobile_up_fraction']:.4f}"
            )

    def _write(self, message: str) -> None:
        if self._bar is not None:
            self._bar.write(message, file=self._stream)
        else:
            print(message, file=self._stream)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


if __name__ == "__main__":
    raise SystemExit(main())
