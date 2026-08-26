import argparse

from adabmDCA.parser import add_args_sample


def create_parser():
    parser = argparse.ArgumentParser(description="Samples from a DCA model.")
    return add_args_sample(parser)


def main():
    args = create_parser().parse_args()

    import pandas as pd
    from tqdm import tqdm

    from adabmDCA.api.results import SamplingProgress
    from adabmDCA.api.sampling import sample_sequences
    from adabmDCA.scripts._utils import ensure_output_dir, require_file
    from adabmDCA.utils import get_device, get_dtype

    print("\n" + "=" * 80)
    print("  SAMPLING FROM DCA MODEL")
    print("=" * 80 + "\n")

    folder = ensure_output_dir(args.output)
    require_file(args.path_params, "Parameters file")
    if args.data is not None:
        require_file(args.data, "Data file")
    if args.weights is not None:
        require_file(args.weights, "Weights file")

    device = get_device(args.device)
    get_dtype(args.dtype)

    print("[CONFIGURATION]")
    print("-" * 80)
    template = "  {0:<28} {1:<50}"
    print(template.format("Parameters file:", args.path_params))
    if args.data is not None:
        print(template.format("Reference data:", args.data))
    print(template.format("Output folder:", str(folder)))
    print(template.format("Output label:", args.label if args.label is not None else "None"))
    print(template.format("Number of samples:", args.ngen))
    print(template.format("Sampler:", args.sampler))
    print(template.format("Beta (temperature):", args.beta))
    print(template.format("Seed:", args.seed))
    print(template.format("Device:", str(device)))
    print(template.format("Data type:", args.dtype))
    print("-" * 80 + "\n")

    print("[SAMPLING]")
    print("-" * 80)
    progress_bar = None

    def report(event: SamplingProgress) -> None:
        nonlocal progress_bar
        if progress_bar is None:
            progress_bar = tqdm(
                total=event.total,
                colour="red",
                dynamic_ncols=True,
                leave=False,
                ascii="-#",
                bar_format="  {desc}: [{bar}] {n}/{total} sweeps [{elapsed}]",
            )
            progress_bar.set_description("Sampling")
        progress_bar.update(event.completed - progress_bar.n)

    result = sample_sequences(
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
        device=str(device),
        dtype=args.dtype,
        progress=report,
    )
    if progress_bar is not None:
        progress_bar.close()
    print(f"  ✓ Sampling completed ({result.num_sweeps} sweeps)")
    print("-" * 80 + "\n")

    print("[OUTPUT]")
    print("-" * 80)
    mean_energy = result.energies.mean()
    std_energy = result.energies.std()
    print(f"  ✓ Mean energy: {mean_energy:.3f} ± {std_energy:.3f}")

    samples_filename = f"{args.label}_samples.fasta" if args.label is not None else "samples.fasta"
    fasta_file = folder / samples_filename
    result.to_fasta(fasta_file)
    print(f"  ✓ Samples saved: {fasta_file}")

    mix_name = f"{args.label}_mix.log" if args.label is not None else "mix.log"
    sampling_name = f"{args.label}_sampling.log" if args.label is not None else "sampling.log"
    mix_log_file = folder / mix_name
    sampling_log_file = folder / sampling_name
    pd.DataFrame.from_dict(result.mixing_history).to_csv(mix_log_file, index=False)
    pd.DataFrame.from_dict(result.sampling_history).to_csv(sampling_log_file, index=False)
    print("  ✓ Logs saved")
    print("-" * 80 + "\n")

    print("=" * 80)
    print("  SAMPLING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n  Results saved in: {folder}")
    print(f"    • Samples: {fasta_file}")
    if result.mixing_history:
        print(f"    • Mixing time log: {mix_log_file}")
    if result.sampling_history:
        print(f"    • Sampling log: {sampling_log_file}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
