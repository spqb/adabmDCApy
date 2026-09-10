"""
Script to plot training metrics from DCA log files.

Usage:
    python plot_training_log.py <log_file> [--output-dir <dir>]
"""

import argparse
import os

from adabmDCA.training_config import DEFAULT_TARGET_PEARSON


def parse_training_log(log_path: str):
    """Parse a version-2 DCA training log file for plotting."""
    import numpy as np

    metadata = {}
    data = {
        "Epochs": [],
        "Pearson": [],
        "Slope": [],
        "LL_train": [],
        "LL_val": [],
        "Pearson_val": [],
        "Slope_val": [],
        "ESS": [],
        "Entropy": [],
        "Density": [],
        "Time": [],
        "Stage": [],
        "Gradient_steps": [],
        "Structure_steps": [],
        "Sweeps": [],
    }
    aliases = {
        "Step": "Epochs",
        "Grad_steps": "Gradient_steps",
        "Struct_steps": "Structure_steps",
        "Chain_ESS_frac": "ESS",
        "Elapsed_s": "Time",
    }
    section = "header"
    header = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line == "adabmDCA training log":
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().lower()
                header = []
                continue
            if line.startswith("Step "):
                header = line.split()
                continue
            if header:
                values = line.split()
                if len(values) != len(header):
                    continue
                try:
                    float(values[0])
                except ValueError:
                    continue
                for label, value in zip(header, values, strict=True):
                    key = aliases.get(label, label)
                    if key == "Stage":
                        data[key].append(value)
                    elif key in data:
                        data[key].append(float(value))
                continue
            if ":" in line:
                key, value = (part.strip() for part in line.split(":", 1))
                qualified = key if section == "header" else f"{section}.{key}"
                metadata[qualified] = value

    parsed_data = {
        key: np.asarray(values, dtype=object if key == "Stage" else float)
        for key, values in data.items()
    }
    return metadata, parsed_data


def create_plots(metadata, data, output_dir):
    """Create and save plots from training data.

    Args:
        metadata (dict): Metadata from the log file.
        data (dict): Training data arrays.
        output_dir (str): Directory to save plots.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from adabmDCA.plot import (
        _DIAGNOSTIC_BLUE,
        _DIAGNOSTIC_CORAL,
        _DIAGNOSTIC_TEAL,
        _DIAGNOSTIC_TEXT,
        _style_diagnostic_axis,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Get the target Pearson value from metadata
    target_pearson = float(metadata.get("optimization.target_pearson", DEFAULT_TARGET_PEARSON))

    # Get base name for output files
    label = metadata.get("run.label", "training")

    plot_dpi = 192

    def style_axis(ax, title):
        _style_diagnostic_axis(ax)
        ax.set_title(title, color=_DIAGNOSTIC_TEXT, pad=10)
        ax.set_xlabel("Training step")

    def plot_series(ax, x, y, *, label, color, marker="o"):
        ax.plot(
            x,
            y,
            marker=marker,
            linewidth=1.8,
            markersize=4,
            markerfacecolor="white",
            markeredgewidth=1.2,
            color=color,
            label=label,
            zorder=2,
        )

    def save_figure(fig, filename):
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=plot_dpi, facecolor="white")
        plt.close(fig)

    # 1. Pearson vs Epochs
    fig, ax = plt.subplots(figsize=(8, 5), dpi=plot_dpi)
    style_axis(ax, f"Pearson correlation - {label}")
    plot_series(ax, data["Epochs"], data["Pearson"], label="Training", color=_DIAGNOSTIC_BLUE)
    if len(data["Pearson_val"]) > 0 and not np.all(np.isnan(data["Pearson_val"])):
        mask = ~np.isnan(data["Pearson_val"])
        if np.any(mask):
            plot_series(
                ax,
                data["Epochs"][mask],
                data["Pearson_val"][mask],
                label="Validation",
                color=_DIAGNOSTIC_CORAL,
                marker="s",
            )
    ax.axhline(
        y=target_pearson,
        color=_DIAGNOSTIC_TEAL,
        linestyle="--",
        linewidth=1.5,
        label=f"Target ({target_pearson:.2f})",
        zorder=1,
    )
    ax.set_ylabel("Pearson correlation")
    ax.legend(frameon=False, loc="best")
    save_figure(fig, f"{label}_pearson.png")

    # 2. Slope vs Epochs
    fig, ax = plt.subplots(figsize=(8, 5), dpi=plot_dpi)
    style_axis(ax, f"Correlation slope - {label}")
    plot_series(ax, data["Epochs"], data["Slope"], label="Training", color=_DIAGNOSTIC_BLUE)
    if len(data["Slope_val"]) > 0 and not np.all(np.isnan(data["Slope_val"])):
        mask = ~np.isnan(data["Slope_val"])
        if np.any(mask):
            plot_series(
                ax,
                data["Epochs"][mask],
                data["Slope_val"][mask],
                color=_DIAGNOSTIC_CORAL,
                label="Validation",
                marker="s",
            )
    ax.axhline(1.0, color="#7A7F85", linestyle="--", linewidth=1.2, label="Ideal slope", zorder=1)
    ax.set_ylabel("Slope")
    ax.legend(frameon=False, loc="best")
    save_figure(fig, f"{label}_slope.png")

    # 3. Log-likelihood per residue vs Epochs (train and validation if available)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=plot_dpi)
    style_axis(ax, f"Log-likelihood per residue - {label}")
    plot_series(ax, data["Epochs"], data["LL_train"], label="Training", color=_DIAGNOSTIC_BLUE)

    # Check if validation data is available and not all NaN
    if len(data["LL_val"]) > 0 and not np.all(np.isnan(data["LL_val"])):
        mask = ~np.isnan(data["LL_val"])
        if np.any(mask):
            plot_series(
                ax,
                data["Epochs"][mask],
                data["LL_val"][mask],
                label="Validation",
                color=_DIAGNOSTIC_CORAL,
                marker="s",
            )

    ax.set_ylabel("Log-likelihood per residue")
    ax.legend(frameon=False, loc="best")
    save_figure(fig, f"{label}_loglikelihood.png")

    # 4. Entropy vs Epochs
    fig, ax = plt.subplots(figsize=(8, 5), dpi=plot_dpi)
    style_axis(ax, f"Model entropy - {label}")
    plot_series(ax, data["Epochs"], data["Entropy"], label="Entropy", color=_DIAGNOSTIC_TEAL)
    ax.set_ylabel("Entropy")
    ax.legend(frameon=False, loc="best")
    save_figure(fig, f"{label}_entropy.png")

    # 5. Sparse-model density and normalized chain ESS
    fig, ax = plt.subplots(figsize=(8, 5), dpi=plot_dpi)
    style_axis(ax, f"Training diagnostics - {label}")
    plot_series(ax, data["Epochs"], data["Density"], label="Graph density", color=_DIAGNOSTIC_TEAL)
    plot_series(
        ax,
        data["Epochs"],
        data["ESS"],
        label="Chain ESS fraction",
        color=_DIAGNOSTIC_BLUE,
        marker="s",
    )
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="best")
    save_figure(fig, f"{label}_diagnostics.png")

    print(f"\nPlots saved to: {output_dir}")
    print(f"  • {label}_pearson.png")
    print(f"  • {label}_slope.png")
    print(f"  • {label}_loglikelihood.png")
    print(f"  • {label}_entropy.png")
    print(f"  • {label}_diagnostics.png")


def create_parser():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from DCA log files.", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log_file", type=str, help="Path to the log file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same directory as log file)",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Check if log file exists
    if not os.path.exists(args.log_file):
        parser.error(f"Log file '{args.log_file}' not found.")

    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(args.log_file)
        if not output_dir:
            output_dir = "."
    else:
        output_dir = args.output_dir

    print(f"Reading log file: {args.log_file}")

    # Parse the log file
    metadata, data = parse_training_log(args.log_file)

    print("\nMetadata:")
    print(f"  Model: {metadata.get('run.model', 'N/A')}")
    print(f"  Label: {metadata.get('run.label', 'N/A')}")
    print(f"  Sequences: {metadata.get('training data.retained_sequences', 'N/A')}")
    print(f"  Sequence length: {metadata.get('training data.sequence_length', 'N/A')}")
    print(f"  Effective sequences: {metadata.get('training data.effective_sequences', 'N/A')}")
    print(f"  Target Pearson: {metadata.get('optimization.target_pearson', 'N/A')}")
    print(f"\nData points: {len(data['Epochs'])}")

    if not len(data["Epochs"]):
        parser.error("The log contains no version-2 training records.")

    # Create plots
    create_plots(metadata, data, output_dir)


if __name__ == "__main__":
    main()
