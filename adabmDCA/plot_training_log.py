#!/usr/bin/env python3
"""
Script to plot training metrics from DCA log files.

Usage:
    python plot_training_log.py <log_file> [--output-dir <dir>]
"""

import argparse
import os


def parse_training_log(log_path: str):
    """Parse a DCA training log file for plotting."""
    import numpy as np

    metadata = {}
    data = {
        'Epochs': [],
        'Pearson': [],
        'Slope': [],
        'LL_train': [],
        'LL_val': [],
        'Pearson_val': [],
        'Slope_val': [],
        'ESS': [],
        'Entropy': [],
        'Density': [],
        'Time': []
    }

    with open(log_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines) and lines[i].strip():
        line = lines[i].strip()
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
            i += 1
        else:
            break

    header = []
    while i < len(lines):
        if lines[i].strip().startswith('Epochs'):
            header = lines[i].strip().split()
            i += 1
            break
        i += 1

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if not line[0].isdigit() and '.' not in line.split()[0]:
            i += 1
            continue

        try:
            values = line.split()
            if len(values) >= len(header):
                for j, key in enumerate(header):
                    if key in data:
                        data[key].append(float(values[j]))
        except (ValueError, IndexError):
            pass

        i += 1

    parsed_data = {key: np.array(values) for key, values in data.items()}
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

    os.makedirs(output_dir, exist_ok=True)
    
    # Get the target Pearson value from metadata
    target_pearson = float(metadata.get('target Pearson Cij', 0.95))
    
    # Get base name for output files
    label = metadata.get('label', 'training')
    
    # Set style
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9
    
    # 1. Pearson vs Epochs
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data['Epochs'], data['Pearson'], 'o-', linewidth=2, markersize=3, label='Pearson train', color='blue')
    if len(data['Pearson_val']) > 0 and not np.all(np.isnan(data['Pearson_val'])):
        # Filter out NaN values for plotting
        mask = ~np.isnan(data['Pearson_val'])
        if np.any(mask):
            ax.plot(data['Epochs'][mask], data['Pearson_val'][mask], 's-',
                   linewidth=2, markersize=3, label='Pearson validation', color='orange')
    ax.axhline(y=target_pearson, color='r', linestyle='--', linewidth=2, label=f'Target ({target_pearson:.2f})')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title(f'Pearson Correlation vs Epochs - {label}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{label}_pearson.png'))
    plt.close()
    
    # 2. Slope vs Epochs
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data['Epochs'], data['Slope'], 'o-', linewidth=2, markersize=3, color='green', label='Slope train')
    if len(data['Slope_val']) > 0 and not np.all(np.isnan(data['Slope_val'])):
        # Filter out NaN values for plotting
        mask = ~np.isnan(data['Slope_val'])
        if np.any(mask):
            ax.plot(data['Epochs'][mask], data['Slope_val'][mask], 's-',
                   linewidth=2, markersize=3, color='brown', label='Slope validation')
            ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Slope')
    ax.set_title(f'Slope vs Epochs - {label}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{label}_slope.png'))
    plt.close()
    
    # 3. Log-likelihood per residue vs Epochs (train and validation if available)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data['Epochs'], data['LL_train'], 'o-', linewidth=2, markersize=3, label='Train', color='blue')
    
    # Check if validation data is available and not all NaN
    if len(data['LL_val']) > 0 and not np.all(np.isnan(data['LL_val'])):
        # Filter out NaN values for plotting
        mask = ~np.isnan(data['LL_val'])
        if np.any(mask):
            ax.plot(data['Epochs'][mask], data['LL_val'][mask], 's-', 
                   linewidth=2, markersize=3, label='Validation', color='orange')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Log-Likelihood per residue')
    ax.set_title(f'Log-Likelihood per residue vs Epochs - {label}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{label}_loglikelihood.png'))
    plt.close()
    
    # 4. Entropy vs Epochs
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data['Epochs'], data['Entropy'], 'o-', linewidth=2, markersize=3, color='purple')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Entropy')
    ax.set_title(f'Entropy vs Epochs - {label}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{label}_entropy.png'))
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}")
    print(f"  • {label}_pearson.png")
    print(f"  • {label}_slope.png")
    print(f"  • {label}_loglikelihood.png")
    print(f"  • {label}_entropy.png")


def create_parser():
    parser = argparse.ArgumentParser(
        description='Plot training metrics from DCA log files.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('log_file', type=str, help='Path to the log file')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same directory as log file)')

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
            output_dir = '.'
    else:
        output_dir = args.output_dir
    
    print(f"Reading log file: {args.log_file}")
    
    # Parse the log file
    metadata, data = parse_training_log(args.log_file)
    
    print(f"\nMetadata:")
    print(f"  Model: {metadata.get('model', 'N/A')}")
    print(f"  Label: {metadata.get('label', 'N/A')}")
    print(f"  Target Pearson: {metadata.get('target Pearson Cij', 'N/A')}")
    print(f"\nData points: {len(data['Epochs'])}")
    
    # Create plots
    create_plots(metadata, data, output_dir)


if __name__ == '__main__':
    main()
