#!/usr/bin/env python3
"""
Script to plot training metrics from DCA log files.

Usage:
    python plot_training_log.py <log_file> [--output-dir <dir>]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from adabmDCA.utils import parse_log_file


def create_plots(metadata, data, output_dir):
    """Create and save plots from training data.
    
    Args:
        metadata (dict): Metadata from the log file.
        data (dict): Training data arrays.
        output_dir (str): Directory to save plots.
    """
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
    if len(data['Pearson_test']) > 0 and not np.all(np.isnan(data['Pearson_test'])):
        # Filter out NaN values for plotting
        mask = ~np.isnan(data['Pearson_test'])
        if np.any(mask):
            ax.plot(data['Epochs'][mask], data['Pearson_test'][mask], 's-', 
                   linewidth=2, markersize=3, label='Pearson test', color='orange')
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
    ax.plot(data['Epochs'], data['Slope'], 'o-', linewidth=2, markersize=3, color='green')
    if len(data['Slope_test']) > 0 and not np.all(np.isnan(data['Slope_test'])):
        # Filter out NaN values for plotting
        mask = ~np.isnan(data['Slope_test'])
        if np.any(mask):
            ax.plot(data['Epochs'][mask], data['Slope_test'][mask], 's-', 
                   linewidth=2, markersize=3, color='brown', label='Slope test')
            ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Slope')
    ax.set_title(f'Slope vs Epochs - {label}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{label}_slope.png'))
    plt.close()
    
    # 3. Log-Likelihood vs Epochs (train and test if available)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data['Epochs'], data['LL_train'], 'o-', linewidth=2, markersize=3, label='Train', color='blue')
    
    # Check if test data is available and not all NaN
    if len(data['LL_test']) > 0 and not np.all(np.isnan(data['LL_test'])):
        # Filter out NaN values for plotting
        mask = ~np.isnan(data['LL_test'])
        if np.any(mask):
            ax.plot(data['Epochs'][mask], data['LL_test'][mask], 's-', 
                   linewidth=2, markersize=3, label='Test', color='orange')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title(f'Log-Likelihood vs Epochs - {label}')
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


def main():
    parser = argparse.ArgumentParser(
        description='Plot training metrics from DCA log files.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('log_file', type=str, help='Path to the log file')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same directory as log file)')
    
    args = parser.parse_args()
    
    # Check if log file exists
    if not os.path.exists(args.log_file):
        print(f"Error: Log file '{args.log_file}' not found.")
        return
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(args.log_file)
        if not output_dir:
            output_dir = '.'
    else:
        output_dir = args.output_dir
    
    print(f"Reading log file: {args.log_file}")
    
    # Parse the log file
    metadata, data = parse_log_file(args.log_file)
    
    print(f"\nMetadata:")
    print(f"  Model: {metadata.get('model', 'N/A')}")
    print(f"  Label: {metadata.get('label', 'N/A')}")
    print(f"  Target Pearson: {metadata.get('target Pearson Cij', 'N/A')}")
    print(f"\nData points: {len(data['Epochs'])}")
    
    # Create plots
    create_plots(metadata, data, output_dir)


if __name__ == '__main__':
    main()
