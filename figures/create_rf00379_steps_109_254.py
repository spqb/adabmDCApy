"""Two model densities with their samples, using the saved fixed training PCA."""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models/RF00379/sampling'
with np.load(OUT / 'pca_ladder_distributions_projections.npz') as f:
    steps = f['training_steps']
    points = [f['models'][np.flatnonzero(steps == step).item()] for step in (109, 254)]
all_points = np.vstack(points)
lo, hi = all_points.min(axis=0), all_points.max(axis=0)
pad = (hi - lo) * .08 + .35
lo -= pad
hi += pad
edges = [np.linspace(lo[j], hi[j], 241) for j in range(2)]
centres = [(e[:-1] + e[1:]) / 2 for e in edges]
sigma = [.22 / (e[1] - e[0]) for e in edges]
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
for ax, p in zip(axes, points):
    histogram = np.histogram2d(*p.T, bins=edges)[0]
    density = gaussian_filter(histogram, sigma=sigma, mode='constant').T
    density /= density.sum()
    ranked = np.sort(density.ravel())[::-1]
    cumulative = ranked.cumsum()
    levels = [ranked[np.searchsorted(cumulative, mass)] for mass in (.95, .80, .50)]
    ax.contourf(*centres, density, levels=[*levels, density.max()*1.001],
                colors=['#fcead9', '#f4c393', '#e69a57'], zorder=1)
    ax.contour(*centres, density, levels=levels, colors='#c9742d', linewidths=.9, zorder=2)
    ax.scatter(*p.T, s=5, color='#285b8c', alpha=.42, linewidths=0,
               rasterized=True, zorder=3)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.985, wspace=.055)
for ext in ('png', 'pdf', 'svg'):
    path = OUT / f'pca_steps_109_254_contours_scatter.{ext}'
    fig.savefig(path, dpi=300, facecolor='white')
    print(path)
