"""Compare every equilibrium ladder sample in a fixed, weighted training PCA."""
from pathlib import Path
import json
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models/RF00379/sampling'
PREFIX = OUT / 'pca_ladder_distributions'
report = json.loads((ROOT / 'models/RF00379/training.json').read_text())['data']['input_report']
seqs = []
for line in (ROOT / report['source']).read_text().splitlines():
    if line.startswith('>'):
        seqs.append('')
    elif line.strip():
        seqs[-1] += line.strip()
with h5py.File(OUT / 'ladder_samples.h5') as f:
    tokens = f.attrs['tokens']
    samples = f['sequences'][()]
    steps = f['models/training_step'][()]
    replicas = f['models/replica'][()]
    model_ids = [s.decode() for s in f['models/model_id'][()]]
train = np.array([[tokens.index(c) for c in seqs[i]] for i in report['retained_indices']])
weights = np.loadtxt(ROOT / 'models/RF00379/weights.dat')
assert weights.shape == (len(train),) and np.all(weights >= 0) and weights.sum() > 0
assert samples.shape[2] == train.shape[1] and samples.min() >= 0 and samples.max() < len(tokens)
weights = weights / weights.sum()
def onehot(a):
    return np.eye(len(tokens))[a].reshape(len(a), -1)
x = onehot(train)
mean = weights @ x
centered = x - mean
covariance = (centered * weights[:, None]).T @ centered
values, vectors = np.linalg.eigh(covariance)
basis = vectors[:, -2:][:, ::-1]
for j in range(2):
    if basis[np.argmax(abs(basis[:, j])), j] < 0:
        basis[:, j] *= -1
reference = centered @ basis
projected = np.array([(onehot(a) - mean) @ basis for a in samples])
all_points = np.vstack([reference, projected.reshape(-1, 2)])
lo, hi = all_points.min(axis=0), all_points.max(axis=0)
pad = (hi - lo) * .10 + .35
lo -= pad
hi += pad
edges = [np.linspace(lo[j], hi[j], 221) for j in range(2)]
centres = [(e[1:] + e[:-1]) / 2 for e in edges]
# Identical smoothing kernel for the data and every model, in fixed PC units.
bandwidth = .22
sigma = [bandwidth / (e[1] - e[0]) for e in edges]
def density(points, w=None):
    hist = np.histogram2d(*points.T, bins=edges, weights=w)[0]
    smooth = gaussian_filter(hist, sigma=sigma, mode='constant')
    return smooth.T / smooth.sum()
def levels(d):
    descending = np.sort(d.ravel())[::-1]
    cumulative = descending.cumsum()
    return np.array([descending[np.searchsorted(cumulative, mass)] for mass in [.95, .80, .50]])
ref_density = density(reference, weights)
ref_levels = levels(ref_density)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
                     'axes.labelcolor': '#34445a', 'text.color': '#172b45'})
fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.2), sharex=True, sharey=True)
blue, orange = '#285b8c', '#df8439'
for index, ax in enumerate(axes.flat):
    ax.set_facecolor('#fafbfd')
    for spine in ax.spines.values():
        spine.set_color('#d6dde5')
    ax.grid(color='#e7ecf1', linewidth=.55, zorder=0)
    if index == 0:
        ax.contourf(*centres, ref_density, levels=[*ref_levels, ref_density.max()*1.001],
                    colors=['#e1eaf3', '#a6c3df', '#5689b8'], zorder=2)
    else:
        i = index - 1
        d = density(projected[i])
        lev = levels(d)
        ax.contourf(*centres, d, levels=[*lev, d.max()*1.001],
                    colors=['#fcead9', '#f4c393', '#e69a57'], zorder=2)
        ax.contour(*centres, d, levels=lev, colors=orange, linewidths=.65, alpha=.8, zorder=3)
        ax.set_title(f'Step {steps[i]:,}', loc='left', weight='bold', pad=12)
    ax.contour(*centres, ref_density, levels=ref_levels, colors=blue,
               linewidths=[1., 1.25, 1.6], zorder=4)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(colors='#68798b', labelsize=8, length=3)
ratio = values[-2:][::-1] / values.sum()
for ax in axes[-1]:
    ax.set_xlabel(f'PC1 · {ratio[0]:.1%} variance', labelpad=9)
for ax in axes[:, 0]:
    ax.set_ylabel(f'PC2 · {ratio[1]:.1%} variance', labelpad=9)
fig.legend(handles=[Line2D([0], [0], color=blue, lw=1.6, label='Training data'),
                    Patch(facecolor='#f4c393', edgecolor=orange, label='Model samples')],
           loc='lower center', bbox_to_anchor=(.5, .015), frameon=False, ncol=2, fontsize=11)
fig.subplots_adjust(left=.055, right=.98, bottom=.13, top=.95, hspace=.25, wspace=.13)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(f'{PREFIX}.{ext}', dpi=240, facecolor='white')
metadata = dict(training_source=report['source'], retained_sequences=len(train),
                weights_source='models/RF00379/weights.dat', samples_source=str(OUT/'ladder_samples.h5'),
                samples_per_model=samples.shape[1], training_steps=steps.tolist(), model_ids=model_ids,
                pca='Exact weighted covariance of retained training sequences; one-hot; centered; no scaling',
                explained_variance_ratio=ratio.tolist(), density='Weighted training histogram / unweighted sample histogram, common Gaussian smoothing',
                bandwidth_pc=bandwidth, enclosed_mass=[.50,.80,.95], device='cpu')
Path(f'{PREFIX}.json').write_text(json.dumps(metadata, indent=2)+'\n')
np.savez_compressed(f'{PREFIX}_projections.npz', reference=reference, reference_weights=weights,
                    models=projected, training_steps=steps, mean=mean, components=basis, explained_variance_ratio=ratio)
print(json.dumps(metadata, indent=2))
