from pathlib import Path
import json
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

def read_fasta(path):
    seqs = []
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            seqs.append('')
        elif line.strip():
            seqs[-1] += line.strip()
    return seqs

alphabet = '-ACGU'
def encode(seqs):
    assert len(set(map(len, seqs))) == 1
    return np.array([[alphabet.index(c) for c in s] for s in seqs])

def onehot(x):
    return np.eye(5, dtype=np.float64)[x].reshape(len(x), -1)

original_train = encode(read_fasta(ROOT / 'example_data/RF00379/splits/RF00379_train.fasta'))
report = json.loads((ROOT / 'models/RF00739/training.json').read_text())['data']['input_report']
train = original_train[np.asarray(report['retained_indices'])]
weights_path = ROOT / 'models/RF00739/weights.dat'
weights = np.loadtxt(weights_path)
assert weights.shape == (len(train),)
assert np.all(np.isfinite(weights)) and np.all(weights >= 0) and weights.sum() > 0
samples = encode(read_fasta(ROOT / 'models/RF00739/sampling/samples.fasta'))
assert train.shape[1] == samples.shape[1]
rng = np.random.default_rng(42)
random = rng.integers(0, 5, size=(2000, train.shape[1]))
resampled_indices = np.random.default_rng(43).choice(len(train), size=10000, replace=True, p=weights / weights.sum())
x = onehot(train[resampled_indices])
mean = x.mean(axis=0)
x -= mean
values, vectors = np.linalg.eigh(x.T @ x / (len(x)-1))
basis = vectors[:, -2:][:, ::-1]
# Fix arbitrary eigenvector signs for reproducibility.
for i in range(2):
    if basis[np.argmax(np.abs(basis[:, i])), i] < 0:
        basis[:, i] *= -1
training_pc = x @ basis
random_pc = (onehot(random) - mean) @ basis
samples_pc = (onehot(samples) - mean) @ basis
all_pc = np.vstack([training_pc, random_pc, samples_pc])
lo, hi = all_pc.min(axis=0), all_pc.max(axis=0)
span = hi - lo
# Compress repeated draws without changing the empirical resampled density.
unique_pc, counts = np.unique(training_pc, axis=0, return_counts=True)
kde = gaussian_kde(unique_pc.T, weights=counts, bw_method=len(training_pc) ** (-1 / 6))
# Extend the grid until the outermost contour is strictly inside every edge.
for margin in [.2, .35, .6, 1.0, 1.5]:
    lo = all_pc.min(axis=0) - span * margin
    hi = all_pc.max(axis=0) + span * margin
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 300), np.linspace(lo[1], hi[1], 300))
    density = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    levels = density.max() * np.array([.025, .065, .13, .23, .38, .56, .77, 1.01])
    edge_max = max(density[0].max(), density[-1].max(), density[:, 0].max(), density[:, -1].max())
    if edge_max < levels[0] * .5:
        break
assert edge_max < levels[0], 'Contour intersects image boundary'
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
for ax, points, color in zip(axes, [random_pc, samples_pc], ['#e58232', '#e58232']):
    ax.contourf(gx, gy, density, levels=levels, cmap='Greys', alpha=.55, zorder=0)
    ax.contour(gx, gy, density, levels=levels[:-1], colors='#7e858b', linewidths=.65, alpha=.65, zorder=1)
    ax.scatter(points[:, 0], points[:, 1], s=7, c=color, alpha=.48, linewidths=0, zorder=2, rasterized=True)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
fig.subplots_adjust(left=.015, right=.985, bottom=.025, top=.975, wspace=.055)
for ext in ['png', 'svg']:
    fig.savefig(OUT / f'rf00379_pca_random_vs_model.{ext}', dpi=300, facecolor='white')
metadata = {'train_count': len(train), 'model_samples_count': len(samples), 'random_count': len(random), 'sequence_length': train.shape[1], 'samples_source': str(ROOT / 'models/RF00739/sampling/samples.fasta'), 'contour_edge_below_lowest_level': bool(edge_max < levels[0]), 'random_seed': 42, 'random_distribution': 'uniform over -ACGU', 'PCA': 'training resampled with replacement according to weights; one-hot, centered, no scaling', 'weights_source': str(weights_path), 'original_train_count': len(original_train), 'resampled_train_count': len(resampled_indices), 'resampling_seed': 43, 'contour': 'KDE of weighted resampled training', 'explained_variance_ratio': (values[-2:][::-1] / values.sum()).tolist()}
(OUT / 'rf00379_pca_random_vs_model.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata))
