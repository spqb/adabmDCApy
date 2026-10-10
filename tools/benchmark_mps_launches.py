"""Measure dispatch sizes for all Metal samplers on dense and sparse graphs.

Run as ``python -m tools.benchmark_mps_launches --output results.json``.
Graphs are regular masks of a short CPU-trained model, not converged models.
The temporary dispatch override is confined to this benchmark process.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
from pathlib import Path
from unittest.mock import patch

import torch

from adabmDCA import load_alignment, train_model
from adabmDCA.mps_kernels import sampling as metal
from adabmDCA.mps_kernels.sparse import sparse_coupling_layout
from tools.benchmark_mps import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--sweeps', type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {'platform': platform.platform(), 'torch': torch.__version__,
              'sweeps': args.sweeps, 'repeats': args.repeats, 'families': {}}
    for family, path in [('RNA', 'example_data/RF00379/splits/RF00379_train.fasta'),
                         ('protein', 'example_data/cm_russ_natural/splits/cm_russ_natural.train.fasta')]:
        train = load_alignment(path)
        with contextlib.redirect_stdout(io.StringIO()):
            result = train_model(train, alphabet=train.tokens, device='cpu', max_gradient_steps=3,
                                 n_chains=2000, n_sweeps=1, no_reweighting=True, seed=42,
                                 target_pearson=0.999999)
        length, q = result.model.params['bias'].shape
        report['families'][family] = {'length': length, 'q': q, 'graphs': {}}
        rows = torch.arange(length)
        for fraction in (1., .15, .30):
            degree = 2 * int(fraction * (length - 1) / 2)
            mask = torch.ones(length, length, dtype=torch.bool) if fraction == 1. else torch.zeros(length, length, dtype=torch.bool)
            if fraction != 1.:
                for offset in range(1, degree // 2 + 1):
                    mask[rows, (rows + offset) % length] = True
                    mask[rows, (rows - offset) % length] = True
            params = {k: v.clone().to('mps') for k, v in result.model.params.items()}
            params['coupling_matrix'] *= mask[:, None, :, None].to('mps')
            sparse = sparse_coupling_layout(params['coupling_matrix'][None], source=params['coupling_matrix'])
            entry = {'sparse_selected': sparse is not None, 'degree': degree, 'populations': {}}
            report['families'][family]['graphs'][str(fraction)] = entry
            for n in (128, 2000):
                x = torch.randint(q, (n, length), generator=torch.Generator().manual_seed(4), dtype=torch.int32).to('mps')
                measurements = {}
                entry['populations'][str(n)] = measurements
                for method in metal.METHODS:
                    measurements[method] = {}
                    for chunk in (64, 128, 256, 512):
                        # Before tuning _run used the constant directly; afterwards
                        # override the selector to reproduce each fixed size.
                        target = '_steps_per_launch' if hasattr(metal, '_steps_per_launch') else '_STEPS_PER_LAUNCH'
                        replacement = (lambda *a, chunk=chunk: chunk) if target == '_steps_per_launch' else chunk
                        with patch.object(metal, target, replacement):
                            measurements[method][str(chunk)] = measure(
                                lambda method=method, x=x, params=params: metal.sample_categorical(method, x, params, args.sweeps),
                                torch.device('mps'), args.repeats)
                best = {m: min(v, key=lambda c: v[c]['median_seconds']) for m, v in measurements.items()}
                print(f'{family} density={fraction} chains={n}: best chunks {best}', flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
