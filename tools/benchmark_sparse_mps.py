"""Compare sparse Metal, dense Metal and Numba on graphs based on real families.

Train a short dense PCD model on each example alignment, then retain symmetric
site blocks on regular graphs of controlled degree. These are throughput
measurements, not converged sparse models. Timings include adapters and, for
changed_model, rebuilding the graph and packed blocks after a tensor update.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
from pathlib import Path

import torch

from adabmDCA import load_alignment, train_model
from adabmDCA.mps_kernels.sparse import sparse_coupling_layout
from adabmDCA.ptt.kernels import _prepare_categorical_sampler, _prepare_exchange_kernel
from tools.benchmark_mps import measure


@contextlib.contextmanager
def sparse_mode(enabled):
    previous = os.environ.get('ADABMDCA_SPARSE')
    os.environ['ADABMDCA_SPARSE'] = '1' if enabled else '0'
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('ADABMDCA_SPARSE', None)
        else:
            os.environ['ADABMDCA_SPARSE'] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chains', type=int, default=2000)
    parser.add_argument('--sweeps', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--rna-train', type=Path, default=Path('example_data/RF00379/splits/RF00379_train.fasta'))
    parser.add_argument('--protein-train', type=Path, default=Path('example_data/cm_russ_natural/splits/cm_russ_natural.train.fasta'))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = {'platform': platform.platform(), 'torch': torch.__version__,
              'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'families': {}}
    for family, path in [('RF00379', args.rna_train), ('cm_russ_natural', args.protein_train)]:
        train = load_alignment(path)
        with contextlib.redirect_stdout(io.StringIO()):
            fitted = train_model(train, alphabet=train.tokens, device='cpu', max_gradient_steps=3,
                                 n_chains=2000, n_sweeps=1, no_reweighting=True, target_pearson=0.999999, seed=42)
        params = fitted.model.params
        length, q = params['bias'].shape
        rows = torch.arange(length)
        generator = torch.Generator().manual_seed(12)
        initial = torch.randint(q, (args.chains, length), generator=generator, dtype=torch.int32)
        family_result = {'length': length, 'q': q, 'graphs': {}}
        report['families'][family] = family_result
        for fraction in (0.05, 0.15, 0.30, 0.50):
            degree = max(2, 2 * int(fraction * (length - 1) / 2))
            mask = torch.zeros(length, length, dtype=torch.bool)
            for offset in range(1, degree // 2 + 1):
                mask[rows, (rows + offset) % length] = True
                mask[rows, (rows - offset) % length] = True
            graph = {k: v.clone() for k, v in params.items()}
            graph['coupling_matrix'] *= mask[:, None, :, None]
            entry = {'degree': degree, 'pair_density': degree / (length - 1), 'sampling': {}, 'exchange': {}}
            family_result['graphs'][str(fraction)] = entry
            print(f'{family}: degree {degree}, density {entry["pair_density"]:.3f}', flush=True)
            for name in ('numba_cpu', 'dense_mps', 'sparse_mps'):
                device = torch.device('cpu' if name == 'numba_cpu' else 'mps')
                with sparse_mode(name != 'dense_mps'):
                    p = {k: v.to(device) for k, v in graph.items()}
                    x = initial.to(device)
                    if name == 'sparse_mps':
                        entry['sparse_selected'] = sparse_coupling_layout(p['coupling_matrix'][None], source=p['coupling_matrix']) is not None
                    for method in ('gibbs', 'metropolis', 'metropolized_gibbs'):
                        sampler = _prepare_categorical_sampler(method, device)
                        timing = measure(lambda sampler=sampler, x=x, p=p: sampler(x, p, args.sweeps), device, args.repeats)

                        def changed_model(sampler=sampler, x=x, p=p):
                            p['coupling_matrix'].add_(0)
                            return sampler(x, p, args.sweeps)

                        timing['changed_model'] = measure(changed_model, device, args.repeats)
                        entry['sampling'].setdefault(method, {})[name] = timing
                    lower = {k: 0.9 * v for k, v in p.items()}
                    exchange = _prepare_exchange_kernel(device)
                    entry['exchange'][name] = measure(lambda exchange=exchange, lower=lower, p=p, x=x:
                                                     exchange(lower, p, x, x.flip(0)), device, args.repeats)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
