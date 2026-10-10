"""Paired timings of fused movement on fixed seven-model CPU PTT ladders.

Run as ``python -m tools.benchmark_numba_swaps --output results.json``.
Local sampling and energy kernels are identical in both paths. Short fitted
models exercise throughput, not convergence or mixing of mature models.
"""

import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numba
import torch

from adabmDCA import PTTConfig, PTTSampler, load_alignment, train_model
from adabmDCA.numba_kernels import is_numba_available


@contextlib.contextmanager
def swap_mode(enabled):
    previous = os.environ.get('ADABMDCA_NUMBA_SWAPS')
    os.environ['ADABMDCA_NUMBA_SWAPS'] = str(int(enabled))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('ADABMDCA_NUMBA_SWAPS', None)
        else:
            os.environ['ADABMDCA_NUMBA_SWAPS'] = previous


def make_ladder(params, tokens, n):
    anchor = {'bias': params['bias'].clone(), 'coupling_matrix': torch.zeros_like(params['coupling_matrix'])}
    sampler = PTTSampler(anchor, tokens=tokens, n_chains=n, seed=8, config=PTTConfig(max_replicas=6))
    models = [{'bias': anchor['bias'].clone(), 'coupling_matrix': params['coupling_matrix'] * (k / 6)} for k in range(7)]
    models[0] = {key: value.clone() for key, value in anchor.items()}
    sampler.models = models
    sampler.ptt_checkpoints = [{'step': k, 'params': p} for k, p in enumerate(models[:-1])]
    sampler.model_version = 6
    sampler.total_models = 7
    sampler.prepare_sampling_ladder(n)
    return sampler


def assert_same(actual, expected):
    for a, b in zip(actual.chains, expected.chains):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for key in ('lineage', 'birth', 'reached_top', '_rng'):
        torch.testing.assert_close(getattr(actual, key), getattr(expected, key), atol=0, rtol=0)
    assert actual.acceptance == expected.acceptance
    assert actual.rounds == expected.rounds and actual.local_sweeps == expected.local_sweeps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chains', type=int, default=2000)
    parser.add_argument('--sweeps', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--rna-train', type=Path, default=Path('example_data/RF00379/splits/RF00379_train.fasta'))
    parser.add_argument('--rna-validation', type=Path, default=Path('example_data/RF00379/splits/RF00379_validation.fasta'))
    parser.add_argument('--protein-train', type=Path, default=Path('example_data/cm_russ_natural/splits/cm_russ_natural.train.fasta'))
    parser.add_argument('--protein-validation', type=Path, default=Path('example_data/cm_russ_natural/splits/cm_russ_natural.test.fasta'))
    args = parser.parse_args()
    if not is_numba_available():
        parser.error('Requires enabled Numba kernels (the cpu extra).')
    torch.set_num_threads(args.threads)
    report = {'platform': platform.platform(), 'torch': torch.__version__, 'numba': numba.__version__,
              'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'order': 'reference/fused on even repetitions, fused/reference on odd', 'families': {}}
    for family, train_path, validation_path in [('RNA', args.rna_train, args.rna_validation),
                                                ('protein', args.protein_train, args.protein_validation)]:
        train, validation = load_alignment(train_path), load_alignment(validation_path)
        with contextlib.redirect_stdout(io.StringIO()):
            fitted = train_model(train, validation_path=validation, alphabet=train.tokens, device='cpu', dtype=args.dtype,
                                 n_chains=args.chains, n_sweeps=1, no_reweighting=True, seed=42,
                                 target_pearson=.999999, max_gradient_steps=3)
        length, q = fitted.model.params['bias'].shape
        entry = report['families'][family] = {'length': length, 'q': q, 'graphs': {}}
        for graph in ('dense', 'sparse'):
            params = {key: value.clone() for key, value in fitted.model.params.items()}
            if graph == 'sparse':
                degree = max(2, 2 * int(.15 * (length - 1) / 2))
                mask = torch.zeros(length, length, dtype=torch.bool)
                rows = torch.arange(length)
                for offset in range(1, degree // 2 + 1):
                    mask[rows, (rows + offset) % length] = mask[rows, (rows - offset) % length] = True
                params['coupling_matrix'] *= mask[:, None, :, None]
            initial = make_ladder(params, train.tokens, args.chains)
            entry['graphs'][graph] = {}
            for sweeps in (0, args.sweeps):
                summary = {name: {'seconds': []} for name in ('reference', 'fused')}
                entry['graphs'][graph][f'{sweeps}_sweeps'] = summary
                ladders = {name: initial.fork() for name in ('reference', 'fused')}
                for name, ladder in ladders.items():
                    if sweeps == 0:
                        # Isolate movement/exchange/profile; public sweeps stay positive.
                        ladder._replica_kernel = None
                        ladder._local_kernel = lambda chains, *a, **kw: chains
                    with swap_mode(name == 'fused'):
                        ladder.advance(rounds=args.rounds, local_sweeps=max(1, sweeps), return_samples=False)
                assert_same(ladders['fused'], ladders['reference'])
                for repeat in range(args.repeats):
                    for name in (('reference', 'fused') if repeat % 2 == 0 else ('fused', 'reference')):
                        with swap_mode(name == 'fused'):
                            start = time.perf_counter()
                            ladders[name].advance(rounds=args.rounds, local_sweeps=max(1, sweeps), return_samples=False)
                            elapsed = time.perf_counter() - start
                        summary[name]['seconds'].append(elapsed)
                    assert_same(ladders['fused'], ladders['reference'])
                for name in ('reference', 'fused'):
                    summary[name]['median_seconds'] = statistics.median(summary[name]['seconds'])
                summary['exact_trajectory_and_metadata'] = True
                summary['speedup'] = summary['reference']['median_seconds'] / summary['fused']['median_seconds']
                print(f'{family} {graph} {sweeps} sweeps: {summary["reference"]["median_seconds"]:.3f}s -> '
                      f'{summary["fused"]["median_seconds"]:.3f}s ({summary["speedup"]:.3f}x)', flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
