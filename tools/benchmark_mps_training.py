"""Compare CPU, previous MPS and tuned MPS for dense and sparse PCD/PTT training.

Run as ``python -m tools.benchmark_mps_training --output results.json``.
Sparse runs continue a short CPU-trained model on a fixed regular graph with
roughly 15% pair density for PCD. Sparse PTT uses eaDCA with one activation
block of 0.01% of coupling elements. All runs include validation and setup. These are
throughput checks over equal numbers of updates, not convergence benchmarks.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch

from adabmDCA import PTTConfig, load_alignment, train_model
from adabmDCA.io import save_params
from adabmDCA.mps_kernels import sampling as metal
from adabmDCA.ptt.optim import _PTTOptimizer
from tools.benchmark_mps import measure, sync


def gather_scores(samples, bias, coupling, chunk_size=256):
    """Previous directional-score implementation, retained for comparison."""
    states = samples.argmax(-1)
    length = states.shape[1]
    sites = torch.arange(length, device=states.device)
    scores_h, scores_j = [], []
    for batch in states.split(chunk_size):
        scores_h.append(bias[sites[None], batch].sum(1))
        scores_j.append(.5 * coupling[sites[None, :, None], batch[:, :, None],
                                     sites[None, None, :], batch[:, None, :]].sum((1, 2)))
    return torch.cat(scores_h), torch.cat(scores_j)


@contextlib.contextmanager
def implementation(name):
    with contextlib.ExitStack() as stack:
        if name == 'previous_mps':
            stack.enter_context(patch.object(metal, '_steps_per_launch', lambda *a: 512))
            stack.enter_context(patch.object(_PTTOptimizer, '_directional_score_components', staticmethod(gather_scores)))
        yield torch.device('cpu' if name == 'numba_cpu' else 'mps')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--chains', type=int, default=2000)
    parser.add_argument('--sweeps', type=int, default=10)
    parser.add_argument('--resume', action='store_true', help='Keep completed cases from the same output/settings.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    ptt = PTTConfig(initialization_rounds=2, equilibration_rounds=2, mixing_chains=100,
                    mixing_initial_rounds=8, mixing_thermalization_rounds=2, mixing_max_rounds=500)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'}
    report = {'platform': platform.platform(), 'torch': torch.__version__, 'settings': settings, 'families': {}}
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text())
        if report['settings'] != settings or report['torch'] != torch.__version__:
            parser.error('Resume requires matching settings and PyTorch version.')
    families = [
        ('RNA', 'example_data/RF00379/splits/RF00379_train.fasta', 'example_data/RF00379/splits/RF00379_validation.fasta'),
        ('protein', 'example_data/cm_russ_natural/splits/cm_russ_natural.train.fasta', 'example_data/cm_russ_natural/splits/cm_russ_natural.test.fasta'),
    ]
    with tempfile.TemporaryDirectory() as directory:
        for family, train_path, validation_path in families:
            train, validation = load_alignment(train_path), load_alignment(validation_path)
            options = {'alphabet': train.tokens, 'validation_path': validation, 'dtype': 'float32',
                       'no_reweighting': True, 'seed': 42, 'n_chains': args.chains, 'n_sweeps': args.sweeps,
                       'target_pearson': 0.999999, 'checkpoint_interval': 1}
            with contextlib.redirect_stdout(io.StringIO()):
                fitted = train_model(train, alphabet=train.tokens, device='cpu', max_gradient_steps=3,
                                     n_chains=2000, n_sweeps=1, no_reweighting=True, seed=42, target_pearson=0.999999)
            length, q = fitted.model.params['bias'].shape
            entry = report['families'].setdefault(family, {'length': length, 'q': q, 'training': {}, 'directional_scores': {}})
            entry['sparse_ptt'] = {'model_type': 'eaDCA', 'activation_fraction': .0001, 'activation_steps': 1000}
            degree = max(2, 2 * int(.15 * (length - 1) / 2))
            mask = torch.zeros(length, length, dtype=torch.bool)
            rows = torch.arange(length)
            for offset in range(1, degree // 2 + 1):
                mask[rows, (rows + offset) % length] = True
                mask[rows, (rows - offset) % length] = True
            params = {k: v.clone() for k, v in fitted.model.params.items()}
            params['coupling_matrix'] *= mask[:, None, :, None]
            sparse_path = Path(directory) / f'{family}.dat.gz'
            save_params(str(sparse_path), params, train.tokens, mask=mask[:, None, :, None].expand_as(params['coupling_matrix']))
            entry['sparse_initial_degree'] = degree
            for graph in ('dense', 'sparse'):
                for algorithm in ('PCD', 'PTT'):
                    case = f'{graph}_{algorithm}'
                    results = entry['training'].setdefault(case, {})
                    for name in ('numba_cpu', 'previous_mps', 'tuned_mps'):
                        if name in results:
                            continue
                        print(f'{family} {case} {name}', flush=True)
                        timings = []
                        with implementation(name) as device, contextlib.redirect_stdout(io.StringIO()):
                            kwargs = {**options, 'device': str(device), 'ptt': ptt if algorithm == 'PTT' else None,
                                      'initial_params_path': sparse_path if graph == 'sparse' and algorithm == 'PCD' else None}
                            if graph == 'sparse' and algorithm == 'PTT':
                                kwargs.update(entry['sparse_ptt'])
                            train_model(train, max_gradient_steps=2, **kwargs)
                            for _ in range(args.repeats):
                                sync(device)
                                started = time.perf_counter()
                                result = train_model(train, max_gradient_steps=args.steps, **kwargs)
                                sync(device)
                                timings.append(time.perf_counter() - started)
                                assert result.gradient_steps == args.steps
                                assert all(bool(torch.isfinite(v).all()) for v in result.model.params.values())
                            coupling = result.model.params['coupling_matrix']
                            pair_graph = (coupling != 0).any(-1).any(-2)
                            if graph == 'sparse':
                                assert int(pair_graph.sum(-1).max()) <= .35 * length
                            entry['training'][case][name] = {'median_seconds': statistics.median(timings), 'seconds': timings,
                                                            'gradient_steps': result.gradient_steps, 'metrics': result.final_metrics,
                                                            'max_degree': int(pair_graph.sum(-1).max())}
                        print(f'  {statistics.median(timings):.3f}s', flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
            population = fitted.chains.to('mps')
            h = fitted.model.params['bias'].to('mps') * .01
            j = fitted.model.params['coupling_matrix'].to('mps') * .01
            for name in ('previous_mps', 'tuned_mps'):
                with implementation(name) as device:
                    entry['directional_scores'][name] = measure(
                        lambda population=population, h=h, j=j:
                        _PTTOptimizer._directional_score_components(population, h, j), device, 10)
            args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
