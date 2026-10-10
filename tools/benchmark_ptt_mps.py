"""Benchmark full PTT training and warmed ladder advances on the example splits.

Run from the repository root. Requires the cpu extra and a Metal-capable PyTorch.
Short runs validate execution and throughput; they do not establish convergence.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from adabmDCA import PTTConfig, PTTSampler, load_alignment, train_model
from adabmDCA.api.ptt import sample_ptt_sequences
from adabmDCA.mps_kernels import is_mps_available
from adabmDCA.numba_kernels import is_numba_available
from tools.benchmark_mps import backend, measure, sync


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--chains', type=int, default=2000)
    parser.add_argument('--sweeps', type=int, default=10)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--rna-train', type=Path, default=Path('example_data/RF00379/splits/RF00379_train.fasta'))
    parser.add_argument('--rna-validation', type=Path, default=Path('example_data/RF00379/splits/RF00379_validation.fasta'))
    parser.add_argument('--protein-train', type=Path, default=Path('example_data/cm_russ_natural/splits/cm_russ_natural.train.fasta'))
    parser.add_argument('--protein-validation', type=Path, default=Path('example_data/cm_russ_natural/splits/cm_russ_natural.test.fasta'))
    args = parser.parse_args()
    if not is_mps_available() or not is_numba_available():
        parser.error('Requires Metal shaders and Numba.')
    torch.set_num_threads(args.threads)
    config = PTTConfig(initialization_rounds=2, equilibration_rounds=2, mixing_chains=100,
                       mixing_initial_rounds=8, mixing_thermalization_rounds=2, mixing_max_rounds=500)
    report = {'platform': platform.platform(), 'torch': torch.__version__,
              'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'ptt_config': asdict(config), 'families': {}}
    for family, train_path, validation_path in [('RF00379', args.rna_train, args.rna_validation),
                                               ('cm_russ_natural', args.protein_train, args.protein_validation)]:
        train, validation = load_alignment(train_path), load_alignment(validation_path)
        options = {'alphabet': train.tokens, 'validation_path': validation, 'dtype': 'float32', 'ptt': config,
                       'no_reweighting': True, 'seed': 42, 'n_chains': args.chains, 'n_sweeps': args.sweeps,
                       'target_pearson': 0.999999, 'checkpoint_interval': 1}
        summary = {'length': train.sequence_length, 'q': len(train.tokens), 'training': {}, 'advance': {}, 'generation': {}}
        report['families'][family] = summary
        models = None
        for name in ['numba_cpu', 'metal_mps']:
            print(f'{family}: training {name}', flush=True)
            with backend(name) as device, contextlib.redirect_stdout(io.StringIO()):
                train_model(train, device=str(device), max_gradient_steps=2, **options)
                sync(device)
                start = time.perf_counter()
                result = train_model(train, device=str(device), max_gradient_steps=args.steps, **options)
                sync(device)
                elapsed = time.perf_counter() - start
                assert result.gradient_steps == args.steps
                assert all(torch.isfinite(value).all() for value in result.model.params.values())
                summary['training'][name] = {'seconds': elapsed, 'gradient_steps': result.gradient_steps,
                                             'metrics': result.final_metrics, 'replicas': len(result.ptt_sampler.models)}
                if name == 'numba_cpu':
                    models = [{k: v.clone() for k, v in p.items()} for p in result.ptt_sampler.models]
                sync(device)
                start = time.perf_counter()
                sampled = sample_ptt_sequences(source=result.ptt_sampler, n_sequences=args.chains,
                                               local_sweeps=args.sweeps, max_rounds=500, seed=11,
                                               reference_fasta=train, no_reweighting=True)
                sync(device)
                summary['generation'][name] = {'seconds': time.perf_counter() - start,
                                               'sequences': len(sampled.sequences), 'diagnostics': sampled.ptt_diagnostics}
            print(f'  training {elapsed:.3f}s, {result.gradient_steps} steps', flush=True)
        # Identical CPU-trained Hamiltonians, three replicas to exercise stacking.
        # Duplicate the endpoint if the short training kept only two rungs.
        if len(models) < 3:
            models.append({k: v.clone() for k, v in models[-1].items()})
        for name in ['numba_cpu', 'metal_mps']:
            print(f'{family}: fixed-ladder advance {name}', flush=True)
            with backend(name) as device:
                ladder = PTTSampler({k: v.to(device) for k, v in models[0].items()}, tokens=train.tokens,
                                    n_chains=args.chains, seed=9)
                ladder.models = [{k: v.to(device) for k, v in p.items()} for p in models]
                ladder.chains = [ladder.chains[0].clone() for _ in models]
                ladder._reset_lineage()
                ladder.acceptance = [1.] * (len(models) - 1)
                summary['advance'][name] = measure(lambda ladder=ladder: ladder.advance(rounds=5, local_sweeps=args.sweeps),
                                                   device, args.repeats)
                summary['advance'][name]['replicas'] = len(models)
                summary['advance'][name]['log_z'] = ladder.partition_estimate().log_z
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
