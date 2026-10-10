"""Separate evolving 2/3-model PTT training from fixed 7-model sampling.

Run as ``python -m tools.benchmark_ptt_swaps --output results.json``.
Uses RNA/protein train and validation splits. Fixed ladders interpolate a
short CPU-trained Hamiltonian; sparse cases retain regular 15% pair graphs.
This measures equal-work throughput, not convergence or mature-model mixing.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch

from adabmDCA import PTTConfig, PTTSampler, load_alignment, train_model
from tools.benchmark_mps import measure


@contextlib.contextmanager
def swap_mode(enabled):
    previous = os.environ.get('ADABMDCA_MPS_SWAPS')
    os.environ['ADABMDCA_MPS_SWAPS'] = str(int(enabled))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('ADABMDCA_MPS_SWAPS', None)
        else:
            os.environ['ADABMDCA_MPS_SWAPS'] = previous


def make_ladder(params, tokens, n, replicas):
    anchor = {'bias': params['bias'].to('mps'), 'coupling_matrix': torch.zeros_like(params['coupling_matrix'], device='mps')}
    sampler = PTTSampler(anchor, tokens=tokens, n_chains=n, seed=8,
                         config=PTTConfig(max_replicas=max(2, replicas - 1)))
    sampler.models = [{'bias': anchor['bias'].clone(), 'coupling_matrix': params['coupling_matrix'].to('mps') * (k / (replicas - 1))}
                      for k in range(replicas)]
    sampler.models[0] = {key: value.clone() for key, value in anchor.items()}
    sampler.ptt_checkpoints = [{'step': k, 'params': p} for k, p in enumerate(sampler.models[:-1])]
    sampler.model_version = replicas - 1
    sampler.total_models = replicas
    # This is the public operation used to build a frozen generation ladder.
    sampler.prepare_sampling_ladder(n)
    if replicas == 3:
        sampler.mode = 'train'
        sampler.birth = sampler.reached_top = None
        sampler.lag_memory = torch.zeros(replicas, n, device='mps')
    return sampler


def state(sampler):
    return {'chains': torch.stack(sampler.chains).cpu(), 'lineage': sampler.lineage.cpu(),
            'birth': None if sampler.birth is None else sampler.birth.cpu(),
            'reached': None if sampler.reached_top is None else sampler.reached_top.cpu(),
            'lag': None if sampler.lag_memory is None else sampler.lag_memory.cpu(),
            'rng': sampler._rng.clone(), 'acceptance': list(sampler.acceptance)}


def assert_same(actual, expected):
    for key, value in actual.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
        else:
            assert value == expected[key], key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chains', type=int, default=2000)
    parser.add_argument('--sweeps', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {'platform': platform.platform(), 'torch': torch.__version__,
              'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'families': {}}
    config = PTTConfig(initialization_rounds=2, equilibration_rounds=2, mixing_chains=100,
                       mixing_initial_rounds=8, mixing_thermalization_rounds=2, mixing_max_rounds=500)
    families = [
        ('RNA', 'example_data/RF00379/splits/RF00379_train.fasta', 'example_data/RF00379/splits/RF00379_validation.fasta'),
        ('protein', 'example_data/cm_russ_natural/splits/cm_russ_natural.train.fasta', 'example_data/cm_russ_natural/splits/cm_russ_natural.test.fasta'),
    ]
    for family, train_path, val_path in families:
        train, validation = load_alignment(train_path), load_alignment(val_path)
        entry = report['families'][family] = {'training': {}, 'ladders': {}}
        with contextlib.redirect_stdout(io.StringIO()):
            fitted = train_model(train, alphabet=train.tokens, device='cpu', n_chains=args.chains, n_sweeps=1,
                                 no_reweighting=True, seed=42, target_pearson=.999999, max_gradient_steps=3)
        length, q = fitted.model.params['bias'].shape
        entry.update(length=length, q=q)
        for graph in ('dense', 'sparse'):
            entry['training'][graph] = {}
            reference = None
            for enabled in (False, True):
                name = 'fused' if enabled else 'reference'
                options = {'alphabet': train.tokens, 'validation_path': validation, 'device': 'mps', 'dtype': 'float32',
                           'ptt': config, 'n_chains': args.chains, 'n_sweeps': args.sweeps, 'no_reweighting': True,
                           'target_pearson': .999999, 'seed': 42, 'checkpoint_interval': 1}
                if graph == 'sparse':
                    options.update(model_type='eaDCA', activation_fraction=.0001, activation_steps=1000)
                timings, sizes = [], []
                original = PTTSampler.advance

                def tracked(self, sizes=sizes, original=original, **kwargs):
                    sizes.append(len(self.models))
                    return original(self, **kwargs)

                with swap_mode(enabled), patch.object(PTTSampler, 'advance', tracked), contextlib.redirect_stdout(io.StringIO()):
                    train_model(train, max_gradient_steps=2, **options)
                    for _ in range(args.repeats):
                        sizes.clear()
                        torch.mps.synchronize()
                        start = time.perf_counter()
                        result = train_model(train, max_gradient_steps=args.steps, **options)
                        torch.mps.synchronize()
                        timings.append(time.perf_counter() - start)
                        assert result.gradient_steps == args.steps and max(sizes) <= 3
                numerical = (state(result.ptt_sampler), {k: v.cpu() for k, v in result.model.params.items()})
                if reference is None:
                    reference = numerical
                else:
                    assert_same(numerical[0], reference[0])
                    assert_same(numerical[1], reference[1])
                entry['training'][graph][name] = {'median_seconds': statistics.median(timings), 'seconds': timings,
                                                 'max_models': max(sizes), 'metrics': result.final_metrics}
                print(f'{family} {graph} training {name}: {statistics.median(timings):.3f}s, max {max(sizes)} models', flush=True)
            params = {k: v.clone() for k, v in fitted.model.params.items()}
            if graph == 'sparse':
                degree = max(2, 2 * int(.15 * (length - 1) / 2))
                mask = torch.zeros(length, length, dtype=torch.bool)
                rows = torch.arange(length)
                for offset in range(1, degree // 2 + 1):
                    mask[rows, (rows + offset) % length] = mask[rows, (rows - offset) % length] = True
                params['coupling_matrix'] *= mask[:, None, :, None]
            entry['ladders'][graph] = {}
            for replicas in (3, 7):
                for sweeps in (0, args.sweeps):
                    case = f'{"evolving_training" if replicas == 3 else "fixed_generation"}_{sweeps}_sweeps'
                    measurements = entry['ladders'][graph][case] = {}
                    initial = make_ladder(params, train.tokens, args.chains, replicas)
                    reference_state = None
                    for enabled in (False, True):
                        ladder = initial.fork()
                        if sweeps == 0:
                            # Isolate swaps/permutations and profile redraws;
                            # the public advance API requires positive sweeps.
                            ladder._replica_kernel = None
                            ladder._local_kernel = lambda chains, *a, **kw: chains

                        def advance(ladder=ladder, replicas=replicas, sweeps=sweeps):
                            if replicas == 3:
                                for _ in range(args.rounds):
                                    # In-place endpoint changes force layout invalidation.
                                    ladder.models[-1]['bias'].add_(.0001)
                                    ladder.models[-1]['coupling_matrix'].mul_(1.0001)
                                    ladder.model_version += 1
                                    ladder.advance(rounds=1, local_sweeps=max(1, sweeps), return_samples=False)
                            else:
                                ladder.advance(rounds=args.rounds, local_sweeps=max(1, sweeps), return_samples=False)

                        with swap_mode(enabled):
                            timing = measure(advance, torch.device('mps'), args.repeats)
                        name = 'fused' if enabled else 'reference'
                        measurements[name] = timing
                        numerical = state(ladder)
                        if reference_state is None:
                            reference_state = numerical
                        else:
                            assert_same(numerical, reference_state)
                            measurements['exact_trajectory_and_metadata'] = True
                        print(f'{family} {graph} {case} {name}: {timing["median_seconds"]:.3f}s', flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
