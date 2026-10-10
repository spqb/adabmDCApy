"""Alternate fused/reference PTT training runs to reduce timing-order bias.

Run as ``python -m tools.benchmark_ptt_swaps_training --output results.json``.
Matches benchmark_ptt_swaps: 20 evolving endpoint updates, at most 3 models,
2,000 chains, ten sweeps, validation, float32 and four CPU threads.
"""

import argparse
import contextlib
import io
import json
import platform
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch

from adabmDCA import PTTConfig, PTTSampler, load_alignment, train_model
from tools.benchmark_ptt_swaps import assert_same, state, swap_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {'platform': platform.platform(), 'torch': torch.__version__, 'repeats': args.repeats, 'steps': 20,
              'chains': 2000, 'sweeps': 10, 'order': 'reference/fused on even repetitions, fused/reference on odd', 'families': {}}
    config = PTTConfig(initialization_rounds=2, equilibration_rounds=2, mixing_chains=100,
                       mixing_initial_rounds=8, mixing_thermalization_rounds=2, mixing_max_rounds=500)
    families = [
        ('RNA', 'example_data/RF00379/splits/RF00379_train.fasta', 'example_data/RF00379/splits/RF00379_validation.fasta'),
        ('protein', 'example_data/cm_russ_natural/splits/cm_russ_natural.train.fasta', 'example_data/cm_russ_natural/splits/cm_russ_natural.test.fasta'),
    ]
    for family, train_path, val_path in families:
        train, validation = load_alignment(train_path), load_alignment(val_path)
        report['families'][family] = {}
        for graph in ('dense', 'sparse'):
            kwargs = {'alphabet': train.tokens, 'validation_path': validation, 'device': 'mps', 'dtype': 'float32',
                      'ptt': config, 'n_chains': 2000, 'n_sweeps': 10, 'no_reweighting': True,
                      'target_pearson': .999999, 'seed': 42, 'checkpoint_interval': 1}
            if graph == 'sparse':
                kwargs.update(model_type='eaDCA', activation_fraction=.0001, activation_steps=1000)
            summary = {name: {'seconds': [], 'max_models': 0} for name in ('reference', 'fused')}
            report['families'][family][graph] = summary
            original = PTTSampler.advance
            sizes = []

            def tracked(self, sizes=sizes, original=original, **kw):
                sizes.append(len(self.models))
                return original(self, **kw)

            with contextlib.redirect_stdout(io.StringIO()):
                for enabled in (False, True):
                    with swap_mode(enabled):
                        train_model(train, max_gradient_steps=2, **kwargs)
            reference = None
            for repeat in range(args.repeats):
                for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                    name = 'fused' if enabled else 'reference'
                    sizes.clear()
                    with swap_mode(enabled), patch.object(PTTSampler, 'advance', tracked), contextlib.redirect_stdout(io.StringIO()):
                        torch.mps.synchronize()
                        start = time.perf_counter()
                        result = train_model(train, max_gradient_steps=20, **kwargs)
                        torch.mps.synchronize()
                        elapsed = time.perf_counter() - start
                    assert result.gradient_steps == 20 and max(sizes) <= 3
                    numerical = (state(result.ptt_sampler), {k: v.cpu() for k, v in result.model.params.items()})
                    if reference is None:
                        reference = numerical
                    else:
                        assert_same(numerical[0], reference[0])
                        assert_same(numerical[1], reference[1])
                    summary[name]['seconds'].append(elapsed)
                    summary[name]['max_models'] = max(summary[name]['max_models'], max(sizes))
                    summary[name]['metrics'] = result.final_metrics
                    del result
                    print(f'{family} {graph} {repeat + 1} {name}: {elapsed:.3f}s', flush=True)
            for name in ('reference', 'fused'):
                summary[name]['median_seconds'] = statistics.median(summary[name]['seconds'])
            summary['exact_trajectory_and_metadata'] = True
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
