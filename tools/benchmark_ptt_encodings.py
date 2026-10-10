"""Compare PTT before/after redundant encoding removal, with exact trajectories.

Run as ``python -m tools.benchmark_ptt_encodings --output results.json``.
Uses the same dense bmDCA and sparse eaDCA cases as benchmark_mps_training.
The previous path is emulated locally; Metal kernels and optimizer are identical.
"""

from __future__ import annotations

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
from adabmDCA.ptt import sampler as sampler_module
from adabmDCA.sampling import sampling_profile


@contextlib.contextmanager
def encoding_path(previous, counts):
    advance = PTTSampler.advance
    endpoint = PTTSampler.endpoint_samples
    profile = sampler_module._profile_states
    one_hot = sampler_module._one_hot

    def counted_hot(chains, params):
        counts['endpoint_and_energy_encodings'] += int(chains.ndim == 2)
        return one_hot(chains, params)

    def legacy_endpoint(self, *, copy=True):
        samples = counted_hot(self.chains[-1], self.models[-1])
        return samples.clone() if copy else samples

    def legacy_advance(self, **kwargs):
        kwargs['return_samples'] = True
        return advance(self, **kwargs)

    def profile_draw(params, n):
        counts['profile_draws'] += 1
        if previous and params['bias'].device.type != 'cpu':
            counts['profile_encodings'] += 1
            return sampling_profile(params, n, 1.).argmax(-1).to(torch.int32)
        return profile(params, n)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(sampler_module, '_one_hot', counted_hot))
        stack.enter_context(patch.object(sampler_module, '_profile_states', profile_draw))
        stack.enter_context(patch.object(PTTSampler, 'endpoint_samples', legacy_endpoint if previous else endpoint))
        if previous:
            stack.enter_context(patch.object(PTTSampler, 'advance', legacy_advance))
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {'platform': platform.platform(), 'torch': torch.__version__, 'steps': args.steps,
              'chains': 2000, 'sweeps': 10, 'repeats': args.repeats, 'families': {}}
    ptt = PTTConfig(initialization_rounds=2, equilibration_rounds=2, mixing_chains=100,
                    mixing_initial_rounds=8, mixing_thermalization_rounds=2, mixing_max_rounds=500)
    families = [
        ('RNA', 'example_data/RF00379/splits/RF00379_train.fasta', 'example_data/RF00379/splits/RF00379_validation.fasta'),
        ('protein', 'example_data/cm_russ_natural/splits/cm_russ_natural.train.fasta', 'example_data/cm_russ_natural/splits/cm_russ_natural.test.fasta'),
    ]
    for family, train_path, val_path in families:
        train, validation = load_alignment(train_path), load_alignment(val_path)
        entry = report['families'][family] = {}
        for graph in ('dense', 'sparse'):
            entry[graph] = {}
            reference = None
            options = {'alphabet': train.tokens, 'validation_path': validation, 'dtype': 'float32', 'device': 'mps',
                       'ptt': ptt, 'seed': 42, 'no_reweighting': True, 'n_chains': 2000, 'n_sweeps': 10,
                       'target_pearson': .999999, 'checkpoint_interval': 1}
            if graph == 'sparse':
                options.update(model_type='eaDCA', activation_fraction=.0001, activation_steps=1000)
            for previous in (True, False):
                name = 'previous' if previous else 'reused_encodings'
                counts = {'endpoint_and_energy_encodings': 0, 'profile_draws': 0, 'profile_encodings': 0}
                timings = []
                with encoding_path(previous, counts), contextlib.redirect_stdout(io.StringIO()):
                    train_model(train, max_gradient_steps=2, **options)
                    for _ in range(args.repeats):
                        for key in counts:
                            counts[key] = 0
                        torch.mps.synchronize()
                        start = time.perf_counter()
                        result = train_model(train, max_gradient_steps=args.steps, **options)
                        torch.mps.synchronize()
                        timings.append(time.perf_counter() - start)
                        assert result.gradient_steps == args.steps
                numerical = ({k: v.cpu().clone() for k, v in result.model.params.items()},
                             result.chains.cpu().clone(), result.ptt_sampler._rng.clone())
                if previous:
                    reference = numerical
                else:
                    for k in numerical[0]:
                        torch.testing.assert_close(numerical[0][k], reference[0][k], atol=0, rtol=0)
                    torch.testing.assert_close(numerical[1], reference[1], atol=0, rtol=0)
                    assert torch.equal(numerical[2], reference[2])
                    entry[graph]['exact_trajectory_and_rng'] = True
                entry[graph][name] = {'median_seconds': statistics.median(timings), 'seconds': timings,
                                     'last_run_counts': dict(counts), 'metrics': result.final_metrics}
                print(f'{family} {graph} {name}: {statistics.median(timings):.3f}s; {counts}', flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
