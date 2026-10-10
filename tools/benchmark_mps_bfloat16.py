"""Synthetic dense-protein BF16 coupling-storage experiment on MPS.

Run as ``python -m tools.benchmark_mps_bfloat16 --output results.json``.
The benchmark uses the production float32 and BF16 shaders. All field/probability arithmetic remains float32; only J storage
changes. Refreshed timings include casting/repacking the full master model.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from adabmDCA.mps_kernels import is_mps_available, sampling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--length', type=int, default=300)
    parser.add_argument('--states', type=int, default=21)
    parser.add_argument('--chains', type=int, default=2000)
    parser.add_argument('--sweeps', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--bias-scale', type=float, default=.3)
    parser.add_argument('--coupling-scale', type=float, default=.02)
    parser.add_argument('--methods', nargs='+', choices=sampling.METHODS, default=list(sampling.METHODS))
    args = parser.parse_args()
    if not is_mps_available():
        parser.error('Requires MPS and Metal shader compilation.')
    if not 1 <= args.length <= 4096 or not 1 <= args.states <= 32 or min(args.chains, args.sweeps, args.repeats) < 1:
        parser.error('Use supported dimensions and positive chain/sweep/repetition counts.')
    torch.set_num_threads(4)
    length, q, n = args.length, args.states, args.chains
    generator = torch.Generator().manual_seed(args.seed)
    bias = torch.randn(length, q, generator=generator) * args.bias_scale
    coupling = torch.randn(length, q, length, q, generator=generator) * args.coupling_scale
    coupling = (coupling + coupling.permute(2, 3, 0, 1)) * .5
    coupling[torch.arange(length), :, torch.arange(length), :] = 0
    states = torch.randint(q, (1, n, length), generator=generator, dtype=torch.int32).to('mps')
    bias, coupling = bias.to('mps')[None], coupling.to('mps')[None]
    rounded = coupling.bfloat16().float()
    delta = rounded - coupling
    report = {'platform': platform.platform(), 'torch': torch.__version__,
              'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'model': {'type': 'synthetic symmetric dense Potts, zero within-site blocks',
                        'fp32_coupling_bytes': coupling.numel() * 4, 'bf16_coupling_bytes': coupling.numel() * 2,
                        'relative_l2_quantization_error': float(delta.norm() / coupling.norm()),
                        'max_abs_quantization_error': float(delta.abs().max())},
              'methods': {},
              'timing': 'paired alternating order; synchronized calls; warm compilation/layouts; states reset each call'}
    print(f'Model: L={length}, q={q}, N={n}, {args.sweeps} sweeps; coupling storage '
          f'{coupling.numel()*4/1024**2:.1f} -> {coupling.numel()*2/1024**2:.1f} MiB', flush=True)
    rng = torch.mps.get_rng_state()
    names = ('fp32_cached', 'bf16_cached', 'fp32_refreshed', 'bf16_refreshed')
    for method in args.methods:
        source = coupling if method == 'metropolis' else coupling.movedim(-3, -1)
        layouts = {'fp32_cached': source.contiguous(),
                   'bf16_cached': source.to(dtype=torch.bfloat16, memory_format=torch.contiguous_format)}
        rounded_layout = layouts['bf16_cached'].float()

        def run(name, override=None, source=source, layouts=layouts, method=method):
            low = name.startswith('bf16')
            layout = override
            if layout is None:
                layout = (source.to(dtype=torch.bfloat16, memory_format=torch.contiguous_format) if low
                          else source.contiguous()) if name.endswith('refreshed') else layouts[name]
            return sampling._run(method, states.clone(), bias, layout, args.sweeps * length, 1.)

        torch.mps.set_rng_state(rng)
        quantized_reference = run('fp32_cached', rounded_layout)
        torch.mps.set_rng_state(rng)
        mixed = run('bf16_cached')
        assert torch.equal(mixed, quantized_reference), 'BF16 loads must match the same rounded couplings with FP32 loads.'
        torch.mps.set_rng_state(rng)
        full_reference = run('fp32_cached')
        changed_fraction = float((mixed != full_reference).float().mean())
        # Gauge-independent field differences: conditional logits for the same
        # configurations/site, centered across residues. Compare to a CPU FP64
        # reference because summed energy/field differences can cancel.
        cpu_x = states[0, :128].cpu().long()
        cpu_j, cpu_rounded = coupling[0].cpu().double(), rounded[0].cpu().double()
        positions = torch.arange(length)
        field_errors, fields = [], []
        for site in (0, length // 2, length - 1):
            exact = cpu_j[site, :, positions[None], cpu_x].sum(-1).T + bias[0, site].cpu().double()
            approximate = cpu_rounded[site, :, positions[None], cpu_x].sum(-1).T + bias[0, site].cpu().double()
            error = approximate - exact
            field_errors.append(error - error.mean(-1, keepdim=True))
            fields.append(exact - exact.mean(-1, keepdim=True))
        error, field = torch.cat(field_errors), torch.cat(fields)
        entry = report['methods'][method] = {
            'matches_rounded_fp32_trajectory': True,
            'final_residue_disagreement_fraction_vs_master': changed_fraction,
            'centered_conditional_logit_rms_error': float(error.square().mean().sqrt()),
            'centered_conditional_logit_relative_l2_error': float(error.norm() / field.norm()),
            'centered_conditional_logit_max_abs_error': float(error.abs().max()),
            'timings': {name: {'seconds': []} for name in names},
        }
        del cpu_j, cpu_rounded, full_reference, quantized_reference, mixed
        for name in names:
            run(name)
        for repeat in range(args.repeats):
            for name in names if repeat % 2 == 0 else names[::-1]:
                torch.mps.set_rng_state(rng)
                torch.mps.synchronize()
                start = time.perf_counter()
                run(name)
                torch.mps.synchronize()
                seconds = time.perf_counter() - start
                entry['timings'][name]['seconds'].append(seconds)
            print(f'{method}: finished paired repetition {repeat+1}/{args.repeats}', flush=True)
        for name, timing in entry['timings'].items():
            timing['median_seconds'] = statistics.median(timing['seconds'])
        entry['cached_speedup'] = entry['timings']['fp32_cached']['median_seconds'] / entry['timings']['bf16_cached']['median_seconds']
        entry['refreshed_speedup'] = entry['timings']['fp32_refreshed']['median_seconds'] / entry['timings']['bf16_refreshed']['median_seconds']
        print(f'{method}: cached {entry["cached_speedup"]:.3f}x, refreshed {entry["refreshed_speedup"]:.3f}x', flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
