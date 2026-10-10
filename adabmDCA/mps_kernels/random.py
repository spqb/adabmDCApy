"""Philox4x32-10 shader code and keys drawn from the PyTorch MPS RNG.

The counter identifies (chain, step, replica); keys and random sites consume
PyTorch's MPS stream, so saving/restoring that stream reproduces sampling.
"""

import torch


def draw_seed(device: torch.device) -> torch.Tensor:
    """Draw the two-word Philox key without synchronizing with the CPU."""
    return torch.randint(0, 2**31 - 1, (2,), device=device, dtype=torch.int32)


SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Philox4x32-10: independent counter (chain, step, replica), keyed by a seed
// drawn from the PyTorch MPS generator. No per-update random buffer is needed.
inline uint4 philox(uint4 counter, uint2 key) {
    for (uint round = 0; round < 10; ++round) {
        const ulong p0 = ulong(0xD2511F53u) * counter.x;
        const ulong p1 = ulong(0xCD9E8D57u) * counter.z;
        counter = uint4(uint(p1 >> 32) ^ counter.y ^ key.x, uint(p1),
                        uint(p0 >> 32) ^ counter.w ^ key.y, uint(p0));
        key += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return counter;
}

"""
