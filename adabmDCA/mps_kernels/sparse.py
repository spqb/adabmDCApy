"""Cached neighbour tables shared by Metal sampling and replica exchange.

Every nonzero site block is retained, including partly activated eaDCA blocks.
Tables have shape (R,L,D); blocks (R,L,D,b,a) store J[i,a,neighbour,b].
Padding uses neighbour -1 and is never read by the shaders. Dense graphs retain
the dense kernels. ADABMDCA_SPARSE=0 disables sparse dispatch.
"""

import torch

from adabmDCA._tensor_cache import cached, sparse_kernels_enabled

# A conservative initial bound; measured crossover is documented in the benchmarks.
MAX_DEGREE_FRACTION = 0.35


def sparse_coupling_layout(couplings, *, source=None, coupling_dtype=None):
    """Return (neighbours, blocks, width) for (R,L,q,L,q), or None if dense.

    Cache by the original coupling tensor, not a temporary unsqueezed view.
    In-place updates invalidate both graph detection and the packed blocks.
    Graph detection precedes any optional storage cast; packed caches include dtype.
    Inference tensors have no version counter and are rebuilt on each call.
    """
    if not sparse_kernels_enabled():
        return None
    source = couplings if source is None else source
    dtype = coupling_dtype or couplings.dtype

    def memo(kind, build):
        return build() if torch.is_inference(source) else cached(source, kind, build)

    def graph():
        pairs = (couplings != 0).any(dim=-1).any(dim=-2)
        degree = pairs.sum(-1)
        largest = int(degree.max()) if degree.numel() else 0
        return pairs, degree, largest

    pairs, degree, largest = memo('mps_sparse_graph', graph)
    replicas, length = couplings.shape[:2]
    if largest > MAX_DEGREE_FRACTION * length:
        return None

    def pack():
        width = max(largest, 1)
        # Sorting integer site keys avoids dependence on stable GPU argsort.
        sites = torch.arange(length, device=couplings.device)
        order = torch.argsort(torch.where(pairs, sites, sites + length), dim=-1)[..., :width]
        valid = torch.arange(width, device=couplings.device) < degree[..., None]
        neighbours = torch.where(valid, order, -1).to(torch.int32).contiguous()
        replica = torch.arange(replicas, device=couplings.device)[:, None, None]
        site = torch.arange(length, device=couplings.device)[None, :, None]
        blocks = couplings.permute(0, 1, 3, 4, 2)[replica, site, order]
        blocks = torch.where(valid[..., None, None], blocks, 0)
        blocks = (blocks.contiguous() if dtype == couplings.dtype else blocks.to(
            dtype=dtype, memory_format=torch.contiguous_format, copy=True))
        return neighbours, blocks, width

    return memo(f'mps_sparse_blocks_{dtype}', pack)
