"""The anchor of PTT training: a tree + low-rank Potts model with exact sampling and normalization.

The model adds the pair terms of a tree to the low-rank model of :mod:`adabmDCA.anchor.lowrank`:

    p(x, z) ∝ exp(-|z|^2 / 2 + sum_i [g_i(x_i) + z . W_i(x_i)] + sum_{(i, j) in tree} theta_ij(x_i, x_j)),

with ``K`` Gaussian hidden units ``z``. Its marginal over ``z`` is a Potts model
with couplings ``W_i(a) . W_j(b)`` plus ``theta`` on the edges of the tree. The
tree holds the strongest local pair statistics; the hidden units hold the
collective modes that no tree can represent.

Given ``z`` the sequence follows a tree model with fields ``g + z . W``, whose
partition function and exact samples come from belief propagation. Given ``x``,
``z`` is a unit Gaussian centred at ``m(x) = sum_i W_i(x_i)``. So the latent
density ``exp(F(z))``, ``F(z) = -|z|^2 / 2 + G(z)`` with ``G`` the log partition
function of the tilted tree, is a mixture of unit Gaussians as in the low-rank
model, and ``G`` is convex. ``log Z`` is a lattice sum; ``z`` is drawn by
rejection from envelopes on cells, where the convexity of ``G`` gives the
bound: ``G`` minus its tangent plane at the centre of a box is largest at a
vertex. With ``K = 0`` the model is a tree alone, and everything is closed-form.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields

import numpy as np
import torch

from adabmDCA.anchor.lowrank import (
    _LATTICE_DEPTH,
    _LATTICE_STEP,
    _MAX_CELLS,
    _MAX_SEEDS,
    _PRUNE,
    _REFINE,
    _SEED_DEPTH,
    _TOLERANCE,
    TargetMoments,
    _lattice_keys,
    _lattice_points,
    hopfield_potts_init,
)

# Largest number of float64 elements of one temporary (256 MB).
_ELEMENTS = 1 << 25
# The fit keeps the beliefs of the lattice points for the gradient when they take at most this many bytes.
_CACHE_BYTES = 2 << 30
# Lattice points within this of the maximum of F enter the gradient (the others carry below exp(-18) of it).
_GRADIENT_DEPTH = 18.0
# The fit starts on a coarser lattice, which gives log Z to about 1e-3 with a third of the points for K = 3.
_COARSE_STEP = 1.5


def chow_liu_tree(frequencies: torch.Tensor, pair_frequencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The spanning tree of largest total mutual information, and its pair terms.

    The tree model ``prod_i f_i(x_i) prod_edges f_ij / (f_i f_j)`` is the
    maximum-likelihood tree for these frequencies (Chow and Liu, 1968).

    Args:
        frequencies: Single-site frequencies (L, q).
        pair_frequencies: Pair frequencies (L, q, L, q), consistent with ``frequencies``.

    Returns:
        The edges, an integer tensor (L - 1, 2), and ``log f_ij / (f_i f_j)`` on them, shape (L - 1, q, q).
    """
    from scipy.sparse.csgraph import minimum_spanning_tree

    ratio = (pair_frequencies / (frequencies[:, :, None, None] * frequencies[None, None, :, :])).clamp_min(1e-300).log()
    information = (pair_frequencies * ratio).sum((1, 3))
    information.fill_diagonal_(0)
    # Minimum spanning tree of the negated information; a tiny shift keeps zero-information pairs as edges.
    tree = minimum_spanning_tree(-(information.cpu().numpy() + 1e-12)).tocoo()
    edges = torch.tensor(np.stack([tree.row, tree.col], 1), dtype=torch.long, device=frequencies.device)
    return edges, ratio[edges[:, 0], :, edges[:, 1], :].contiguous()


def _log_gaussian_mass(slope, half_width):
    """``log of the integral of exp(slope t - t^2 / 2) over [-h, h]``, stably."""
    low, high = -half_width - slope, half_width - slope  # limits of the standard normal variable t - slope
    flip = low + high > 0
    low, high = torch.where(flip, -high, low), torch.where(flip, -low, high)
    log_low, log_high = torch.special.log_ndtr(low), torch.special.log_ndtr(high)
    difference = log_high + torch.log1p(-torch.exp(log_low - log_high).clamp_max(1 - 1e-16))
    return 0.5 * math.log(2 * math.pi) + 0.5 * slope ** 2 + difference


def _draw_gaussian(slope, half_width, uniform):
    """Draws of the density proportional to ``exp(slope t - t^2 / 2)`` on [-h, h], by inverting its distribution."""
    low, high = -half_width - slope, half_width - slope
    flip = low + high > 0
    low, high = torch.where(flip, -high, low), torch.where(flip, -low, high)
    cumulative_low, cumulative_high = torch.special.ndtr(low), torch.special.ndtr(high)
    value = torch.special.ndtri((cumulative_low + uniform * (cumulative_high - cumulative_low)).clamp(1e-300, 1 - 1e-16))
    value = torch.minimum(torch.maximum(value, low), high)
    return (torch.where(flip, -value, value) + slope).clamp(-half_width, half_width)


@dataclass(frozen=True)
class TreeLatentCells:
    """Boxes covering the latent density of a tree + low-rank model, with the proposal of the exact sampler.

    A cell proposes uniformly (``tilted`` false), with ``F <= envelope``, or from
    the truncated Gaussian proportional to ``exp(slope . d - |d|^2 / 2)``
    (``tilted`` true), with ``F(c + d) <= F(c) + slope . d - |d|^2 / 2 + envelope``.

    Attributes:
        centers: Cell centres, shape (n, K).
        half_widths: Cell half widths, shape (n, K).
        center_values: ``F`` at the centres.
        slopes: Gradient of ``F`` at the centres, shape (n, K).
        tilted: Whether the cell proposes from the truncated Gaussian.
        envelope: Flat cells: upper bound of ``F``; tilted cells: largest gap between ``G`` and its tangent plane.
        log_mass: Log of the proposal mass of each cell (an upper bound of its share of ``Z``).
        log_mass_lower: Log of a lower bound of each cell's share of ``Z``.
    """

    centers: torch.Tensor
    half_widths: torch.Tensor
    center_values: torch.Tensor
    slopes: torch.Tensor
    tilted: torch.Tensor
    envelope: torch.Tensor
    log_mass: torch.Tensor
    log_mass_lower: torch.Tensor


def _take(cells, mask):
    return TreeLatentCells(*(getattr(cells, field.name)[mask] for field in dataclass_fields(TreeLatentCells)))


def _join(first, second):
    return TreeLatentCells(*(torch.cat([getattr(first, field.name), getattr(second, field.name)])
                             for field in dataclass_fields(TreeLatentCells)))


class _Tree:
    """A forest over the sites, rooted at the centre of every component, with belief propagation on it."""

    def __init__(self, edges: torch.Tensor, length: int, device) -> None:
        pairs = [tuple(edge) for edge in edges.tolist()]
        neighbours = [[] for _ in range(length)]
        for index, (i, j) in enumerate(pairs):
            if i == j or not (0 <= i < length and 0 <= j < length):
                raise ValueError("Tree edges must join two different sites.")
            neighbours[i].append((j, index, False))
            neighbours[j].append((i, index, True))

        def distances(root):
            depth, order, queue = {root: 0}, [root], deque([root])
            while queue:
                i = queue.popleft()
                for j, _, _ in neighbours[i]:
                    if j not in depth:
                        depth[j] = depth[i] + 1
                        order.append(j)
                        queue.append(j)
            return depth, order

        parent, edge, flip, depth, roots = [-1] * length, [-1] * length, [False] * length, [-1] * length, []
        reached = 0
        for start in range(length):
            if depth[start] >= 0:
                continue
            # The centre of the component: the site that minimizes the distance to both ends of a diameter.
            _, order = distances(start)
            first, order = distances(order[-1])
            second, _ = distances(order[-1])
            root = min(order, key=lambda site: max(first[site], second[site]))
            roots.append(root)
            depth[root] = 0
            queue = deque([root])
            while queue:
                i = queue.popleft()
                reached += 1
                for j, index, flipped in neighbours[i]:
                    if depth[j] < 0:
                        depth[j], parent[j], edge[j], flip[j] = depth[i] + 1, i, index, flipped
                        queue.append(j)
        if len(pairs) != length - len(roots):
            raise ValueError("The edges must form a forest (no cycles).")
        self.roots = torch.tensor(roots, device=device)
        self.levels = []
        for level in range(1, max(depth) + 1):
            nodes = [i for i in range(length) if depth[i] == level]
            self.levels.append((torch.tensor(nodes, device=device), torch.tensor([parent[i] for i in nodes], device=device),
                                torch.tensor([edge[i] for i in nodes], device=device),
                                torch.tensor([flip[i] for i in nodes], device=device)))

    @staticmethod
    def oriented(pairs, edge, flip):
        """Pair terms as (parent state, child state): ``pairs[e][a_i, a_j]`` belongs to the edge (i, j)."""
        selected = pairs[edge]
        return torch.where(flip[:, None, None], selected.transpose(1, 2), selected)

    def upward(self, z, fields, weights, pairs, messages=False):
        """Beliefs of every subtree, shape (n, L, q), and ``G(z)``; with ``messages`` also the messages to the parents."""
        beliefs = fields[None] + torch.einsum("nk,iak->nia", z, weights)
        sent = torch.zeros_like(beliefs) if messages else None
        for nodes, parents, edge, flip in reversed(self.levels):
            message = torch.logsumexp(self.oriented(pairs, edge, flip)[None] + beliefs[:, nodes][:, :, None, :], -1)
            if messages:
                sent[:, nodes] = message
            beliefs.index_add_(1, parents, message)
        value = torch.logsumexp(beliefs[:, self.roots], -1).sum(-1)
        return (beliefs, value, sent) if messages else (beliefs, value)

    def downward(self, beliefs, sent, pairs, on_edges=None):
        """Single-site marginals (n, L, q) from the beliefs and messages of :meth:`upward`.

        ``on_edges(edge, flip, joint)`` receives the pair marginals of every level, shape (n, nodes, q_parent, q_child).
        """
        count, length, _ = beliefs.shape
        down = torch.zeros_like(beliefs)
        norm = torch.zeros((count, length), dtype=beliefs.dtype, device=beliefs.device)  # log Z of each site's component
        norm[:, self.roots] = torch.logsumexp(beliefs[:, self.roots], -1)
        marginals = torch.zeros_like(beliefs)
        marginals[:, self.roots] = torch.softmax(beliefs[:, self.roots], -1)
        for nodes, parents, edge, flip in self.levels:
            norm[:, nodes] = norm[:, parents]
            cavity = beliefs[:, parents] + down[:, parents] - sent[:, nodes]  # the parent without this child
            joint = torch.exp(cavity[..., None] + self.oriented(pairs, edge, flip)[None] + beliefs[:, nodes][:, :, None, :]
                              - norm[:, nodes][:, :, None, None])
            child = joint.sum(2)
            marginals[:, nodes] = child
            down[:, nodes] = torch.log(child.clamp_min(1e-300)) - beliefs[:, nodes] + norm[:, nodes][:, :, None]
            if on_edges is not None:
                on_edges(edge, flip, joint)
        return marginals

    def descend(self, beliefs, pairs):
        """Exact draws of the tree model from the beliefs of :meth:`upward`, as integers (n, L)."""
        count, length, _ = beliefs.shape
        states = torch.zeros((count, length), dtype=torch.long, device=beliefs.device)

        def draw(logits):  # Gumbel-max
            return (logits - torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-300)))).argmax(-1)

        states[:, self.roots] = draw(beliefs[:, self.roots])
        for nodes, parents, edge, flip in self.levels:
            rows = self.oriented(pairs, edge, flip)[torch.arange(len(nodes), device=beliefs.device)[None, :],
                                                    states[:, parents]]
            states[:, nodes] = draw(rows + beliefs[:, nodes])
        return states


def _chunk(length, num_states):
    return max(1, _ELEMENTS // (length * num_states * num_states))


@torch.no_grad()
def _lattice_sum(tree, fields, weights, pairs, start=None, *, step=_LATTICE_STEP, depth=_LATTICE_DEPTH,
                 gradient=False, workspace=None):
    """``log of the integral of exp(F)`` as a lattice sum, as in :func:`adabmDCA.anchor.lowrank._lattice_sum`.

    With ``gradient`` also its derivatives with respect to ``fields``, ``weights``
    and ``pairs``: the single-site and pair marginals of the tilted tree, averaged
    over the lattice points with their weights ``exp(F)``. They come from a
    second, downward pass of belief propagation.

    ``workspace`` is a dict that keeps the buffer of the gradient between calls (the fit passes one).

    Returns:
        The log integral, the next ``start`` (a tuple of the spacing and the keys), the number of points,
        and with ``gradient`` the three derivatives.
    """
    length, num_states, rank = weights.shape
    device = weights.device
    if rank == 0:
        zero = torch.zeros((1, 0), dtype=weights.dtype, device=device)
        beliefs, value, sent = tree.upward(zero, fields, weights, pairs, messages=True)
        if not gradient:
            return float(value[0]), None, 1
        edge_marginals = torch.zeros_like(pairs)

        def collect(edge, flip, joint):
            edge_marginals.index_add_(0, edge, torch.where(flip[:, None, None], joint[0].transpose(1, 2), joint[0]))

        marginals = tree.downward(beliefs, sent, pairs, collect)
        return float(value[0]), None, 1, marginals[0], torch.zeros_like(weights), edge_marginals
    chunk = _chunk(length, num_states)
    stride = max(1, round(3.0 / step))
    point_bytes = length * num_states * 16
    workspace = {} if workspace is None else workspace

    def points(keys):
        return _lattice_points(keys, rank).to(weights.dtype) * step

    granule = max(16, chunk // 16)

    def padded(count):
        """Round CUDA batch sizes so the allocator can reuse memory across batches."""
        if device.type != "cuda":
            return count
        if count < granule:
            return 1 << max(4, (count - 1).bit_length())
        return min(chunk, -(-count // granule) * granule)

    def upward(z, messages=False):
        count = len(z)
        full = torch.zeros((padded(count), rank), dtype=z.dtype, device=device)
        full[:count] = z
        result = tree.upward(full, fields, weights, pairs, messages=messages)
        return tuple(item[:count] for item in result)

    class Cache:
        """Beliefs and messages of the evaluated points, in a buffer kept in ``workspace`` between calls.

        It grows by half its size, up to the cache limit, and never shrinks, so the fit allocates it only a few times.
        """

        def __init__(self):
            self.used = 0

        def append(self, beliefs, sent):
            count = len(beliefs)
            held = workspace.get("beliefs")
            if held is None or self.used + count > len(held[0]) or held[0].shape[1:] != beliefs.shape[1:]:
                grown_size = 3 * len(held[0]) // 2 if held is not None else 0
                capacity = max(self.used + count, min(max(chunk, grown_size), _CACHE_BYTES // point_bytes))
                grown = [torch.empty((capacity, length, num_states), dtype=weights.dtype, device=device)
                         for _ in range(2)]
                if held is not None and self.used:
                    grown[0][:self.used], grown[1][:self.used] = held[0][:self.used], held[1][:self.used]
                workspace["beliefs"] = held = grown
                if device.type == "cuda":
                    torch.cuda.empty_cache()  # the old buffer has a size that no later tensor reuses
            held[0][self.used:self.used + count] = beliefs
            held[1][self.used:self.used + count] = sent
            self.used += count

    def evaluate(keys, cache=None):
        values = []
        for begin in range(0, len(keys), chunk):
            z = points(keys[begin:begin + chunk])
            result = upward(z, messages=cache is not None)
            if cache is not None:
                cache.append(result[0], result[2])
            values.append(result[1] - 0.5 * (z ** 2).sum(-1))
        return torch.cat(values)

    # Seeds: the coarse lattice over the box that contains every m(x).
    low = torch.floor(weights.amin(1).sum(0) / (stride * step)).long()
    high = torch.ceil(weights.amax(1).sum(0) / (stride * step)).long()
    if int((high - low + 1).prod()) > _MAX_SEEDS:
        raise RuntimeError(f"The latent density of the rank-{rank} model spreads over too large a box.")
    axes = [torch.arange(int(low[k]), int(high[k]) + 1, device=device) * stride for k in range(rank)]
    keys = _lattice_keys(torch.cartesian_prod(*axes).reshape(-1, rank))
    coarse = evaluate(keys)
    keys = keys[coarse > coarse.max() - _SEED_DEPTH]
    if start is not None and start[0] == step:
        keys = torch.unique(torch.cat([start[1], keys]))
    neighbours = torch.zeros((2 * rank + 1, rank), dtype=torch.int64, device=device)
    for k in range(rank):
        neighbours[2 * k + 1, k], neighbours[2 * k + 2, k] = 1, -1
    offsets = _lattice_keys(neighbours) - _lattice_keys(neighbours[:1])
    # The beliefs are kept for the gradient unless they would take too much memory.
    cache = Cache() if gradient and len(keys) * point_bytes <= _CACHE_BYTES else None
    values = evaluate(keys, cache)
    while True:
        shell = torch.unique((keys[values > values.max() - depth][:, None] + offsets[None]).reshape(-1))
        fresh = shell[~torch.isin(shell, keys)]
        if not len(fresh):
            break
        if cache is not None and (len(keys) + len(fresh)) * point_bytes > _CACHE_BYTES:
            cache = None
            workspace.pop("beliefs", None)
        values = torch.cat([values, evaluate(fresh, cache)])
        keys = torch.cat([keys, fresh])
    log_integral = float(torch.logsumexp(values, 0)) + rank * math.log(step)
    if not gradient:
        return log_integral, (step, shell), len(keys)
    use = values > values.max() - _GRADIENT_DEPTH
    mass = torch.softmax(values, 0)
    field_marginals = torch.zeros_like(fields)
    weight_marginals = torch.zeros_like(weights)
    edge_marginals = torch.zeros_like(pairs)

    def accumulate(z, share, beliefs, sent):
        def collect(edge, flip, joint):
            summed = torch.einsum("n,nipc->ipc", share, joint)
            edge_marginals.index_add_(0, edge, torch.where(flip[:, None, None], summed.transpose(1, 2), summed))

        marginals = tree.downward(beliefs, sent, pairs, collect)
        field_marginals.add_(torch.einsum("n,nia->ia", share, marginals))
        weight_marginals.add_(torch.einsum("n,nk,nia->iak", share, z, marginals))

    if cache is not None:
        beliefs, sent = workspace["beliefs"]
        # The cache is read in blocks of a padded size; the padding rows repeat a point with no weight.
        for begin in range(0, len(keys), chunk):
            rows = use[begin:begin + chunk].nonzero().squeeze(1)
            if not len(rows):
                continue
            share = torch.zeros(padded(len(rows)), dtype=mass.dtype, device=device)
            share[:len(rows)] = mass[begin + rows]
            rows = torch.cat([rows, rows.new_zeros(len(share) - len(rows))]) + begin
            accumulate(points(keys[rows]), share, beliefs[rows], sent[rows])
    else:
        selected, mass = points(keys[use]), mass[use]
        for begin in range(0, len(selected), chunk):
            count = len(selected[begin:begin + chunk])
            z = torch.zeros((padded(count), rank), dtype=selected.dtype, device=device)
            z[:count] = selected[begin:begin + chunk]
            share = torch.zeros(len(z), dtype=mass.dtype, device=device)
            share[:count] = mass[begin:begin + count]
            beliefs, _, sent = tree.upward(z, fields, weights, pairs, messages=True)
            accumulate(z, share, beliefs, sent)
    return log_integral, (step, shell), len(keys), field_marginals, weight_marginals, edge_marginals


class TreeLowRankPotts:
    """A tree + rank-``K`` Potts model with exact sampling and normalization.

    Parameterized by the fields ``g`` (L, q), the weights ``W`` (L, q, K) of the
    Gaussian hidden units, the edges of a forest over the sites and their pair
    terms ``theta`` (E, q, q). The Potts couplings are ``W_i(a) . W_j(b)`` plus
    ``theta`` on the edges, and the fields ``g_i(a) + |W_i(a)|^2 / 2``. With
    ``K = 0`` the model is the tree alone.

    Example:
        >>> anchor = fit_tree_low_rank_potts(sequences, weights, num_states=21, rank=2)
        >>> params = anchor.potts_params(torch.float32)   # {"bias", "coupling_matrix"}
        >>> states = anchor.sample(1000)                   # exact draws, int32 (1000, L)
        >>> anchor.log_z                                   # log partition function
    """

    def __init__(self, fields: torch.Tensor, weights: torch.Tensor, edges: torch.Tensor, pairs: torch.Tensor, *,
                 tolerance: float = _TOLERANCE) -> None:
        if fields.ndim != 2 or weights.ndim != 3 or weights.shape[:2] != fields.shape:
            raise ValueError("Tree + low-rank fields must be (L, q) and weights (L, q, K).")
        num_states = fields.shape[1]
        if edges.ndim != 2 or edges.shape[1] != 2 or pairs.shape != (len(edges), num_states, num_states):
            raise ValueError("Tree edges must be (E, 2) and their pair terms (E, q, q).")
        device = torch.device(fields.device)
        device = device if device.type in ("cpu", "cuda") else torch.device("cpu")
        self.fields = fields.detach().to(device=device, dtype=torch.float64).contiguous()
        self.weights = weights.detach().to(device=device, dtype=torch.float64).contiguous()
        self.pairs = pairs.detach().to(device=device, dtype=torch.float64).contiguous()
        self.edges = edges.detach().to(device=device, dtype=torch.long).contiguous()
        if not all(bool(torch.isfinite(tensor).all()) for tensor in (self.fields, self.weights, self.pairs)):
            raise ValueError("Tree + low-rank parameters must be finite.")
        self.tolerance = tolerance
        self._tree = _Tree(self.edges, fields.shape[0], device)
        self._cells = None
        self._log_z = None

    @property
    def rank(self) -> int:
        return self.weights.shape[2]

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.fields.shape)

    def to(self, device) -> TreeLowRankPotts:
        """The same model on ``device``; MPS models compute on CPU."""
        return TreeLowRankPotts(self.fields.to(device), self.weights.to(device), self.edges.to(device),
                                self.pairs.to(device), tolerance=self.tolerance)

    def potts_params(self, dtype: torch.dtype = torch.float32, device=None) -> dict[str, torch.Tensor]:
        """The model as Potts parameters ``{"bias", "coupling_matrix"}`` with zero within-site blocks."""
        length = self.fields.shape[0]
        couplings = torch.einsum("iak,jbk->iajb", self.weights, self.weights)
        couplings[torch.arange(length), :, torch.arange(length), :] = 0.0
        couplings[self.edges[:, 0], :, self.edges[:, 1], :] += self.pairs
        couplings[self.edges[:, 1], :, self.edges[:, 0], :] += self.pairs.transpose(1, 2)
        device = self.fields.device if device is None else device
        return {"bias": (self.fields + 0.5 * (self.weights ** 2).sum(-1)).to(device=device, dtype=dtype).contiguous(),
                "coupling_matrix": couplings.to(device=device, dtype=dtype).contiguous()}

    def _convex_part(self, z, gradient=False):
        """``G(z)``, the log partition function of the tilted tree; with ``gradient`` also ``grad G = E[m(x) | z]``."""
        values, gradients = [], []
        chunk = _chunk(*self.shape)
        for begin in range(0, len(z), chunk):
            block = z[begin:begin + chunk]
            if gradient:
                beliefs, value, sent = self._tree.upward(block, self.fields, self.weights, self.pairs, messages=True)
                marginals = self._tree.downward(beliefs, sent, self.pairs)
                gradients.append(torch.einsum("nia,iak->nk", marginals, self.weights))
            else:
                value = self._tree.upward(block, self.fields, self.weights, self.pairs)[1]
            values.append(value)
        return (torch.cat(values), torch.cat(gradients)) if gradient else torch.cat(values)

    @torch.no_grad()
    def latent_log_density(self, z: torch.Tensor) -> torch.Tensor:
        """``F(z)``, the unnormalized log density of the latent coordinates, for ``z`` of shape (n, K)."""
        z = z.to(device=self.fields.device, dtype=torch.float64)
        return self._convex_part(z) - 0.5 * (z ** 2).sum(-1)

    @property
    def log_z(self) -> float:
        """log Z of the Potts model: exact for a tree, a lattice sum of the latent density otherwise."""
        if self._log_z is None:
            log_integral = _lattice_sum(self._tree, self.fields, self.weights, self.pairs)[0]
            self._log_z = log_integral - 0.5 * self.rank * math.log(2 * math.pi)
        return self._log_z

    @torch.no_grad()
    def _bound_cells(self, centers, half_widths):
        """Proposal masses (upper bounds) and lower bounds of every cell's share of ``Z``.

        ``G`` is convex, so on a box it lies above its tangent plane at the centre,
        and the gap ``r`` is largest at a vertex. With the exact Gaussian factor,
        ``F(c + d)`` lies between ``F(c) + s . d - |d|^2 / 2`` and the same plus
        ``r``: a truncated Gaussian envelope. On large cells a flat bound can be
        smaller: the largest vertex value of ``G`` minus the smallest ``|z|^2 / 2``.
        """
        rank = centers.shape[1]
        value, gradient = self._convex_part(centers, gradient=True)
        signs = torch.tensor([[1.0 if (vertex >> k) & 1 else -1.0 for k in range(rank)] for vertex in range(2 ** rank)],
                             dtype=centers.dtype, device=centers.device)
        offsets = signs[None] * half_widths[:, None, :]
        vertex_values = self._convex_part((centers[:, None, :] + offsets).reshape(-1, rank)).reshape(len(centers), -1)
        gap = (vertex_values - value[:, None] - (gradient[:, None, :] * offsets).sum(-1)).amax(1).clamp_min(0.0)
        # Rounding must not push F above a bound that holds exactly.
        gap = gap + 1e-9 * (1.0 + value.abs())
        center_values = value - 0.5 * (centers ** 2).sum(-1)
        slopes = gradient - centers
        lower = center_values + _log_gaussian_mass(slopes, half_widths).sum(-1)
        closest = (centers.abs() - half_widths).clamp_min(0.0)
        flat = vertex_values.amax(1) - 0.5 * (closest ** 2).sum(-1)
        flat = torch.maximum(flat, center_values) + 1e-9 * (1.0 + flat.abs())
        flat_mass = flat + torch.log(2 * half_widths).sum(-1)
        tilted_mass = lower + gap
        tilted = tilted_mass <= flat_mass
        return TreeLatentCells(centers, half_widths, center_values, slopes, tilted, torch.where(tilted, gap, flat),
                               torch.minimum(flat_mass, tilted_mass), lower)

    @property
    def cells(self) -> TreeLatentCells | None:
        """The latent cells of the exact sampler, built on first use (``None`` for a tree alone).

        As for the low-rank model: a box outside which ``F`` is negligible is cut
        into ``8^K`` cells, which are halved along their longest side until the
        proposal mass exceeds the lower bound of the cell's mass by at most the
        tolerance, wherever the cell can carry more than ``exp(-20)`` of the mass;
        cells that cannot carry more than ``exp(-50)`` of it are dropped.
        """
        if self.rank == 0 or self._cells is not None:
            return self._cells
        rank, dtype, device = self.rank, torch.float64, self.fields.device
        # |grad G| <= S_k in direction k, so F(z) - F(0) <= sum_k (S_k |z_k| - z_k^2 / 2).
        slopes = self.weights.abs().amax(1).sum(0)
        radii = []
        for k in range(rank):
            others = float((slopes ** 2).sum() - slopes[k] ** 2) / 2
            radii.append(float(slopes[k]) + math.sqrt(float(slopes[k]) ** 2 + 2 * (_PRUNE + others)) + 1.0)
        mids = []
        for radius in radii:
            edge = torch.linspace(-radius, radius, 9, dtype=dtype, device=device)
            mids.append(0.5 * (edge[1:] + edge[:-1]))
        centers = torch.cartesian_prod(*mids).reshape(-1, rank)
        half_widths = torch.tensor([radius / 8 for radius in radii], dtype=dtype, device=device)
        cells = self._bound_cells(centers, half_widths.expand(len(centers), -1).contiguous())
        while True:
            floor = float(torch.logsumexp(cells.log_mass_lower, 0))
            keep = cells.log_mass > floor - _PRUNE
            refine = keep & (cells.log_mass - cells.log_mass_lower > self.tolerance) & (cells.log_mass > floor - _REFINE)
            if not refine.any():
                self._cells = _take(cells, keep)
                return self._cells
            stay = keep & ~refine
            if int(stay.sum()) + 2 * int(refine.sum()) > _MAX_CELLS:
                raise RuntimeError(f"The latent density of the rank-{rank} model needs more than {_MAX_CELLS} cells.")
            half = cells.half_widths[refine]
            direction = half.argmax(1, keepdim=True)
            child_half = half.scatter(1, direction, half.gather(1, direction) / 2)
            shift = torch.zeros_like(child_half).scatter(1, direction, child_half.gather(1, direction))
            parents = cells.centers[refine]
            children = self._bound_cells(torch.cat([parents - shift, parents + shift]), torch.cat([child_half, child_half]))
            cells = _join(_take(cells, stay), children)

    @property
    def log_z_bounds(self) -> tuple[float, float]:
        """Rigorous lower and upper bounds of log Z (up to cells dropped below exp(-50) of Z each)."""
        if self.rank == 0:
            return self.log_z, self.log_z
        gaussian = 0.5 * self.rank * math.log(2 * math.pi)
        return (float(torch.logsumexp(self.cells.log_mass_lower, 0)) - gaussian,
                float(torch.logsumexp(self.cells.log_mass, 0)) - gaussian)

    @property
    def acceptance(self) -> float:
        """Expected acceptance of the latent rejection sampler (1 for a tree alone)."""
        return 1.0 if self.rank == 0 else math.exp(self.log_z - self.log_z_bounds[1])

    @torch.no_grad()
    def sample(self, n_samples: int, device=None) -> torch.Tensor:
        """Exact draws of the model, int32 of shape (n, L), on ``device`` (default: the parameters' device).

        The latent coordinates are drawn by rejection from the cell envelopes,
        then the sequence by ancestral sampling of the tilted tree. Uses the
        global torch random generator of the parameters' device.
        """
        own = self.fields.device
        chunk = _chunk(*self.shape)
        drawn, missing = [], n_samples
        if self.rank == 0:
            while missing > 0:
                z = torch.zeros((min(chunk, missing), 0), dtype=torch.float64, device=own)
                drawn.append(self._tree.descend(self._tree.upward(z, self.fields, self.weights, self.pairs)[0], self.pairs))
                missing -= len(drawn[-1])
        else:
            cells = self.cells
            probabilities = (cells.log_mass - cells.log_mass.max()).exp()
            acceptance = self.acceptance
            while missing > 0:
                batch = min(chunk, max(64, int(missing / acceptance * 1.1) + 16))
                index = torch.multinomial(probabilities, batch, replacement=True)
                half, slope, tilted = cells.half_widths[index], cells.slopes[index], cells.tilted[index]
                uniform = torch.rand((batch, self.rank), dtype=torch.float64, device=own)
                offsets = torch.where(tilted[:, None], _draw_gaussian(slope, half, uniform), (2 * uniform - 1) * half)
                z = cells.centers[index] + offsets
                beliefs, value = self._tree.upward(z, self.fields, self.weights, self.pairs)
                bound = torch.where(
                    tilted,
                    cells.center_values[index] + (slope * offsets).sum(-1) - 0.5 * (offsets ** 2).sum(-1)
                    + cells.envelope[index],
                    cells.envelope[index],
                )
                keep = torch.rand(batch, dtype=torch.float64, device=own).log() < value - 0.5 * (z ** 2).sum(-1) - bound
                drawn.append(self._tree.descend(beliefs[keep], self.pairs)[:missing])
                missing -= len(drawn[-1])
        states = (torch.cat(drawn) if drawn else torch.empty((0, self.shape[0]), dtype=torch.long, device=own))
        states = states.to(torch.int32)
        return states if device is None else states.to(device)

    def _scores(self, states):
        """``sum_i g_i(x_i) + |m(x)|^2 / 2 + sum_edges theta(x_i, x_j)`` of integer sequences (N, L)."""
        sites = torch.arange(states.shape[1], device=states.device)
        projection = self.weights[sites, states].sum(1)
        on_edges = self.pairs[torch.arange(len(self.edges), device=states.device),
                              states[:, self.edges[:, 0]], states[:, self.edges[:, 1]]].sum(1)
        return self.fields[sites, states].sum(1) + 0.5 * (projection ** 2).sum(-1) + on_edges

    def log_likelihood(self, states: torch.Tensor, weights: torch.Tensor | None = None) -> float:
        """Mean log-likelihood of integer sequences (N, L), weighted by ``weights``."""
        score = self._scores(states.to(device=self.fields.device).long())
        if weights is None:
            return float(score.mean()) - self.log_z
        weights = weights.to(device=self.fields.device, dtype=torch.float64).reshape(-1)
        return float((weights * score).sum() / weights.sum()) - self.log_z


def fit_tree_low_rank_potts(
    states: torch.Tensor,
    sequence_weights: torch.Tensor | None,
    *,
    num_states: int,
    rank: int = 2,
    pseudocount: float = 0.0,
    l2: float = 1e-4,
    max_iterations: int = 5000,
    tolerance: float = 0.01,
    device=None,
    verbose: bool = False,
    callback: Callable[[dict], None] | None = None,
) -> TreeLowRankPotts:
    """Fit a tree + rank-``K`` Potts model to integer sequences by maximum likelihood.

    The tree is the Chow-Liu tree of the pseudocounted pair frequencies. With
    ``rank = 0`` the fit ends there: the tree model is closed-form. Otherwise the
    fields, the weights and the pair terms of the tree are fitted together by
    L-BFGS, starting from the tree and from the mean-field Hopfield-Potts patterns
    scaled to the size that the likelihood prefers. The fit runs first on a coarse
    latent lattice, then on the lattice of spacing 1.

    Args:
        states: Integer sequences, shape (N, L).
        sequence_weights: Sequence weights (N,), or ``None`` for equal weights.
        num_states: Alphabet size ``q``.
        rank: Number of Gaussian hidden units ``K`` (0 to 3).
        pseudocount: Weight of the uniform distribution in the fitted target.
        l2: L2 penalty on the weights, in the gauge where the weights of every site sum to zero.
        max_iterations: Largest number of evaluations of the likelihood.
        tolerance: Stop a stage when a block of L-BFGS iterations lowers the negative log-likelihood per sequence
            by less than this. Blocks contain up to 25 iterations on the coarse lattice and 100 on the fine lattice.
        device: Device of the fit; defaults to that of ``states``.
        verbose: Print the objective every 100 evaluations.
        callback: Called after every evaluation with a dict: ``evaluation``, ``max_evaluations``,
            ``log_likelihood`` (per sequence, with the pseudocount and the penalty) and ``lattice_points``.

    Returns:
        The fitted :class:`TreeLowRankPotts`.
    """
    if not 0 <= rank <= 3:
        raise ValueError("Tree + low-rank Potts models support 0 to 3 latent directions.")
    device = torch.device(device) if device is not None else states.device
    if device.type not in ("cpu", "cuda"):
        device = torch.device("cpu")
    states = states.to(device)
    if sequence_weights is None:
        sequence_weights = torch.ones(len(states), dtype=torch.float64, device=device)
    length = states.shape[1]
    columns = length * num_states
    # The tree needs the pair frequencies themselves.
    moments = TargetMoments(states, sequence_weights.reshape(-1), num_states, pseudocount, pair_matrix=True)
    frequencies = moments.frequencies
    pair_frequencies = moments.matrix.view(length, num_states, length, num_states)
    edges, pairs = chow_liu_tree(frequencies.clamp_min(1e-300), pair_frequencies)
    edge_frequencies = pair_frequencies[edges[:, 0], :, edges[:, 1], :].contiguous()
    profile = frequencies.clamp_min(1e-8).log()
    if rank == 0:
        return TreeLowRankPotts(profile, torch.zeros((length, num_states, 0), dtype=torch.float64, device=device),
                                edges, pairs)
    _, patterns = hopfield_potts_init(moments, rank)
    tree = _Tree(edges, length, device)
    gaussian = 0.5 * rank * math.log(2 * math.pi)
    sizes = (columns, columns * rank)
    progress = {"evaluations": 0, "points": 0, "start": None}
    workspace = {}

    def unpack(x):
        return (x[:sizes[0]].view(length, num_states), x[sizes[0]:sizes[0] + sizes[1]].view(length, num_states, rank),
                x[sizes[0] + sizes[1]:].view(len(edges), num_states, num_states))

    def objective(x, step, gradient=True):
        """Negative log-likelihood per sequence of the flattened parameters, and its gradient."""
        fields, weights, pair_terms = unpack(x)
        flat = weights.reshape(columns, rank)
        correlated = moments.product(flat)
        centred = (weights - weights.mean(1, keepdim=True)).reshape(columns, rank)
        result = _lattice_sum(tree, fields, weights, pair_terms, progress["start"], step=step, gradient=gradient,
                              workspace=workspace)
        progress["start"], progress["points"] = result[1], result[2]
        loss = (result[0] - gaussian - float((frequencies * fields).sum()) - 0.5 * float((flat * correlated).sum())
                - float((edge_frequencies * pair_terms).sum()) + 0.5 * l2 * float((centred ** 2).sum()))
        if not gradient:
            return loss, None
        return loss, torch.cat([(result[3] - frequencies).reshape(-1),
                                (result[4].reshape(columns, rank) - correlated + l2 * centred).reshape(-1),
                                (result[5] - edge_frequencies).reshape(-1)])

    # The tree already holds the strongest pairs: start from the scale of the patterns that the likelihood prefers.
    candidates = []
    for scale in [2.0 ** -k for k in range(7)]:
        weights = scale * patterns
        x = torch.cat([(profile - 0.5 * (weights ** 2).sum(-1)).reshape(-1), weights.reshape(-1), pairs.reshape(-1)])
        progress["start"] = None
        candidates.append((objective(x, _COARSE_STEP, gradient=False)[0], x))
    parameters = min(candidates, key=lambda candidate: candidate[0])[1].clone().requires_grad_(True)

    for step, max_iter, max_eval in ((_COARSE_STEP, 25, 32), (_LATTICE_STEP, 100, 125)):
        progress["start"] = None

        def closure(step=step):
            loss, slope = objective(parameters.detach(), step)
            parameters.grad = slope
            progress["evaluations"] += 1
            if callback is not None:
                callback({"evaluation": progress["evaluations"], "max_evaluations": max_iterations,
                          "log_likelihood": -loss, "lattice_points": progress["points"]})
            if verbose and progress["evaluations"] % 100 == 0:
                print(f"tree + low-rank fit, evaluation {progress['evaluations']}: negative log-likelihood {loss:.6f} "
                      f"({progress['points']} lattice points, spacing {step})")
            return torch.tensor(loss, dtype=torch.float64, device=device)

        def new_optimizer():
            return torch.optim.LBFGS([parameters], max_iter=max_iter, max_eval=max_eval, line_search_fn="strong_wolfe",
                                     tolerance_grad=1e-10, tolerance_change=1e-14, history_size=20)

        optimizer = new_optimizer()
        previous = math.inf
        while progress["evaluations"] < max_iterations:
            try:
                loss = float(optimizer.step(closure))  # the objective at the start of the block
            except IndexError:  # a failed line search of torch's L-BFGS: restart it from the current point
                optimizer = new_optimizer()
                continue
            if previous - loss < tolerance:
                break
            previous = loss
    fields, weights, pair_terms = unpack(parameters.detach())
    workspace.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()  # hand the memory of the lattice back to the GPU before training starts
    # Return the model in the gauge where the weights of every site sum to zero.
    shift = weights.mean(1)
    weights = weights - shift[:, None, :]
    return TreeLowRankPotts(fields + weights @ shift.sum(0), weights, edges, pair_terms)
