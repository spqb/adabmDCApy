"""The tree + low-rank anchor restricted to a Hamming window, with exact draws and normalization.

The restricted law is

    p_W(x) ∝ exp(s_A(x)) 1{l <= d(x) <= u},

with ``s_A`` the score of :class:`~adabmDCA.anchor.anchor.TreeLowRankPotts` and
``d`` a sum of 0/1 site increments (the substitutions from a reference). At a
fixed latent coordinate ``z`` the anchor is a tree, and belief propagation with
messages indexed by the parent state *and* by the count of the child's subtree
gives

    C_z(c) = sum over x with d(x) = c of exp(s_tree(x; z)),

truncated at ``u``: counts never decrease up the tree, so larger counts can be
dropped as soon as they appear. ``log Z_W`` is the lattice sum of
``-|z|^2 / 2 + log sum_{c=l..u} C_z(c)``. For one sequence the integrand is a
unit Gaussian centred at ``m(x)``, so the lattice sum has the same relative
error for every sequence, whatever the window selects (below 1e-8 at unit
spacing for ``K <= 3``). Drawing a lattice point with its weight, a total count
in the window, and then the tree top-down, allocating the remaining count among
the subtrees of each node, gives independent draws of the same lattice mixture.
The lattice is searched with the restricted integrand, so a window that selects
a latent region the unrestricted anchor rarely visits is still covered.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from adabmDCA.anchor.anchor import _CACHE_BYTES, TreeLowRankPotts
from adabmDCA.anchor.lowrank import (
    _LATTICE_DEPTH,
    _LATTICE_STEP,
    _MAX_SEEDS,
    _SEED_DEPTH,
    _SEED_STRIDE,
    LowRankPotts,
    _lattice_keys,
    _lattice_points,
)

# Largest number of float64 elements of one count table batch (128 MB).
_ELEMENTS = 1 << 24
# Lattice points below this many nats from the largest weight are not drawn.
_DRAW_DEPTH = 40.0


def _log_convolve(first, second):
    """``log sum_j exp(first[c - j] + second[j])`` along the last axis, truncated to its width."""
    width = first.shape[-1]
    result = first + second[..., :1]
    for shift in range(1, width):
        result[..., shift:] = torch.logaddexp(result[..., shift:], first[..., :width - shift] + second[..., shift:shift + 1])
    return result


def _gumbel_argmax(logits):
    return (logits - torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-300)))).argmax(-1)


class WindowedAnchor:
    """A tree + low-rank anchor restricted to ``minimum <= sum_i increments[i, x_i] <= maximum``.

    Args:
        anchor: :class:`~adabmDCA.anchor.anchor.TreeLowRankPotts` or :class:`~adabmDCA.anchor.lowrank.LowRankPotts`.
        increments: 0/1 array (L, q) in the anchor's alphabet, typically ``x_i != reference_i``.
        minimum: Smallest allowed total.
        maximum: Largest allowed total.

    The count tables are as wide as the smaller of ``maximum`` and ``L - minimum``
    (counting matches instead of mismatches), less the counts that every sequence
    carries. Their cost grows with that width, not with the width of the window.

    Example:
        >>> restricted = WindowedAnchor(anchor, states != reference, 0, 15)
        >>> restricted.log_z                 # log of the restricted partition function
        >>> restricted.sample(1000)          # independent draws in the window, int32 (1000, L)
    """

    def __init__(self, anchor, increments, minimum: int, maximum: int) -> None:
        if isinstance(anchor, LowRankPotts):
            num_states = anchor.shape[1]
            anchor = TreeLowRankPotts(anchor.fields, anchor.weights,
                                      torch.zeros((0, 2), dtype=torch.long, device=anchor.fields.device),
                                      torch.zeros((0, num_states, num_states), dtype=torch.float64,
                                                  device=anchor.fields.device))
        if not isinstance(anchor, TreeLowRankPotts):
            raise TypeError("A windowed anchor needs a tree + low-rank or low-rank anchor.")
        self.anchor = anchor
        length, num_states = anchor.shape
        device = anchor.fields.device
        increments = torch.as_tensor(np.asarray(increments), device=device).long()
        if increments.shape != (length, num_states) or bool(((increments != 0) & (increments != 1)).any()):
            raise ValueError("Window increments must be a 0/1 array of the anchor's shape.")
        if not 0 <= minimum <= maximum:
            raise ValueError("The window needs 0 <= minimum <= maximum.")
        maximum = min(maximum, length)
        if maximum > length - minimum:  # count matches, which are fewer
            increments, minimum, maximum = 1 - increments, length - maximum, length - minimum
        forced = int(increments.amin(1).sum())
        increments = increments - increments.amin(1, keepdim=True)
        self.minimum, self.maximum = max(0, minimum - forced), maximum - forced
        if self.maximum < 0 or self.minimum > int(increments.amax(1).sum()):
            raise ValueError("The window and the allowed alphabet leave no sequence.")
        self.increments = increments
        self.width = self.maximum + 1

        # Children of each parent are multiplied in rounds, one child per parent per round.
        tree = anchor._tree
        self.roots = tree.roots
        self.levels = []
        for nodes, parents, edge, flip in tree.levels:
            seen, order = {}, []
            for parent in parents.tolist():
                order.append(seen.get(parent, 0))
                seen[parent] = order[-1] + 1
            order = torch.tensor(order, device=device)
            rounds = [torch.nonzero(order == r).squeeze(1) for r in range(int(order.max()) + 1)]
            self.levels.append((nodes, parents, tree.oriented(anchor.pairs, edge, flip), rounds))
        self._search()

    @property
    def fields(self) -> torch.Tensor:
        return self.anchor.fields

    @property
    def rank(self) -> int:
        return self.anchor.rank

    @property
    def shape(self) -> tuple[int, int]:
        return self.anchor.shape

    def _chunk(self):
        length, num_states = self.shape
        return max(1, _ELEMENTS // (length * num_states * self.width))

    def _upward(self, z, tables=False):
        """Restricted ``G_W(z)``; with ``tables`` also the count tables of the top-down draw."""
        count = len(z)
        length, num_states = self.shape
        width = self.width
        logits = self.anchor.fields[None] + torch.einsum("nk,iak->nia", z, self.anchor.weights)
        beliefs = torch.full((count, length, num_states, width + 1), -math.inf, dtype=logits.dtype, device=z.device)
        beliefs.scatter_(3, self.increments[None, :, :, None].expand(count, -1, -1, 1), logits[..., None])
        beliefs = beliefs[..., :width].contiguous()
        prefix = torch.full_like(beliefs, -math.inf) if tables else None
        messages = torch.full_like(beliefs, -math.inf) if tables else None
        for nodes, parents, pairs, rounds in reversed(self.levels):
            message = torch.logsumexp(pairs[None, :, :, :, None] + beliefs[:, nodes][:, :, None], 3)
            if tables:
                messages[:, nodes] = message
            for selected in rounds:
                children, owners = nodes[selected], parents[selected]
                if tables:
                    prefix[:, children] = beliefs[:, owners]
                beliefs[:, owners] = _log_convolve(beliefs[:, owners], message[:, selected])
        root_counts = torch.logsumexp(beliefs[:, self.roots], 2)
        root_prefix = torch.full_like(root_counts, -math.inf)
        total = root_counts[:, 0]
        for r in range(1, root_counts.shape[1]):
            root_prefix[:, r] = total
            total = _log_convolve(total, root_counts[:, r])
        value = torch.logsumexp(total[:, self.minimum:], -1)
        if not tables:
            return value
        return value, (beliefs, prefix, messages, root_counts, root_prefix, total)

    def _evaluate(self, keys):
        values = []
        for begin in range(0, len(keys), self._chunk()):
            z = self._points(keys[begin:begin + self._chunk()])
            values.append(self._upward(z) - 0.5 * (z ** 2).sum(-1))
        return torch.cat(values)

    def _points(self, keys):
        if self.rank == 0:
            return torch.zeros((len(keys), 0), dtype=torch.float64, device=keys.device)
        return _lattice_points(keys, self.rank).to(torch.float64) * _LATTICE_STEP

    @torch.no_grad()
    def _search(self):
        """Lattice points of the restricted latent density within ``_LATTICE_DEPTH`` of its maximum."""
        rank, device = self.rank, self.fields.device
        if rank == 0:
            self.keys = torch.zeros(1, dtype=torch.int64, device=device)
            self.values = self._upward(torch.zeros((1, 0), dtype=torch.float64, device=device))
            self.log_z = float(self.values[0])
        else:
            weights = self.anchor.weights
            stride = _SEED_STRIDE * _LATTICE_STEP
            low = torch.floor(weights.amin(1).sum(0) / stride).long()
            high = torch.ceil(weights.amax(1).sum(0) / stride).long()
            if int((high - low + 1).prod()) > _MAX_SEEDS:
                raise RuntimeError(f"The latent density of the rank-{rank} anchor spreads over too large a box.")
            axes = [torch.arange(int(low[k]), int(high[k]) + 1, device=device) * _SEED_STRIDE for k in range(rank)]
            keys = _lattice_keys(torch.cartesian_prod(*axes).reshape(-1, rank))
            coarse = self._evaluate(keys)
            keys = keys[coarse > coarse.max() - _SEED_DEPTH]
            neighbours = torch.zeros((2 * rank + 1, rank), dtype=torch.int64, device=device)
            for k in range(rank):
                neighbours[2 * k + 1, k], neighbours[2 * k + 2, k] = 1, -1
            offsets = _lattice_keys(neighbours) - _lattice_keys(neighbours[:1])
            values = self._evaluate(keys)
            while True:
                shell = torch.unique((keys[values > values.max() - _LATTICE_DEPTH][:, None] + offsets[None]).reshape(-1))
                fresh = shell[~torch.isin(shell, keys)]
                if not len(fresh):
                    break
                values = torch.cat([values, self._evaluate(fresh)])
                keys = torch.cat([keys, fresh])
            self.log_z = (float(torch.logsumexp(values, 0)) + rank * math.log(_LATTICE_STEP)
                          - 0.5 * rank * math.log(2 * math.pi))
            keep = values > values.max() - _DRAW_DEPTH
            self.keys, self.values = keys[keep], values[keep]
        if not math.isfinite(self.log_z):
            raise ValueError("The window and the allowed alphabet leave no sequence.")
        length, num_states = self.shape
        table_bytes = 3 * len(self.keys) * length * num_states * self.width * 8
        self._tables = self._build(torch.arange(len(self.keys), device=device)) if table_bytes <= _CACHE_BYTES else None

    def _build(self, indices):
        parts = []
        for begin in range(0, len(indices), self._chunk()):
            parts.append(self._upward(self._points(self.keys[indices[begin:begin + self._chunk()]]), tables=True)[1])
        return tuple(torch.cat(items) for items in zip(*parts))

    @property
    def log_probability(self) -> float:
        """``log P_A(window)``, the log probability of the window under the unrestricted anchor."""
        return self.log_z - self.anchor.log_z

    @torch.no_grad()
    def sample(self, n_samples: int, device=None) -> torch.Tensor:
        """Independent draws of the restricted anchor, int32 (n, L), with the global torch generator."""
        own = self.fields.device
        index = torch.multinomial(torch.softmax(self.values, 0), n_samples, replacement=True)
        if self._tables is not None:
            states = self._descend(self._tables, index)
        else:
            unique, inverse = torch.unique(index, return_inverse=True)
            drawn = torch.empty((n_samples, self.shape[0]), dtype=torch.long, device=own)
            for begin in range(0, len(unique), self._chunk()):
                block = unique[begin:begin + self._chunk()]
                rows = torch.nonzero((inverse >= begin) & (inverse < begin + len(block))).squeeze(1)
                drawn[rows] = self._descend(self._build(block), inverse[rows] - begin)
            states = drawn
        states = states.to(torch.int32)
        return states if device is None else states.to(device)

    def _descend(self, tables, index):
        beliefs, prefix, messages, root_counts, root_prefix, total = tables
        count = len(index)
        length = self.shape[0]
        device = index.device
        span = torch.arange(self.width, device=device)
        states = torch.zeros((count, length), dtype=torch.long, device=device)
        remaining = torch.zeros((count, length), dtype=torch.long, device=device)

        def split(own, before, budget):
            # Count of the last factor of a product, given the total budget and the product of the others.
            gap = budget[..., None] - span
            logits = own + before.gather(-1, gap.clamp_min(0))
            return _gumbel_argmax(logits.masked_fill(gap < 0, -math.inf))

        logits = total[index].clone()
        logits[:, :self.minimum] = -math.inf
        budget = _gumbel_argmax(logits)
        for r in reversed(range(len(self.roots))):
            share = budget if r == 0 else split(root_counts[index, r], root_prefix[index, r], budget)
            budget = budget - share
            root = self.roots[r]
            states[:, root] = _gumbel_argmax(beliefs[index, root].gather(-1, share[:, None, None].expand(-1, beliefs.shape[2], 1))
                                             .squeeze(-1))
            remaining[:, root] = share
        rows = index[:, None]
        for nodes, parents, pairs, rounds in self.levels:
            for selected in reversed(rounds):
                children, owners = nodes[selected], parents[selected]
                above = states[:, owners]
                share = split(messages[rows, children[None], above], prefix[rows, children[None], above],
                              remaining[:, owners])
                remaining[:, owners] -= share
                remaining[:, children] = share
            subtree = beliefs[rows, nodes[None]].gather(
                -1, remaining[:, nodes][:, :, None, None].expand(-1, -1, beliefs.shape[2], 1)).squeeze(-1)
            coupling = pairs[torch.arange(len(nodes), device=device)[None], states[:, parents]]
            states[:, nodes] = _gumbel_argmax(coupling + subtree)
        return states

    def score(self, states: torch.Tensor) -> torch.Tensor:
        """``s_A(x)`` of integer sequences (N, L) up to a constant, in float64."""
        return self.anchor._scores(states.to(device=self.fields.device).long())
