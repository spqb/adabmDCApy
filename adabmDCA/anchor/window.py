"""The tree + low-rank anchor restricted to Hamming windows, with exact draws and normalization.

The restricted law is

    p_W(x) ∝ exp(s_A(x)) prod_r 1{l_r <= d_r(x) <= u_r},

with ``s_A`` the score of :class:`~adabmDCA.anchor.anchor.TreeLowRankPotts` and
``d_r`` the number of substitutions from reference ``r``. At a fixed latent
coordinate ``z`` the anchor is a tree, and belief propagation with messages
indexed by the parent state *and* by a vector of counts of the child's subtree
gives, for every count vector ``c``,

    C_z(c) = sum over x with counts c of exp(s_tree(x; z)).

The counts are additive with nonnegative site increments, so they never
decrease up the tree and every count above its bound is dropped as soon as it
appears. With one window the single count is the number of substitutions, or
of matches when those are fewer. With several, the reference with the smallest
upper bound is primary: ``n`` counts substitutions from it, and every other
reference ``j`` adds ``t_j = k_j + n - d_j``, with ``k_j`` the number of sites
where the two references differ. ``t_j`` only grows at those sites (by 2 for the
letter of reference ``j``, 1 for a third letter), and ``t_j <= k_j + u - l_j``,
so the table stays small when the primary window is close. Only count vectors
that some subtree can reach are combined.

``log Z_W`` is the lattice sum of ``-|z|^2 / 2 + log sum_{c in W} C_z(c)``. For one
sequence the integrand is a unit Gaussian centred at ``m(x)``, so the lattice
sum has the same relative error for every sequence, whatever the windows select
(below 1e-8 at unit spacing for ``K <= 3``). Drawing a lattice point with its
weight, a count vector in the windows, and then the tree top-down, allocating
the remaining counts among the subtrees of each node, gives independent draws
of the same lattice mixture. The lattice is searched with the restricted
integrand, so windows that select a latent region the unrestricted anchor
rarely visits are still covered.
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
# Largest number of entries of the count table of one site and state.
_MAX_KEYS = 1 << 14
# Lattice points below this many nats from the largest weight are not drawn.
_DRAW_DEPTH = 40.0


def _gumbel_argmax(logits):
    return (logits - torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-300)))).argmax(-1)


class _Counts:
    """A box of count vectors, flattened, with the shifts that a factor can add."""

    def __init__(self, dims):
        self.dims = tuple(int(dim) for dim in dims)
        self.size = math.prod(self.dims)
        self.strides = np.array([math.prod(self.dims[a + 1:]) for a in range(len(self.dims))], dtype=np.int64)

    def flat(self, vectors):
        return np.asarray(vectors, dtype=np.int64) @ self.strides

    def boolean_convolve(self, first, second):
        result = np.zeros(self.dims, dtype=bool)
        for shift in np.argwhere(second):
            result[tuple(slice(s, None) for s in shift)] |= first[tuple(slice(0, d - s) for d, s in zip(self.dims, shift))]
        return result

    def log_convolve(self, first, second, shifts):
        """``log sum_s exp(first[c - s] + second[s])`` over ``shifts``, on flattened tables (..., size)."""
        lead = first.shape[:-1]
        box = first.reshape(*lead, *self.dims)
        result = torch.full_like(box, -math.inf)
        for shift in shifts:
            target = (Ellipsis, *(slice(s, None) for s in shift))
            source = (Ellipsis, *(slice(0, d - s) for d, s in zip(self.dims, shift)))
            term = box[source] + second[..., int(self.flat(shift))].reshape(*lead, *(1,) * len(self.dims))
            result[target] = torch.logaddexp(result[target], term)
        return result.reshape(*lead, self.size)


class WindowedAnchor:
    """A tree + low-rank anchor restricted to Hamming windows from one or more references.

    Args:
        anchor: :class:`~adabmDCA.anchor.anchor.TreeLowRankPotts` or :class:`~adabmDCA.anchor.lowrank.LowRankPotts`.
        references: Reference sequences (R, L) as token ids of the original alphabet.
        minimum: Smallest allowed number of substitutions from each reference (R,).
        maximum: Largest allowed number of substitutions from each reference (R,).
        tokens: Original token id of each anchor state (q,). The default is the identity.
            A reference letter outside ``tokens`` is a substitution for every state.

    The cost grows with the bound of the primary count (the smallest maximum, or
    with one window the smaller of ``maximum`` and ``L - minimum``) and with the
    secondary bounds ``k_j + u - l_j``, not with the width of the windows.

    Example:
        >>> restricted = WindowedAnchor(anchor, [reference], [0], [15])
        >>> restricted.log_z                 # log of the restricted partition function
        >>> restricted.sample(1000)          # independent draws in the windows, int32 (1000, L)
    """

    def __init__(self, anchor, references, minimum, maximum, tokens=None) -> None:
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
        references = np.asarray(references, dtype=np.int64).reshape(-1, length)
        minimum, maximum = np.asarray(minimum, dtype=np.int64).ravel(), np.asarray(maximum, dtype=np.int64).ravel()
        tokens = np.arange(num_states) if tokens is None else np.asarray(tokens, dtype=np.int64)
        if not len(references) or minimum.shape != (len(references),) or maximum.shape != minimum.shape:
            raise ValueError("Give one minimum and one maximum for each reference.")
        if tokens.shape != (num_states,) or (minimum < 0).any() or (minimum > maximum).any():
            raise ValueError("The windows need 0 <= minimum <= maximum and one token per anchor state.")
        maximum = np.minimum(maximum, length)
        references, minimum, maximum = self._reduce(references, minimum, maximum, tokens)
        mismatches = (tokens[None, None, :] != references[:, :, None]).astype(np.int64)  # (R, L, q)
        if len(references) == 1:
            if maximum[0] > length - minimum[0]:  # count matches, which are fewer
                increments, low, high = 1 - mismatches[0], length - maximum[0], length - minimum[0]
            else:
                increments, low, high = mismatches[0], minimum[0], maximum[0]
            increments = increments[:, :, None]
            limits = [high]
        else:
            primary = int(np.argmin(maximum))
            others = [j for j in range(len(references)) if j != primary]
            differ = references[others] != references[primary]  # (R - 1, L)
            increments = [mismatches[primary]] + [
                differ[a][:, None] + mismatches[primary] - mismatches[j] for a, j in enumerate(others)]
            increments = np.stack(increments, -1)
            limits = [maximum[primary]] + [differ[a].sum() + maximum[primary] - minimum[j] for a, j in enumerate(others)]
        offsets = increments.min(1).sum(0)
        increments = increments - increments.min(1, keepdims=True)
        dims = np.minimum(np.array(limits) - offsets, increments.max(1).sum(0)) + 1
        if (dims < 1).any():
            raise ValueError("The windows and the allowed alphabet leave no sequence.")
        if math.prod(dims.tolist()) > _MAX_KEYS:
            raise RuntimeError("The joint Hamming count table is too large.")
        counts = self.counts = _Counts(dims)

        # Count vectors that satisfy every window, in true (unshifted) counts.
        grid = np.indices(counts.dims).reshape(len(counts.dims), -1).T + offsets
        if len(references) == 1:
            accept = (grid[:, 0] >= low) & (grid[:, 0] <= high)
        else:
            n = grid[:, 0]
            accept = (n >= minimum[primary]) & (n <= maximum[primary])
            for a, j in enumerate(others):
                distance = differ[a].sum() + n - grid[:, a + 1]
                accept &= (distance >= minimum[j]) & (distance <= maximum[j])
        self.accept = torch.as_tensor(accept, device=device)

        inside = (increments < dims).all(-1)  # (L, q): states whose own counts stay in the box
        keys = np.where(inside, increments @ counts.strides, counts.size)
        self.keys_of_states = torch.as_tensor(keys, device=device)
        reach = [np.zeros(counts.dims, dtype=bool) for _ in range(length)]
        for site in range(length):
            for state in np.flatnonzero(inside[site]):
                reach[site][tuple(increments[site, state])] = True

        def shifts(nodes):
            union = np.zeros(counts.dims, dtype=bool)
            for node in nodes:
                union |= reach[node]
            vectors = np.argwhere(union)
            return [tuple(v) for v in vectors], torch.as_tensor(counts.flat(vectors), device=device)

        # Children of each parent are multiplied in rounds, one child per parent per round.
        tree = anchor._tree
        self.roots = tree.roots
        self.levels = [None] * len(tree.levels)
        for depth in reversed(range(len(tree.levels))):
            nodes, parents, edge, flip = tree.levels[depth]
            seen, order = {}, []
            for parent in parents.tolist():
                order.append(seen.get(parent, 0))
                seen[parent] = order[-1] + 1
            order = torch.tensor(order, device=device)
            rounds = []
            for r in range(int(order.max()) + 1):
                selected = torch.nonzero(order == r).squeeze(1)
                children, owners = nodes[selected].tolist(), parents[selected].tolist()
                rounds.append((selected, *shifts(children)))
                for child, owner in zip(children, owners):
                    reach[owner] = counts.boolean_convolve(reach[owner], reach[child])
            self.levels[depth] = (nodes, parents, tree.oriented(anchor.pairs, edge, flip), rounds)
        roots = self.roots.tolist()
        self.root_shifts = [shifts([root]) for root in roots]
        total = reach[roots[0]]
        for root in roots[1:]:
            total = counts.boolean_convolve(total, reach[root])
        if not (total.reshape(-1) & accept).any():
            raise ValueError("The windows and the allowed alphabet leave no sequence.")
        self._search()

    @staticmethod
    def _reduce(references, minimum, maximum, tokens):
        """Merge windows on the same reference and drop those that the tightest window implies.

        A reference letter outside the alphabet is a substitution for every state,
        so only references that agree on the letters inside it are merged.
        """
        inside = np.isin(references, tokens)
        keys = np.where(inside, references, -1)
        merged = {}
        for key, low, high in zip(map(bytes, keys.astype(np.int64)), minimum, maximum):
            previous = merged.get(key)
            merged[key] = (low, high) if previous is None else (max(previous[0], low), min(previous[1], high))
        keys = list(merged)
        references = np.stack([np.frombuffer(key, dtype=np.int64) for key in keys])
        minimum = np.array([merged[key][0] for key in keys])
        maximum = np.array([merged[key][1] for key in keys])
        if (minimum > maximum).any():
            raise ValueError("The windows and the allowed alphabet leave no sequence.")
        # |d_j - k_j| <= d_p by the triangle inequality, with k_j the distance between the references.
        primary = int(np.argmin(maximum))
        differ = (references != references[primary]).sum(1)
        kept = [j for j in range(len(references)) if j == primary
                or not (differ[j] - maximum[primary] >= minimum[j] and differ[j] + maximum[primary] <= maximum[j])]
        return references[kept], minimum[kept], maximum[kept]

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
        return max(1, _ELEMENTS // (length * num_states * self.counts.size))

    def _upward(self, z, tables=False):
        """Restricted ``G_W(z)``; with ``tables`` also the count tables of the top-down draw."""
        count = len(z)
        length, num_states = self.shape
        size = self.counts.size
        logits = self.anchor.fields[None] + torch.einsum("nk,iak->nia", z, self.anchor.weights)
        beliefs = torch.full((count, length, num_states, size + 1), -math.inf, dtype=logits.dtype, device=z.device)
        beliefs.scatter_(3, self.keys_of_states[None, :, :, None].expand(count, -1, -1, 1), logits[..., None])
        beliefs = beliefs[..., :size].contiguous()
        prefix = torch.full_like(beliefs, -math.inf) if tables else None
        messages = torch.full_like(beliefs, -math.inf) if tables else None
        for nodes, parents, pairs, rounds in reversed(self.levels):
            message = torch.logsumexp(pairs[None, :, :, :, None] + beliefs[:, nodes][:, :, None], 3)
            if tables:
                messages[:, nodes] = message
            for selected, shifts, _ in rounds:
                children, owners = nodes[selected], parents[selected]
                if tables:
                    prefix[:, children] = beliefs[:, owners]
                beliefs[:, owners] = self.counts.log_convolve(beliefs[:, owners], message[:, selected], shifts)
        root_counts = torch.logsumexp(beliefs[:, self.roots], 2)
        root_prefix = torch.full_like(root_counts, -math.inf)
        total = root_counts[:, 0]
        for r in range(1, root_counts.shape[1]):
            root_prefix[:, r] = total
            total = self.counts.log_convolve(total, root_counts[:, r], self.root_shifts[r][0])
        value = torch.logsumexp(total.masked_fill(~self.accept, -math.inf), -1)
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
            raise ValueError("The windows and the allowed alphabet leave no sequence.")
        length, num_states = self.shape
        table_bytes = 3 * len(self.keys) * length * num_states * self.counts.size * 8
        self._tables = self._build(torch.arange(len(self.keys), device=device)) if table_bytes <= _CACHE_BYTES else None

    def _build(self, indices):
        parts = []
        for begin in range(0, len(indices), self._chunk()):
            parts.append(self._upward(self._points(self.keys[indices[begin:begin + self._chunk()]]), tables=True)[1])
        return tuple(torch.cat(items) for items in zip(*parts))

    @property
    def log_probability(self) -> float:
        """``log P_A(windows)``, the log probability of the windows under the unrestricted anchor."""
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
        length, num_states = self.shape
        device = index.device
        strides = torch.as_tensor(self.counts.strides, device=device)
        dims = torch.as_tensor(self.counts.dims, device=device)
        states = torch.zeros((count, length), dtype=torch.long, device=device)
        remaining = torch.zeros((count, length), dtype=torch.long, device=device)

        def split(own, before, budget, keys):
            # Counts of the last factor of a product, given the total and the product of the others.
            vectors = (keys[:, None] // strides) % dims                      # (k, A)
            left = (budget[..., None] // strides) % dims                     # (..., A)
            valid = (left[..., None, :] >= vectors).all(-1)                  # (..., k)
            gap = (budget[..., None] - keys).clamp_min(0)
            logits = own.gather(-1, keys.expand(*own.shape[:-1], -1)) + before.gather(-1, gap)
            return keys[_gumbel_argmax(logits.masked_fill(~valid, -math.inf))]

        logits = total[index].masked_fill(~self.accept, -math.inf)
        budget = _gumbel_argmax(logits)
        for r in reversed(range(len(self.roots))):
            share = budget if r == 0 else split(root_counts[index, r], root_prefix[index, r], budget,
                                                self.root_shifts[r][1])
            budget = budget - share
            root = self.roots[r]
            states[:, root] = _gumbel_argmax(beliefs[index, root].gather(-1, share[:, None, None].expand(-1, num_states, 1))
                                             .squeeze(-1))
            remaining[:, root] = share
        rows = index[:, None]
        for nodes, parents, pairs, rounds in self.levels:
            for selected, _, keys in reversed(rounds):
                children, owners = nodes[selected], parents[selected]
                above = states[:, owners]
                share = split(messages[rows, children[None], above], prefix[rows, children[None], above],
                              remaining[:, owners], keys)
                remaining[:, owners] -= share
                remaining[:, children] = share
            subtree = beliefs[rows, nodes[None]].gather(
                -1, remaining[:, nodes][:, :, None, None].expand(-1, -1, num_states, 1)).squeeze(-1)
            coupling = pairs[torch.arange(len(nodes), device=device)[None], states[:, parents]]
            states[:, nodes] = _gumbel_argmax(coupling + subtree)
        return states

    def score(self, states: torch.Tensor) -> torch.Tensor:
        """``s_A(x)`` of integer sequences (N, L) up to a constant, in float64."""
        return self.anchor._scores(states.to(device=self.fields.device).long())
