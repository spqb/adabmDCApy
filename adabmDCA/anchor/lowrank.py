"""Low-rank Potts models with exact sampling and normalization, used as the bottom of a PTT ladder.

A Potts model whose couplings have rank ``K``,

    J_ij(a, b) = sum_k W_ik(a) W_jk(b)        (i != j),

is a restricted Boltzmann machine with ``K`` Gaussian hidden units (the
Hubbard-Stratonovich transform). With ``g_i(a) = h_i(a) - |W_i(a)|^2 / 2`` and
the energy ``-sum_i h_i(x_i) - 1/2 sum_{i != j} J_ij(x_i, x_j)`` of
:func:`adabmDCA.statmech.compute_energy`,

    exp(-E(x)) = exp(sum_i g_i(x_i)) E_z[exp(z . m(x))],   m(x) = sum_i W_i(x_i),   z ~ N(0, I_K).

Summing out the sequence leaves a density on the ``K`` latent coordinates,

    p(z) ∝ exp(F(z)),   F(z) = -|z|^2 / 2 + sum_i log sum_a exp(g_i(a) + z . W_i(a)),

and given ``z`` the sites are independent: ``p(x_i = a | z) ∝ exp(g_i(a) + z . W_i(a))``.
``exp(F)`` is a mixture of unit Gaussians centred at the projections ``m(x)``, so
its sum over a regular lattice of spacing 1 gives ``log Z`` and its derivatives
to about ``1e-8`` (:func:`_lattice_sum`). For sampling, the latent density is
covered by adaptive cells (:class:`LatentCells`): ``F`` is a convex function minus
``|z|^2 / 2``, which gives rigorous upper and lower bounds of ``F`` on every cell.
Rejection sampling from the upper bounds draws ``z`` exactly, and the sequence is
then drawn site by site, so samples are exact without Markov chains.

Only positive semi-definite couplings can be written this way. The rank-``K``
model is fitted by maximum likelihood (:func:`fit_low_rank_potts`) from the
single-site and pair frequencies of the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable
from dataclasses import fields as dataclass_fields

import torch

# Cells whose largest possible mass is below exp(-PRUNE) times the smallest possible log Z are dropped.
# Their total mass is at most (number of cells) * exp(-PRUNE) of Z: below 1e-15 for a million cells.
_PRUNE = 50.0
# Cells are refined until the upper bound of their mass exceeds the lower bound by at most this (rejection
# sampling then accepts at least exp(-tolerance) of the proposals in each cell, and about 0.95 overall) ...
_TOLERANCE = 0.3
# ... unless their largest possible mass is below exp(-_REFINE) of Z: they barely affect acceptance.
_REFINE = 20.0
_MAX_CELLS = 4_000_000
# Spacing of the lattice on which exp(F) is summed. exp(F) is a mixture of unit Gaussians, so the sum
# misses the integral by a relative 2 K exp(-2 pi^2 / spacing^2) at most: 2e-8 for K = 3.
_LATTICE_STEP = 1.0
# Lattice points are kept while F is within this of its maximum ...
_LATTICE_DEPTH = 32.0
# ... and searched from the nodes of a coarse lattice (every _SEED_STRIDE-th point) within _SEED_DEPTH of it.
_SEED_STRIDE = 3
_SEED_DEPTH = 30.0
_MAX_SEEDS = 20_000_000
_KEY_BASE = 1 << 20
# Largest number of float64 elements of one temporary (128 MB).
_ELEMENTS = 1 << 24
# The (L q) x (L q) matrix of pair frequencies is formed when it takes at most this many bytes; above,
# its products are computed from the sequences.
_PAIR_MATRIX_BYTES = 1 << 30
# Subspace iteration for the top modes of the correlation matrix: extra vectors, iterations, tolerance.
_EXTRA_MODES = 8
_MODE_ITERATIONS = 500
_MODE_TOLERANCE = 1e-9


def _chunk(fields):
    """Number of latent points whose (L, q) site tables fit in one temporary."""
    return max(1, _ELEMENTS // fields.numel())


def _site_logits(fields, weights, z):
    """Logits ``g_i(a) + z . W_i(a)`` of every site, shape (n, L, q), for latent points ``z`` (n, K)."""
    return fields + torch.einsum("nk,iak->nia", z, weights)


def _convex_part(fields, weights, z, order=0):
    """``G(z) = sum_i log sum_a exp(g_i(a) + z . W_i(a))``; with ``order`` 1 also its gradient ``sum_i E[W_i]``."""
    values, gradients = [], []
    chunk = _chunk(fields)
    for start in range(0, len(z), chunk):
        logits = _site_logits(fields, weights, z[start:start + chunk])
        values.append(torch.logsumexp(logits, -1).sum(-1))
        if order:
            gradients.append(torch.einsum("nia,iak->nk", logits.softmax(-1), weights))
    if not order:
        return torch.cat(values)
    return torch.cat(values), torch.cat(gradients)


def _site_curvature(weights):
    """``D_i^2 / 4`` per site, with ``D_i`` the diameter of {W_i(a)}: ``Cov(W_i) <= D_i^2 / 4 I`` under any distribution.

    The covariance of a vector confined to a set of diameter D is at most D^2/4
    in every direction (Popoviciu's inequality).
    """
    differences = weights[:, :, None, :] - weights[:, None, :, :]
    return (differences ** 2).sum(-1).amax((1, 2)) / 4


def _cell_statistics(fields, weights, centers, half_widths, popoviciu):
    """``G`` and its gradient at the cell centres, and matrices enclosing the Hessian of ``G`` over each cell.

    For ``z = c + d`` in the cell, the site probabilities change by at most
    ``exp(-delta_a - rho^2 / 8) <= p(a | z) / p(a | c) <= exp(delta_a)``, with
    ``delta_a = sum_k h_k |W_k(a) - E_c[W_k]|`` and ``rho = sum_k h_k (max_a W_k(a) - min_a W_k(a))``
    (Jensen's inequality and Hoeffding's lemma for ``E_c[exp(d . W)]``). Since a
    covariance is at most the second moment about any point, and at least the
    weighted covariance of any smaller weights,

        sum_a w_a (W(a) - mean_w) (W(a) - mean_w)' <= Cov_z(W_i) <= sum_a p(a|c) e^{delta_a} (W(a) - E_c W)(W(a) - E_c W)'

    with ``w_a = p(a|c) exp(-delta_a - rho^2/8)``. A site whose upper matrix is
    larger than Popoviciu's bound ``D_i^2 / 4 I`` (large cells) uses the latter.

    Returns:
        ``G(c)``, ``grad G(c)`` (n, K), and the upper and lower matrices (n, K, K), summed over sites.
    """
    length, num_states, rank = weights.shape
    ranges = weights.amax(1) - weights.amin(1)  # (L, K)
    eye = torch.eye(rank, dtype=weights.dtype, device=weights.device)
    chunk = max(1, _ELEMENTS // (fields.numel() * rank))
    values, gradients, uppers, lowers = [], [], [], []
    for start in range(0, len(centers), chunk):
        half = half_widths[start:start + chunk]
        logits = _site_logits(fields, weights, centers[start:start + chunk])
        normalizers = torch.logsumexp(logits, -1, keepdim=True)
        values.append(normalizers.sum((1, 2)))
        log_probabilities = logits - normalizers
        means = torch.einsum("nia,iak->nik", log_probabilities.exp(), weights)
        gradients.append(means.sum(1))
        deviations = weights[None] - means[:, :, None, :]  # (n, L, q, K)
        spread = (deviations.abs() * half[:, None, None, :]).sum(-1)  # delta_a, (n, L, q)
        upper = torch.einsum("nia,niak,nial->nikl", (log_probabilities + spread).exp(), deviations, deviations)
        # Overflowing or useless local bounds (NaN or inf traces fail the comparison) fall back to Popoviciu.
        local = upper.diagonal(0, -2, -1).sum(-1) <= rank * popoviciu
        uppers.append(torch.where(local[..., None, None], upper, popoviciu[:, None, None] * eye).sum(1))
        rho = (half[:, None, :] * ranges[None]).sum(-1)  # (n, L)
        lower_weights = (log_probabilities - spread - rho[..., None] ** 2 / 8).exp()
        mass = lower_weights.sum(-1)
        first = torch.einsum("nia,iak->nik", lower_weights, weights)
        second = torch.einsum("nia,iak,ial->nikl", lower_weights, weights, weights)
        lower = second - first[..., :, None] * first[..., None, :] / mass.clamp_min(1e-300)[..., None, None]
        lowers.append(lower.sum(1))
    return torch.cat(values), torch.cat(gradients), torch.cat(uppers), torch.cat(lowers)


def _vertex_quadratic(matrices, half_widths):
    """``max over the box of d' A d <= max over its vertices of d' A_+ d``, with ``A_+`` the positive part of ``A``.

    Also returns ``h_k^2 (A_+)_kk``, the share of each direction, which picks the direction to split.
    """
    # On the CPU: the batched GPU eigensolver can ask for gigabytes of workspace for small matrices.
    eigenvalues, vectors = torch.linalg.eigh(matrices.cpu())
    positive = ((vectors * eigenvalues.clamp_min(0.0)[:, None, :]) @ vectors.mT).to(matrices.device)
    rank = matrices.shape[-1]
    signs = torch.tensor([[1.0 if (vertex >> k) & 1 else -1.0 for k in range(rank)] for vertex in range(2 ** rank)],
                         dtype=matrices.dtype, device=matrices.device)
    vertices = signs[None] * half_widths[:, None, :]
    quadratic = torch.einsum("nvk,nkl,nvl->nv", vertices, positive, vertices).amax(-1)
    return quadratic, half_widths ** 2 * positive.diagonal(0, -2, -1)


def _log_sinhc(slope, half_width):
    """``log of the integral of exp(slope t) over [-h, h]`` = log(2 sinh(slope h) / slope), stably."""
    x = (slope * half_width).abs()
    small = x < 1e-3
    safe = torch.where(small, torch.ones_like(x), x)
    large = safe + torch.log1p(-torch.exp(-2 * safe)) - torch.log(2 * safe)
    return torch.log(2 * half_width) + torch.where(small, x ** 2 / 6, large)


@dataclass(frozen=True)
class LatentCells:
    """Boxes covering the latent density, with the proposal of the exact sampler on each.

    A cell either proposes uniformly (``tilted`` false), with the envelope
    ``envelope`` on ``F``, or proportionally to ``exp(F(c) + slope . (z - c))``
    (``tilted`` true), with ``F(z) <= F(c) + slope . (z - c) + envelope``.

    Attributes:
        centers: Cell centres, shape (n, K).
        half_widths: Cell half widths, shape (n, K).
        center_values: ``F`` at the centres.
        slopes: Gradient of ``F`` at the centres, shape (n, K).
        tilted: Whether the cell proposes from the tilted density.
        envelope: Flat cells: upper bound of ``F``; tilted cells: bound of the second-order remainder.
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

    @property
    def log_volumes(self) -> torch.Tensor:
        return torch.log(2 * self.half_widths).sum(-1)


def _bound_cells(fields, weights, centers, half_widths, popoviciu):
    """Proposal masses (upper bounds) and lower bounds of every cell's share of ``Z``.

    With ``L <= Hessian(G) <= U`` on the cell (:func:`_cell_statistics`), ``F``
    lies between its tangent plane at the centre plus ``d' (L - I) d / 2`` and
    plus ``d' (U - I) d / 2``. The quadratic terms are bounded over the box by
    :func:`_vertex_quadratic`, which leaves a tilted bound whose mass is a product
    of one-dimensional exponential integrals. On large cells a flat bound can be
    smaller: ``G`` is convex, so its maximum is at a vertex, and ``-|z|^2/2`` is
    largest at the point closest to the origin. Each cell proposes from the
    smaller of the two.

    Returns:
        The :class:`LatentCells` and, for every cell, the direction along which to halve it.
    """
    rank = centers.shape[1]
    value, gradient, upper, lower = _cell_statistics(fields, weights, centers, half_widths, popoviciu)
    eye = torch.eye(rank, dtype=centers.dtype, device=centers.device)
    center_values = value - 0.5 * (centers ** 2).sum(-1)
    slopes = gradient - centers
    above, above_share = _vertex_quadratic(upper - eye, half_widths)
    below, below_share = _vertex_quadratic(eye - lower, half_widths)
    # Rounding must not push F above a bound that holds exactly.
    remainder = 0.5 * above + 1e-9 * (1.0 + center_values.abs())
    tilt = center_values + _log_sinhc(slopes, half_widths).sum(-1)
    log_volumes = torch.log(2 * half_widths).sum(-1)
    tilted_mass = tilt + remainder
    flat = torch.full_like(tilt, math.inf)
    # The flat bound can only win on cells whose tilted bound is loose.
    loose = remainder > 1.0
    if loose.any():
        signs = torch.tensor([[1.0 if (vertex >> k) & 1 else -1.0 for k in range(rank)]
                              for vertex in range(2 ** rank)], dtype=centers.dtype, device=centers.device)
        vertices = centers[loose][:, None, :] + signs[None] * half_widths[loose][:, None, :]
        vertex_values = _convex_part(fields, weights, vertices.reshape(-1, rank)).reshape(len(vertices), -1)
        closest = (centers[loose].abs() - half_widths[loose]).clamp_min(0.0)
        bound = vertex_values.max(-1).values - 0.5 * (closest ** 2).sum(-1)
        flat[loose] = torch.maximum(bound, center_values[loose]) + 1e-9 * (1.0 + bound.abs())
    flat_mass = flat + log_volumes
    tilted = tilted_mass <= flat_mass
    log_mass = torch.minimum(flat_mass, tilted_mass)
    envelope = torch.where(tilted, remainder, flat)
    log_mass_lower = tilt - 0.5 * below
    split = (above_share + below_share).argmax(-1)
    cells = LatentCells(centers, half_widths, center_values, slopes, tilted, envelope, log_mass, log_mass_lower)
    return cells, split


def _select(cells, mask):
    return LatentCells(*(getattr(cells, field.name)[mask] for field in dataclass_fields(LatentCells)))


def _concatenate(first, second):
    return LatentCells(*(torch.cat([getattr(first, field.name), getattr(second, field.name)])
                         for field in dataclass_fields(LatentCells)))


def latent_cells(fields: torch.Tensor, weights: torch.Tensor, *, tolerance: float = _TOLERANCE) -> LatentCells:
    """Cover the latent density of a low-rank model by cells, refined where it has mass.

    Starts from a box outside which ``F`` is provably negligible and halves cells,
    one direction at a time, until, on every cell that can carry more than
    ``exp(-20)`` of the mass, the proposal mass exceeds the lower bound of the
    cell's mass by at most ``tolerance``. Cells that cannot carry more than
    ``exp(-50)`` of the mass are dropped.

    Args:
        fields: ``g`` of shape (L, q), float64.
        weights: ``W`` of shape (L, q, K), float64.
        tolerance: Refinement tolerance, in log mass.

    Returns:
        The final :class:`LatentCells`.
    """
    rank = weights.shape[2]
    dtype, device = weights.dtype, weights.device
    popoviciu = _site_curvature(weights)
    # G(z) <= A + sum_k S_k |z_k|, so F <= A + sum_k (S_k |z_k| - z_k^2 / 2): a box outside which
    # F stays exp(-_PRUNE) below F(0).
    num_states = fields.shape[1]
    peak = float(fields.max(-1).values.sum()) + fields.shape[0] * math.log(num_states)
    slopes = weights.abs().amax(1).sum(0)
    zero = torch.zeros((1, rank), dtype=dtype, device=device)
    excess = peak - float(_convex_part(fields, weights, zero)[0]) + _PRUNE
    radii = []
    for k in range(rank):
        others = float((slopes ** 2).sum() - slopes[k] ** 2) / 2
        slope = float(slopes[k])
        radii.append(slope + math.sqrt(slope ** 2 + 2 * (excess + others)) + 1.0)
    # Start with 8 cells per direction.
    mids = []
    for radius in radii:
        edge = torch.linspace(-radius, radius, 9, dtype=dtype, device=device)
        mids.append(0.5 * (edge[1:] + edge[:-1]))
    centers = torch.cartesian_prod(*mids).reshape(-1, rank)
    half_widths = torch.tensor([radius / 8 for radius in radii], dtype=dtype, device=device)
    half_widths = half_widths.expand(len(centers), -1).contiguous()
    cells, split = _bound_cells(fields, weights, centers, half_widths, popoviciu)
    while True:
        floor = float(torch.logsumexp(cells.log_mass_lower, 0))  # log Z is at least this
        keep = cells.log_mass > floor - _PRUNE
        refine = keep & (cells.log_mass - cells.log_mass_lower > tolerance) & (cells.log_mass > floor - _REFINE)
        if not refine.any():
            return _select(cells, keep)
        stay = keep & ~refine
        if int(stay.sum()) + 2 * int(refine.sum()) > _MAX_CELLS:
            raise RuntimeError(f"The latent density of the rank-{rank} model needs more than {_MAX_CELLS} cells.")
        # Halve every refined cell along the direction that contributes most to the gap between its bounds.
        direction = split[refine][:, None]
        children_half = cells.half_widths[refine].scatter(1, direction, cells.half_widths[refine].gather(1, direction) / 2)
        offsets = torch.zeros_like(children_half).scatter(1, direction, children_half.gather(1, direction))
        children_centers = torch.cat([cells.centers[refine] - offsets, cells.centers[refine] + offsets])
        children, children_split = _bound_cells(fields, weights, children_centers,
                                                torch.cat([children_half, children_half]), popoviciu)
        cells = _concatenate(_select(cells, stay), children)
        split = torch.cat([split[stay], children_split])


def _lattice_keys(index):
    """One integer per lattice point (n, K); neighbours differ by fixed offsets of the keys."""
    keys = torch.zeros(len(index), dtype=torch.int64, device=index.device)
    for k in range(index.shape[1]):
        keys = keys * _KEY_BASE + (index[:, k] + _KEY_BASE // 2)
    return keys


def _lattice_points(keys, rank):
    columns = []
    for _ in range(rank):
        columns.append(keys % _KEY_BASE - _KEY_BASE // 2)
        keys = keys // _KEY_BASE
    return torch.stack(columns[::-1], 1)


def _lattice_sum(fields, weights, start=None, gradient=False):
    """``log of the integral of exp(F)`` as a sum over the lattice of spacing ``_LATTICE_STEP``.

    ``exp(F)`` is a mixture of unit Gaussians centred at the projections ``m(x)``,
    so its Fourier transform decays as ``exp(-|k|^2 / 2)`` and the lattice sum
    converges to the integral faster than any power of the spacing. Only the
    points within ``_LATTICE_DEPTH`` of the maximum of ``F`` are summed. They are
    found by growing the set from ``start`` (the points of a previous call) and
    from the nodes of a coarse lattice over the box that contains every ``m(x)``,
    which sees every mode of the mixture.

    Returns:
        The log integral, the keys of the summed points with their neighbours (the ``start`` of the
        next call), their number, and with ``gradient`` the derivatives with respect to ``fields`` and ``weights``.
    """
    length, num_states, rank = weights.shape
    dtype, device = weights.dtype, weights.device
    columns = length * num_states
    flat_fields = fields.reshape(columns)
    flat_weights = weights.reshape(columns, rank)
    chunk = _chunk(fields)

    def evaluate(keys, reference=None, moments=None):
        """``F`` on lattice points; with ``moments`` also adds their weights and weighted site marginals to it."""
        values = []
        for begin in range(0, len(keys), chunk):
            z = _lattice_points(keys[begin:begin + chunk], rank).to(dtype) * _LATTICE_STEP
            logits = torch.addmm(flat_fields[None], z, flat_weights.T).view(len(z), length, num_states)
            normalizer = torch.logsumexp(logits, -1, keepdim=True)
            value = normalizer.sum((1, 2)) - 0.5 * (z ** 2).sum(-1)
            values.append(value)
            if moments is not None:
                mass = (value - reference).exp()
                moments[0] += mass.sum()
                if gradient:
                    probabilities = logits.sub_(normalizer).exp_().view(len(z), columns)
                    moments[1] += torch.cat([mass[:, None], mass[:, None] * z], 1).T @ probabilities
        return torch.cat(values)

    # Every m(x) lies in the box [sum_i min_a W_i(a), sum_i max_a W_i(a)], and a unit Gaussian centred in
    # it loses at most exp(-K (stride * spacing)^2 / 8) at the nearest coarse node.
    low = torch.floor(weights.amin(1).sum(0) / (_SEED_STRIDE * _LATTICE_STEP)).long()
    high = torch.ceil(weights.amax(1).sum(0) / (_SEED_STRIDE * _LATTICE_STEP)).long()
    if int((high - low + 1).prod()) > _MAX_SEEDS:
        raise RuntimeError(f"The latent density of the rank-{rank} model spreads over too large a box.")
    axes = [torch.arange(int(low[k]), int(high[k]) + 1, device=device) * _SEED_STRIDE for k in range(rank)]
    keys = _lattice_keys(torch.cartesian_prod(*axes).reshape(-1, rank))
    coarse = evaluate(keys)
    reference = coarse.max()
    keys = keys[coarse > reference - _SEED_DEPTH]
    if start is not None:
        keys = torch.unique(torch.cat([start, keys]))
    neighbours = torch.zeros((2 * rank + 1, rank), dtype=torch.int64, device=device)
    for k in range(rank):
        neighbours[2 * k + 1, k], neighbours[2 * k + 2, k] = 1, -1
    offsets = _lattice_keys(neighbours) - _lattice_keys(neighbours[:1])
    moments = [torch.zeros((), dtype=dtype, device=device),
               torch.zeros((rank + 1, columns), dtype=dtype, device=device)]
    values = evaluate(keys, reference, moments)
    while True:
        shell = torch.unique((keys[values > values.max() - _LATTICE_DEPTH][:, None] + offsets[None]).reshape(-1))
        fresh = shell[~torch.isin(shell, keys)]
        if not len(fresh):
            break
        values = torch.cat([values, evaluate(fresh, reference, moments)])
        keys = torch.cat([keys, fresh])
    log_integral = float(reference + moments[0].log()) + rank * math.log(_LATTICE_STEP)
    if not gradient:
        return log_integral, shell, len(keys)
    derivative = (moments[1] / moments[0]).view(rank + 1, length, num_states)
    return log_integral, shell, len(keys), derivative[0], derivative[1:].permute(1, 2, 0)


class LowRankPotts:
    """A rank-``K`` Potts model with exact sampling and normalization.

    Parameterized by the hidden-unit fields ``g`` (L, q) and the weights ``W``
    (L, q, K): the Potts couplings are ``J_ij(a, b) = W_i(a) . W_j(b)`` for
    ``i != j`` and the fields ``h_i(a) = g_i(a) + |W_i(a)|^2 / 2``. Latent
    computations run in float64 on the device of the parameters (on CPU for MPS).

    Example:
        >>> anchor = fit_low_rank_potts(sequences, weights, num_states=21, rank=2)
        >>> params = anchor.potts_params(torch.float32)   # {"bias", "coupling_matrix"}
        >>> states = anchor.sample(1000)                   # exact draws, int32 (1000, L)
        >>> anchor.log_z                                   # log partition function
    """

    def __init__(self, fields: torch.Tensor, weights: torch.Tensor, *, tolerance: float = _TOLERANCE) -> None:
        if fields.ndim != 2 or weights.ndim != 3 or weights.shape[:2] != fields.shape:
            raise ValueError("Low-rank Potts fields must be (L, q) and weights (L, q, K).")
        if weights.shape[2] < 1:
            raise ValueError("A low-rank Potts model needs at least one latent direction.")
        device = torch.device(fields.device)
        device = device if device.type in ("cpu", "cuda") else torch.device("cpu")
        self.fields = fields.detach().to(device=device, dtype=torch.float64).contiguous()
        self.weights = weights.detach().to(device=device, dtype=torch.float64).contiguous()
        if not (torch.isfinite(self.fields).all() and torch.isfinite(self.weights).all()):
            raise ValueError("Low-rank Potts parameters must be finite.")
        self.tolerance = tolerance
        self._cells = None
        self._log_z = None

    @property
    def cells(self) -> LatentCells:
        """The latent cells of the exact sampler, built on first use."""
        if self._cells is None:
            self._cells = latent_cells(self.fields, self.weights, tolerance=self.tolerance)
        return self._cells

    @property
    def log_z(self) -> float:
        """log Z of the Potts model, from the lattice sum of the latent density (it needs no cells)."""
        if self._log_z is None:
            gaussian = 0.5 * self.rank * math.log(2 * math.pi)  # the normalization of z ~ N(0, I)
            self._log_z = _lattice_sum(self.fields, self.weights)[0] - gaussian
        return self._log_z

    @property
    def log_z_bounds(self) -> tuple[float, float]:
        """Rigorous lower and upper bounds of log Z (up to cells dropped below exp(-50) of Z each)."""
        gaussian = 0.5 * self.rank * math.log(2 * math.pi)
        return (float(torch.logsumexp(self.cells.log_mass_lower, 0)) - gaussian,
                float(torch.logsumexp(self.cells.log_mass, 0)) - gaussian)

    @property
    def acceptance(self) -> float:
        """Expected acceptance of the latent rejection sampler."""
        return math.exp(self.log_z - self.log_z_bounds[1])

    def to(self, device) -> LowRankPotts:
        """The same model on ``device``; MPS models compute on CPU."""
        return LowRankPotts(self.fields.to(device), self.weights.to(device), tolerance=self.tolerance)

    @property
    def rank(self) -> int:
        return self.weights.shape[2]

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.fields.shape)

    def potts_params(self, dtype: torch.dtype = torch.float32, device=None) -> dict[str, torch.Tensor]:
        """The model as Potts parameters ``{"bias", "coupling_matrix"}`` with zero within-site blocks."""
        length = self.fields.shape[0]
        bias = self.fields + 0.5 * (self.weights ** 2).sum(-1)
        couplings = torch.einsum("iak,jbk->iajb", self.weights, self.weights)
        couplings[torch.arange(length), :, torch.arange(length), :] = 0.0
        device = self.fields.device if device is None else device
        return {"bias": bias.to(device=device, dtype=dtype).contiguous(),
                "coupling_matrix": couplings.to(device=device, dtype=dtype).contiguous()}

    def latent_log_density(self, z: torch.Tensor) -> torch.Tensor:
        """``F(z)``, the unnormalized log density of the latent coordinates, for ``z`` of shape (n, K)."""
        z = z.to(device=self.fields.device, dtype=torch.float64)
        return _convex_part(self.fields, self.weights, z) - 0.5 * (z ** 2).sum(-1)

    def sample_latent(self, n_samples: int) -> torch.Tensor:
        """Exact draws of the latent coordinates, shape (n, K), by rejection from the cell envelopes.

        Uses the global torch random generator of the parameters' device.
        """
        cells = self.cells
        device = self.fields.device
        accepted = []
        missing = n_samples
        probabilities = (cells.log_mass - cells.log_mass.max()).exp()
        while missing > 0:
            batch = max(64, int(missing / self.acceptance * 1.1) + 16)
            index = torch.multinomial(probabilities, batch, replacement=True)
            half = cells.half_widths[index]
            slope = cells.slopes[index]
            tilted = cells.tilted[index]
            uniform = torch.rand((batch, self.rank), dtype=torch.float64, device=device)
            # Inverse CDF of the density proportional to exp(slope t) on [-h, h], per direction.
            x = slope * half
            steep = x.abs() > 1e-6
            safe = torch.where(steep, x, torch.ones_like(x))
            sign = torch.sign(safe)
            exponential = sign * (1 + torch.log(uniform + (1 - uniform) * torch.exp(-2 * safe.abs())) / safe.abs())
            u = torch.where(tilted[:, None] & steep, exponential, 2 * uniform - 1)
            offsets = u * half
            z = cells.centers[index] + offsets
            bound = torch.where(
                tilted,
                cells.center_values[index] + (slope * offsets).sum(-1) + cells.envelope[index],
                cells.envelope[index],
            )
            log_ratio = self.latent_log_density(z) - bound
            keep = torch.rand(batch, dtype=torch.float64, device=device).log() < log_ratio
            accepted.append(z[keep][:missing])
            missing -= len(accepted[-1])
        return torch.cat(accepted)

    def sample_given_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Draw sequences from the independent site distributions given latent points ``z``; int32 (n, L)."""
        states = []
        chunk = _chunk(self.fields)
        for start in range(0, len(z), chunk):
            logits = _site_logits(self.fields, self.weights, z[start:start + chunk])
            # Gumbel-max draws from softmax(logits) at every site.
            gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-300)))
            states.append((logits + gumbel).argmax(-1).to(torch.int32))
        return torch.cat(states) if states else torch.empty((0, self.shape[0]), dtype=torch.int32,
                                                              device=self.fields.device)

    def sample(self, n_samples: int, device=None) -> torch.Tensor:
        """Exact draws of the model, int32 of shape (n, L), on ``device`` (default: the parameters' device)."""
        states = self.sample_given_latent(self.sample_latent(n_samples))
        return states if device is None else states.to(device)

    def log_likelihood(self, states: torch.Tensor, weights: torch.Tensor | None = None) -> float:
        """Mean log-likelihood of integer sequences (N, L), weighted by ``weights``."""
        states = states.to(device=self.fields.device).long()
        score = _sequence_scores(self.fields, self.weights, states)
        if weights is None:
            return float(score.mean()) - self.log_z
        weights = weights.to(device=self.fields.device, dtype=torch.float64)
        return float((weights * score).sum() / weights.sum()) - self.log_z


def _sequence_scores(fields, weights, states):
    """``sum_i g_i(x_i) + |m(x)|^2 / 2`` of integer sequences (N, L): log weights before normalization."""
    sites = torch.arange(states.shape[1], device=states.device)
    projection = weights[sites, states].sum(1)  # (N, K)
    return fields[sites, states].sum(1) + 0.5 * (projection ** 2).sum(-1)


def target_moments(states: torch.Tensor, sequence_weights: torch.Tensor, num_states: int,
                   pseudocount: float) -> tuple[torch.Tensor, torch.Tensor]:
    """First and second moments of the one-hot sequences: all that the likelihood needs from the data.

    The pseudocount mixes the data with the uniform distribution over sequences.

    Returns:
        The frequencies, shape (L, q), and the second moments ``<x_i(a) x_j(b)>``, shape (L q, L q),
        whose within-site blocks are the diagonal matrices of the frequencies. Both float64.
    """
    device = states.device
    length = states.shape[1]
    columns = length * num_states
    sequence_weights = sequence_weights.to(device=device, dtype=torch.float64)
    sequence_weights = sequence_weights / sequence_weights.sum()
    offsets = torch.arange(length, device=device) * num_states
    if device.type == "cuda":
        # Float64 matrix products are slow on GPUs: add the L^2 entries of every sequence instead.
        second = torch.zeros(columns * columns, dtype=torch.float64, device=device)
        chunk = max(1, _ELEMENTS // length ** 2)
        for start in range(0, len(states), chunk):
            index = states[start:start + chunk].long() + offsets
            keys = index[:, :, None] * columns + index[:, None, :]
            weight = sequence_weights[start:start + chunk, None, None].expand_as(keys)
            second.index_add_(0, keys.reshape(-1), weight.reshape(-1))
        second = second.view(columns, columns)
    else:
        second = torch.zeros((columns, columns), dtype=torch.float64, device=device)
        chunk = max(1, _ELEMENTS // columns)
        for start in range(0, len(states), chunk):
            index = states[start:start + chunk].long() + offsets
            one_hot = torch.zeros((len(index), columns), dtype=torch.float64, device=device)
            one_hot.scatter_(1, index, 1.0)
            second.addmm_((one_hot * sequence_weights[start:start + chunk, None]).T, one_hot)
    second.mul_(1 - pseudocount).add_(pseudocount / num_states ** 2)
    # So far the diagonal holds (1 - pseudocount) f + pseudocount / q^2: complete the within-site blocks.
    frequencies = second.diagonal().view(length, num_states) + pseudocount * (num_states - 1) / num_states ** 2
    blocks = second.view(length, num_states, length, num_states)
    sites = torch.arange(length, device=device)
    blocks[sites, :, sites, :] = torch.diag_embed(frequencies)
    return frequencies, second


class TargetMoments:
    """Single-site frequencies of the target and products with its matrix of pair frequencies.

    The target mixes the weighted sequences with the uniform distribution
    (weight ``pseudocount``). Its pair matrix ``S``, of shape (L q, L q), has the
    blocks ``f_ij(a, b)`` and, within sites, ``diag(f_i)``. The likelihood of a
    low-rank model needs ``S`` only through products ``S @ M`` with thin matrices.
    These use ``S`` itself when it fits in ``_PAIR_MATRIX_BYTES``, and otherwise
    the sequences: ``sum_n w_n x_n (x_n' M)``, at a cost of ``O(N L)`` per column
    and no ``(L q)^2`` memory.

    Args:
        states: Integer sequences (N, L).
        sequence_weights: Sequence weights (N,).
        num_states: Alphabet size ``q``.
        pseudocount: Weight of the uniform distribution.
        pair_matrix: Form ``S`` (True), never form it (False), or decide from its size (None).
    """

    def __init__(self, states, sequence_weights, num_states, pseudocount, *, pair_matrix=None):
        self.length = states.shape[1]
        self.num_states = num_states
        self.pseudocount = pseudocount
        columns = self.length * num_states
        if pair_matrix is None:
            pair_matrix = columns ** 2 * 8 <= _PAIR_MATRIX_BYTES
        if pair_matrix:
            self.frequencies, self.matrix = target_moments(states, sequence_weights, num_states, pseudocount)
            return
        self.matrix = None
        device = states.device
        self.states = states
        self.weights = sequence_weights.to(device=device, dtype=torch.float64)
        self.weights = self.weights / self.weights.sum()
        self.offsets = torch.arange(self.length, device=device) * num_states
        counts = torch.zeros(columns, dtype=torch.float64, device=device)
        for index, weight in self._chunks(1):
            counts.index_add_(0, index.reshape(-1), weight.repeat_interleave(self.length))
        self.frequencies = ((1 - pseudocount) * counts + pseudocount / num_states).view(self.length, num_states)

    def _chunks(self, width):
        """Flat one-hot column indices (n, L) and normalized weights of chunks of sequences."""
        chunk = max(1, _ELEMENTS // (self.length * width))
        for start in range(0, len(self.states), chunk):
            yield self.states[start:start + chunk].long() + self.offsets, self.weights[start:start + chunk]

    def product(self, matrix: torch.Tensor) -> torch.Tensor:
        """``S @ matrix`` for ``matrix`` of shape (L q, b)."""
        if self.matrix is not None:
            # Column by column: float64 products of a matrix with several columns are slow on GPUs.
            return torch.stack([torch.mv(self.matrix, matrix[:, k]) for k in range(matrix.shape[1])], 1)
        width = matrix.shape[1]
        result = torch.zeros_like(matrix)
        for index, weight in self._chunks(width):
            projection = torch.nn.functional.embedding_bag(index, matrix, mode="sum")  # x_n' M, (n, b)
            result.index_add_(0, index.reshape(-1), (projection * weight[:, None]).repeat_interleave(self.length, 0))
        # The uniform part: lambda / q^2 everywhere, and within sites lambda (I / q - 1 / q^2).
        q, mixture = self.num_states, self.pseudocount
        blocks = matrix.view(self.length, q, width)
        result.mul_(1 - mixture).add_(mixture / q ** 2 * matrix.sum(0, keepdim=True))
        result.add_((mixture * (blocks / q - blocks.sum(1, keepdim=True) / q ** 2)).view_as(matrix))
        return result


def _top_modes(operator, start, count):
    """The ``count`` largest eigenvalues and eigenvectors of a symmetric positive operator, by subspace iteration.

    Args:
        operator: Maps a matrix (n, b) to the operator applied to its columns.
        start: Initial block (n, b), with b > count.
        count: Number of modes.
    """
    basis = torch.linalg.qr(start).Q
    previous = None
    for _ in range(_MODE_ITERATIONS):
        image = operator(basis)
        small = basis.T @ image
        values, vectors = torch.linalg.eigh((0.5 * (small + small.T)).cpu())
        values, vectors = values.flip(0).to(basis.device), vectors.flip(1).to(basis.device)
        top = values[:count]
        if previous is not None and ((top - previous).abs() <= _MODE_TOLERANCE * top.abs()).all():
            break
        previous = top
        basis = torch.linalg.qr(image).Q
    return top, basis @ vectors[:, :count]


def hopfield_potts_init(moments: TargetMoments, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial ``(g, W)`` from the top modes of the normalized correlation matrix (mean field).

    The Hopfield-Potts mean-field solution (Cocco, Monasson and Weigt, PLoS
    Comput. Biol. 2013): patterns ``sqrt(1 - 1/lambda) v / sqrt(f)`` from the
    largest eigenvalues ``lambda`` of ``D^{-1/2} C D^{-1/2}``, in the gauge where
    the last state carries no pattern; fields from the single-site frequencies.
    Only the ``rank`` top modes are computed, by subspace iteration with products
    by the pair matrix (:meth:`TargetMoments.product`).

    Args:
        moments: The target moments.
        rank: Number of patterns.
    """
    frequencies = moments.frequencies
    length, num_states = frequencies.shape
    device = frequencies.device
    flat = frequencies.reshape(-1)
    # The gauge without the last state: its rows and columns are zero.
    reduced = (torch.arange(length * num_states, device=device) % num_states) != num_states - 1
    scale = torch.where(reduced, flat.clamp_min(1e-300).rsqrt(), torch.zeros_like(flat))

    def correlation(block):
        scaled = scale[:, None] * block
        return scale[:, None] * (moments.product(scaled) - flat[:, None] * (flat @ scaled)[None, :])

    width = min(rank + _EXTRA_MODES, int(reduced.sum()))
    generator = torch.Generator().manual_seed(0)
    start = torch.randn(len(flat), width, generator=generator, dtype=torch.float64).to(device) * reduced[:, None]
    eigenvalues, eigenvectors = _top_modes(correlation, start, rank)
    # W = 0 is a stationary point of the likelihood: modes without excess variance start small, not at zero.
    patterns = (1 - 1 / eigenvalues).clamp_min(0.01).sqrt() * eigenvectors * scale[:, None]
    # States absent from the data carry only the pseudocount: dividing by its square root would turn
    # numerical noise into huge weights, so they start without a pattern.
    observed = flat > (1 + 1e-6) * moments.pseudocount / num_states
    patterns = patterns * observed[:, None]
    weights = patterns.reshape(length, num_states, rank).contiguous()
    # Unobserved states without a pseudocount: a finite start, which the fit then lowers.
    fields = frequencies.clamp_min(1e-8).log() - 0.5 * (weights ** 2).sum(-1)
    return fields, weights


def fit_low_rank_potts(
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
    pair_matrix: bool | None = None,
    verbose: bool = False,
    callback: Callable[[dict], None] | None = None,
) -> LowRankPotts:
    """Fit a rank-``K`` Potts model to integer sequences by maximum likelihood.

    The likelihood depends on the data through the single-site frequencies and
    through products of the pair-frequency matrix with the weights
    (:class:`TargetMoments`); an evaluation also costs a sum over the latent lattice. The
    pseudocount mixes the data with the uniform distribution, as for the
    frequencies of the full Potts model. ``l2`` penalizes ``|W|^2 / 2`` per sequence, in the gauge where the
    weights of every site sum to zero.

    Args:
        states: Integer sequences, shape (N, L).
        sequence_weights: Sequence weights (N,), or ``None`` for equal weights.
        num_states: Alphabet size ``q``.
        rank: Number of latent directions ``K`` (1 to 3).
        pseudocount: Weight of the uniform distribution in the fitted target.
        l2: L2 penalty on the weights.
        max_iterations: Largest number of evaluations of the likelihood.
        tolerance: Stop when 100 L-BFGS iterations lower the negative log-likelihood per sequence by less
            than this. The last stretch of the fit gains little: 0.01 stops a few hundredths of a nat
            above the optimum, far below the gap between training and validation sequences.
        device: Device of the fit; defaults to that of ``states``.
        pair_matrix: Form the (L q) x (L q) pair-frequency matrix (True), work from the sequences
            (False), or form it when it takes at most 1 GiB (None).
        verbose: Print the objective every 100 evaluations.
        callback: Called after every evaluation with a dict: ``evaluation``, ``max_evaluations``,
            ``log_likelihood`` (per sequence, with the pseudocount and the penalty) and ``lattice_points``.

    Returns:
        The fitted :class:`LowRankPotts`.
    """
    if not 1 <= rank <= 3:
        raise ValueError("Low-rank Potts models support 1 to 3 latent directions.")
    device = torch.device(device) if device is not None else states.device
    if device.type not in ("cpu", "cuda"):
        device = torch.device("cpu")
    states = states.to(device)
    if sequence_weights is None:
        sequence_weights = torch.ones(len(states), dtype=torch.float64, device=device)
    moments = TargetMoments(states, sequence_weights, num_states, pseudocount, pair_matrix=pair_matrix)
    frequencies = moments.frequencies
    fields, weights = hopfield_potts_init(moments, rank)
    length = states.shape[1]
    columns = length * num_states
    gaussian = 0.5 * rank * math.log(2 * math.pi)

    def objective(fields, weights, lattice=None):
        """Negative log-likelihood per sequence, without its gradient."""
        log_integral, lattice, _ = _lattice_sum(fields, weights, lattice)
        flat = weights.reshape(columns, rank)
        centred = (weights - weights.mean(1, keepdim=True)).reshape(columns, rank)
        return (log_integral - gaussian - float((frequencies * fields).sum())
                - 0.5 * float((flat * moments.product(flat)).sum()) + 0.5 * l2 * float((centred ** 2).sum()))

    # The mean-field patterns can be far too strong, notably on rare states, whose weights are divided by
    # the square root of their frequency: start from the scale of the patterns that the likelihood prefers.
    scales = [2.0 ** -k for k in range(6)]
    starts = [(frequencies.clamp_min(1e-8).log() - 0.5 * scale ** 2 * (weights ** 2).sum(-1), scale * weights)
              for scale in scales]
    fields, weights = min(starts, key=lambda start: objective(*start))
    parameters = torch.cat([fields.reshape(-1), weights.reshape(-1)]).requires_grad_(True)
    lattice = None
    evaluations = 0

    def closure():
        """Negative log-likelihood per sequence; leaves its gradient on the flattened parameters."""
        nonlocal lattice, evaluations
        with torch.no_grad():
            fields = parameters[:columns].view(length, num_states)
            weights = parameters[columns:].view(length, num_states, rank)
            log_integral, lattice, points, marginals, projections = _lattice_sum(fields, weights, lattice,
                                                                                 gradient=True)
            flat = weights.reshape(columns, rank)
            correlated = moments.product(flat)
            # The penalty acts on the weights of every site minus their mean, which no gauge change moves.
            centred = (weights - weights.mean(1, keepdim=True)).reshape(columns, rank)
            loss = (log_integral - gaussian - (frequencies * fields).sum() - 0.5 * (flat * correlated).sum()
                    + 0.5 * l2 * (centred ** 2).sum())
            parameters.grad = torch.cat([(marginals - frequencies).reshape(-1),
                                         (projections.reshape(columns, rank) - correlated + l2 * centred).reshape(-1)])
        evaluations += 1
        if callback is not None:
            callback({"evaluation": evaluations, "max_evaluations": max_iterations,
                      "log_likelihood": -float(loss), "lattice_points": points})
        if verbose and evaluations % 100 == 0:
            print(f"low-rank fit, evaluation {evaluations}: negative log-likelihood {float(loss):.6f} "
                  f"({points} lattice points)")
        return loss

    def new_optimizer():
        return torch.optim.LBFGS([parameters], max_iter=100, max_eval=125, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-10, tolerance_change=1e-14, history_size=20)

    optimizer = new_optimizer()
    previous = math.inf
    while evaluations < max_iterations:
        try:
            loss = float(optimizer.step(closure))  # the objective before these 100 iterations
        except IndexError:  # a failed line search of torch's L-BFGS: restart it from the current point
            optimizer = new_optimizer()
            continue
        if previous - loss < tolerance:
            break
        previous = loss
    solution = parameters.detach()
    fields = solution[:columns].view(length, num_states)
    weights = solution[columns:].view(length, num_states, rank)
    # Return the model in the gauge where the weights of every site sum to zero.
    shift = weights.mean(1)
    weights = weights - shift[:, None, :]
    return LowRankPotts(fields + weights @ shift.sum(0), weights)
