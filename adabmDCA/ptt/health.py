"""Bidirectional reweighting diagnostics between adjacent PTT replicas.

For neighbouring models k and k+1, every configuration has the energy
difference ``W = E_{k+1} - E_k``: the population-annealing log-weight is
``-W`` and swaps accept with ``min(1, exp(W(x) - W(y)))`` for ``x`` moving
down from k+1 and ``y`` moving up from k. With equilibrium populations of
both replicas these quantities give, without extra sampling:

- free-energy differences ``dF = F_{k+1} - F_k = -log(Z_{k+1} / Z_k)`` from
  the lower replica (forward, the historical bridge), from the upper replica
  (reverse) and from both (Bennett's acceptance ratio, BAR);
- the effective sample size of one reweighting step in each direction and
  the weight share of the top 1% of configurations (heavy tails);
- the Crooks identity ``log[h_{k+1}(W) / h_k(W)] = dF - W`` for the two
  W histograms, whose slope must be -1 at equilibrium;
- the mobility spectrum: each configuration's swap probability averaged
  over the other replica's population. Its low tail identifies
  configurations that cannot leave their replica by exchanges, which the
  mean acceptance hides.

All free energies are in nats; errors are bootstrap standard deviations.
"""

from __future__ import annotations

import math

import torch

IMMOBILE_THRESHOLD = 1e-4


def exp_forward(w_lower: torch.Tensor) -> torch.Tensor:
    """dF from lower-replica samples: -log mean exp(-W). Batched over leading dimensions."""
    return -(torch.logsumexp(-w_lower, -1) - math.log(w_lower.shape[-1]))


def exp_reverse(w_upper: torch.Tensor) -> torch.Tensor:
    """dF from upper-replica samples: log mean exp(W). Batched over leading dimensions."""
    return torch.logsumexp(w_upper, -1) - math.log(w_upper.shape[-1])


def bar(w_lower: torch.Tensor, w_upper: torch.Tensor, iterations: int = 100) -> torch.Tensor:
    """Bennett acceptance ratio estimate of dF, batched over leading dimensions.

    Solves ``sum_lower s(dF - M - W) = sum_upper s(M + W - dF)`` with ``s`` the
    logistic function and ``M = log(n_lower / n_upper)``; the left side
    increases and the right side decreases with dF, so bisection converges.
    """
    if w_lower.device.type == "mps":
        w_lower, w_upper = w_lower.cpu().double(), w_upper.cpu().double()
    shift = math.log(w_lower.shape[-1] / w_upper.shape[-1])
    lo = torch.minimum(w_lower.min(-1).values, w_upper.min(-1).values) - 50.0
    hi = torch.maximum(w_lower.max(-1).values, w_upper.max(-1).values) + 50.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        balance = (torch.sigmoid(mid.unsqueeze(-1) - shift - w_lower).sum(-1)
                   - torch.sigmoid(shift + w_upper - mid.unsqueeze(-1)).sum(-1))
        hi = torch.where(balance > 0, mid, hi)
        lo = torch.where(balance > 0, lo, mid)
    return 0.5 * (lo + hi)


def effective_fraction(log_weights: torch.Tensor) -> float:
    """Effective sample size of normalized weights exp(log_weights), as a fraction of the samples."""
    w = torch.softmax(log_weights, 0)
    return float(1.0 / (w.square().sum() * len(w)))


def top_share(log_weights: torch.Tensor, fraction: float = 0.01) -> float:
    """Share of the total weight carried by the heaviest ``fraction`` of samples."""
    w = torch.softmax(log_weights, 0)
    count = max(1, int(len(w) * fraction))
    return float(torch.topk(w, count).values.sum())


def crooks_slope(w_lower: torch.Tensor, w_upper: torch.Tensor, bins: int = 30, min_count: int = 20):
    """Weighted least-squares slope of log[h_upper(W) / h_lower(W)] against W.

    Bins are quantiles of the pooled W values, so the overlap region is
    resolved; only bins with ``min_count`` samples from both replicas enter.
    Returns ``(slope, error, used_bins)``; slope and error are None with
    fewer than three usable bins. Equilibrium populations give slope -1.
    """
    if w_lower.device.type == "mps":
        w_lower, w_upper = w_lower.cpu().double(), w_upper.cpu().double()
    pooled = torch.cat([w_lower, w_upper]).double()
    edges = torch.unique(torch.quantile(pooled, torch.linspace(0, 1, bins + 1, dtype=torch.float64)))
    if len(edges) < 4:
        return None, None, 0
    lower_bin = torch.bucketize(w_lower.double(), edges[1:-1])
    upper_bin = torch.bucketize(w_upper.double(), edges[1:-1])
    n_bins = len(edges) - 1
    lower_counts = torch.bincount(lower_bin, minlength=n_bins).double()
    upper_counts = torch.bincount(upper_bin, minlength=n_bins).double()
    centers = torch.bincount(torch.bucketize(pooled, edges[1:-1]), weights=pooled, minlength=n_bins)
    centers = centers / torch.bincount(torch.bucketize(pooled, edges[1:-1]), minlength=n_bins).clamp_min(1)
    usable = (lower_counts >= min_count) & (upper_counts >= min_count)
    if int(usable.sum()) < 3:
        return None, None, int(usable.sum())
    y = torch.log(upper_counts[usable] / len(w_upper)) - torch.log(lower_counts[usable] / len(w_lower))
    variance = 1.0 / upper_counts[usable] + 1.0 / lower_counts[usable]
    design = torch.stack([centers[usable], torch.ones_like(y)], 1)
    weight = 1.0 / variance
    normal = design.T @ (design * weight[:, None])
    covariance = torch.linalg.inv(normal)
    slope = (covariance @ (design.T @ (weight * y)))[0]
    return float(slope), float(covariance[0, 0].sqrt()), int(usable.sum())


def mobility(w_lower: torch.Tensor, w_upper: torch.Tensor):
    """Per-configuration swap probabilities against a random partner of the other replica.

    Returns ``(down, up)``: for each upper configuration the mean over lower
    partners of ``min(1, exp(W_upper - W_lower))``, and for each lower
    configuration the mean over upper partners. Their common mean is the
    expected acceptance of a random pairing.
    """
    down = torch.empty_like(w_upper)
    up = torch.zeros_like(w_lower)
    chunk = max(1, 4_000_000 // max(len(w_lower), 1))
    for start in range(0, len(w_upper), chunk):
        acceptance = torch.exp((w_upper[start:start + chunk, None] - w_lower[None, :]).clamp_max(0.0))
        down[start:start + chunk] = acceptance.mean(1)
        up += acceptance.sum(0)
    return down, up / len(w_upper)


def pair_health(w_lower: torch.Tensor, w_upper: torch.Tensor, *, bootstrap: int = 200, generator=None,
                ages: torch.Tensor | None = None, immobile_threshold: float = IMMOBILE_THRESHOLD) -> dict:
    """All diagnostics for one adjacent pair; ``ages`` are upper-replica ages in rounds, if tracked."""
    w_lower = w_lower.cpu().double()
    w_upper = w_upper.cpu().double()
    forward, reverse = float(exp_forward(w_lower)), float(exp_reverse(w_upper))
    bennett = float(bar(w_lower, w_upper))
    lower_draw = torch.randint(len(w_lower), (bootstrap, len(w_lower)), generator=generator)
    upper_draw = torch.randint(len(w_upper), (bootstrap, len(w_upper)), generator=generator)
    boot_lower, boot_upper = w_lower[lower_draw], w_upper[upper_draw]
    slope, slope_error, slope_bins = crooks_slope(w_lower, w_upper)
    down, up = mobility(w_lower, w_upper)
    immobile = down < immobile_threshold
    result = {
        "dF_forward": forward, "dF_forward_error": float(exp_forward(boot_lower).std()),
        "dF_reverse": reverse, "dF_reverse_error": float(exp_reverse(boot_upper).std()),
        "dF_bar": bennett, "dF_bar_error": float(bar(boot_lower, boot_upper).std()),
        "hysteresis": forward - reverse,
        "ess_forward": effective_fraction(-w_lower), "ess_reverse": effective_fraction(w_upper),
        "top1_share_forward": top_share(-w_lower), "top1_share_reverse": top_share(w_upper),
        "crooks_slope": slope, "crooks_slope_error": slope_error, "crooks_bins": slope_bins,
        "mean_acceptance": float(down.mean()),
        "immobile_down_fraction": float(immobile.double().mean()),
        "immobile_up_fraction": float((up < immobile_threshold).double().mean()),
        "log10_mobility_down_q01": float(torch.quantile(down.clamp_min(1e-300).log10(), 0.01)),
        "log10_mobility_down_median": float(down.clamp_min(1e-300).log10().median()),
        "immobile_threshold": immobile_threshold,
    }
    # Per-configuration acceptance: equal means can hide a tail of configurations that never cross.
    for name, values in (("down", down), ("up", up)):
        for quantile, label in ((0.5, "q50"), (0.1, "q10"), (0.01, "q01")):
            result[f"acceptance_{name}_{label}"] = float(torch.quantile(values, quantile))
    if ages is not None and immobile.any() and (~immobile).any():
        ages = ages.cpu().double()
        result["immobile_mean_age"] = float(ages[immobile].mean())
        result["mobile_mean_age"] = float(ages[~immobile].mean())
    return result
