"""PRIVET: per-sample memorization and overfitting test from nearest-neighbour distances.

Follows Szatkownik et al., "PRIVET: PRoximIty leakage detection Via Extreme
value Theory" (arXiv:2510.24233; reference below). The distances from each training sequence to its
nearest other training sequence give the null distribution of nearest-neighbour
distances, fitted with an extreme-value law (Weibull or Gumbel for minima,
whichever has the higher likelihood). For a generated sample at rank ``r`` among
the ``M`` distances to the training set, the probability that the ``r``-th
smallest of ``M`` independent null distances is that small is binomial; the
excess of smaller samples is discounted, so each sample is judged on its own.
The same is done with distances to a held-out set, rescaled to the training-set
size, and

    delta_p = log10(p_train / p_test)

is very negative for a sample far closer to the training set than to unseen
sequences of the family: a sample the model has memorized. ``N_pleaks`` counts
the excess of generated samples near the training set compared with the held-out
set.

Hamming distances are discrete: every fitted probability is taken over the whole
bin of a distance, ``d +- resolution / 2``.

Departures from the paper, for alignments of homologous sequences:

- The null law is fitted to the *reweighted* training distances. A DCA model
  is fitted to the reweighted alignment; unweighted, redundant homologues make
  close neighbours look common.
- The window ``[q1, q2]`` is fitted by maximum likelihood with the distances
  below and above it censored (they anchor the level without imposing the
  shape), and starts by default at the 1% quantile rather than 10-30%: the
  lower tail decides the test, and the law fitted higher up over-predicts it.
- ``excess_near_training`` scores the most improbable *group* of samples close
  to the training set. With a few thousand samples and the short, diverse
  distances of protein or RNA families, the per-sample scores cannot reach
  ``LEAK_THRESHOLD`` even for exact copies: the null gives a fresh family
  member about a 1e-4 to 1e-2 chance of lying that close.
- The excess below a rank is never negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, stats

# Reference: A. Szatkownik, A. Decelle, B. Seoane, N. Béreux, L. Planche,
# G. Charpiat, B. Yelmen, F. Jay and C. Furtlehner, "PRIVET: PRoximIty leakage
# detection Via Extreme value Theory", arXiv:2510.24233 (2025),
# https://arxiv.org/abs/2510.24233, also at
# https://hal-ciheam.iamm.fr/LISN-AO/hal-05326013v2. Section and equation
# pointers below refer to that paper; see the module docstring for where this
# implementation departs from it.

LEAK_THRESHOLD = -3.0
DEFAULT_WINDOW = (0.01, 0.5)
_LN10 = np.log(10.0)


@dataclass(frozen=True)
class ExtremeValueFit:
    """Fitted law of nearest-neighbour distances, ``P(d <= u) = 1 - exp(-H(u))``.

    ``H(u) = scale * u**shape`` (Weibull) or ``scale * exp(shape * u)`` (Gumbel).
    """

    family: str
    scale: float
    shape: float
    log_likelihood: float
    alternative_log_likelihood: float
    window: tuple[float, float]

    def hazard(self, u: np.ndarray, size_ratio: float = 1.0) -> np.ndarray:
        """Cumulative hazard; ``size_ratio`` rescales it to a reference set ``size_ratio`` times larger."""
        u = np.asarray(u, dtype=np.float64)
        if self.family == "weibull":
            value = self.scale * np.clip(u, 0.0, None) ** self.shape
        else:
            value = self.scale * np.exp(self.shape * u)
        return size_ratio * value

    def cdf(self, u: np.ndarray, size_ratio: float = 1.0) -> np.ndarray:
        """Probability that a nearest-neighbour distance is at most ``u``.

        The nearest of ``n`` independent points has a hazard proportional to
        ``n``: with ``size_ratio = n_set / n_train`` this is the law of distances
        to a set of ``n_set`` sequences.
        """
        return -np.expm1(-self.hazard(u, size_ratio))

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "scale": self.scale, "shape": self.shape,
                "log_likelihood": self.log_likelihood,
                "alternative_log_likelihood": self.alternative_log_likelihood,
                "window": list(self.window)}


def _hazard(family: str, parameters: np.ndarray, u: np.ndarray) -> np.ndarray:
    log_scale, shape = parameters
    if family == "weibull":
        return np.exp(log_scale + np.exp(shape) * np.log(np.clip(u, 1e-300, None))) * (u > 0)
    return np.exp(log_scale + shape * u)


def _negative_log_likelihood(parameters, family, lower, upper, weights, below, above, low_edge, high_edge):
    """Weighted interval likelihood inside the window, censored weights below and above it."""
    h_lower = np.where(np.isfinite(lower), _hazard(family, parameters, np.where(np.isfinite(lower), lower, 0.0)), 0.0)
    h_upper = _hazard(family, parameters, upper)
    # log(S(lower) - S(upper)) with S = exp(-H), stably.
    inside = -h_lower + np.log(-np.expm1(np.clip(h_lower - h_upper, None, -1e-300)))
    total = (weights * inside).sum()
    if below:
        total += below * np.log(-np.expm1(-_hazard(family, parameters, np.array([low_edge]))[0]) + 1e-300)
    if above:
        total -= above * _hazard(family, parameters, np.array([high_edge]))[0]
    return -total if np.isfinite(total) else np.inf


# Extreme-value law of the train-train nearest-neighbour distances: Weibull or
# Gumbel, chosen by likelihood (PRIVET, Eq. 1 and Algorithm 1, steps 2-3).
def fit_nearest_distances(distances: np.ndarray, *, resolution: float,
                          window: tuple[float, float] = DEFAULT_WINDOW,
                          weights: np.ndarray | None = None) -> ExtremeValueFit:
    """Fit Weibull and Gumbel minimum laws to nearest-neighbour distances; keep the likelier.

    Args:
        distances: Nearest-neighbour distances of the reference set to itself
            (each point to its nearest other point).
        resolution: Spacing of the distance values (``1 / L`` for Hamming
            distances as a fraction of the sites).
        window: Quantiles ``(q1, q2)`` of the distances whose shape is fitted;
            distances outside are censored.
        weights: Optional weight of each distance (e.g. the reweighting weights
            of the sequences); ``None`` weighs them equally.
    """
    distances = np.asarray(distances, dtype=np.float64)
    weights = np.ones_like(distances) if weights is None else np.asarray(weights, dtype=np.float64)
    order = np.argsort(distances, kind="stable")
    distances = distances[order]
    weights = weights[order] * (len(distances) / weights.sum())
    q1, q2 = window
    if not 0.0 <= q1 < q2 <= 1.0:
        raise ValueError("The PRIVET window needs 0 <= q1 < q2 <= 1.")
    if resolution <= 0:
        raise ValueError("Pass the spacing of the distance values (1 / L for Hamming distances).")
    cumulative = np.cumsum(weights) / weights.sum()
    low, high = (distances[min(np.searchsorted(cumulative, level), len(distances) - 1)] for level in (q1, q2))
    half = 0.5 * resolution
    inside = (distances >= low) & (distances <= high)
    selected, selected_weights = distances[inside], weights[inside]
    below = float(weights[distances < low].sum())
    above = float(weights[distances > high].sum())
    # Each distance stands for its bin; the bin of 0 extends to -inf (the Gumbel law has mass there).
    lower = np.where(selected - half <= 0.0, -np.inf, selected - half)
    upper = selected + half

    # Starting point: a straight line of log H against log u (Weibull) or u (Gumbel), H from the eCDF.
    unique, inverse = np.unique(distances, return_inverse=True)
    ecdf = np.cumsum(np.bincount(inverse, weights=weights)) / (weights.sum() * (1 + 1 / len(distances)))
    keep = (unique >= low) & (unique <= high) & (unique + half > 0)
    edges = unique[keep] + half
    log_h = np.log(-np.log1p(-ecdf[keep]))
    fits = {}
    for family in ("weibull", "gumbel"):
        x = np.log(edges) if family == "weibull" else edges
        if len(x) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, log_h, 1)
        else:
            slope, intercept = 1.0, float(log_h.mean()) if len(log_h) else 0.0
        start = np.array([intercept, np.log(max(slope, 1e-3))]) if family == "weibull" else \
            np.array([intercept, max(slope, 1e-3)])
        arguments = (family, lower, upper, selected_weights, below, above, low - half, high + half)
        result = optimize.minimize(_negative_log_likelihood, start, args=arguments, method="Nelder-Mead",
                                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
        fits[family] = (result.x, -float(result.fun))
    best = max(fits, key=lambda family: fits[family][1])
    other = "gumbel" if best == "weibull" else "weibull"
    log_scale, shape = fits[best][0]
    return ExtremeValueFit(
        family=best, scale=float(np.exp(log_scale)),
        shape=float(np.exp(shape)) if best == "weibull" else float(shape),
        log_likelihood=fits[best][1], alternative_log_likelihood=fits[other][1],
        window=(float(low), float(high)),
    )


# Binomial probability of the r-th smallest of M distances (PRIVET, Eq. 2), with
# the excess below rank r discounted as in the per-sample score (Eq. 7).
def rank_log10_p(distances: np.ndarray, fit: ExtremeValueFit, *, resolution: float,
                 size_ratio: float = 1.0) -> np.ndarray:
    """log10 probability that each sample's rank-``r`` distance is that small under the null.

    Args:
        distances: Nearest distances of ``M`` samples to a reference set.
        fit: Null law of nearest distances to the training set.
        resolution: Spacing of the distance values.
        size_ratio: Size of the reference set divided by that of the training set.

    Returns:
        One value per sample, in the input order. The excess of samples below
        rank ``r`` is discounted: the ``r``-th smallest distance is compared with
        the expected count at the previous one, so a sample is judged on its own.
    """
    distances = np.asarray(distances, dtype=np.float64)
    count = len(distances)
    order = np.argsort(distances, kind="stable")
    probability = fit.cdf(distances[order] + 0.5 * resolution, size_ratio)
    rank = np.arange(1, count + 1)
    expected = np.floor(count * probability)
    excess = np.zeros(count)
    excess[1:] = np.maximum(0.0, (rank[:-1]) - expected[:-1])
    effective = np.maximum(1, rank - excess)
    log_p = stats.binom.logsf(effective - 1, count, probability) / _LN10
    result = np.empty(count)
    result[order] = np.minimum(log_p, 0.0)
    return result


def excess_near_training(distances: np.ndarray, fit: ExtremeValueFit, *, resolution: float) -> dict[str, float]:
    """The most improbable group of samples close to the training set.

    For each rank ``r`` with a distance inside or below the fitted window, the
    probability that at least ``r`` of the ``M`` null distances fall at or below
    the ``r``-th smallest one, without the excess correction of
    :func:`rank_log10_p`: it measures the group, not one sample. The smallest
    value is returned with its count, the count expected under the null and the
    distance. Scanning the ranks makes it a minimum over many tests, so treat
    values above about -3 as unremarkable.
    """
    distances = np.sort(np.asarray(distances, dtype=np.float64))
    count = len(distances)
    probability = fit.cdf(distances + 0.5 * resolution)
    log_p = stats.binom.logsf(np.arange(count), count, probability) / _LN10
    log_p[distances > fit.window[1]] = 0.0
    best = int(np.argmin(log_p))
    return {"log10_p": float(min(log_p[best], 0.0)), "count": best + 1,
            "expected": float(count * probability[best]), "distance": float(distances[best])}


# n_pleaks(r) and N_pleaks = max_r n_pleaks(r) (PRIVET, Algorithm 1, step 7).
def _excess_curve(train: np.ndarray, test: np.ndarray, fit: ExtremeValueFit, *, resolution: float,
                  test_ratio: float) -> np.ndarray:
    """n_pleaks(r): expected rank of the r-th distance to the test set minus that to the training set."""
    count = len(train)
    to_train = np.floor(count * fit.cdf(np.sort(train) + 0.5 * resolution))
    to_test = np.floor(count * fit.cdf(np.sort(test) + 0.5 * resolution, test_ratio))
    return np.minimum(np.arange(1, count + 1), to_test - to_train)


def privet(train_train: np.ndarray, generated_train: np.ndarray, generated_test: np.ndarray | None = None,
           test_train: np.ndarray | None = None, *, n_train: int, n_test: int | None = None,
           resolution: float, window: tuple[float, float] = DEFAULT_WINDOW,
           train_weights: np.ndarray | None = None, threshold: float = LEAK_THRESHOLD) -> dict[str, Any]:
    """PRIVET scores of generated samples.

    Args:
        train_train: Distance of each training sequence to its nearest other one.
        generated_train: Distance of each generated sequence to its nearest training one.
        generated_test: Distance of each generated sequence to its nearest held-out one.
        test_train: Distance of each held-out sequence to its nearest training one,
            scored like a generated set as a control.
        n_train: Number of training sequences searched.
        n_test: Number of held-out sequences searched.
        resolution: Spacing of the distance values (``1 / L``).
        window: Quantiles of ``train_train`` fitted by the extreme-value law.
        train_weights: Weights of the training sequences in the null law. A DCA
            model is fitted to the reweighted alignment, so its samples are
            compared with the reweighted distribution; unweighted, redundant
            homologues make close neighbours look common.
        threshold: log10 probability below which a sample is flagged.

    Returns:
        The fit, cumulative distributions on the distance grid (``cdf``: the
        weighted training null, its fit, generated and held-out distances to the
        training set), the most improbable group of
        samples near the training set (``excess_train``), per-sample ``log10_p_train``
        (and, with a held-out set, ``log10_p_test`` and ``delta_p``), the number
        and fraction of flagged samples, ``mean_delta_p`` and ``n_pleaks``
        (clipped at 0: a negative excess means the samples are farther from
        the training set than from the held-out set).
    """
    fit = fit_nearest_distances(train_train, resolution=resolution, window=window, weights=train_weights)
    grid = np.arange(0.0, 1.0 + resolution / 2, resolution)
    weights = np.ones(len(train_train)) if train_weights is None else np.asarray(train_weights, dtype=np.float64)

    def ecdf(values, value_weights=None):
        values = np.asarray(values, dtype=np.float64)
        value_weights = np.ones(len(values)) if value_weights is None else value_weights
        order = np.argsort(values, kind="stable")
        cumulative = np.concatenate([[0.0], np.cumsum(value_weights[order]) / value_weights.sum()])
        return cumulative[np.searchsorted(values[order], grid + resolution / 2, side="right")].tolist()

    log_p_train = rank_log10_p(generated_train, fit, resolution=resolution)
    result: dict[str, Any] = {
        "fit": fit.to_dict(),
        "quantile_window": list(window),
        "threshold": threshold,
        "n_generated": len(generated_train), "n_train": int(n_train), "n_test": int(n_test or 0),
        "cdf": {"distance": grid.tolist(), "train_train": ecdf(train_train, weights),
                "fitted": fit.cdf(grid + resolution / 2).tolist(), "generated_train": ecdf(generated_train)},
        "log10_p_train": log_p_train.tolist(),
        "n_memorized": int((log_p_train < threshold).sum()),
        "excess_train": excess_near_training(generated_train, fit, resolution=resolution),
    }
    if test_train is not None:
        control = rank_log10_p(test_train, fit, resolution=resolution)
        result["cdf"]["test_train"] = ecdf(test_train)
        result["control"] = {"n": len(control), "n_flagged": int((control < threshold).sum()),
                             "log10_p_train": control.tolist(),
                             "excess_train": excess_near_training(test_train, fit, resolution=resolution)}
    if generated_test is not None:
        ratio = n_test / n_train  # set-size rescaling of the test distances (Algorithm 1, step 4)
        log_p_test = rank_log10_p(generated_test, fit, resolution=resolution, size_ratio=ratio)
        delta = log_p_train - log_p_test  # Delta p_r = log10(p_train / p_test), PRIVET Eq. 7
        curve = _excess_curve(generated_train, generated_test, fit, resolution=resolution, test_ratio=ratio)
        result.update({
            "log10_p_test": log_p_test.tolist(),
            "delta_p": delta.tolist(),
            "n_flagged": int((delta < threshold).sum()),
            "flagged_fraction": float((delta < threshold).mean()),
            "mean_delta_p": float(delta.mean()),
            "n_pleaks": int(max(0.0, curve.max())),
            "n_pleaks_curve": curve.tolist(),
        })
    return result
