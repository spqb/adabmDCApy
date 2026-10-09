"""Replica-index autocorrelation analysis used by PTT's TRWA experiment.

This follows ``ptt_paper.implement._process_experiment``: center replica
labels at their known uniform mean, compute their autocorrelation with an
FFT, use a self-consistent window for tau_int, and fit the exponential tail
for tau_exp. Times are in exchange rounds, not individual local sweeps.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import OptimizeWarning, curve_fit


@dataclass(frozen=True)
class MixingEstimate:
    """Outcome of one PTT mixing check.

    Attributes:
        tau_int: Integrated autocorrelation time of the replica index, in rounds
            (autocorrelation method), else ``None``.
        tau_exp: Exponential autocorrelation time, in rounds, else ``None``.
        rounds: Rounds measured.
        required_rounds: Rounds the check needed to pass.
        status: ``"converged"``, or why it did not converge (e.g. ``"budget_exceeded"``).
        acceptance: Swap acceptance of each adjacent replica pair.
        local_sweeps: Local sweeps spent on the check.
        model_version: Training update of the endpoint when checked.
        ladder_version: Version of the ladder checked.
        replicas: Number of replicas checked.
        full_ladder: Whether the full historical ladder was checked.
        method: ``"autocorrelation"`` or ``"renewal"``.
        warmup_rounds: Renewal method: rounds until the initial populations were
            replaced, or ``None`` if the budget ran out first.
        renewal_rounds: Renewal method: rounds until the check passed, or ``None``.
        trapped_fraction: Renewal method: fraction of endpoint configurations still
            older than the reference when the check ended (large when it failed).
        immobile: Renewal method, when the check failed: for each replica above
            the bottom that still holds old configurations, a dict with
            ``replica`` (counting from the bottom of the active ladder),
            ``chains``, ``old`` (configurations born before the reference),
            ``immobile`` (old configurations whose swap acceptance toward the
            replica below is under ``health.IMMOBILE_THRESHOLD``, so that only
            local moves can free them)
            and ``blocking`` (whether the immobile ones alone exceed the renewal
            tolerance).
    """

    tau_int: float | None
    tau_exp: float | None
    rounds: int
    required_rounds: int
    status: str
    acceptance: tuple[float, ...] = ()
    local_sweeps: int = 0
    model_version: int = 0
    ladder_version: int = 0
    replicas: int = 0
    full_ladder: bool = False
    method: str = "autocorrelation"
    warmup_rounds: int | None = None
    renewal_rounds: int | None = None
    trapped_fraction: float | None = None
    immobile: tuple[dict, ...] = ()

    @property
    def converged(self) -> bool:
        """Whether the ladder passed the check."""
        return self.status == "converged"


@dataclass(frozen=True)
class RenewalEstimate:
    """Population renewal measured with configuration birth rounds.

    Rung 0 is redrawn exactly every round, so a configuration's birth round
    is the round in which it was drawn there. Relative to a reference round,
    ``ladder_old`` is the fraction of all configurations born before it (it
    can only decrease) and ``endpoint_fresh`` the fraction of endpoint
    configurations born after it. A phase ends once at most
    ``tolerance * chains_per_model`` old configurations remain anywhere in
    the ladder, which bounds the endpoint old fraction by ``tolerance`` from
    then on.

    The warmup phase measures renewal of the initial state. The stationary
    phase restarts the reference at the end of the warmup and advances in
    fixed blocks of ``chunk_rounds`` (a fraction of ``warmup_rounds``) until
    the ladder is renewed again, so the retained population descends entirely
    from draws made after the ladder had forgotten its initialization.
    ``warmup_decay_rounds`` and ``stationary_decay_rounds`` are the decay
    times of the old fraction fitted at the end of each phase (see
    ``renewal_forecast``). Times are exchange rounds.
    """

    status: str
    tolerance: float
    warmup_rounds: int | None
    renewal_rounds: int | None
    stationary_rounds: int
    chunk_rounds: int
    trapped_fraction: float
    warmup_ladder_old: tuple[float, ...] = ()
    warmup_endpoint_fresh: tuple[float, ...] = ()
    stationary_ladder_old: tuple[float, ...] = ()
    stationary_endpoint_fresh: tuple[float, ...] = ()
    old_fraction_by_model: tuple[float, ...] = ()
    acceptance: tuple[float, ...] = ()
    local_sweeps: int = 0
    replicas: int = 0
    warmup_decay_rounds: float | None = None
    stationary_decay_rounds: float | None = None
    immobile: tuple[dict, ...] = ()

    @property
    def converged(self) -> bool:
        """Whether the ladder passed the check."""
        return self.status == "converged"

    @property
    def rounds(self):
        return len(self.warmup_ladder_old) + len(self.stationary_ladder_old)


@dataclass(frozen=True)
class RenewalForecast:
    """Exponential fit of the old fraction G(t) of a renewal phase.

    ``decay_rounds`` is the fitted decay time (``None`` if G is not
    decaying), ``predicted_round`` the round of the phase at which the fit
    reaches ``threshold``, and the fit is ``log G = intercept - t / decay``
    over rounds ``fit_start``..``fit_end``.
    """

    decay_rounds: float | None
    predicted_round: float | None
    intercept: float
    fit_start: int
    fit_end: int


def renewal_forecast(ladder_old, threshold, *, min_points=20):
    """Fit ``log G`` against rounds over the recent half of the decay and predict the renewal round.

    Only rounds with ``0 < G < 0.5`` enter the fit: the first rounds, before
    fresh configurations reach every replica, and the counting floor are
    excluded. The fit uses the most recent half of those rounds, so it follows
    the slowest remaining configurations as the faster ones leave; an earlier
    fit tends to predict renewal too early. Returns ``None`` until
    ``min_points`` rounds are usable. The forecast only informs; renewal is
    decided by counting old configurations.
    """
    values = np.asarray(ladder_old, dtype=float)
    usable = np.flatnonzero((values > 0) & (values < 0.5))
    if len(usable) < min_points:
        return None
    recent = usable[len(usable) // 2:]
    rounds = recent + 1.0
    slope, intercept = np.polyfit(rounds, np.log(values[recent]), 1)
    if slope >= 0:
        return RenewalForecast(None, None, float(intercept), int(rounds[0]), int(rounds[-1]))
    predicted = (math.log(threshold) - intercept) / slope
    return RenewalForecast(float(-1.0 / slope), float(max(predicted, len(values))), float(intercept),
                           int(rounds[0]), int(rounds[-1]))


def exponential_decay(t, amplitude, tau_exp):
    return amplitude * np.exp(-t / tau_exp)


def replica_autocorrelation(indices: torch.Tensor) -> torch.Tensor:
    """Normalized C(t) for (rounds, replicas, chains) origin labels.

    Labels must be replica numbers, not globally unique chain identifiers.
    Use zero padding to at least twice the trajectory length to avoid circular
    correlations. The reference's biased (unadjusted for lag) FFT estimator is
    retained; no empirical per-chain mean is subtracted.
    """
    if indices.ndim != 3 or indices.shape[0] < 8 or indices.shape[1] < 2 or indices.shape[2] < 1:
        raise ValueError("PTT mixing needs at least 8 rounds, 2 replicas and 1 chain.")
    if not torch.isfinite(indices).all() or indices.min() < 0 or indices.max() >= indices.shape[1]:
        raise ValueError("PTT mixing labels must be replica indices.")
    n = indices.shape[0]
    x = indices.double().reshape(n, -1) - (indices.shape[1] - 1) / 2
    spectrum = torch.fft.rfft(x, n=1 << (2 * n - 1).bit_length(), dim=0)
    correlation = torch.fft.irfft(spectrum.abs().square(), n=1 << (2 * n - 1).bit_length(), dim=0)
    correlation = correlation[: n // 2].mean(1)
    if correlation[0] <= 0 or not torch.isfinite(correlation).all():
        raise ValueError("Degenerate PTT replica-index autocorrelation.")
    return correlation / correlation[0]


def integrated_autocorrelation_time(correlation: torch.Tensor) -> float:
    """Self-consistent window t >= 6 tau_int(t), with C(0)/2."""
    c = correlation.double().clone()
    if c.ndim != 1 or len(c) < 2 or not torch.isfinite(c).all():
        raise ValueError("A finite autocorrelation vector is required.")
    c[0] = 0.5
    cumulative = c.cumsum(0)
    windows = torch.where(torch.arange(len(c), device=c.device) >= 6 * cumulative)[0]
    return max(1.0, float(cumulative[windows[0] if len(windows) else -1]))


def exponential_autocorrelation_time(correlation: torch.Tensor) -> float:
    """Fit the reference tail window: first zero / 3 through twice that zero.

    For a trajectory without a zero crossing, use half the available lags as
    the window endpoint, as in the reference. Positive fit bounds and explicit
    fit failure avoid accepting a negative or non-finite relaxation time.
    """
    c = correlation.detach().double().cpu().numpy()
    if c.ndim != 1 or len(c) < 4 or not np.isfinite(c).all():
        raise ValueError("At least four finite autocorrelation lags are required.")
    zeros = np.flatnonzero(c <= 0)
    end = int(zeros[0]) if len(zeros) else len(c) // 2
    # Completely decorrelated or alternating replica labels resolve only a
    # one-round time scale (also the lower bound returned by reference TRWA).
    if end <= 1:
        return 1.0
    start = max(1, end // 3)
    stop = min(len(c), 2 * end)
    x = np.arange(start, stop, dtype=float)
    y = c[start:stop]
    if len(x) < 3 or np.max(y) <= 0:
        raise ValueError("Insufficient PTT autocorrelation tail for a fit.")
    # Fit in local coordinates with unit-scale observations. Otherwise a
    # clean, long exponential tail can be so small that the least-squares
    # solver accepts its initial guess without taking a step.
    x -= x[0]
    y = y / np.max(np.abs(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            fit, _ = curve_fit(
                exponential_decay, x, y,
                p0=(max(float(y[0]), 1e-6), max(end / 20, 1.0)),
                bounds=((0.0, 1e-6), (np.inf, np.inf)), maxfev=10000,
            )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            raise ValueError("PTT exponential autocorrelation fit failed.") from exc
    tau = float(fit[1])
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Invalid PTT exponential autocorrelation time.")
    return max(1.0, tau)


def process_replica_experiment(indices: torch.Tensor, *, n_thermalization=0, reference=False):
    """Return (tau_int, tau_exp, C) without changing the recorded labels."""
    if not isinstance(n_thermalization, int) or not 0 <= n_thermalization < len(indices):
        raise ValueError("Invalid PTT thermalization length.")
    if reference:
        labels = indices[n_thermalization:]
        if labels.ndim != 3 or len(labels) < 8:
            raise ValueError("PTT mixing needs at least eight recorded rounds.")
        x = labels.reshape(len(labels), -1).float() - (labels.shape[1] - 1) / 2
        n_fft = 1 << len(labels).bit_length()
        spectrum = torch.fft.fft(x, n=n_fft, dim=0)
        c = torch.fft.ifft(spectrum.abs().square(), dim=0).real[:len(labels) // 2].mean(1)
        c = c / c[0]
        if not torch.isfinite(c).all():
            raise ValueError("Non-finite reference replica autocorrelation.")
        negative = torch.where(c < 0)[0]
        end = int(negative[0]) if len(negative) else len(c) // 2
        tau_int = integrated_autocorrelation_time(c)
        # Reference _tau_int modifies C(0) before the exponential fit.
        c[0] = 0.5
        if end <= 1:
            return tau_int, 1.0, c
        start, stop = end // 3, min(len(c), 2 * end)
        if stop - start < 3:
            raise ValueError("Insufficient reference autocorrelation tail.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                fit, _ = curve_fit(exponential_decay, np.arange(start, stop),
                                   c[start:stop].cpu().numpy(), p0=[float(c[start]), end / 20])
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            raise ValueError("Reference exponential autocorrelation fit failed.") from exc
        if not math.isfinite(float(fit[1])):
            raise ValueError("Non-finite reference exponential mixing time.")
        return tau_int, max(1.0, float(fit[1])), c
    c = replica_autocorrelation(indices[n_thermalization:])
    return integrated_autocorrelation_time(c), exponential_autocorrelation_time(c), c
