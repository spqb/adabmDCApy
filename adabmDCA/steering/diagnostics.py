"""Stationarity and adjacent normalization diagnostics for steered runs."""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import expit, ndtri
from scipy.stats import rankdata

from adabmDCA.ptt.health import bar


def _split(chains):
    half = chains.shape[1] // 2
    return np.concatenate((chains[:, :half], chains[:, -half:]))


def _rank(chains):
    return ndtri((rankdata(chains).reshape(chains.shape) - 0.375) / (chains.size + 0.25))


def _rhat(chains):
    length = chains.shape[1]
    within = chains.var(axis=1, ddof=1).mean()
    between = chains.mean(axis=1).var(ddof=1)
    if within == 0:
        return 1.0 if between == 0 else float("inf")
    return float(np.sqrt((length - 1) / length + between / within))


def _ess(chains):
    length = chains.shape[1]
    centered = chains - chains.mean(axis=1, keepdims=True)
    transform = np.fft.rfft(centered, n=2 * length)
    covariance = np.fft.irfft(np.abs(transform) ** 2)[:, :length].mean(axis=0) / length
    within = covariance[0] * length / (length - 1)
    variance = covariance[0] + chains.mean(axis=1).var(ddof=1)
    if variance == 0:
        return float(chains.size)
    rho = 1 - (within - covariance) / variance
    rho[0] = 1
    count = length // 2
    pairs = np.minimum.accumulate(rho[:2 * count].reshape(count, 2).sum(axis=1))
    return float(chains.size / max(1, -1 + 2 * pairs[pairs > 0].sum()))


def summarize_traces(history, names):
    """Diagnostics on independent groups' mean time series."""
    traces = np.asarray(history)
    diagnostics = {}
    for name, values in zip(names, np.moveaxis(traces, -1, 0)):
        chains = _split(values.T)
        ranks = _rank(chains)
        folded = _rank(np.abs(chains - np.median(chains)))
        lower, upper = np.quantile(chains, [0.05, 0.95])
        diagnostics[name] = {
            "rhat": max(_rhat(ranks), _rhat(folded)),
            "bulk_ess": _ess(ranks),
            "tail_ess": min(_ess(chains <= lower), _ess(chains >= upper)),
        }
    return diagnostics


def partition_point(log_z0, work, endpoint, *, keep=slice(None)):
    """Estimate the partition function, entropy, and overlap from BAR bridges."""
    log_z = log_z0
    overlap = []
    for lower, upper in work:
        a, b = lower[:, keep].ravel(), upper[:, keep].ravel()
        difference = float(bar(torch.from_numpy(a), torch.from_numpy(b)))
        log_z -= difference
        overlap.append(float((expit(difference - a).mean() + expit(b - difference).mean()) * 0.5))
    return {
        "log_z": float(log_z), "entropy_nats": float(log_z + endpoint[:, keep].mean()),
        "bar_overlap": overlap, "overlap_ok": all(value >= 0.01 for value in overlap),
    }


def partition(log_z0, work, endpoint, groups):
    """Sum BAR bridges and jackknife independent groups jointly over all rungs."""
    result = partition_point(log_z0, work, endpoint)

    labels = np.empty(endpoint.shape[1], dtype=int)
    for label, group in enumerate(groups):
        labels[group] = label
    estimates = (partition_point(log_z0, work, endpoint, keep=labels != group) for group in range(len(groups)))
    leave_one_out = np.array([[point["log_z"], point["entropy_nats"]] for point in estimates])
    error = np.sqrt((len(groups) - 1) / len(groups) * np.square(leave_one_out - leave_one_out.mean(axis=0)).sum(axis=0))
    return {**result, "log_z_mcse": float(error[0]), "entropy_mcse": float(error[1])}
