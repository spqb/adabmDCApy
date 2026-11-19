from typing import Literal

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

_DIAGNOSTIC_BLUE = "#31688E"
_DIAGNOSTIC_TEAL = "#35B779"
_DIAGNOSTIC_CORAL = "#E76F51"
_DIAGNOSTIC_TEXT = "#263238"


def _style_diagnostic_axis(ax: Axes) -> None:
    """Apply a restrained, consistent style to sampling diagnostics."""
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, color="#D9DEE3", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_DIAGNOSTIC_TEXT, labelsize=10)
    ax.xaxis.label.set_color(_DIAGNOSTIC_TEXT)
    ax.yaxis.label.set_color(_DIAGNOSTIC_TEXT)
    sns.despine(ax=ax)


def _plot_scatter_labels(
    ax: Axes,
    data1: np.ndarray,
    pc1: int = 0,
    pc2: int = 1,
    data2: np.ndarray | None = None,
    labels: list[str] | str | None = "Data",
    colors: list[str] | str = "black",
) -> Axes:
    color1 = colors[0] if isinstance(colors, list) else colors
    label1 = labels[0] if isinstance(labels, list) else labels
    ax.scatter(
        data1[:, pc1],
        data1[:, pc2],
        color=color1,
        s=22,
        label=label1,
        zorder=1,
        alpha=0.42,
        edgecolors="none",
        rasterized=True,
    )
    if data2 is not None:
        color2 = colors[1] if isinstance(colors, list) and len(colors) > 1 else _DIAGNOSTIC_CORAL
        label2 = labels[1] if isinstance(labels, list) and len(labels) > 1 else None
        ax.scatter(
            data2[:, pc1],
            data2[:, pc2],
            color=color2,
            label=label2,
            s=18,
            zorder=2,
            edgecolors="white",
            marker="o",
            alpha=0.48,
            linewidths=0.25,
            rasterized=True,
        )

    return ax


def _plot_hist(
    ax: Axes,
    data1: np.ndarray,
    pc: int,
    data2: np.ndarray | None = None,
    colors: list[str] | str = "black",
    labels: list[str] | str | None = "Data",
    orientation: Literal["vertical", "horizontal"] = "vertical",
) -> Axes:
    label1 = labels[0] if isinstance(labels, list) else labels
    color1 = colors[0] if isinstance(colors, list) else colors
    values = [data1[:, pc]]
    if data2 is not None:
        values.append(data2[:, pc])
    finite_values = np.concatenate([value[np.isfinite(value)] for value in values])
    if finite_values.size:
        num_bins = int(np.clip(np.ceil(np.sqrt(finite_values.size)), 5, 40))
        bins = np.histogram_bin_edges(finite_values, bins=num_bins)
    else:
        bins = 10
    ax.hist(
        data1[:, pc], bins=bins, color=color1, histtype="stepfilled", alpha=0.18,
        density=True, orientation=orientation, linewidth=0,
    )
    ax.hist(
        data1[:, pc], bins=bins, color=color1, histtype="step", label=label1,
        density=True, orientation=orientation, linewidth=1.5,
    )
    if data2 is not None:
        color2 = colors[1] if isinstance(colors, list) and len(colors) > 1 else _DIAGNOSTIC_CORAL
        label2 = labels[1] if isinstance(labels, list) and len(labels) > 1 else None
        ax.hist(
            data2[:, pc], bins=bins, color=color2, histtype="stepfilled", alpha=0.14,
            density=True, orientation=orientation, linewidth=0,
        )
        ax.hist(
            data2[:, pc], bins=bins, color=color2, histtype="step", label=label2,
            density=True, orientation=orientation, linewidth=1.5,
        )
    ax.set_facecolor("#FAFAFA")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    sns.despine(ax=ax, left=True, bottom=True)
    return ax


def plot_PCA(
    fig: Figure,
    data1: np.ndarray,
    pc1: int = 0,
    pc2: int = 1,
    data2: np.ndarray | None = None,
    labels: list[str] | str = "Natural",
    colors: list[str] | str = _DIAGNOSTIC_BLUE,
    title: str | None = None,
    explained_variance_ratio: np.ndarray | None = None,
) -> Figure:
    """Makes the scatter plot of the components (pc1, pc2) of the input data and shows the histograms of the components.

    Args:
        fig (Figure): Figure to plot the data.
        data1 (np.ndarray): Data to plot.
        pc1 (int, optional): First principal direction. Defaults to 0.
        pc2 (int, optional): Second principal direction. Defaults to 1.
        data2 (Optional[np.ndarray], optional): Data to be superimposed to data1. Defaults to None.
        labels (Union[List[str], str], optional): Labels to put in the legend. Defaults to "Data".
        colors (Union[List[str], str], optional): Colors to be used. Defaults to "black".
        title (Optional[str], optional): Title of the plot. Defaults to None.

    Returns:
        Figure: Updated figure.
    """
    
    data1 = np.asarray(data1)
    if data1.ndim != 2 or data1.shape[1] <= max(pc1, pc2):
        raise ValueError("data1 must contain the requested principal components")
    if data2 is not None:
        data2 = np.asarray(data2)
        if data2.ndim != 2 or data2.shape[1] <= max(pc1, pc2):
            raise ValueError("data2 must contain the requested principal components")

    fig.clear()
    gs = GridSpec(
        4,
        4,
        figure=fig,
        width_ratios=(1, 1, 1, 0.72),
        height_ratios=(0.72, 1, 1, 1),
        hspace=0.04,
        wspace=0.04,
    )
    ax_scatter = fig.add_subplot(gs[1:, :3])
    ax_hist_x = fig.add_subplot(gs[0, :3], sharex=ax_scatter)
    ax_hist_y = fig.add_subplot(gs[1:, 3], sharey=ax_scatter)
        
    ax_scatter = _plot_scatter_labels(
        ax=ax_scatter,
        data1=data1,
        pc1=pc1,
        pc2=pc2,
        data2=data2,
        labels=labels,
        colors=colors,
    )
    ax_hist_x = _plot_hist(
        ax=ax_hist_x,
        data1=data1,
        pc=pc1,
        data2=data2,
        colors=colors,
        labels=labels,
        orientation="vertical",
    )
    ax_hist_y = _plot_hist(
        ax=ax_hist_y,
        data1=data1,
        pc=pc2,
        data2=data2,
        colors=colors,
        labels=None,
        orientation="horizontal",
    )

    _style_diagnostic_axis(ax_scatter)
    variance = None if explained_variance_ratio is None else np.asarray(explained_variance_ratio)
    x_label = f"PC {pc1 + 1}"
    y_label = f"PC {pc2 + 1}"
    if variance is not None and variance.size > max(pc1, pc2):
        x_label += f" ({100 * variance[pc1]:.1f}%)"
        y_label += f" ({100 * variance[pc2]:.1f}%)"
    ax_scatter.set_xlabel(x_label)
    ax_scatter.set_ylabel(y_label)
    if title is not None:
        fig.suptitle(title, color=_DIAGNOSTIC_TEXT, y=0.995)
    handles, legend_labels = ax_scatter.get_legend_handles_labels()
    if handles:
        ax_scatter.legend(handles, legend_labels, frameon=False, loc="best")

    return fig
    
    
def plot_pearson_sampling(
    ax: Axes,
    checkpoints: np.ndarray,
    pearsons: np.ndarray,
    pearson_training: float | None = None
) -> Axes:
    """Plots the Pearson correlation coefficient over sampling time.

    Args:
        ax (Axes): Axes to plot the data.
        checkpoints (np.ndarray): Checkpoints of the sampling.
        pearsons (np.ndarray): Pearson correlation coefficients at different checkpoints.
        pearson_training (Optional[float], optional): Pearson correlation coefficient obtained during training. Defaults to None.

    Returns:
        Axes: Updated axes.
    """
    
    checkpoints = np.asarray(checkpoints)
    pearsons = np.asarray(pearsons)
    if checkpoints.size == 0 or pearsons.size == 0 or checkpoints.shape != pearsons.shape:
        raise ValueError("checkpoints and pearsons must be non-empty arrays with the same shape")
    _style_diagnostic_axis(ax)
    if pearson_training is not None:
        ax.axhline(
            y=pearson_training,
            ls="dashed",
            color=_DIAGNOSTIC_CORAL,
            label="Training",
            lw=1.4,
            zorder=1,
        )
        annotation_text = f"Training: {pearson_training:.3f}\nSampling: {pearsons[-1]:.3f}"
    else:
        annotation_text = f"Sampling: {pearsons[-1]:.3f}"
    ax.plot(
        checkpoints,
        pearsons,
        "-o",
        label="Generated samples",
        lw=1.8,
        ms=4,
        color=_DIAGNOSTIC_BLUE,
        markerfacecolor="white",
        markeredgewidth=1.2,
        zorder=2,
    )
    if np.all(checkpoints > 0):
        ax.set_xscale("log")
    ax.set_title("Pairwise-correlation convergence", color=_DIAGNOSTIC_TEXT, pad=10)
    ax.set_xlabel("Sampling time [sweeps]")
    ax.set_ylabel(r"Pearson $C_{ij}(a,b)$")
    ax.annotate(
        annotation_text,
        xy=(0.97, 0.08),
        xycoords="axes fraction",
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#C8CDD2", "alpha": 0.9},
    )
    ax.legend(frameon=False, loc="best")
    return ax


def plot_autocorrelation(
    ax: Axes,
    checkpoints: np.ndarray,
    autocorr: np.ndarray,
    gen_seqid: float | np.ndarray,
    data_seqid: float | None = None,
    *,
    autocorr_std: np.ndarray | None = None,
    independent_std: np.ndarray | None = None,
) -> Axes:
    """Plots the time-autocorrelation curve of the sequence identity and the generated and data sequence identities.
    
    Args:
        ax (Axes): Axes to plot the data.
        checkpoints (np.ndarray): Checkpoints of the sampling.
        autocorr (np.ndarray): Time-autocorrelation of the sequence identity.
        gen_seqid (float or np.ndarray): Independent-chain sequence identity, either as a level or a curve.
        data_seqid (float, optional): Reference-data sequence identity level.
        autocorr_std (np.ndarray, optional): Uncertainty of the autocorrelation curve.
        independent_std (np.ndarray, optional): Uncertainty of the independent-chain curve.

    Returns:
        Axes: Updated axes.
    """
    checkpoints = np.asarray(checkpoints)
    autocorr = np.asarray(autocorr)
    independent = np.asarray(gen_seqid)
    if checkpoints.size == 0 or autocorr.shape != checkpoints.shape:
        raise ValueError("checkpoints and autocorr must be non-empty arrays with the same shape")

    _style_diagnostic_axis(ax)
    ax.plot(
        checkpoints,
        autocorr,
        "-o",
        c=_DIAGNOSTIC_BLUE,
        lw=1.8,
        ms=4,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label=r"Same chains: $t$ vs $t/2$",
        zorder=3,
    )
    if autocorr_std is not None:
        autocorr_std = np.asarray(autocorr_std)
        if autocorr_std.shape != checkpoints.shape:
            raise ValueError("autocorr_std must have the same shape as checkpoints")
        ax.fill_between(
            checkpoints,
            np.clip(autocorr - autocorr_std, 0.0, 1.0),
            np.clip(autocorr + autocorr_std, 0.0, 1.0),
            color=_DIAGNOSTIC_BLUE,
            alpha=0.26,
            linewidth=0,
            label=r"Same chains: $\pm 1$ std",
            zorder=2,
        )

    if independent.ndim == 0:
        comparison = np.full(checkpoints.shape, float(independent))
        ax.axhline(
            y=float(independent), color=_DIAGNOSTIC_TEAL, lw=1.6, ls="dashed", label="Independent chains", zorder=2
        )
    else:
        if independent.shape != checkpoints.shape:
            raise ValueError("array-valued gen_seqid must have the same shape as checkpoints")
        comparison = independent
        ax.plot(
            checkpoints,
            independent,
            "-o",
            color=_DIAGNOSTIC_TEAL,
            lw=1.8,
            ms=4,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="Independent chains",
            zorder=3,
        )
        if independent_std is not None:
            independent_std = np.asarray(independent_std)
            if independent_std.shape != checkpoints.shape:
                raise ValueError("independent_std must have the same shape as checkpoints")
            ax.fill_between(
                checkpoints,
                np.clip(independent - independent_std, 0.0, 1.0),
                np.clip(independent + independent_std, 0.0, 1.0),
                color=_DIAGNOSTIC_TEAL,
                alpha=0.22,
                linewidth=0,
                label=r"Independent: $\pm 1$ std",
                zorder=2,
            )
    if data_seqid is not None:
        ax.axhline(y=data_seqid, color=_DIAGNOSTIC_CORAL, lw=1.4, ls="dashed", label="Reference data", zorder=1)
    ax.set_title("Sampling mixing diagnostic", color=_DIAGNOSTIC_TEXT, pad=10)
    ax.set_xlabel(r"$\tau$ [sweeps]")
    ax.set_ylabel(r"$\langle \mathrm{SeqID}(t, t-\tau) \rangle$")
    # ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="best", fontsize=9)

    crossings = np.flatnonzero(autocorr <= comparison)
    if crossings.size:
        mixing_time = checkpoints[crossings[0]]
        annotation_text = r"$\tau_{\mathrm{mix}}$" + f": {mixing_time} sweeps"
        ax.annotate(
            annotation_text,
            xy=(0.95, 0.65),
            xycoords="axes fraction",
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#C8CDD2", "alpha": 0.9},
        )
    
    return ax


def plot_ptt_autocorrelation(
    ax: Axes,
    correlation: np.ndarray,
    tau_int: float | None,
    tau_exp: float | None,
) -> Axes:
    """Plot replica-index autocorrelation and the two PTT time estimates."""
    correlation = np.asarray(correlation, dtype=float).reshape(-1)
    _style_diagnostic_axis(ax)
    if correlation.size:
        lags = np.arange(correlation.size)
        positive = np.where(correlation > 0, correlation, np.nan)
        ax.plot(lags, positive, color=_DIAGNOSTIC_BLUE, lw=2.0, label="Replica index", zorder=3)
    else:
        ax.text(
            0.5, 0.5, "Autocorrelation unavailable within round budget",
            transform=ax.transAxes, ha="center", va="center", color=_DIAGNOSTIC_TEXT,
        )
    ax.axhline(np.exp(-1.0), color="#AAB0B6", lw=1.0, ls=":", label=r"$e^{-1}$", zorder=1)
    if tau_int is not None and np.isfinite(tau_int) and tau_int > 0:
        ax.axvline(
            tau_int, color=_DIAGNOSTIC_TEAL, lw=1.7, ls="--",
            label=rf"$\tau_{{\mathrm{{int}}}}={tau_int:.2f}$", zorder=2,
        )
    if tau_exp is not None and np.isfinite(tau_exp) and tau_exp > 0:
        ax.axvline(
            tau_exp, color=_DIAGNOSTIC_CORAL, lw=1.7, ls="-.",
            label=rf"$\tau_{{\mathrm{{exp}}}}={tau_exp:.2f}$", zorder=2,
        )
    if tau_int is None or tau_exp is None:
        ax.text(
            0.98, 0.95, "Mixing time unresolved",
            transform=ax.transAxes, ha="right", va="top", color=_DIAGNOSTIC_CORAL,
        )
    ax.set_title("PTT replica mixing", color=_DIAGNOSTIC_TEXT, pad=10)
    ax.set_xlabel("Lag [exchange rounds]")
    ax.set_ylabel("Replica-index autocorrelation")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.05)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, loc="best", fontsize=9)
    return ax


def plot_ptt_renewal(
    figure: Figure,
    history: dict[str, np.ndarray],
    *,
    tolerance: float,
    n_models: int,
    warmup_rounds: int | None,
    renewal_rounds: int | None,
    chunk_rounds: int = 0,
    stationary: bool = True,
) -> Figure:
    """Plot PTT population renewal for the warmup and, if it ran, the stationary phase.

    ``history`` holds per-round ``{phase}_ladder_old`` (G, fraction of all
    ladder configurations born before the phase reference) and
    ``{phase}_endpoint_fresh`` (F, fraction of endpoint configurations born
    after it) for ``phase`` in ``warmup`` and ``stationary``. The top row
    shows G and F; the bottom row shows the old fractions G and 1 - F on a
    logarithmic scale with the stopping thresholds, and the exponential fit of
    G over the recent half of each phase (see ``renewal_forecast``). Renewal
    is decided by G reaching ``tolerance / n_models``; the endpoint fraction
    1 - F can rise again when old configurations climb back up the ladder.
    """
    from adabmDCA.ptt.mixing import renewal_forecast

    phases = (
        ("warmup", "Renewal of the initial state", warmup_rounds, "Rounds since start"),
        ("stationary", "Stationary renewal", renewal_rounds, "Rounds since warmup end"),
    )[:2 if stationary else 1]
    axes = figure.subplots(2, len(phases), sharex="col", squeeze=False)
    ladder_limit = tolerance / max(n_models, 1)
    for column, (phase, title, renewed, xlabel) in enumerate(phases):
        ladder_old = np.asarray(history.get(f"{phase}_ladder_old", ()), dtype=float)
        endpoint_fresh = np.asarray(history.get(f"{phase}_endpoint_fresh", ()), dtype=float)
        top, bottom = axes[0, column], axes[1, column]
        for ax in (top, bottom):
            _style_diagnostic_axis(ax)
        if ladder_old.size and renewed is None:
            title = f"{title}\nnot renewed within the round budget"
        top.set_title(title, color=_DIAGNOSTIC_CORAL if ladder_old.size and renewed is None else _DIAGNOSTIC_TEXT,
                      pad=10)
        if ladder_old.size == 0:
            for ax in (top, bottom):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.5, 0.5, "Phase not reached within round budget", transform=ax.transAxes,
                        ha="center", va="center", color=_DIAGNOSTIC_CORAL)
            bottom.set_xlabel(xlabel)
            continue
        rounds = np.arange(1, ladder_old.size + 1)
        top.plot(rounds, endpoint_fresh, color=_DIAGNOSTIC_CORAL, lw=1.8, label=r"$F(t)$: endpoint fresh")
        top.plot(rounds, ladder_old, color=_DIAGNOSTIC_BLUE, lw=1.8, label=r"$G(t)$: ladder old")
        top.axhline(1.0 - tolerance, color="#AAB0B6", lw=1.0, ls=":", label=rf"$1-\epsilon$, $\epsilon={tolerance:g}$")
        top.set_ylim(-0.02, 1.02)
        top.set_ylabel("Fraction of configurations")

        bottom.plot(rounds, np.where(1.0 - endpoint_fresh > 0, 1.0 - endpoint_fresh, np.nan),
                    color=_DIAGNOSTIC_CORAL, lw=1.8, label=r"$1-F(t)$: endpoint old")
        bottom.plot(rounds, np.where(ladder_old > 0, ladder_old, np.nan),
                    color=_DIAGNOSTIC_BLUE, lw=1.8, label=r"$G(t)$: ladder old")
        bottom.axhline(tolerance, color=_DIAGNOSTIC_CORAL, lw=1.0, ls=":", label=r"$\epsilon$")
        bottom.axhline(ladder_limit, color=_DIAGNOSTIC_BLUE, lw=1.0, ls=":",
                       label=r"$\epsilon / n_{\mathrm{models}}$: stop when $G$ reaches it")
        forecast = renewal_forecast(ladder_old, ladder_limit)
        if forecast is not None and forecast.decay_rounds is not None:
            span = np.arange(forecast.fit_start, ladder_old.size + 1)
            bottom.plot(span, np.exp(forecast.intercept - span / forecast.decay_rounds), color="#263238", lw=1.1,
                        ls="--", label=f"fit of $G$: decay {forecast.decay_rounds:.0f} rounds")
        bottom.set_yscale("log")
        floor = min(ladder_limit, 1.0 / max(ladder_old.size, 1)) / 10
        positive = np.concatenate([ladder_old[ladder_old > 0], 1.0 - endpoint_fresh[endpoint_fresh < 1]])
        if positive.size:
            floor = min(floor, positive.min() / 2)
        bottom.set_ylim(floor, 1.5)
        bottom.set_ylabel("Old fraction")
        bottom.set_xlabel(xlabel)

        if renewed is not None:
            for ax in (top, bottom):
                ax.axvline(renewed, color=_DIAGNOSTIC_TEAL, lw=1.5, ls="--",
                           label=f"renewed: {renewed} rounds" if ax is top else None)
        if phase == "stationary" and chunk_rounds > 0:
            for boundary in range(chunk_rounds, ladder_old.size + 1, chunk_rounds):
                bottom.axvline(boundary, color="#AAB0B6", lw=0.8, ls="-", alpha=0.6, zorder=0)
        top.legend(frameon=True, framealpha=0.85, edgecolor="none", loc="center right", fontsize=8)
        bottom.legend(frameon=True, framealpha=0.85, edgecolor="none", loc="lower left", fontsize=8)
    figure.suptitle("PTT population renewal", color=_DIAGNOSTIC_TEXT)
    return figure


_TRAPPING_COLORS = {"none": "#AAB0B6", "noise": "#AAB0B6", "slow": "#E9A23B", "blocking": "#C0392B"}


def _p_text(p: float) -> str:
    """p-values of the age test; its resolution is about 1e-4."""
    return "p < 2e-4" if p < 2e-4 else f"p {p:.0e}" if p < 0.01 else f"p {p:.2f}"


def plot_ptt_ladder_health(figure: Figure, health: dict, mixing: dict | None = None) -> Figure:
    """Plot how configurations move through a PTT ladder.

    Panels:

    - swap acceptance per link: the mean and the median, 10% and 1% quantiles
      of the per-configuration acceptance against random partners, for upper
      configurations moving down and lower ones moving up. The ladder is built
      so that means sit near the 0.25 target; low quantiles reveal
      configurations that rarely cross;
    - trapped configurations per link: mean age of the configurations that
      practically cannot move down (swap probability below the immobility
      threshold) divided by that of the others. The tick on each bar is the
      95% quantile of the same ratio for random groups of configurations of
      the same size (the null of the age test,
      ``adabmDCA.ptt.health.age_permutation_test``). Colours: grey, within
      chance; amber, immobile configurations older than chance but renewed
      (slow link); red, older than chance and more of them older than the
      renewal time than the renewal tolerance allows (blocking link). Labels: share of
      immobile configurations, how many of them are older than the renewal
      time, and the p-value of the age test.

    The title gives the renewal time, or says in red that renewal failed and
    how many configurations were never renewed. Free energies, effective
    sample sizes and Crooks slopes are in the log only.
    """
    pairs = health["pairs"]
    axes = figure.subplots(1, 2)
    for ax in axes:
        _style_diagnostic_axis(ax)
    if not pairs:
        for ax in axes:
            ax.text(0.5, 0.5, "No adjacent pairs", transform=ax.transAxes, ha="center", va="center")
        return figure
    x = np.arange(len(pairs))
    labels = [f"{pair['lower']}→{pair['upper']}" for pair in pairs]
    floor = 1e-6

    ax = axes[0]
    for offset, direction, color, name in ((-0.12, "down", _DIAGNOSTIC_CORAL, "upper replica, moving down"),
                                           (0.12, "up", _DIAGNOSTIC_TEAL, "lower replica, moving up")):
        if f"acceptance_{direction}_q50" not in pairs[0]:
            continue
        q01 = np.array([max(pair[f"acceptance_{direction}_q01"], floor) for pair in pairs])
        q10 = np.array([max(pair[f"acceptance_{direction}_q10"], floor) for pair in pairs])
        q50 = np.array([max(pair[f"acceptance_{direction}_q50"], floor) for pair in pairs])
        ax.vlines(x + offset, q01, q50, color=color, lw=1.5, alpha=0.7)
        ax.scatter(x + offset, q50, color=color, s=28, zorder=3, label=f"{name}: median")
        ax.scatter(x + offset, q10, color=color, s=22, marker="v", zorder=3)
        ax.scatter(x + offset, q01, color=color, s=22, marker="x", zorder=3)
    ax.scatter(x, [max(pair["mean_acceptance"], floor) for pair in pairs], color=_DIAGNOSTIC_BLUE, marker="D",
               s=30, zorder=4, label="mean (both directions)")
    ax.axhline(0.25, color="#AAB0B6", lw=1.0, ls="--", label="snapshot target 0.25")
    ax.scatter([], [], color="#666", marker="v", s=22, label="10% quantile")
    ax.scatter([], [], color="#666", marker="x", s=22, label="1% quantile")
    ax.set_yscale("log")
    ax.set_ylim(floor / 2, 1.5)
    ax.set_ylabel("per-configuration swap acceptance")
    ax.set_title("Swap acceptance per link: mean and tail", color=_DIAGNOSTIC_TEXT, fontsize=9)
    # Below the axes: the 1% quantiles that matter most sit at the floor.
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)

    ax = axes[1]
    ratios = []
    for pair in pairs:
        immobile_age, mobile_age = pair.get("immobile_mean_age"), pair.get("mobile_mean_age")
        ratios.append(np.nan if immobile_age is None or not mobile_age else immobile_age / mobile_age)
    heights = np.nan_to_num(np.asarray(ratios, dtype=float), nan=0.0)
    classified = all("trapping" in pair for pair in pairs)
    if classified:
        colors = [_TRAPPING_COLORS[pair["trapping"]] for pair in pairs]
    else:  # Results saved before the age test: the former fixed threshold of 2.
        colors = [_DIAGNOSTIC_CORAL if ratio > 2 else _DIAGNOSTIC_BLUE for ratio in heights]
    ax.bar(x, heights, color=colors, width=0.6)
    nulls = [pair.get("immobile_age_null_q95") for pair in pairs]
    for position, null in zip(x, nulls):
        if null is not None:
            ax.hlines(null, position - 0.36, position + 0.36, color=_DIAGNOSTIC_TEXT, lw=1.6, zorder=3)
    ax.axhline(1.0, color="#AAB0B6", lw=1.0, ls="--")
    if not classified:
        ax.axhline(2.0, color=_DIAGNOSTIC_CORAL, lw=0.8, ls=":")
    peak = max([float(np.nanmax(heights))] + [null for null in nulls if null is not None])
    top = max(2.5, peak * 1.3)
    for position, ratio, null, pair in zip(x, ratios, nulls, pairs):
        fraction = pair["immobile_down_fraction"]
        lines = ["none" if fraction == 0 else f"{fraction:.1%}"]
        if pair.get("immobile_old") is not None:
            lines.append(f"{pair['immobile_old']} old")
        if pair.get("immobile_age_p") is not None:
            lines.append(_p_text(pair["immobile_age_p"]))
        above = max(0.0 if np.isnan(ratio) else ratio, null or 0.0)
        ax.text(position, above + top * 0.015, "\n".join(lines), ha="center",
                va="bottom", fontsize=6.5, color=_DIAGNOSTIC_TEXT, linespacing=1.1)
    ax.set_ylim(0, top)
    ax.set_ylabel("age of immobile / age of mobile configurations")
    ax.set_title(f"Trapped configurations (swap probability < {health['immobile_threshold']:g})\n"
                 + ("label: share immobile · of them old (missed a renewal) · age-test p" if classified
                    else "label: share immobile"),
                 color=_DIAGNOSTIC_TEXT, fontsize=9)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Line2D([], [], color="#AAB0B6", lw=1.0, ls="--", label="ratio 1: no age difference")]
    if any(null is not None for null in nulls):
        handles.append(Line2D([], [], color=_DIAGNOSTIC_TEXT, lw=1.6, label="95% of random groups (null)"))
    if classified:
        level = pairs[0].get("trapping_level")
        handles += [
            Patch(color=_TRAPPING_COLORS["noise"], label="within chance"
                  + ("" if level is None else f" (p ≥ {level:.0e})")),
            Patch(color=_TRAPPING_COLORS["slow"], label="slow: older than chance, renewed"),
            Patch(color=_TRAPPING_COLORS["blocking"], label="blocking: older than chance, and the old ones\n"
                  "exceed the renewal tolerance"),
        ]
    else:
        handles.append(Line2D([], [], color=_DIAGNOSTIC_CORAL, lw=0.8, ls=":", label="trapped (2)"))
    # Below the axes, like the left panel: tall bars and their labels fill the top.
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)

    for ax in axes:
        # Tilted so that two-digit links (10→11) do not run into each other.
        ax.set_xticks(x, labels, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlabel("adjacent models")

    title = f"PTT ladder health · endpoint log Z (BAR) {health['log_z_bar']:.2f} ± {health['log_z_bar_error']:.2f}"
    color = _DIAGNOSTIC_TEXT
    if mixing and mixing.get("warmup_rounds") is not None:
        decay = mixing.get("warmup_decay_rounds")
        title += f" · renewal after {mixing['warmup_rounds']} rounds"
        if decay is not None:
            title += f" (decay time {decay:.0f})"
    elif mixing and mixing.get("method") == "renewal" and mixing.get("status") not in (None, "converged"):
        color = _TRAPPING_COLORS["blocking"]
        rounds = mixing.get("rounds")
        title += "\nrenewal NOT reached" + ("" if rounds is None else f" within {rounds} rounds")
        chains = mixing.get("n_chains") or health.get("n_chains")
        old = mixing.get("old_fraction_by_model")
        if chains and old:
            title += (f": {round(sum(old) * chains)} configurations never renewed "
                      f"(tolerance {mixing['tolerance'] * chains:.0f})")
    figure.suptitle(title, color=color)
    return figure


def plot_distance_comparison(figure: Figure, distances: dict) -> Figure:
    """Hamming distances within and between natural and generated sequences.

    Left: all-pair distance distributions within the training sequences, within
    the samples, and between the two; matching curves mean the samples reproduce
    the family's diversity. Right: nearest-neighbour distances. A model that
    copies its training set puts generated -> training below training ->
    training and, with a held-out set, below held-out -> training, the distance
    at which unseen members of the family sit; generated -> generated near zero
    means the samples collapse onto few sequences. Natural sequences carry their
    reweighting weights.
    """
    axes = figure.subplots(1, 2)
    for ax in axes:
        _style_diagnostic_axis(ax)
    edges = np.asarray(distances["bin_edges"])
    summary = distances["summary"]

    def upper(curves):
        last = max((np.flatnonzero(np.asarray(curve) > 0).max(initial=0) for curve in curves), default=0)
        return min(1.0, edges[min(last + 1, len(edges) - 1)] + 0.05)

    ax = axes[0]
    pairs = distances["all_pairs"]
    ax.stairs(pairs["natural"], edges, color="#31688E", lw=1.8, label="training - training")
    ax.stairs(pairs["generated"], edges, color="#E76F51", lw=1.8, label="generated - generated")
    ax.stairs(pairs["natural_generated"], edges, color="#6C757D", lw=1.5, ls="--", label="training - generated")
    ax.set_xlim(0.0, upper(pairs.values()))
    ax.set_xlabel("Hamming distance (fraction of sites)")
    ax.set_ylabel("density")
    ax.set_title("All pairs: the samples should reproduce the family's diversity",
                 color=_DIAGNOSTIC_TEXT, fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    nearest = distances["nearest"]
    curves = [
        ("natural_to_natural", "training → nearest other training", "#31688E", "-"),
        ("test_to_natural", "held-out → nearest training", "#2A9D8F", "-"),
        ("generated_to_natural", "generated → nearest training", "#E76F51", "-"),
        ("generated_to_test", "generated → nearest held-out", "#2A9D8F", "--"),
        ("generated_to_generated", "generated → nearest other generated", "#E76F51", ":"),
    ]
    for key, label, color, style in curves:
        if key not in nearest:
            continue
        stats = summary[key]
        text = f"{label} (median {stats['median']:.3f}"
        text += f", {stats['identical']:.1%} identical)" if stats["identical"] > 0 else ")"
        ax.stairs(nearest[key], edges, color=color, lw=2.2 if key == "generated_to_natural" else 1.5, ls=style,
                  label=text)
    ax.set_xlim(0.0, upper(nearest.values()))
    ax.set_ylim(0.0, 1.5 * max(max(curve) for curve in nearest.values()))
    ax.set_xlabel("distance to the nearest sequence (fraction of sites)")
    ax.set_ylabel("density")
    reference = "held-out → training" if "test_to_natural" in nearest else "training → other training"
    ax.set_title(f"Nearest neighbours: generated → training far below {reference} signals copying",
                 color=_DIAGNOSTIC_TEXT, fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    return figure


def _log10_text(value: float) -> str:
    if not np.isfinite(value):
        return "< -300"
    return "0" if value > -0.05 else f"{value:.1f}"


_PRIVET_OVER = "#C0392B"
_PRIVET_UNDER = "#4C5FD5"


def _cdf_median(grid: np.ndarray, values) -> float | None:
    """Smallest grid distance at which a cumulative distribution reaches one half."""
    reach = np.flatnonzero(np.asarray(values, dtype=float) >= 0.5)
    return float(grid[reach[0]]) if len(reach) else None


def plot_privet(figure: Figure, privet: dict) -> Figure:
    """PRIVET: the null law of nearest-neighbour distances and the samples against it.

    Left: cumulative distribution of the (reweighted) training sequences'
    distances to their nearest other training sequence (steps) and the fitted
    extreme-value law (line), on a log scale to show the lower tail; the band
    is the fitted window. The box gives the largest ratio between fit and data
    inside the window: the p-values are only as good as the fit at its left edge.

    Right: the same law against the distances of the generated and held-out
    sequences to their nearest training sequence, with their medians in the
    legend. Above the null (red area) samples lie closer to the training set
    than training sequences lie to each other: an excess there is memorization,
    scored in the title (most improbable group, log10 probability under the
    null). Below the held-out curve (blue area) samples lie farther from the
    training set than the held-out sequences: the model generates sequences
    outside the family. With a random split the held-out curve follows the
    null and the two areas meet. A split by identity clustering keeps held-out
    sequences away from the training set, so the blue area starts late; when
    the held-out median exceeds the null median by more than a quarter, a note
    says so, and samples right of the null are already farther than a typical
    family member.

    Per-sample scores (``delta_p``, ``n_pleaks``) are in ``privet`` and in the
    sample table, not plotted.
    """
    axes = figure.subplots(1, 2)
    for ax in axes:
        _style_diagnostic_axis(ax)
    fit = privet["fit"]
    cdf = privet["cdf"]
    grid = np.asarray(cdf["distance"])
    observed = np.asarray(cdf["train_train"])
    fitted = np.asarray(cdf["fitted"])
    reach = np.flatnonzero(observed >= 0.999)
    right = min(1.0, 1.25 * grid[reach[0]]) if len(reach) else 1.0
    floor = 0.2 / max(privet["n_generated"], 1)
    top = 1.5
    box = {"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#CED4DA", "alpha": 0.9}

    def curve(ax, values, **style):
        ax.step(grid, np.clip(np.asarray(values), floor, None), where="post", **style)

    def finish(ax, title, legend_loc):
        ax.set_yscale("log")
        ax.set_ylim(floor, top)
        ax.set_xlim(0.0, right)
        ax.set_xlabel("distance to the nearest training sequence (fraction of sites)")
        ax.set_ylabel("cumulative fraction")
        ax.set_title(title, color=_DIAGNOSTIC_TEXT, fontsize=9)
        ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=7, loc=legend_loc)

    def median_text(values):
        median = _cdf_median(grid, values)
        return "" if median is None else f" (median {median:.2f})"

    ax = axes[0]
    low, high = fit["window"]
    ax.axvspan(low, high, color="#ADB5BD", alpha=0.25, lw=0, label="fitted window")
    curve(ax, observed, color="#31688E", lw=1.8, label="training → nearest other training (weighted)")
    ax.plot(grid, np.clip(fitted, floor, None), color="#6C757D", lw=1.5,
            label=f"{fit['family']} fit (shape {fit['shape']:.2f})")
    inside = (grid >= low) & (grid <= high) & (observed > 0) & (fitted > 0)
    lines = []
    if inside.any():
        ratio = fitted[inside] / observed[inside]
        worst = int(np.argmax(np.abs(np.log(ratio))))
        lines.append(f"largest fit/data ratio in the window: ×{ratio[worst]:.2g} "
                     f"at d = {grid[inside][worst]:.2f}")
    q1, q2 = privet.get("quantile_window", (None, None))
    if q1 is not None:
        lines.append(f"window: {q1:.0%} to {q2:.0%} quantiles of the data")
    lines.append("The line must follow the steps inside the band,\n"
                 "above all at its left edge: the p-values use it there.")
    ax.text(0.97, 0.2, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color=_DIAGNOSTIC_TEXT, bbox=box)
    finish(ax, "Null law of nearest-neighbour distances", "lower right")

    ax = axes[1]
    has_test = "test_train" in cdf
    ax.fill_between(grid, np.clip(fitted, floor, top), top, color=_PRIVET_OVER, alpha=0.13, lw=0,
                    label="overfitting: closer than training sequences are to each other")
    if has_test:
        ax.fill_between(grid, floor, np.clip(np.asarray(cdf["test_train"]), floor, top), step="post",
                        color=_PRIVET_UNDER, alpha=0.13, lw=0,
                        label="underfitting: farther than the held-out sequences")
    ax.plot(grid, np.clip(fitted, floor, None), color="#6C757D", lw=1.5,
            label="null (fit)" + median_text(fitted))
    if has_test:
        curve(ax, cdf["test_train"], color="#2A9D8F", lw=1.5,
              label="held-out → nearest training" + median_text(cdf["test_train"]))
    curve(ax, cdf["generated_train"], color="#E76F51", lw=2.0,
          label="generated → nearest training" + median_text(cdf["generated_train"]))
    ax.text(0.03, 0.97, "OVERFITTING", transform=ax.transAxes, ha="left", va="top", fontsize=9,
            weight="bold", color=_PRIVET_OVER, alpha=0.85)
    if has_test:
        ax.text(0.97, 0.45, "UNDERFITTING", transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                weight="bold", color=_PRIVET_UNDER, alpha=0.85)
        null_median, test_median = _cdf_median(grid, fitted), _cdf_median(grid, cdf["test_train"])
        if null_median and test_median and test_median > 1.25 * null_median:
            ax.text(0.97, 0.43,
                    f"Held-out sequences lie farther than the null\n"
                    f"(median {test_median:.2f} vs {null_median:.2f}), as after a split by\n"
                    f"identity: the blue area starts late. Samples right\n"
                    f"of the null are already farther than a typical\n"
                    f"family member.",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=_DIAGNOSTIC_TEXT, bbox=box)
    else:
        ax.text(0.97, 0.45, "Give a held-out alignment (--test)\nto mark the underfitting side",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=_DIAGNOSTIC_TEXT)
    excess = privet["excess_train"]
    title = ("No excess of samples near the training set" if excess["log10_p"] > -1 else
             f"Closest group of samples: {excess['count']} within {excess['distance']:.3f} "
             f"(null {excess['expected']:.1f}), log10 p {_log10_text(excess['log10_p'])}")
    control = privet.get("control")
    if control is not None and has_test:
        title += f"\nheld-out control: log10 p {_log10_text(control['excess_train']['log10_p'])}"
    finish(ax, title, "lower right")
    return figure


def plot_data_comparison(figure: Figure, comparison: dict, pca_reference=None, pca_generated=None) -> Figure:
    """Compare samples with the data: clusters of the data and energy distributions.

    Panels: the data on its first two principal components, coloured by
    cluster (k-means on the principal components of the PCA plots); the share
    of data and samples in each cluster; and the energy distributions of data
    and samples under the model. A cluster whose share differs is a region
    the model, or its sampling, over- or under-weights. Training sequences have
    lower energies than samples of a fitted model, so a shift of the energy
    distributions is expected; their shapes, and held-out sequences, are what
    to compare.
    """
    axes = figure.subplots(1, 3)
    for ax in axes:
        _style_diagnostic_axis(ax)
    clusters = comparison["clusters"]
    centres = np.asarray(clusters["centres"])
    data_fraction = np.asarray(clusters["data_fraction"])
    sample_fraction = np.asarray(clusters["sample_fraction"])
    k = len(data_fraction)
    palette = matplotlib.colormaps["tab10"](np.arange(k) % 10)

    ax = axes[0]
    if pca_reference is not None and len(centres):
        points = np.asarray(pca_reference)[:, :centres.shape[1]]
        labels = ((points[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
        ax.scatter(points[:, 0], points[:, 1], c=palette[labels], s=4, alpha=0.4, rasterized=True)
        for index, centre in enumerate(centres):
            ax.text(centre[0], centre[1], str(index + 1), ha="center", va="center", fontsize=9, weight="bold",
                    bbox={"boxstyle": "circle,pad=0.2", "facecolor": "white", "edgecolor": palette[index]})
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Clusters of the data (k-means on the principal components)", color=_DIAGNOSTIC_TEXT, fontsize=9)

    ax = axes[1]
    positions = np.arange(k)
    floor = 0.5 / max(comparison["n_samples"], 1)
    ax.bar(positions - 0.2, np.maximum(data_fraction, floor), width=0.4, color="#31688E", label="data")
    ax.bar(positions + 0.2, np.maximum(sample_fraction, floor), width=0.4, color="#E76F51", label="samples")
    for position, data, sample in zip(positions, data_fraction, sample_fraction):
        if data > 0:
            ax.text(position, max(data, sample, floor) * 1.15, f"×{sample / data:.2f}", ha="center", va="bottom",
                    fontsize=7, color=_DIAGNOSTIC_TEXT)
    ax.set_yscale("log")
    ax.set_xticks(positions, [str(index + 1) for index in positions])
    ax.set_xlabel("cluster")
    ax.set_ylabel("share of sequences")
    ax.set_title("Share of each cluster (labels: samples / data)", color=_DIAGNOSTIC_TEXT, fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    energy = comparison["energy"]
    edges = np.asarray(energy["bin_edges"])
    ax.stairs(energy["data_density"], edges, color="#31688E", lw=1.8, label=f"data (mean {energy['data_mean']:.1f})")
    ax.stairs(energy["sample_density"], edges, color="#E76F51", lw=1.8,
              label=f"samples (mean {energy['sample_mean']:.1f})")
    ax.set_xlabel("DCA energy")
    ax.set_ylabel("density")
    ax.set_title(f"Energies under the model (KS distance {energy['ks_distance']:.3f})\n"
                 "training data sit lower: the model is fitted to them", color=_DIAGNOSTIC_TEXT, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    figure.suptitle(f"Samples against the data ({comparison['n_samples']} samples, "
                    f"{comparison['n_reference']} weighted data sequences)", color=_DIAGNOSTIC_TEXT)
    return figure


def plot_energy_cde_scatter(
    ax: Axes,
    cde_sum: np.ndarray,
    energies: np.ndarray,
    fit: dict[str, float | int] | None,
) -> Axes:
    """Plot sequence energies against summed CDE and their fitted line."""
    entropy = np.asarray(cde_sum, dtype=np.float64).reshape(-1)
    energy = np.asarray(energies, dtype=np.float64).reshape(-1)
    if entropy.size == 0 or entropy.shape != energy.shape:
        raise ValueError("CDE sums and energies must be non-empty arrays with the same shape")

    _style_diagnostic_axis(ax)
    ax.scatter(
        entropy, energy, color=_DIAGNOSTIC_BLUE, s=24, alpha=0.5,
        edgecolors="white", linewidths=0.3, rasterized=True,
        label="Generated sequences", zorder=2,
    )
    lower, upper = float(entropy.min()), float(entropy.max())
    padding = 0.05 * (upper - lower) if upper > lower else 0.05
    limits = np.asarray([lower - padding, upper + padding])
    ax.set_xlim(limits)
    if fit is not None:
        slope = float(fit["lambda"])
        intercept = float(fit["intercept"])
        ax.plot(
            limits, intercept + slope * limits, color=_DIAGNOSTIC_CORAL,
            lw=2.0, label="Linear fit", zorder=3,
        )
        annotation = rf"$\lambda = {slope:.3g}$" + "\n" + rf"$R^2 = {float(fit['r_squared']):.3f}$"
    else:
        annotation = r"$\lambda$: unavailable" + "\n" + r"$R^2$: unavailable"
    ax.annotate(
        annotation, xy=(0.05, 0.95), xycoords="axes fraction",
        verticalalignment="top", horizontalalignment="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#C8CDD2", "alpha": 0.92},
    )
    ax.set_xlabel("Summed context-dependent entropy (nats)")
    ax.set_ylabel("DCA energy")
    ax.set_title("Energy and local entropy", color=_DIAGNOSTIC_TEXT, pad=10)
    ax.legend(frameon=False, loc="best")
    return ax


def plot_cij_scatter(
    ax: Axes,
    Cij_data: np.ndarray,
    Cij_gen: np.ndarray,
    pearson: float | None = None,
) -> Axes:
    """Plot reference versus generated connected two-site correlations.

    The box gives the Pearson correlation, the R² about the identity (which
    also penalizes a slope away from 1) and the slope.
    """
    Cij_data = np.asarray(Cij_data).reshape(-1)
    Cij_gen = np.asarray(Cij_gen).reshape(-1)
    if Cij_data.size == 0 or Cij_data.shape != Cij_gen.shape:
        raise ValueError("Cij_data and Cij_gen must be non-empty arrays with the same shape")
    if pearson is None:
        pearson = float(np.corrcoef(Cij_data, Cij_gen)[0, 1])

    _style_diagnostic_axis(ax)
    centered = Cij_data - Cij_data.mean()
    denominator = float(centered @ centered)
    if denominator > 0.0:
        slope = float(centered @ (Cij_gen - Cij_gen.mean()) / denominator)
        intercept = float(Cij_gen.mean() - slope * Cij_data.mean())
    else:
        slope = float("nan")
        intercept = float("nan")

    lower = min(float(Cij_data.min()), float(Cij_gen.min()))
    upper = max(float(Cij_data.max()), float(Cij_gen.max()))
    span = upper - lower
    padding = 0.04 * span if span > 0.0 else 0.05
    limits = (lower - padding, upper + padding)
    ax.scatter(
        Cij_data,
        Cij_gen,
        alpha=0.55,
        s=16,
        color=_DIAGNOSTIC_BLUE,
        edgecolors="white",
        linewidths=0.25,
        rasterized=True,
        label="Correlations",
        zorder=2,
    )
    ax.plot(limits, limits, ls="dashed", color="#7A7F85", lw=1.2, label="Identity", zorder=1)
    if np.isfinite(slope):
        fit_x = np.asarray(limits)
        ax.plot(
            fit_x,
            slope * fit_x + intercept,
            color=_DIAGNOSTIC_CORAL,
            lw=1.8,
            label="Linear fit",
            zorder=3,
        )
    ax.set_xlabel(r"$C_{ij}$ reference")
    ax.set_ylabel(r"$C_{ij}$ generated")
    ax.set_title("Final pairwise correlations", color=_DIAGNOSTIC_TEXT, pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    slope_text = f"{slope:.3f}" if np.isfinite(slope) else "n/a"
    ax.annotate(
        r"$\rho = $" + f"{pearson:.3f}\n" + r"$R^2 = $" + f"{_r2_about_identity(Cij_data, Cij_gen):.3f}\n"
        + r"slope $= $" + slope_text,
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        verticalalignment="top",
        horizontalalignment="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#C8CDD2", "alpha": 0.92},
    )
    ax.legend(frameon=False, loc="lower right")
    return ax

def _r2_about_identity(reference: np.ndarray, generated: np.ndarray) -> float:
    """1 - sum (generated - reference)^2 / sum (reference - mean)^2: 1 only on the diagonal."""
    reference = np.asarray(reference, dtype=float).reshape(-1)
    generated = np.asarray(generated, dtype=float).reshape(-1)
    spread = float(((reference - reference.mean()) ** 2).sum())
    return float(1 - ((generated - reference) ** 2).sum() / spread) if spread > 0 else float("nan")


def _scatter_against_reference(ax: Axes, reference, generated, *, color, size=16) -> tuple[float, float]:
    """Scatter generated against reference values with the identity and a least-squares line; return (pearson, slope)."""
    reference = np.asarray(reference, dtype=float).reshape(-1)
    generated = np.asarray(generated, dtype=float).reshape(-1)
    _style_diagnostic_axis(ax)
    centered = reference - reference.mean()
    denominator = float(centered @ centered)
    slope = float(centered @ (generated - generated.mean()) / denominator) if denominator > 0 else float("nan")
    intercept = float(generated.mean() - slope * reference.mean()) if denominator > 0 else float("nan")
    varied = len(reference) > 1 and reference.std() > 0 and generated.std() > 0
    pearson = float(np.corrcoef(reference, generated)[0, 1]) if varied else float("nan")
    lower, upper = min(reference.min(), generated.min()), max(reference.max(), generated.max())
    padding = 0.04 * (upper - lower) if upper > lower else 0.05
    limits = (lower - padding, upper + padding)
    ax.scatter(reference, generated, alpha=0.55, s=size, color=color, edgecolors="white", linewidths=0.25,
               rasterized=True, zorder=2)
    ax.plot(limits, limits, ls="dashed", color="#7A7F85", lw=1.2, label="Identity", zorder=1)
    if np.isfinite(slope):
        ax.plot(limits, slope * np.asarray(limits) + intercept, color=_DIAGNOSTIC_CORAL, lw=1.8,
                label="Linear fit", zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    return pearson, slope


def _stat_box(ax: Axes, lines: list[str]) -> None:
    ax.annotate("\n".join(lines), xy=(0.05, 0.95), xycoords="axes fraction", va="top", ha="left", fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#C8CDD2", "alpha": 0.92})


def plot_fi_scatter(ax: Axes, fi_data: np.ndarray, fi_gen: np.ndarray) -> Axes:
    """Plot reference versus generated single-site frequencies.

    Training fits them directly, so a departure points at the sampling
    (equilibration, temperature) more than at the model.
    """
    pearson, slope = _scatter_against_reference(ax, fi_data, fi_gen, color=_DIAGNOSTIC_TEAL, size=10)
    ax.set_xlabel(r"$f_i$ reference")
    ax.set_ylabel(r"$f_i$ generated")
    ax.set_title("Single-site frequencies", color=_DIAGNOSTIC_TEXT, pad=10)
    _stat_box(ax, [r"$\rho = $" + f"{pearson:.4f}", r"$R^2 = $" + f"{_r2_about_identity(fi_data, fi_gen):.4f}",
                   r"slope $= $" + f"{slope:.3f}"])
    return ax


def plot_three_point_scatter(ax: Axes, three_point: dict, *, min_triplets: int = 20) -> Axes:
    """Plot reference versus generated connected three-point correlations.

    Uses the triplets of :func:`adabmDCA.stats.compare_three_point_correlations`:
    no strong pair among the three states, non-gap states of intermediate
    frequency, and a signal-to-noise ratio of at least its ``z_threshold``.
    The box gives the Pearson correlation, the R² about the identity (1 only
    when the points lie on the diagonal: it also penalizes a slope away from 1)
    and the slope; the first two also between two random halves of the data,
    the agreement the data allow. With fewer than ``min_triplets`` resolved triplets the panel
    says the data cannot resolve three-point correlations.
    """
    _style_diagnostic_axis(ax)
    ax.set_title("Three-point correlations, no strong pair", color=_DIAGNOSTIC_TEXT, pad=10)
    shown = len(three_point.get("data", []))
    if shown < min_triplets:
        ax.text(0.5, 0.5,
                f"{three_point.get('n_resolved', 0)} of {three_point.get('n_candidates', 0)} candidate triplets\n"
                f"have a signal-to-noise ratio above {three_point.get('z_threshold', 5):g}:\n"
                "too few to compare three-point correlations.",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color=_DIAGNOSTIC_TEXT)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax
    _scatter_against_reference(ax, three_point["data"], three_point["samples"], color="#7B5EA7", size=12)
    ax.set_xlabel(r"$C_{ijk}$ reference")
    ax.set_ylabel(r"$C_{ijk}$ generated")
    ceiling = three_point.get("noise_ceiling")
    _stat_box(ax, [
        r"$\rho = $" + f"{three_point['pearson']:.3f}" + ("" if ceiling is None else f"  (data halves {ceiling:.3f})"),
        r"$R^2 = $" + f"{three_point['r2']:.3f}"
        + ("" if three_point.get("noise_ceiling_r2") is None else f"  (data halves {three_point['noise_ceiling_r2']:.3f})"),
        r"slope $= $" + f"{three_point['slope']:.3f}",
        f"{shown} shown of {three_point['n_resolved']} resolved",
        f"(|r| < {three_point['max_pair_correlation']:g} for each pair, z ≥ {three_point['z_threshold']:g})",
    ])
    return ax


def plot_statistics_scatter(
    figure: Figure,
    fi_data: np.ndarray,
    fi_gen: np.ndarray,
    Cij_data: np.ndarray,
    Cij_gen: np.ndarray,
    three_point: dict | None = None,
    pearson: float | None = None,
) -> Figure:
    """Single-site frequencies, connected two-site and three-point correlations, reference against samples.

    One row: :func:`plot_fi_scatter`, :func:`plot_cij_scatter` and, when
    ``three_point`` is given, :func:`plot_three_point_scatter`.
    """
    axes = figure.subplots(1, 3 if three_point is not None else 2)
    plot_fi_scatter(axes[0], fi_data, fi_gen)
    plot_cij_scatter(axes[1], Cij_data, Cij_gen, pearson=pearson)
    if three_point is not None:
        plot_three_point_scatter(axes[2], three_point)
    return figure


def plot_scatter_correlations(
    ax: tuple[Axes, Axes],
    Cij_data: np.ndarray,
    Cij_gen: np.ndarray,
    Cijk_data: np.ndarray,
    Cijk_gen: np.ndarray,
    pearson_Cij: float,
    pearson_Cijk: float,
) -> tuple[Axes, Axes]:
    """Plots the scatter plot of the data and generated Cij and Cijk values.
    
    Args:
        ax (Tuple[Axes, Axes]): Tuple of 2 Axes to plot the data.
        Cij_data (np.ndarray): Data Cij values.
        Cij_gen (np.ndarray): Generated Cij values.
        Cijk_data (np.ndarray): Data Cijk values.
        Cijk_gen (np.ndarray): Generated Cijk values.
        pearson_Cij (float): Pearson correlation coefficient of Cij.
        pearson_Cijk (float): Pearson correlation coefficient of Cijk.
        
    Returns:
        Tuple[Axes, Axes]: Updated axes.
    """
    if not isinstance(ax, tuple) or len(ax) != 2:
        raise ValueError("The 'ax' parameter must be a tuple of 2 Axes objects.")
    
    color_line = "#50424F"
    color_scatter = "#FF6275"

    plot_cij_scatter(ax[0], Cij_data, Cij_gen, pearson_Cij)

    x = np.linspace(Cijk_data.min(), Cijk_data.max(), 100)
    ax[1].scatter(Cijk_data, Cijk_gen, alpha=0.5, color=color_scatter)
    ax[1].plot(x, x, ls="dashed", color=color_line)
    ax[1].set_xlabel(r"$C_{ijk}$ data")
    ax[1].set_ylabel(r"$C_{ijk}$ generated")

    ax[1].annotate(
        r"$\rho=$" + f"{pearson_Cijk:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="left",
        bbox={"facecolor": "white", "alpha": 0.5},
    )
    
    return ax


def plot_contact_map(
    ax: Axes,
    cm: np.ndarray,
    title: str | None = None,
) -> Axes:
    """Plots the contact map.

    Args:
        ax (Axes): Axes to plot the contact map.
        cm (np.ndarray): Contact map to plot.
        title (Optional[str], optional): Title of the plot. Defaults to None.

    Returns:
        Axes: Updated axes.
    """
    sns.heatmap(cm, ax=ax, cmap="coolwarm", cbar_kws={'label': 'Frobenius norm'})
    ax.invert_yaxis()
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    if title is not None:
        ax.set_title(title)

    return ax
