from typing import Literal

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
    bins = np.histogram_bin_edges(finite_values, bins="auto") if finite_values.size else 10
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
    ax.set_ylim(0.0, 1.0)
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


def plot_cij_scatter(
    ax: Axes,
    Cij_data: np.ndarray,
    Cij_gen: np.ndarray,
    pearson: float | None = None,
) -> Axes:
    """Plot reference versus generated connected two-site correlations."""
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
        r"$\rho = $" + f"{pearson:.3f}\n" + r"slope $= $" + slope_text,
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        verticalalignment="top",
        horizontalalignment="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#C8CDD2", "alpha": 0.92},
    )
    ax.legend(frameon=False, loc="lower right")
    return ax

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
