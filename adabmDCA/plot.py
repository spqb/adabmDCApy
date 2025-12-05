
from typing import List, Tuple, Optional, Union, Literal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns

def _plot_scatter_labels(
    ax: Axes,
    data1: np.ndarray,
    pc1: int = 0,
    pc2: int = 1,
    data2: Optional[np.ndarray] = None,
    labels: Union[List[str], str, None] = "Data",
    colors: Union[List[str], str] = "black",
) -> Axes:

    if isinstance(labels, str):
        labels = [labels]
    if isinstance(colors, str):
        colors = [colors]
    if labels is None:
        labels = ["Data", "Generated"]
    ax.scatter(data1[:, pc1], data1[:, pc2], color=colors[0], s=50, label=labels[0], zorder=0, alpha=0.3)
    if data2 is not None:
        if len(labels) == 1:
            labels.append("Generated")
        if len(colors) == 1:
            colors.append("red")
        ax.scatter(data2[:, pc1], data2[:, pc2], color=colors[1], label=labels[1], s=20, zorder=2, edgecolor='black', marker='o', alpha=1, linewidth=0.4)
    
    return ax
    
def _plot_hist(
    ax: Axes,
    data1: np.ndarray,
    pc: int,
    data2: Optional[np.ndarray] = None,
    colors: Union[List[str], str] = "black",
    labels: Union[List[str], str, None] = "Data",
    orientation: Literal['vertical', 'horizontal'] = 'vertical',
) -> Axes:
    if isinstance(labels, str):
        labels = [labels]
    if labels is None:
        labels = ["Data", "Generated"]
    if isinstance(colors, str):
        colors = [colors]
    ax.hist(data1[:, pc], bins=40, color=colors[0], histtype='step', label=labels[0], zorder=0, density=True, orientation=orientation, lw=1)
    if data2 is not None:
        if len(labels) == 1:
            labels.append("Generated")
        if len(colors) == 1:
            colors.append("red")
        ax.hist(data2[:, pc], bins=40, color=colors[1], histtype='step', label=labels[1], zorder=1, density=True, orientation=orientation, lw=1.5)
    ax.axis('off')
    
    return ax


def plot_PCA(
    fig: Figure,
    data1: np.ndarray,
    pc1: int = 0,
    pc2: int = 1,
    data2: Optional[np.ndarray] = None,
    labels: Union[List[str], str] = "Data",
    colors: Union[List[str], str] = "black",
    title: Optional[str] = None,
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
    
    gs = GridSpec(4, 4)
    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_hist_x = fig.add_subplot(gs[0, 0:3])
    ax_hist_y = fig.add_subplot(gs[1:4, 3])
        
    ax_scatter = _plot_scatter_labels(
        ax=ax_scatter,
        data1=data1,
        pc1=pc1,
        pc2=pc2,
        data2=data2,
        labels=None,
        colors=colors,
    )
    ax_hist_x = _plot_hist(
        ax=ax_hist_x,
        data1=data1,
        pc=pc1,
        data2=data2,
        colors=colors,
        labels=labels,
        orientation='vertical',
    )
    ax_hist_y = _plot_hist(
        ax=ax_hist_y,
        data1=data1,
        pc=pc2,
        data2=data2,
        colors=colors,
        labels=None,
        orientation='horizontal',
    )
    
    ax_scatter.set_xlabel(f"PC {pc1 + 1}")
    ax_scatter.set_ylabel(f"PC {pc2 + 1}")
    if title is not None:
        fig.suptitle(title)
    fig.legend(fontsize=12, bbox_to_anchor=(1, 1))
    
    return fig
    
    
def plot_pearson_sampling(
    ax: Axes,
    checkpoints: np.ndarray,
    pearsons: np.ndarray,
    pearson_training: Optional[float] = None
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
    
    if pearson_training is not None:
        ax.axhline(y=pearson_training, ls="dashed", color="red", label="Training", lw=1, zorder=0)
        annotation_text = f"Training: {pearson_training:.3f}\nSampling: {pearsons[-1]:.3f}"
    else:
        annotation_text = f"Sampling: {pearsons[-1]:.3f}"
    ax.plot(checkpoints, pearsons, "-o", label="Resampling", lw=0.5, color="royalblue", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Sampling time [sweeps]")
    ax.set_ylabel(r"Pearson $C_{ij}(a,b)$")
    ax.annotate(annotation_text, xy=(0.95, 0.53), xycoords='axes fraction',
                verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    ax.legend()
    return ax


def plot_autocorrelation(
    ax: Axes,
    checkpoints: np.ndarray,
    autocorr: np.ndarray,
    gen_seqid: float,
    data_seqid: float
) -> Axes:
    """Plots the time-autocorrelation curve of the sequence identity and the generated and data sequence identities.
    
    Args:
        ax (Axes): Axes to plot the data.
        checkpoints (np.ndarray): Checkpoints of the sampling.
        autocorr (np.ndarray): Time-autocorrelation of the sequence identity.
        gen_seqid (float): Sequence identity of the generated data.
        data_seqid (float): Sequence identity of the data.

    Returns:
        Axes: Updated axes.
    """
    ax.plot(checkpoints, autocorr, "-o", c="royalblue", lw=0.5, label="Time-autocorrelation", zorder=2)
    ax.axhline(y=gen_seqid, color="navy", lw=1, ls="dashed", label="Generated seqID", zorder=1)
    ax.axhline(y=data_seqid, color="red", lw=1, ls="dashed", label="Data seqID", zorder=0)
    ax.set_xlabel(r"$\tau$ [sweeps]")
    ax.set_ylabel(r"$\langle \mathrm{SeqID}(t, t-\tau) \rangle$")
    ax.legend()

    if np.any(checkpoints[autocorr <= gen_seqid]):
        mixing_time = checkpoints[autocorr <= gen_seqid][0]
        print(f"Mixing time: {mixing_time} sweeps")
        annotation_text = r"$\tau_{\mathrm{mix}}$" + f": {mixing_time} sweeps"
        plt.annotate(annotation_text, xy=(0.95, 0.65), xycoords='axes fraction',
                verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    else:
        print(f"The mixing time could not be computed within {checkpoints[-1]} sweeps")
    
    return ax

def plot_scatter_correlations(
    ax: Tuple[Axes, Axes],
    Cij_data: np.ndarray,
    Cij_gen: np.ndarray,
    Cijk_data: np.ndarray,
    Cijk_gen: np.ndarray,
    pearson_Cij: float,
    pearson_Cijk: float,
) -> Tuple[Axes, Axes]:
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

    x = np.linspace(Cij_data.min(), Cij_data.max(), 100)
    ax[0].scatter(Cij_data, Cij_gen, alpha=0.5, color=color_scatter)
    ax[0].plot(x, x, ls="dashed", color=color_line)
    ax[0].set_xlabel(r"$C_{ij}$ data")
    ax[0].set_ylabel(r"$C_{ij}$ generated")

    x = np.linspace(Cijk_data.min(), Cijk_data.max(), 100)
    ax[1].scatter(Cijk_data, Cijk_gen, alpha=0.5, color=color_scatter)
    ax[1].plot(x, x, ls="dashed", color=color_line)
    ax[1].set_xlabel(r"$C_{ijk}$ data")
    ax[1].set_ylabel(r"$C_{ijk}$ generated")

    ax[0].annotate(r"$\rho=$" + f"{pearson_Cij:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                    verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

    ax[1].annotate(r"$\rho=$" + f"{pearson_Cijk:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                    verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    
    return ax


def plot_contact_map(
    ax: Axes,
    cm: np.ndarray,
    title: Optional[str] = None,
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