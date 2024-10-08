
from typing import List
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def plot_scatter_labels(ax, data1, data2, dim1, dim2, labels):    
    ax.scatter(data1[:, dim1], data1[:, dim2], color='black', s=50, label=labels[0], zorder=0, alpha=0.3)
    ax.scatter(data2[:, dim1], data2[:, dim2], color='red', label=labels[1], s=20, zorder=2, edgecolor='black', marker='o', alpha=1, linewidth=0.4)
    
def plot_hist(ax, data1, data2, color, dim, labels, orientation='vertical'):    
    ax.hist(data1[:, dim], bins=40, color='black', histtype='step', label=labels[0], zorder=0, density=True, orientation=orientation, lw=1)
    ax.hist(data2[:, dim], bins=40, color=color, histtype='step', label=labels[1], zorder=1, density=True, orientation=orientation, lw=1.5)
    ax.axis('off')
    
def plot_PCA(
    fig : plt.figure,
    data1 : np.array,
    data2 : np.array,
    dim1 : int,
    dim2 : int,
    labels : List[str],
    title : str
):
    gs = GridSpec(4, 4)

    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_hist_x = fig.add_subplot(gs[0, 0:3])
    ax_hist_y = fig.add_subplot(gs[1:4, 3])
        
    plot_scatter_labels(ax_scatter, data1, data2, dim1, dim2, labels=labels)
    plot_hist(ax_hist_x, data1, data2, 'red', dim1, labels=labels)
    plot_hist(ax_hist_y, data1, data2, 'red', dim2, orientation='horizontal', labels=labels)
    
    ax_scatter.set_xlabel(f"PC {dim1 + 1}")
    ax_scatter.set_ylabel(f"PC {dim2 + 1}")
    
    fig.suptitle(title)

    ax_hist_x.legend(fontsize=12, bbox_to_anchor=(1, 1));
    h, l = ax_scatter.get_legend_handles_labels()
    return fig
    
    
def plot_pearson_sampling(
    ax : plt.Axes,
    checkpoints : np.array,
    pearsons : np.array,
    pearson_training : np.array = None
):
    
    if pearson_training:
        ax.axhline(y=pearson_training, ls="dashed", color="red", label="Training", lw=1, zorder=0)
        annotation_text = f"Training: {pearson_training:.3f}\nSampling: {pearsons[-1]:.3f}"
    else:
        annotation_text = annotation_text = f"Sampling: {pearsons[-1]:.3f}"
    ax.plot(checkpoints, pearsons, "-o", label="Resampling", lw=0.5, color="royalblue", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Sampling time [sweeps]")
    ax.set_ylabel(r"Pearson $C_{ij}(a,b)$")
    ax.annotate(annotation_text, xy=(0.95, 0.53), xycoords='axes fraction',
                verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    ax.legend()
    return ax


def plot_autocorrelation(
    ax : plt.Axes,
    checkpoints : np.array,
    autocorr : np.array,
    gen_seqid : np.array,
    data_seqid : np.array
):
    ax.plot(checkpoints, autocorr, "-o", c="royalblue", lw=0.5, label="Time-autocorrelation", zorder=2)
    ax.axhline(y=gen_seqid, color="navy", lw=1, ls="dashed", label="Generated seqID", zorder=1)
    ax.axhline(y=data_seqid, color="red", lw=1, ls="dashed", label="Data seqID", zorder=0)
    ax.set_xlabel(r"$\tau$ [sweeps]")
    ax.set_ylabel(r"$\langle \mathrm{SeqID}(t, t-\tau) \rangle$")
    ax.legend();

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
    ax : plt.Axes,
    Cij_data : np.array,
    Cij_gen : np.array,
    Cijk_data : np.array,
    Cijk_gen : np.array,
    pearson_Cij : float,
    pearson_Cijk : float
) -> plt.Axes:
    
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