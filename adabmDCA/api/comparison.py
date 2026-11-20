"""Compare generated sequences with an alignment on the principal components of the alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from adabmDCA._validation import validate_integer, validate_seed
from adabmDCA.exceptions import InputValidationError
from adabmDCA.input_loading import AlignmentInput, load_alignment

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Colours of the natural and generated sequences in every diagnostic plot.
PCA_COLORS = ("#31688E", "#E76F51")


def pca_figure(reference_scores, generated_scores, explained_variance_ratio=None, *, components=(0, 1),
               labels=("Natural", "Generated"), title=None, figure=None, dpi=192) -> Figure:
    """Scatter of two principal components with their marginal histograms, natural and generated superimposed."""
    import matplotlib.pyplot as plt

    from adabmDCA.plot import plot_PCA

    pc1, pc2 = components
    figure = plt.figure(dpi=dpi, figsize=(7, 6.5)) if figure is None else figure
    return plot_PCA(
        figure, reference_scores, pc1=pc1, pc2=pc2, data2=generated_scores, labels=list(labels),
        colors=list(PCA_COLORS),
        title=title or f"{labels[0]} and {labels[1].lower()} sequences: PC{pc1 + 1} vs PC{pc2 + 1}",
        explained_variance_ratio=explained_variance_ratio,
    )


@dataclass(frozen=True)
class PCAComparison:
    """Natural and generated sequences on the principal components of the natural ones.

    Attributes:
        reference_scores: Natural sequences, drawn with their weights, projected on the components; ``(N, k)``.
        generated_scores: Generated sequences projected on the same components; ``(M, k)``.
        explained_variance_ratio: Share of the variance of the natural sequences carried by each component.
        tokens: Alphabet of both alignments.
    """

    reference_scores: np.ndarray = field(repr=False)
    generated_scores: np.ndarray = field(repr=False)
    explained_variance_ratio: np.ndarray
    tokens: str

    def plot(self, *, components: tuple[int, int] = (0, 1), labels: tuple[str, str] = ("Natural", "Generated"),
             title: str | None = None, figure: Figure | None = None) -> Figure:
        """Scatter of two components (zero-based) with marginal histograms; returns the figure."""
        return pca_figure(self.reference_scores, self.generated_scores, self.explained_variance_ratio,
                          components=components, labels=labels, title=title, figure=figure)

    def save_plots(self, directory: str | Path, *, label: str | None = None,
                   labels: tuple[str, str] = ("Natural", "Generated")) -> dict[str, Path]:
        """Write ``pca_1_2.png`` and, with four components, ``pca_3_4.png``; returns the paths."""
        import matplotlib.pyplot as plt

        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f"{label}_" if label else ""
        pairs = [(0, 1)] + ([(2, 3)] if self.reference_scores.shape[1] >= 4 else [])
        paths = {}
        for pc1, pc2 in pairs:
            path = folder / f"{prefix}pca_{pc1 + 1}_{pc2 + 1}.png"
            figure = self.plot(components=(pc1, pc2), labels=labels)
            figure.savefig(path, dpi=figure.dpi, facecolor="white", bbox_inches="tight")
            plt.close(figure)
            paths[f"pca_{pc1 + 1}_{pc2 + 1}_plot"] = path
        return paths


def compare_pca(
    data: AlignmentInput,
    samples: AlignmentInput,
    *,
    alphabet: str | None = None,
    data_weights: np.ndarray | torch.Tensor | None = None,
    clustering_seqid: float = 0.8,
    no_reweighting: bool = False,
    n_reference: int = 10_000,
    n_components: int = 4,
    seed: int = 0,
) -> PCAComparison:
    """Project an alignment and generated sequences on the principal components of the alignment.

    The natural sequences are first drawn with replacement according to their
    weights, as the model sees them, so that redundant clusters do not dominate
    the components or the histograms; ``n_reference`` of them are drawn (at most
    the size of the alignment). The components are fitted on these draws alone
    and the generated sequences are projected on them. Only the leading
    ``n_components`` are computed, by randomized low-rank PCA
    (:func:`torch.pca_lowrank`) on the one-hot sequences.

    Args:
        data: Natural alignment: path, :class:`Alignment` or sequences.
        samples: Generated (or any other) sequences of the same length.
        alphabet: Alphabet of both; detected from ``data`` by default.
        data_weights: Weights of the retained natural sequences; computed by default.
        clustering_seqid: Identity above which natural sequences share weight.
        no_reweighting: Give every natural sequence the same weight.
        n_reference: Natural sequences drawn for the comparison.
        n_components: Leading components to compute.
        seed: Seed of the draw of the natural sequences.

    Returns:
        A :class:`PCAComparison`; ``.plot()`` and ``.save_plots(folder)`` draw it.

    Example:
        >>> from adabmDCA import compare_pca
        >>> comparison = compare_pca("train.fasta", "samples/samples.fasta")
        >>> comparison.save_plots("comparison")
    """
    from adabmDCA.api.sampling import _compute_pca_scores
    from adabmDCA.fasta import compute_weights

    validate_integer("n_reference", n_reference)
    validate_integer("n_components", n_components)
    validate_seed(seed)
    natural = load_alignment(data, alphabet=alphabet)
    generated = load_alignment(samples, alphabet=natural.tokens)
    natural_states = torch.as_tensor(natural.encoded_sequences, dtype=torch.long)
    generated_states = torch.as_tensor(generated.encoded_sequences, dtype=torch.long)
    if natural_states.shape[1] != generated_states.shape[1]:
        raise InputValidationError(
            "The natural and generated sequences must have the same length.",
            details={"natural": natural_states.shape[1], "generated": generated_states.shape[1]},
        )
    if data_weights is not None:
        weights = torch.as_tensor(np.asarray(data_weights), dtype=torch.float64).reshape(-1)
        if len(weights) != len(natural_states) or not torch.all(weights >= 0) or weights.sum() <= 0:
            raise InputValidationError("data_weights must hold one non-negative weight per retained sequence.")
    elif no_reweighting:
        weights = torch.ones(len(natural_states), dtype=torch.float64)
    else:
        weights = compute_weights(natural_states, th=clustering_seqid, dtype=torch.float64).reshape(-1)
    generator = torch.Generator().manual_seed(seed)
    drawn = torch.multinomial(weights, min(n_reference, len(natural_states)), replacement=True, generator=generator)
    num_states = len(natural.tokens)
    one_hot = torch.nn.functional.one_hot
    reference_scores, generated_scores, explained = _compute_pca_scores(
        one_hot(natural_states[drawn], num_states).float(), one_hot(generated_states, num_states).float(),
        n_components=n_components,
    )
    return PCAComparison(reference_scores.cpu().numpy(), generated_scores.cpu().numpy(), explained.cpu().numpy(),
                         natural.tokens)
