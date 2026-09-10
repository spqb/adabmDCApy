import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import FillBetweenPolyCollection

from adabmDCA.plot import plot_autocorrelation, plot_cij_scatter


def test_autocorrelation_renders_both_standard_deviation_bands():
    figure, axis = plt.subplots()
    checkpoints = np.asarray([1, 2, 3])

    plot_autocorrelation(
        axis,
        checkpoints,
        np.asarray([0.9, 0.7, 0.5]),
        np.asarray([0.3, 0.35, 0.4]),
        autocorr_std=np.asarray([0.05, 0.04, 0.03]),
        independent_std=np.asarray([0.02, 0.03, 0.02]),
    )

    bands = [collection for collection in axis.collections if isinstance(collection, FillBetweenPolyCollection)]
    assert len(bands) == 2
    assert all(collection.get_alpha() >= 0.2 for collection in bands)
    plt.close(figure)


def test_cij_scatter_shows_linear_fit_and_slope():
    figure, axis = plt.subplots()
    reference = np.asarray([-0.2, -0.1, 0.1, 0.2])
    generated = 1.5 * reference + 0.05

    plot_cij_scatter(axis, reference, generated)

    assert {line.get_label() for line in axis.lines} >= {"Identity", "Linear fit"}
    annotation = "\n".join(text.get_text() for text in axis.texts)
    assert "slope" in annotation
    assert "1.500" in annotation
    assert axis.get_title() == "Final pairwise correlations"
    assert axis.get_xlim() == pytest.approx(axis.get_ylim())
    plt.close(figure)
