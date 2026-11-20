"""compare_pca: natural sequences drawn with their weights, generated ones projected on their components."""

import numpy as np
import pytest

from adabmDCA import Alignment, compare_pca
from adabmDCA.exceptions import InputValidationError


def _write(path, sequences):
    path.write_text("".join(f">s{i}\n{sequence}\n" for i, sequence in enumerate(sequences)))
    return path


def test_compare_pca_projects_samples_on_the_data_components(tmp_path):
    rng = np.random.default_rng(0)
    # Two clusters of RNA sequences, one a single repeated sequence.
    first = ["ACGUACGUAC"] * 30
    second = ["UGCAUGCAUG", "UGCAUGCAUC", "UGCAUGCAAG"]
    data = _write(tmp_path / "data.fasta", first + second)
    samples = _write(tmp_path / "samples.fasta", ["".join(rng.choice(list("ACGU"), 10)) for _ in range(20)])
    comparison = compare_pca(data, samples, n_reference=400, seed=1)
    assert comparison.tokens == "-ACGU"
    assert comparison.generated_scores.shape == (20, 4) and comparison.reference_scores.shape[1] == 4
    assert comparison.explained_variance_ratio[0] > 0.5
    assert comparison.reference_scores.shape[0] == 33  # at most the retained sequences are drawn
    # The natural sequences enter with their weights: without weight, the second cluster is never drawn.
    only_first = compare_pca(data, samples, data_weights=[1] * 30 + [0] * 3, seed=1).reference_scores
    assert np.ptp(only_first) == pytest.approx(0, abs=1e-5)
    paths = comparison.save_plots(tmp_path / "plots", label="run")
    assert sorted(path.name for path in paths.values()) == ["run_pca_1_2.png", "run_pca_3_4.png"]
    assert all(path.stat().st_size > 0 for path in paths.values())


def test_compare_pca_rejects_mismatched_inputs(tmp_path):
    data = _write(tmp_path / "data.fasta", ["ACGU", "ACGA", "UGCA"])
    with pytest.raises(InputValidationError, match="same length"):
        compare_pca(data, _write(tmp_path / "short.fasta", ["ACG"]))
    with pytest.raises(InputValidationError, match="data_weights"):
        compare_pca(data, data, data_weights=[1.0, 1.0])
    assert compare_pca(data, Alignment(["s"], ["ACGU"]), n_components=2).generated_scores.shape == (1, 2)
