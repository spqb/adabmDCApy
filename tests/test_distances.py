from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adabmDCA import DCAModel, InputValidationError, sample_sequences
from adabmDCA.api.sampling import _distance_comparison


def _hamming(a, b):
    return (a[:, None, :] != b[None, :, :]).float().mean(-1)


def _nearest_median(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][min(np.searchsorted(cumulative, 0.5), len(values) - 1)])


def test_distances_match_brute_force_with_weights_and_self_exclusion():
    generator = torch.Generator().manual_seed(0)
    natural = torch.randint(4, (30, 10), generator=generator)
    held_out = torch.randint(4, (12, 10), generator=generator)
    generated = torch.randint(4, (25, 10), generator=generator)
    weights = torch.rand(30, generator=generator) + 0.1
    dataset = SimpleNamespace(data=natural, weights=weights)
    test = SimpleNamespace(data=held_out, weights=torch.ones(12))
    samples = torch.nn.functional.one_hot(generated, 4).float()

    result = _distance_comparison(dataset, samples, test=test)

    edges = np.asarray(result["bin_edges"])
    width = edges[1] - edges[0]
    w = (weights / weights.sum()).double()
    within = _hamming(natural, natural)
    pair_weight = w[:, None] * w[None, :]
    pair_weight.fill_diagonal_(0)
    assert len(edges) == 12  # one bin per mismatch count 0..10 (L = 10 < 50 bins)
    bins = (within * 10).round().long()
    expected = np.bincount(bins.numpy().ravel(), weights=pair_weight.numpy().ravel(), minlength=11)
    np.testing.assert_allclose(result["all_pairs"]["natural"], expected / (expected.sum() * width), atol=1e-9)

    nearest_natural = (within + torch.eye(30) * 10).min(1).values.numpy()
    assert result["summary"]["natural_to_natural"]["median"] == pytest.approx(
        _nearest_median(nearest_natural, w.numpy()))
    nearest_generated = _hamming(generated, natural).min(1).values.numpy()
    assert result["summary"]["generated_to_natural"]["median"] == pytest.approx(
        _nearest_median(nearest_generated, np.full(25, 1 / 25)))
    nearest_test = _hamming(held_out, natural).min(1).values.numpy()
    assert result["summary"]["test_to_natural"]["median"] == pytest.approx(
        _nearest_median(nearest_test, np.full(12, 1 / 12)))
    assert result["n_test"] == 12
    assert set(result["nearest"]) == {"natural_to_natural", "generated_to_natural", "generated_to_generated",
                                      "test_to_natural", "generated_to_test"}
    for curve in result["nearest"].values():
        assert np.sum(curve) * width == pytest.approx(1.0)


def test_copies_of_training_sequences_are_reported_as_identical():
    natural = torch.randint(3, (20, 8), generator=torch.Generator().manual_seed(1))
    dataset = SimpleNamespace(data=natural, weights=torch.ones(20))
    samples = torch.nn.functional.one_hot(natural[:10], 3).float()

    result = _distance_comparison(dataset, samples)

    assert result["summary"]["generated_to_natural"]["identical"] == pytest.approx(1.0)
    assert result["summary"]["generated_to_natural"]["median"] == 0.0
    assert "test_to_natural" not in result["nearest"]


def test_sampling_with_held_out_alignment_saves_distance_plot(tmp_path):
    reference = tmp_path / "train.fasta"
    reference.write_text("".join(f">t{i}\n{s}\n" for i, s in enumerate(["AAB", "ABA", "BAA", "ABB", "BBA"])))
    held_out = tmp_path / "test.fasta"
    held_out.write_text(">v1\nBBB\n>v2\nAAA\n")
    model = DCAModel({"bias": torch.zeros(3, 3), "coupling_matrix": torch.zeros(3, 3, 3, 3)}, alphabet="AB-")

    result = sample_sequences(model=model, n_sequences=16, n_sweeps=4, reference_fasta=reference,
                              test_fasta=held_out, collect_diagnostics=True, device="cpu")

    assert result.distance_comparison["n_natural"] == 5
    assert result.distance_comparison["n_test"] == 2
    assert "distance_comparison" in result.to_dict()["data"]
    artifacts = result.save_bundle(tmp_path / "out")
    artifacts.update(result.save_diagnostic_plots(tmp_path / "out"))
    assert artifacts["distances_plot"].is_file()
    assert artifacts["distances_log"].is_file()


def test_held_out_alignment_requires_diagnostics(tmp_path):
    reference = tmp_path / "train.fasta"
    reference.write_text(">a\nAB\n>b\nBA\n")
    model = DCAModel({"bias": torch.zeros(2, 3), "coupling_matrix": torch.zeros(2, 3, 2, 3)}, alphabet="AB-")
    with pytest.raises(InputValidationError, match="collect_diagnostics"):
        sample_sequences(model=model, n_sequences=2, n_sweeps=2, reference_fasta=reference, test_fasta=reference)
