"""End-of-sampling plotting diagnostics must work without MPS CPU fallback."""

import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adabmDCA import sample_sequences, train_model
from adabmDCA.api.sampling import _compare_with_data, _compute_pca_scores, _distance_comparison
from tests.test_ptt import training_case

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Requires Apple MPS GPU')


@pytest.mark.parametrize('n_reference', [1, 3, 20])
def test_pca_runs_on_cpu_and_matches_cpu_reference(n_reference, monkeypatch):
    generator = torch.Generator().manual_seed(14)
    reference = torch.nn.functional.one_hot(torch.randint(3, (n_reference, 5), generator=generator), 3).float()
    generated = torch.nn.functional.one_hot(torch.randint(3, (11, 5), generator=generator), 3).float()
    original = torch.pca_lowrank

    def cpu_only(matrix, **kwargs):
        assert matrix.device.type == 'cpu'
        return original(matrix, **kwargs)

    monkeypatch.setattr(torch, 'pca_lowrank', cpu_only)
    torch.manual_seed(13)
    expected = _compute_pca_scores(reference, generated)
    torch.manual_seed(13)
    actual = _compute_pca_scores(reference.to('mps'), generated.to('mps'))
    for a, b in zip(actual, expected):
        assert a.device.type == 'cpu'
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_cluster_and_energy_comparison_accepts_mps_scores():
    generator = torch.Generator().manual_seed(19)
    reference = torch.nn.functional.one_hot(torch.randint(3, (20, 5), generator=generator), 3).float()
    generated = torch.nn.functional.one_hot(torch.randint(3, (17, 5), generator=generator), 3).float()
    scores = (torch.randn(20, 4, generator=generator), torch.randn(17, 4, generator=generator))
    bias = torch.randn(5, 3, generator=generator)
    params = {'bias': bias, 'coupling_matrix': torch.zeros(5, 3, 5, 3)}
    expected = _compare_with_data(reference, generated, *scores, params, seed=13)
    actual = _compare_with_data(reference.to('mps'), generated.to('mps'),
                                *(x.to('mps') for x in scores), {k: v.to('mps') for k, v in params.items()}, seed=13)
    assert actual['clusters'] == expected['clusters']
    for key in ('data_mean', 'sample_mean', 'ks_distance'):
        assert actual['energy'][key] == pytest.approx(expected['energy'][key], abs=1e-6)


@pytest.mark.parametrize('n_measure', [7, 100])
def test_weighted_distances_and_held_out_alignment_match_cpu(n_measure):
    generator = torch.Generator().manual_seed(19)
    data = torch.randint(3, (20, 5), generator=generator)
    held_out = torch.randint(3, (11, 5), generator=generator)
    weights = torch.rand(20, generator=generator) + .1
    samples = torch.nn.functional.one_hot(torch.randint(3, (17, 5), generator=generator), 3).float()
    natural = SimpleNamespace(data=data, weights=weights)
    test = SimpleNamespace(data=held_out, weights=torch.ones(11))
    expected = _distance_comparison(natural, samples, test=test, n_measure=n_measure, seed=23)
    natural_mps = SimpleNamespace(data=data.to('mps'), weights=weights.to('mps'))
    test_mps = SimpleNamespace(data=held_out.to('mps'), weights=test.weights.to('mps'))
    actual = _distance_comparison(natural_mps, samples.to('mps'), test=test_mps, n_measure=n_measure, seed=23)
    assert actual == expected


def test_ptt_archive_generation_saves_all_diagnostic_plots(tmp_path):
    path, config = training_case.__wrapped__(tmp_path)
    trained = train_model(path, config=replace(config, device='mps', dtype='float32', max_epochs=1),
                          output_dir=tmp_path / 'trained')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = sample_sequences(model=trained.artifacts['ptt_archive'], ptt=True, device='mps',
                                  n_sequences=32, ptt_local_sweeps=1, reference_fasta=path, test_fasta=path,
                                  n_measure=16, ptt_max_rounds=100, no_reweighting=True,
                                  collect_diagnostics=True, seed=11)
    assert not any('fall back' in str(w.message) or 'MPS backend' in str(w.message) for w in caught)
    assert len(result.sequences) == 32
    assert result.pca_reference.shape == (min(16, result.data_comparison['n_reference']), 4)
    assert result.pca_generated.shape == (32, 4)
    assert np.isfinite(result.pca_generated).all()
    assert result.distance_comparison['n_test'] > 0
    result.save_bundle(tmp_path / 'sampled')
    artifacts = result.save_diagnostic_plots(tmp_path / 'sampled')
    for key in ('pca_1_2_plot', 'pca_3_4_plot', 'distances_plot', 'data_vs_samples_plot', 'cij_scatter_plot'):
        assert artifacts[key].is_file()
