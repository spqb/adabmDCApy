"""Low-rank Potts models: exact normalization and sampling, fitting, and their use as the bottom of a PTT ladder."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from adabmDCA import PTTConfig, PTTSampler, train_model
from adabmDCA.api import TrainingConfig
from adabmDCA.exceptions import InputValidationError
from adabmDCA.lowrank import (LowRankPotts, TargetMoments, _cell_statistics, _lattice_sum, _site_curvature,
                              _site_logits, fit_low_rank_potts, hopfield_potts_init, target_moments)
from adabmDCA.statmech import compute_energy


def random_low_rank(length, num_states, rank, *, scale=1.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    fields = 0.5 * torch.randn(length, num_states, generator=generator, dtype=torch.float64)
    weights = scale * torch.randn(length, num_states, rank, generator=generator, dtype=torch.float64)
    return LowRankPotts(fields, weights)


def enumerate_model(model):
    """All sequences, their log probabilities under the Potts parameters, and log Z."""
    length, num_states = model.shape
    sequences = torch.tensor(list(product(range(num_states), repeat=length)))
    one_hot = torch.nn.functional.one_hot(sequences, num_states).double()
    log_weights = -compute_energy(one_hot, model.potts_params(torch.float64))
    log_z = float(torch.logsumexp(log_weights, 0))
    return sequences, log_weights - log_z, log_z


def codes(sequences, num_states):
    length = sequences.shape[1]
    return (sequences.long() * num_states ** torch.arange(length - 1, -1, -1)).sum(1)


@pytest.mark.parametrize("rank, scale", [(1, 1.0), (2, 1.0), (2, 2.0), (3, 0.7)])
def test_log_z_and_its_bounds_match_enumeration(rank, scale):
    model = random_low_rank(4, 3, rank, scale=scale, seed=rank)
    _, _, log_z = enumerate_model(model)
    assert model.log_z == pytest.approx(log_z, abs=1e-4)
    lower, upper = model.log_z_bounds
    assert lower <= log_z <= upper


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_lattice_sum_gives_the_marginals_and_latent_correlations(rank):
    model = random_low_rank(4, 3, rank, scale=1.5, seed=20 + rank)
    sequences, log_probabilities, log_z = enumerate_model(model)
    one_hot = torch.nn.functional.one_hot(sequences, 3).double()
    probabilities = log_probabilities.exp()
    projections = model.weights[torch.arange(4), sequences].sum(1)  # m(x), the mean of z given x
    log_integral, start, _, marginals, correlations = _lattice_sum(model.fields, model.weights, gradient=True)
    assert log_integral - 0.5 * rank * np.log(2 * np.pi) == pytest.approx(log_z, abs=1e-7)
    np.testing.assert_allclose(marginals, torch.einsum("n,nia->ia", probabilities, one_hot), atol=1e-7)
    np.testing.assert_allclose(correlations, torch.einsum("n,nia,nk->iak", probabilities, one_hot, projections),
                               atol=1e-6)
    # Starting from the points of a previous call changes nothing.
    assert _lattice_sum(model.fields, model.weights, start)[0] == pytest.approx(log_integral, abs=1e-12)


def test_lattice_sum_finds_separated_modes():
    # Two modes 40 apart: the valley between them is far deeper than the depth of the lattice.
    weights = torch.zeros(40, 2, 1, dtype=torch.float64)
    weights[:, 0], weights[:, 1] = 0.5, -0.5
    model = LowRankPotts(torch.zeros(40, 2, dtype=torch.float64), weights)
    # m(x) = (40 - 2 n) / 2 for n sites in the second state: Z = sum_n C(40, n) exp(m^2 / 2).
    counts = torch.arange(41, dtype=torch.float64)
    log_binomial = torch.lgamma(torch.tensor(41.0)) - torch.lgamma(counts + 1) - torch.lgamma(41 - counts)
    log_z = float(torch.logsumexp(log_binomial + 0.5 * (20 - counts) ** 2, 0))
    assert model.log_z == pytest.approx(log_z, abs=1e-6)


def test_target_moments_match_the_weighted_one_hot_averages():
    generator = torch.Generator().manual_seed(4)
    states = torch.randint(0, 3, (50, 5), generator=generator)
    sequence_weights = torch.rand(50, generator=generator, dtype=torch.float64)
    frequencies, second = target_moments(states, sequence_weights, 3, 0.2)
    one_hot = torch.nn.functional.one_hot(states, 3).double().reshape(50, 15)
    normalized = sequence_weights / sequence_weights.sum()
    np.testing.assert_allclose(frequencies.reshape(-1), 0.8 * normalized @ one_hot + 0.2 / 3, atol=1e-12)
    expected = 0.8 * (one_hot * normalized[:, None]).T @ one_hot + 0.2 / 9
    blocks = second.view(5, 3, 5, 3)
    for i in range(5):
        for j in range(5):
            block = torch.diag(frequencies[i]) if i == j else expected.view(5, 3, 5, 3)[i, :, j, :]
            np.testing.assert_allclose(blocks[i, :, j, :], block, atol=1e-12)
    if torch.cuda.is_available():
        on_gpu = target_moments(states.cuda(), sequence_weights.cuda(), 3, 0.2)
        np.testing.assert_allclose(on_gpu[1].cpu(), second, atol=1e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_products_from_the_sequences_match_the_pair_matrix(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no GPU")
    generator = torch.Generator().manual_seed(6)
    states = torch.randint(0, 4, (300, 7), generator=generator).to(device)
    sequence_weights = torch.rand(300, generator=generator, dtype=torch.float64).to(device)
    dense = TargetMoments(states, sequence_weights, 4, 0.3, pair_matrix=True)
    sparse = TargetMoments(states, sequence_weights, 4, 0.3, pair_matrix=False)
    assert sparse.matrix is None
    np.testing.assert_allclose(sparse.frequencies.cpu(), dense.frequencies.cpu(), atol=1e-14)
    block = torch.randn(28, 5, generator=generator, dtype=torch.float64).to(device)
    np.testing.assert_allclose(sparse.product(block).cpu(), (dense.matrix @ block).cpu(), atol=1e-13)


def test_initial_patterns_are_the_top_modes_of_the_correlation_matrix():
    generator = torch.Generator().manual_seed(7)
    model = random_low_rank(12, 3, 2, scale=0.6, seed=8)
    torch.manual_seed(7)
    states = model.sample(5000)
    moments = TargetMoments(states, torch.ones(5000, dtype=torch.float64), 3, 0.01, pair_matrix=True)
    _, weights = hopfield_potts_init(moments, 2)
    # Reference: the full eigendecomposition in the gauge without the last state.
    reduced = (torch.arange(36) % 3) != 2
    mean = moments.frequencies.reshape(-1)[reduced]
    covariance = moments.matrix[reduced][:, reduced] - mean[:, None] * mean[None, :]
    values, vectors = torch.linalg.eigh(covariance / mean.sqrt()[:, None] / mean.sqrt()[None, :])
    expected = (1 - 1 / values[-2:].flip(0)).sqrt() * vectors[:, -2:].flip(1) / mean.sqrt()[:, None]
    found = weights.reshape(36, 2)[reduced]
    assert not weights[:, 2].any()
    for k in range(2):  # eigenvectors are defined up to their sign
        sign = torch.sign((found[:, k] * expected[:, k]).sum())
        np.testing.assert_allclose(found[:, k], sign * expected[:, k], rtol=1e-4, atol=1e-6)


def test_fit_without_the_pair_matrix_matches_the_fit_with_it():
    torch.manual_seed(9)
    data = random_low_rank(10, 3, 1, scale=0.8, seed=9).sample(3000)
    best = {}
    for pair_matrix in (True, False):
        values = []
        # Converged far beyond the default stopping rule, so that both fits reach the optimum.
        fit_low_rank_potts(data, None, num_states=3, rank=1, pseudocount=0.01, pair_matrix=pair_matrix,
                           tolerance=1e-9, callback=lambda info: values.append(info["log_likelihood"]))
        best[pair_matrix] = max(values)
    assert best[False] == pytest.approx(best[True], abs=1e-7)


def test_potts_parameters_are_symmetric_with_empty_site_blocks():
    model = random_low_rank(5, 3, 2)
    params = model.potts_params(torch.float32)
    couplings = params["coupling_matrix"]
    assert torch.equal(couplings, couplings.permute(2, 3, 0, 1))
    assert not couplings[torch.arange(5), :, torch.arange(5), :].any()


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_samples_follow_the_exact_distribution(rank):
    torch.manual_seed(1)
    model = random_low_rank(4, 3, rank, scale=1.2, seed=10 + rank)
    sequences, log_probabilities, _ = enumerate_model(model)
    samples = model.sample(200_000)
    assert samples.dtype == torch.int32 and samples.shape == (200_000, 4)
    empirical = torch.bincount(codes(samples, 3), minlength=81).double() / len(samples)
    expected = log_probabilities.exp()[torch.argsort(codes(sequences, 3))]
    np.testing.assert_allclose(empirical.numpy(), expected.numpy(), atol=4e-3)


@pytest.mark.parametrize("rank, scale", [(1, 1.5), (2, 1.0), (3, 0.8)])
def test_cell_bounds_enclose_the_latent_density_and_its_curvature(rank, scale):
    torch.manual_seed(5)
    model = random_low_rank(30, 4, rank, scale=scale, seed=30 + rank)
    cells = model.cells
    # Random points, and the vertices, of every cell.
    index = torch.arange(len(cells.centers)).repeat(64)
    offsets = (2 * torch.rand(len(index), rank, dtype=torch.float64) - 1)
    offsets[: len(cells.centers)] = torch.sign(offsets[: len(cells.centers)])
    offsets = offsets * cells.half_widths[index]
    values = model.latent_log_density(cells.centers[index] + offsets)
    tangent = cells.center_values[index] + (cells.slopes[index] * offsets).sum(-1)
    upper = torch.where(cells.tilted[index], tangent + cells.envelope[index], cells.envelope[index])
    assert (values <= upper).all()
    # The Hessian of G at those points lies between the matrices of its cell.
    _, _, above, below = _cell_statistics(model.fields, model.weights, cells.centers[index], cells.half_widths[index],
                                          _site_curvature(model.weights))
    probabilities = _site_logits(model.fields, model.weights, cells.centers[index] + offsets).softmax(-1)
    means = torch.einsum("nia,iak->nik", probabilities, model.weights)
    hessians = (torch.einsum("nia,iak,ial->nkl", probabilities, model.weights, model.weights)
                - torch.einsum("nik,nil->nkl", means, means))
    assert torch.linalg.eigvalsh(above - hessians).min() > -1e-9
    assert torch.linalg.eigvalsh(hessians - below).min() > -1e-9
    assert model.acceptance > 0.8


def test_log_likelihood_matches_the_normalized_potts_model():
    model = random_low_rank(4, 3, 2, seed=3)
    sequences, log_probabilities, _ = enumerate_model(model)
    assert model.log_likelihood(sequences[:7]) == pytest.approx(float(log_probabilities[:7].mean()), abs=1e-4)


def test_fit_approaches_the_generating_model():
    torch.manual_seed(2)
    # A ferromagnetic model: the first two states of every site pull the latent coordinate apart.
    weights = torch.zeros(8, 3, 1, dtype=torch.float64)
    weights[:, 0], weights[:, 1] = 0.6, -0.6
    truth = LowRankPotts(0.3 * torch.randn(8, 3, dtype=torch.float64), weights)
    data = truth.sample(20_000)
    fitted = fit_low_rank_potts(data, None, num_states=3, rank=1, l2=0.0)
    profile_frequencies = torch.nn.functional.one_hot(data.long(), 3).double().mean(0)
    profile = float(profile_frequencies.log().gather(1, data.long().T).sum(0).mean())
    assert truth.log_likelihood(data) > profile + 0.2
    assert fitted.log_likelihood(data) == pytest.approx(truth.log_likelihood(data), abs=0.005)
    # The couplings are recovered up to the gauge: compare connected couplings.
    def connected(model):
        couplings = model.potts_params(torch.float64)["coupling_matrix"]
        return couplings - couplings.mean(1, keepdim=True) - couplings.mean(3, keepdim=True) \
            + couplings.mean((1, 3), keepdim=True)
    assert (connected(fitted) - connected(truth)).abs().max() < 0.1


def test_ladder_starts_at_the_anchor_and_survives_an_archive(tmp_path):
    torch.manual_seed(3)
    anchor = random_low_rank(4, 2, 1, scale=0.8, seed=5)
    params = anchor.potts_params(torch.float64)
    sampler = PTTSampler(params, tokens="AB", n_chains=64, seed=1, anchor=anchor)
    assert sampler.anchor_log_z == pytest.approx(anchor.log_z)
    sampler.advance(rounds=3, local_sweeps=1, return_samples=False)
    # Identical models: the bridge adds nothing to the anchor's log Z.
    assert sampler.partition_estimate().log_z == pytest.approx(anchor.log_z, abs=1e-9)
    path = sampler.save_archive(tmp_path / "ptt.h5")
    loaded = PTTSampler.from_archive(path, device="cpu", mode="generate")
    assert loaded.anchor is not None and loaded.anchor_log_z == pytest.approx(anchor.log_z)
    assert torch.equal(loaded.anchor.weights, anchor.weights)
    loaded.prepare_sampling_ladder(32)
    loaded.advance(rounds=2, local_sweeps=1, return_samples=False)
    with pytest.raises(InputValidationError, match="low-rank anchor"):
        PTTSampler(random_low_rank(4, 2, 1, seed=6).potts_params(torch.float64), tokens="AB", n_chains=8,
                   anchor=anchor)


def test_rank_setting_is_validated():
    assert PTTConfig(anchor_rank=2).anchor_rank == 2
    for rank in (-1, 4):
        with pytest.raises(InputValidationError):
            PTTConfig(anchor_rank=rank)


def test_training_starts_from_a_fitted_anchor(tmp_path):
    path = tmp_path / "data.fasta"
    sequences = ["AAAA", "AAAB", "AABA", "BBBB", "BBBA", "BBAB", "ABAB"]
    path.write_text("".join(f">s{i}\n{seq}\n" for i, seq in enumerate(sequences)))
    config = TrainingConfig(
        alphabet="AB", device="cpu", dtype="float64", n_chains=60, n_sweeps=1, max_epochs=3,
        target_pearson=0.999999, no_reweighting=True, pseudocount=0.1,
        ptt=PTTConfig(equilibration_rounds=2, initialization_rounds=2, mixing_thermalization_rounds=2,
                      mixing_chains=20, anchor_rank=1),
        checkpoint_interval=1,
    )
    events = []
    result = train_model(path, config=config, output_dir=tmp_path / "out", stage_progress=events.append)
    fit = [event for event in events if event.stage == "anchor_fit"]
    assert [event.kind for event in fit[:2]] == ["start", "progress"] and fit[-1].kind == "event"
    summary = fit[-1].details
    assert summary["rank"] == 1 and summary["cells"] > 0 and 0 < summary["acceptance"] <= 1
    assert summary["log_likelihood"] > summary["profile_log_likelihood"]
    loaded = PTTSampler.from_archive(result.artifacts["ptt_archive"], device="cpu", mode="inspect")
    assert loaded.anchor is not None and loaded.anchor.rank == 1
    assert torch.count_nonzero(loaded.profile_params["coupling_matrix"])
    with pytest.raises(InputValidationError, match="bmDCA"):
        train_model(path, config=replace(config, model_type="eaDCA"), output_dir=tmp_path / "ea")
