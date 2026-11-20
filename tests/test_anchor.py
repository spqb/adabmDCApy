"""The anchor, a tree + low-rank Potts model: exact normalization and sampling, fitting, and their use at the bottom of a PTT ladder."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from adabmDCA import PTTConfig, PTTSampler, train_model
from adabmDCA.api import TrainingConfig
from adabmDCA.training.config import CLI_DEFAULT_ANCHOR
from adabmDCA.exceptions import InputValidationError
from adabmDCA.anchor.lowrank import LowRankPotts, TargetMoments
from adabmDCA.statmech import compute_energy
from adabmDCA.anchor.anchor import TreeLowRankPotts, _lattice_sum, chow_liu_tree, fit_tree_low_rank_potts

# A forest with two components and edges stored in both orientations.
EDGES = torch.tensor([(0, 1), (2, 1), (1, 3), (4, 5)])


def random_model(rank, *, length=6, num_states=3, scale=0.9, seed=0, edges=EDGES):
    generator = torch.Generator().manual_seed(seed)
    fields = 0.5 * torch.randn(length, num_states, generator=generator, dtype=torch.float64)
    weights = scale * torch.randn(length, num_states, rank, generator=generator, dtype=torch.float64)
    pairs = torch.randn(len(edges), num_states, num_states, generator=generator, dtype=torch.float64)
    return TreeLowRankPotts(fields, weights, edges, pairs)


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


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_log_z_and_its_bounds_match_enumeration(rank):
    model = random_model(rank, seed=rank)
    _, _, log_z = enumerate_model(model)
    assert model.log_z == pytest.approx(log_z, abs=1e-6)
    lower, upper = model.log_z_bounds
    assert lower - 1e-9 <= log_z <= upper + 1e-9


def test_potts_parameters_are_symmetric_with_empty_site_blocks():
    params = random_model(2).potts_params(torch.float32)
    couplings = params["coupling_matrix"]
    assert torch.equal(couplings, couplings.permute(2, 3, 0, 1))
    assert not couplings[torch.arange(6), :, torch.arange(6), :].any()


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_samples_follow_the_exact_distribution(rank):
    torch.manual_seed(1)
    model = random_model(rank, seed=10 + rank)
    sequences, log_probabilities, _ = enumerate_model(model)
    samples = model.sample(300_000)
    assert samples.dtype == torch.int32 and samples.shape == (300_000, 6)
    empirical = torch.bincount(codes(samples, 3), minlength=729).double() / len(samples)
    expected = log_probabilities.exp()[torch.argsort(codes(sequences, 3))]
    # Sequences frequent enough for Gaussian counting noise; the rare ones are checked together.
    frequent = expected > 1e-4
    noise = (expected * (1 - expected) / len(samples)).sqrt()
    assert ((empirical - expected).abs() <= 5 * noise)[frequent].all()
    rare = float(expected[~frequent].sum())
    assert float(empirical[~frequent].sum()) == pytest.approx(rare, abs=5 * (rare / len(samples)) ** 0.5 + 1e-6)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_cell_envelopes_lie_above_the_latent_density(rank):
    torch.manual_seed(2)
    model = random_model(rank, length=20, num_states=4, scale=0.6, seed=20 + rank,
                         edges=torch.tensor([(i, i + 1) for i in range(19) if i % 5 != 4]))
    cells = model.cells
    index = torch.arange(len(cells.centers)).repeat(32)
    offsets = 2 * torch.rand(len(index), rank, dtype=torch.float64) - 1
    offsets[: len(cells.centers)] = torch.sign(offsets[: len(cells.centers)])  # the vertices
    offsets = offsets * cells.half_widths[index]
    values = model.latent_log_density(cells.centers[index] + offsets)
    tangent = cells.center_values[index] + (cells.slopes[index] * offsets).sum(-1) - 0.5 * (offsets ** 2).sum(-1)
    assert (values <= torch.where(cells.tilted[index], tangent + cells.envelope[index], cells.envelope[index])).all()
    # The Gaussian tangent is a lower bound: G is convex.
    assert (values >= tangent - 1e-9)[cells.tilted[index]].all()
    assert model.acceptance > 0.7


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_lattice_gradient_gives_the_marginals_of_the_model(rank):
    model = random_model(rank, seed=30 + rank)
    sequences, log_probabilities, _ = enumerate_model(model)
    probabilities = log_probabilities.exp()
    one_hot = torch.nn.functional.one_hot(sequences, 3).double()
    projections = model.weights[torch.arange(6), sequences].sum(1)  # m(x), the mean of z given x
    _, _, _, site, latent, edge = _lattice_sum(model._tree, model.fields, model.weights, model.pairs, gradient=True)
    np.testing.assert_allclose(site, torch.einsum("n,nia->ia", probabilities, one_hot), atol=1e-6)
    np.testing.assert_allclose(latent, torch.einsum("n,nia,nk->iak", probabilities, one_hot, projections), atol=1e-5)
    expected = torch.einsum("n,nea,neb->eab", probabilities, one_hot[:, model.edges[:, 0]], one_hot[:, model.edges[:, 1]])
    np.testing.assert_allclose(edge, expected, atol=1e-6)


def test_chow_liu_tree_keeps_the_strongest_pairs_and_reproduces_them():
    torch.manual_seed(3)
    data = torch.randint(0, 3, (4000, 5))
    data[:, 1] = data[:, 0]                                       # sites 0 and 1 identical
    data[:3000, 4] = data[:3000, 3]                               # sites 3 and 4 mostly identical
    moments = TargetMoments(data, torch.ones(4000, dtype=torch.float64), 3, 0.01, pair_matrix=True)
    edges, pairs = chow_liu_tree(moments.frequencies, moments.matrix.view(5, 3, 5, 3))
    assert len(edges) == 4
    found = {tuple(sorted(edge)) for edge in edges.tolist()}
    assert {(0, 1), (3, 4)} <= found
    tree = fit_tree_low_rank_potts(data, None, num_states=3, rank=0, pseudocount=0.01)
    assert tree.rank == 0 and tree.cells is None and tree.log_z == pytest.approx(0.0, abs=1e-9)
    samples = tree.sample(200_000).long()
    joint = torch.zeros(3, 3, dtype=torch.float64)
    joint.index_put_((samples[:, 3], samples[:, 4]), torch.ones(len(samples), dtype=torch.float64), accumulate=True)
    np.testing.assert_allclose(joint / len(samples), moments.matrix.view(5, 3, 5, 3)[3, :, 4, :], atol=5e-3)


def test_fit_approaches_the_generating_model():
    torch.manual_seed(4)
    # A collective mode that no tree holds (the first two states of every site pull the latent coordinate
    # apart), on top of pair terms on a few edges.
    edges = torch.tensor([(0, 1), (2, 3), (4, 5), (6, 7)])
    weights = torch.zeros(8, 3, 1, dtype=torch.float64)
    weights[:, 0], weights[:, 1] = 0.6, -0.6
    generator = torch.Generator().manual_seed(40)
    truth = TreeLowRankPotts(0.3 * torch.randn(8, 3, generator=generator, dtype=torch.float64), weights, edges,
                             0.8 * torch.randn(4, 3, 3, generator=generator, dtype=torch.float64))
    data = truth.sample(20_000)
    fitted = fit_tree_low_rank_potts(data, None, num_states=3, rank=1, pseudocount=1e-4, l2=0.0, tolerance=1e-6)
    tree = fit_tree_low_rank_potts(data, None, num_states=3, rank=0, pseudocount=1e-4)
    assert fitted.log_likelihood(data) == pytest.approx(truth.log_likelihood(data), abs=0.01)
    assert fitted.log_likelihood(data) > tree.log_likelihood(data) + 0.1
    assert fitted.weights.mean(1).abs().max() < 1e-12


@pytest.mark.parametrize("rank", [0, 1])
def test_ladder_starts_at_the_anchor_and_survives_an_archive(tmp_path, rank):
    torch.manual_seed(5)
    anchor = random_model(rank, length=4, num_states=2, scale=0.8, seed=50, edges=torch.tensor([(0, 1), (3, 2)]))
    sampler = PTTSampler(anchor.potts_params(torch.float64), tokens="AB", n_chains=64, seed=1, anchor=anchor)
    assert sampler.anchor_log_z == pytest.approx(anchor.log_z)
    sampler.advance(rounds=3, local_sweeps=1, return_samples=False)
    assert sampler.partition_estimate().log_z == pytest.approx(anchor.log_z, abs=1e-9)
    path = sampler.save_archive(tmp_path / "ptt.h5")
    loaded = PTTSampler.from_archive(path, device="cpu", mode="generate")
    assert isinstance(loaded.anchor, TreeLowRankPotts) and loaded.anchor.rank == rank
    assert torch.equal(loaded.anchor.pairs, anchor.pairs) and torch.equal(loaded.anchor.edges, anchor.edges)
    assert loaded.anchor_log_z == pytest.approx(anchor.log_z)
    loaded.prepare_sampling_ladder(32)
    loaded.advance(rounds=2, local_sweeps=1, return_samples=False)
    other = random_model(rank, length=4, num_states=2, seed=51, edges=torch.tensor([(0, 1), (3, 2)]))
    with pytest.raises(InputValidationError, match="anchor"):
        PTTSampler(other.potts_params(torch.float64), tokens="AB", n_chains=8, anchor=anchor)


def test_anchor_setting_is_validated_and_parsed():
    from adabmDCA.training.cli import anchor_options, create_parser

    # Python starts from the profile, the command line from the tree + low-rank anchor.
    config = TrainingConfig()
    assert config.anchor == "profile" and config.anchor_rank == 2
    assert TrainingConfig(ptt=PTTConfig()).anchor == "profile"
    assert TrainingConfig(anchor=CLI_DEFAULT_ANCHOR).anchor == "tree-low-rank"
    with pytest.raises(InputValidationError):
        TrainingConfig(anchor="spline")
    for rank in (0, 4):
        with pytest.raises(InputValidationError):
            TrainingConfig(anchor_rank=rank)
    # Left at its default the anchor falls back to the profile for the sparse models; asked for, it is an error.
    assert TrainingConfig(model_type="eaDCA", anchor=CLI_DEFAULT_ANCHOR).anchor == "profile"
    assert TrainingConfig(model_type="edDCA", anchor="profile").anchor == "profile"
    with pytest.raises(InputValidationError, match="bmDCA"):
        TrainingConfig(model_type="eaDCA", anchor="tree")
    base = ["-d", "x.fasta", "-o", "out"]
    assert anchor_options(create_parser().parse_args(base)) == {"anchor": "tree-low-rank"}
    assert anchor_options(create_parser().parse_args([*base, "--anchor", "low-rank", "--anchor-rank", "3"])) == {
        "anchor": "low-rank", "anchor_rank": 3}
    for removed in (["--ptt-anchor"], ["--anchor", "spline"]):
        with pytest.raises(SystemExit):
            create_parser().parse_args([*base, *removed])


def _alignment(tmp_path):
    path = tmp_path / "data.fasta"
    sequences = ["AAAA", "AAAB", "AABA", "BBBB", "BBBA", "BBAB", "ABAB"]
    path.write_text("".join(f">s{i}\n{seq}\n" for i, seq in enumerate(sequences)))
    return path


@pytest.mark.parametrize(("kind", "rank", "model_class"), [("tree-low-rank", 1, TreeLowRankPotts),
                                                           ("low-rank", 1, LowRankPotts),
                                                           ("tree", 0, TreeLowRankPotts)])
def test_ptt_training_starts_from_the_anchor(tmp_path, kind, rank, model_class):
    path = _alignment(tmp_path)
    config = TrainingConfig(
        alphabet="AB", device="cpu", dtype="float64", n_chains=60, n_sweeps=1, max_epochs=3,
        target_pearson=0.999999, no_reweighting=True, pseudocount=0.1, anchor=kind, anchor_rank=max(rank, 1),
        ptt=PTTConfig(equilibration_rounds=2, initialization_rounds=2, mixing_thermalization_rounds=2,
                      mixing_chains=20),
        checkpoint_interval=1,
    )
    events = []
    result = train_model(path, config=config, output_dir=tmp_path / "out", stage_progress=events.append)
    fit = [event for event in events if event.stage == "anchor_fit"]
    assert fit[0].kind == "start" and fit[-1].kind == "event"
    summary = fit[-1].details
    assert summary["anchor"] == kind and summary["rank"] == rank and 0 < summary["acceptance"] <= 1
    assert (summary["cells"] > 0) == (rank > 0)
    assert summary["log_likelihood"] > summary["profile_log_likelihood"]
    loaded = PTTSampler.from_archive(result.artifacts["ptt_archive"], device="cpu", mode="inspect")
    assert type(loaded.anchor) is model_class and loaded.anchor.rank == rank
    assert torch.count_nonzero(loaded.profile_params["coupling_matrix"])


def test_pcd_training_starts_from_the_anchor(tmp_path):
    path = _alignment(tmp_path)
    # A negligible learning rate keeps the model at the anchor: couplings on the tree only.
    config = TrainingConfig(alphabet="AB", device="cpu", dtype="float64", n_chains=50, n_sweeps=1, max_epochs=2,
                            target_pearson=0.999999, no_reweighting=True, pseudocount=0.1, anchor="tree",
                            learning_rate=1e-12)
    events = []
    result = train_model(path, config=config, stage_progress=events.append)
    assert [event.kind for event in events if event.stage == "anchor_fit"] == ["start", "event"]
    couplings = result.model.params["coupling_matrix"].abs().amax(dim=(1, 3))
    tree = couplings > 1e-6
    assert torch.equal(tree, tree.T) and int(tree.sum()) == 2 * 3
    assert result.chains.shape == (50, 4, 2)


def test_the_default_anchor_falls_back_to_the_profile(tmp_path):
    path = _alignment(tmp_path)
    config = TrainingConfig(alphabet="AB", device="cpu", dtype="float64", n_chains=20, n_sweeps=1, max_epochs=1,
                            target_pearson=0.999999, no_reweighting=True, pseudocount=0.1, anchor="profile")
    first = train_model(path, config=config, output_dir=tmp_path / "first")
    events = []
    warm = train_model(path, config=replace(config, anchor=CLI_DEFAULT_ANCHOR), stage_progress=events.append,
                       initial_params_path=first.artifacts["params"])
    assert warm.config.anchor == "profile" and not events
    with pytest.raises(InputValidationError, match="warm start"):
        train_model(path, config=replace(config, anchor="tree"), initial_params_path=first.artifacts["params"])


def test_archives_with_the_former_anchor_options_load(tmp_path):
    import json

    import h5py

    anchor = random_model(1, length=4, num_states=2, scale=0.8, seed=50, edges=torch.tensor([(0, 1), (3, 2)]))
    sampler = PTTSampler(anchor.potts_params(torch.float64), tokens="AB", n_chains=16, seed=1, anchor=anchor)
    path = sampler.save_archive(tmp_path / "ptt.h5")
    with h5py.File(path, "r+") as archive:
        meta = json.loads(archive.attrs["metadata"])
        meta["config"].update({"anchor": True, "anchor_rank": 1, "anchor_tree": True})
        archive.attrs["metadata"] = json.dumps(meta)
    loaded = PTTSampler.from_archive(path, device="cpu", mode="inspect")
    assert isinstance(loaded.anchor, TreeLowRankPotts) and loaded.anchor.rank == 1
    assert not hasattr(loaded.config, "anchor")
