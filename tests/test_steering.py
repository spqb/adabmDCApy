"""Steered (importance) sampling, checked against exact enumeration of a small model."""

from itertools import product

import numpy as np
import pytest
import torch

from adabmDCA import DCAModel, PTTSampler, sample_sequences
from adabmDCA.api.exceptions import InputValidationError
from adabmDCA.statmech import compute_energy
from adabmDCA.steering import Steering, importance_summary

L, Q, TOKENS = 3, 2, "AB"


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def coupled():
    params = {"bias": torch.tensor([[0.3, -0.2], [-0.5, 0.4], [0.1, 0.2]], dtype=torch.float64),
              "coupling_matrix": torch.zeros(L, Q, L, Q, dtype=torch.float64)}
    block = torch.tensor([[0.9, -0.4], [-0.4, 0.6]], dtype=torch.float64)
    for i, j in ((0, 1), (1, 2)):
        params["coupling_matrix"][i, :, j, :] = block
        params["coupling_matrix"][j, :, i, :] = block.T
    return params


SEQUENCES = ["".join(s) for s in product(TOKENS, repeat=L)]


def score(sequence):
    """Number of B tokens, plus a bonus when both ends are B."""
    return sequence.count("B") + 1.5 * (sequence[0] == "B" and sequence[2] == "B")


def onehot_potential(x, strength):
    return strength * (x[:, :, 1].sum(1) + 1.5 * x[:, 0, 1] * x[:, 2, 1])


def string_potential(sequences, strength):
    return [strength * score(s) for s in sequences]


def exact(strength):
    states = torch.nn.functional.one_hot(torch.tensor(list(product(range(Q), repeat=L))), Q).double()
    log_p = -compute_energy(states, coupled()) - torch.tensor([strength * score(s) for s in SEQUENCES],
                                                              dtype=torch.float64)
    return (log_p - torch.logsumexp(log_p, 0)).exp().numpy(), float(torch.logsumexp(log_p, 0))


def total_variation(sequences, probabilities):
    counts = np.array([sequences.count(s) for s in SEQUENCES], dtype=float) / len(sequences)
    return 0.5 * np.abs(counts - probabilities).sum()


@pytest.mark.parametrize("steering_input, potential", [("onehot", onehot_potential),
                                                       ("sequences", string_potential)])
def test_ordinary_steered_sampling_matches_exact_distribution(steering_input, potential):
    strength = 1.5
    result = sample_sequences(model=DCAModel(coupled(), alphabet=TOKENS), n_sequences=20_000, n_sweeps=40,
                              sampler="gibbs", device="cpu", steering_potential=potential,
                              steering_strength=strength, steering_input=steering_input)
    target, _ = exact(strength)
    assert total_variation(list(result.sequences), target) < 0.02
    np.testing.assert_allclose(result.steering_potentials,
                               [strength * score(s) for s in result.sequences], atol=1e-12)
    assert result.steering["weights"] == "self_normalized"
    assert np.exp(result.log_importance_weights).mean() == pytest.approx(1.0)
    frame = result.to_dataframe()
    assert {"steering_potential", "log_importance_weight"} <= set(frame.columns)


def test_importance_weights_recover_unsteered_averages():
    # Weak steering keeps the two distributions overlapping, so the weights are well behaved.
    result = sample_sequences(model=DCAModel(coupled(), alphabet=TOKENS), n_sequences=20_000, n_sweeps=40,
                              sampler="gibbs", device="cpu", steering_potential=string_potential,
                              steering_strength=0.3)
    assert result.steering["effective_sample_size"] > 10_000
    unsteered, _ = exact(0.0)
    observable = np.array([score(s) for s in result.sequences])
    reweighted = (np.exp(result.log_importance_weights) * observable).mean()
    assert reweighted == pytest.approx(float(unsteered @ np.array([score(s) for s in SEQUENCES])), abs=0.03)


def test_ptt_steered_sampling_adds_rungs_and_estimates_log_z_ratio(tmp_path):
    strength = 3.0
    reference = tmp_path / "reference.fasta"
    reference.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(SEQUENCES)))
    profile = {"bias": coupled()["bias"].clone(), "coupling_matrix": torch.zeros(L, Q, L, Q, dtype=torch.float64)}
    backend = PTTSampler(profile, tokens=TOKENS, n_chains=1000, sampler="gibbs", seed=1)
    assert backend.transition_target(coupled(), local_sweeps=2)[0]
    result = sample_sequences(model=backend, n_sequences=4_000, reference_fasta=reference,
                              ptt_local_sweeps=2, ptt_max_rounds=3000, device="cpu",
                              steering_potential=onehot_potential, steering_strength=strength,
                              steering_input="onehot", ptt_steering_acceptance=0.5)
    target, log_z_steered = exact(strength)
    _, log_z = exact(0.0)
    strengths = result.steering["strengths"]
    assert len(strengths) >= 2 and strengths == sorted(strengths) and strengths[-1] == strength
    assert total_variation(list(result.sequences), target) < 0.04
    assert result.steering["log_z_ratio"] == pytest.approx(log_z_steered - log_z, abs=0.05)
    assert result.steering["weights"] == "bridge_normalized"
    flags = [row["flag"] for row in result.ptt_diagnostics["models"]]
    assert flags.count("steered") == len(strengths)
    # Weights are normalized by the bridge estimate of the partition-function ratio.
    np.testing.assert_allclose(result.log_importance_weights,
                               result.steering_potentials + result.steering["log_z_ratio"])


def test_steered_ladder_cannot_be_saved(tmp_path):
    backend = PTTSampler({"bias": coupled()["bias"].clone(),
                          "coupling_matrix": torch.zeros(L, Q, L, Q, dtype=torch.float64)},
                         tokens=TOKENS, n_chains=50, sampler="gibbs")
    backend.mode = "generate"
    backend.prepare_sampling_ladder(50)
    backend.prepare_steering(Steering(onehot_potential, tokens=TOKENS, steering_input="onehot"), 1.0,
                             trial_rounds=2)
    with pytest.raises(InputValidationError, match="cannot be saved"):
        backend.save_archive(tmp_path / "steered.h5")


@pytest.mark.parametrize("potential, message", [
    (lambda seqs, s: [1.0] * len(seqs), "must be 0"),
    (lambda seqs, s: [s] * (len(seqs) + 1), "expected"),
    (lambda seqs, s: [float("nan") if s else 0.0] * len(seqs), "non-finite"),
    (lambda seqs, s: "not numbers", "one number per sequence"),
])
def test_invalid_steering_potentials_are_rejected(potential, message):
    with pytest.raises(InputValidationError, match=message):
        sample_sequences(model=DCAModel(coupled(), alphabet=TOKENS), n_sequences=10, n_sweeps=2,
                         device="cpu", steering_potential=potential)


def test_steering_arguments_are_validated():
    model = DCAModel(coupled(), alphabet=TOKENS)
    with pytest.raises(InputValidationError, match="non-zero"):
        sample_sequences(model=model, n_sequences=5, n_sweeps=1, device="cpu",
                         steering_potential=string_potential, steering_strength=0.0)
    with pytest.raises(InputValidationError, match="steering_input"):
        sample_sequences(model=model, n_sequences=5, n_sweeps=1, device="cpu",
                         steering_potential=string_potential, steering_input="tokens")


def test_sequences_are_decoded_in_alignment_order():
    steering = Steering(string_potential, tokens="-AB", steering_input="sequences")
    states = torch.tensor([[0, 1, 2], [2, 2, 1]])
    assert steering.decode(states) == ["-AB", "BBA"]


def test_importance_summary_normalization():
    log_w, ess = importance_summary(np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(log_w, 0.0, atol=1e-12)
    assert ess == pytest.approx(3.0)
    log_w, ess = importance_summary(np.array([1.0, 2.0]), log_z_ratio=-0.5)
    np.testing.assert_allclose(log_w, [0.5, 1.5])
    assert 1.0 < ess < 2.0


@pytest.mark.parametrize("proposal_steps", [1, 2, 6, 15])
def test_fixed_proposal_blocks_are_exact_for_short_and_long_blocks(proposal_steps):
    # Blocks longer than a sweep share their sites across chains; only a palindromic
    # site order keeps the whole population exact (a fixed random order biases it).
    from adabmDCA.steering import SteeredKernel

    torch.manual_seed(proposal_steps)
    kernel = SteeredKernel(Steering(onehot_potential, tokens=TOKENS, steering_input="onehot"), 1.5,
                           length=L, device=torch.device("cpu"), proposal_steps=proposal_steps)
    chains = torch.nn.functional.one_hot(torch.randint(0, Q, (40_000, L)), Q).double()
    for _ in range(40):
        chains = kernel(chains, coupled(), nsweeps=5)
    sequences = ["".join(TOKENS[i] for i in row) for row in chains.argmax(-1).tolist()]
    target, _ = exact(1.5)
    assert total_variation(sequences, target) < 0.012
