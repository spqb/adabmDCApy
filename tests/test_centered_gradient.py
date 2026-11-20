import pytest
import torch

from adabmDCA import train_model
from adabmDCA.api import TrainingConfig
from adabmDCA.exceptions import InputValidationError
from adabmDCA.stats import get_freq_two_points
from adabmDCA.training import center_gradient, update_params


def _statistics(length=4, states=3):
    torch.manual_seed(0)
    mask = torch.ones(length, states, length, states, dtype=torch.float64)
    mask[torch.arange(length), :, torch.arange(length), :] = 0

    def frequencies():
        samples = torch.nn.functional.one_hot(torch.randint(states, (200, length)), states).double()
        return samples.mean(0), get_freq_two_points(samples) * mask

    (fi, fij), (pi, pij) = frequencies(), frequencies()
    return fi, fij, pi, pij, mask


def _params(length=4, states=3):
    return {"bias": torch.zeros(length, states, dtype=torch.float64),
            "coupling_matrix": torch.zeros(length, states, length, states, dtype=torch.float64)}


def test_pcd_update_is_centred_by_default():
    fi, fij, pi, pij, mask = _statistics()
    centred = update_params(fi, fij, pi, pij, _params(), mask, lr=0.1)
    raw = update_params(fi, fij, pi, pij, _params(), mask, lr=0.1, centered=False)
    expected = center_gradient({"bias": fi - pi, "coupling_matrix": (fij - pij) * mask}, fi=fi, pi=pi, mask=mask)
    torch.testing.assert_close(centred["bias"], 0.1 * expected["bias"])
    torch.testing.assert_close(centred["coupling_matrix"], 0.1 * expected["coupling_matrix"])
    torch.testing.assert_close(raw["bias"], 0.1 * (fi - pi))
    torch.testing.assert_close(raw["coupling_matrix"], 0.1 * (fij - pij) * mask)


def test_pcd_update_centres_the_l2_term_with_the_couplings():
    fi, fij, pi, pij, mask = _statistics()
    params = _params()
    params["coupling_matrix"] = torch.randn_like(params["coupling_matrix"]) * mask
    params["coupling_matrix"] = 0.5 * (params["coupling_matrix"] + params["coupling_matrix"].permute(2, 3, 0, 1))
    start = {key: value.clone() for key, value in params.items()}
    updated = update_params(fi, fij, pi, pij, params, mask, lr=0.1, l2_reg=0.01)
    raw = {"bias": fi - pi, "coupling_matrix": (fij - pij - 0.01 * start["coupling_matrix"]) * mask}
    expected = center_gradient(raw, fi=fi, pi=pi, mask=mask)
    for key in start:
        torch.testing.assert_close(updated[key], start[key] + 0.1 * expected[key])


def test_centered_gradient_setting_and_flag():
    from adabmDCA.training.cli import create_parser

    assert TrainingConfig().centered_gradient is True
    with pytest.raises(InputValidationError, match="centered_gradient"):
        TrainingConfig(centered_gradient="no")
    for strategy in ("pcd", "ptt"):
        base = ["-d", "x.fasta", "--strategy", strategy]
        assert create_parser().parse_args(base).centered_gradient is True
        assert create_parser().parse_args([*base, "--no-centered-gradient"]).centered_gradient is False


@pytest.mark.parametrize("model_type", ["bmDCA", "eaDCA", "edDCA"])
def test_pcd_training_follows_the_setting(tmp_path, model_type):
    path = tmp_path / "data.fasta"
    sequences = ["AAAA", "AAAB", "AABA", "BBBB", "BBBA", "BBAB", "ABAB"]
    path.write_text("".join(f">s{i}\n{seq}\n" for i, seq in enumerate(sequences)))
    config = TrainingConfig(model_type=model_type, alphabet="AB", device="cpu", dtype="float64", n_chains=20,
                            n_sweeps=1, max_epochs=2, target_pearson=0.999999, no_reweighting=True,
                            pseudocount=0.1, anchor="profile", activation_fraction=0.5, activation_steps=2,
                            target_density=0.9)
    centred = train_model(path, config=config)
    raw = train_model(path, config=TrainingConfig(**{**config.as_dict(), "centered_gradient": False}))
    assert centred.config.centered_gradient is True and raw.config.centered_gradient is False
    assert not torch.allclose(centred.model.params["bias"], raw.model.params["bias"])
