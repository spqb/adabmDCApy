"""PTT edge corrections, adaptive pseudocounts, and archive recovery."""

from dataclasses import replace

import pytest
import torch

from adabmDCA import PTTConfig, PTTSampler, train_model
from adabmDCA.api import TrainingConfig
from adabmDCA.exceptions import ConvergenceError
from adabmDCA.ptt.optim import _PTTEdgeOptimizer


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def edge_case(tmp_path):
    data = tmp_path / "data.fasta"
    data.write_text(">s0\nAAA\n>s1\nAAB\n>s2\nABA\n>s3\nBAA\n>s4\nBBB\n>s5\nBAB\n")
    config = TrainingConfig(
        model_type="edgeDCA", alphabet="AB", device="cpu", dtype="float64",
        n_chains=80, n_sweeps=1, max_epochs=3, target_pearson=0.999999,
        no_reweighting=True, pseudocount=0.1,
        ptt=PTTConfig(equilibration_rounds=2, initialization_rounds=2, swaps=1,
                      mixing_thermalization_rounds=2, mixing_chains=20),
        checkpoint_interval=1,
    )
    return data, config


def pair_statistics():
    target = torch.zeros(2, 2, 2, 2, dtype=torch.float64)
    model = torch.zeros_like(target)
    target_pair = torch.tensor([[0.8, 0.05], [0.05, 0.1]], dtype=torch.float64)
    model_pair = torch.full((2, 2), 0.25, dtype=torch.float64)
    target[0, :, 1, :] = target_pair
    target[1, :, 0, :] = target_pair.T
    model[0, :, 1, :] = model_pair
    model[1, :, 0, :] = model_pair.T
    return target, model



def test_one_edge_kl_matches_exact_distribution():
    old_pair = torch.tensor([[0.4, 0.1], [0.2, 0.3]], dtype=torch.float64)
    correction = torch.tensor([[0.6, -0.3], [0.1, -0.2]], dtype=torch.float64)
    new_pair = old_pair * correction.exp()
    new_pair /= new_pair.sum()
    exact = float((new_pair * (new_pair / old_pair).log()).sum())
    assert _PTTEdgeOptimizer._candidate_kl(old_pair, correction) == pytest.approx(exact)


def test_edge_pseudocount_adapts_and_active_edge_can_be_reselected(monkeypatch):
    target, model = pair_statistics()
    params = {"bias": torch.zeros(2, 2, dtype=torch.float64),
              "coupling_matrix": torch.zeros(2, 2, 2, 2, dtype=torch.float64)}
    mask = torch.zeros_like(params["coupling_matrix"], dtype=torch.bool)
    adaptive = _PTTEdgeOptimizer(
        TrainingConfig(model_type="edgeDCA", pseudocount=0.1,
                       ptt=PTTConfig(trust_radius=0.001)), 0.1,
    )
    monkeypatch.setattr(adaptive, "_record_lags", lambda sampler: None)
    candidate, activated = adaptive.step(fij_raw=target, pij_raw=model, params=params,
                                         mask=mask, sampler=object())
    assert 0.1 < adaptive.pseudocount < 1.0
    assert adaptive.trust_diagnostics["predicted_kl"] <= 0.001
    assert adaptive.last_edge == [0, 1] and adaptive.last_edge_new
    again, repeated_mask = adaptive.step(fij_raw=target, pij_raw=model, params=candidate,
                                         mask=activated, sampler=object())
    assert not adaptive.last_edge_new
    torch.testing.assert_close(repeated_mask, activated)
    assert not torch.equal(again["coupling_matrix"], candidate["coupling_matrix"])
    fixed = _PTTEdgeOptimizer(
        TrainingConfig(model_type="edgeDCA", pseudocount=0.1,
                       ptt=PTTConfig(optimizer="sgd", trust_radius=0.001)), 0.1,
    )
    fixed.step(fij_raw=target, pij_raw=model, params=params, mask=mask)
    assert fixed.pseudocount == 0.1
    assert fixed.trust_diagnostics["predicted_kl"] > 0.001


def test_ptt_edge_training_and_exact_resume(tmp_path):
    data, config = edge_case(tmp_path)
    complete = train_model(data, config=config)
    partial = train_model(data, config=replace(config, max_epochs=1), output_dir=tmp_path / "partial")
    resumed = train_model(data, config=config, ptt_resume=partial.artifacts["ptt_archive"])
    for key in ("bias", "coupling_matrix"):
        torch.testing.assert_close(complete.model.params[key], resumed.model.params[key], rtol=0, atol=0)
    for key in ("edge_pseudocount", "edge_i", "edge_j", "edge_new", "Structure_steps"):
        assert complete.history[key] == resumed.history[key]
    assert complete.history["Epochs"] == [0, 1, 2, 3]
    assert complete.history["Structure_steps"][-1] <= 3
    assert any(not value for value in complete.history["edge_new"][1:])
    saved = PTTSampler.from_archive(partial.artifacts["ptt_archive"], mode="resume")
    assert saved.training_state["active_edges"]
    assert saved.optimizer_state["pseudocount"] > config.pseudocount



def test_ptt_edge_validation_event_uses_total_update_step(tmp_path):
    import json

    data, config = edge_case(tmp_path)
    config = replace(config, ptt=replace(config.ptt, validation_stop=True,
                                         validation_window=1, validation_min_gain=10.0))
    result = train_model(data, validation_path=data, config=config,
                         output_dir=tmp_path / "validation_stop")
    events = [json.loads(line) for line in result.artifacts["events"].read_text().splitlines()]
    stop = next(event for event in events if event["event"] == "ptt_validation_plateau")
    assert result.gradient_steps == stop["step"] == 2
    assert result.structure_steps < result.gradient_steps


def test_edge_recovery_increases_pseudocount_and_restores_mask(tmp_path, monkeypatch):
    data, config = edge_case(tmp_path)

    def reject(self, *args, **kwargs):
        self.last_failure = {
            "reason": "mixing_budget_exceeded", "model_version": self.model_version + 1,
            "mixing": {"tau_int": 1500.0, "tau_exp": 2000.0, "rounds": 100,
                       "required_rounds": 40000, "status": "budget_exceeded"},
            "events": [],
        }
        return False, 7

    monkeypatch.setattr(PTTSampler, "transition_target", reject)
    with pytest.raises(ConvergenceError, match="replicas failed to mix again"):
        train_model(data, config=config, output_dir=tmp_path / "failure")
    saved = PTTSampler.from_archive(tmp_path / "failure/ptt.h5", mode="resume")
    assert saved.training_state["gradient_steps"] == 0
    assert saved.training_state["structure_steps"] == 0
    assert saved.training_state["active_edges"] == []
    expected = 1.0 - (1.0 - config.pseudocount) / 2 ** config.ptt.max_recoveries
    assert saved.optimizer_state["nominal_pseudocount"] == pytest.approx(expected)
