"""PTT eaDCA: transactional element activation, graph-block counters, recovery and resume."""

import json
from dataclasses import replace

import pytest
import torch

from adabmDCA import PTTConfig, PTTSampler, train_model
from adabmDCA.api import TrainingConfig, sample_sequences
from adabmDCA.exceptions import ConvergenceError, InputValidationError
from adabmDCA.graph import select_inactive_elements
from adabmDCA.ptt.policies import _ElementActivationPolicy, _pack_elements, _unpack_elements
from adabmDCA.training_control import TrainingController
from adabmDCA.utils import get_mask_save


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


GSTEPS = 2


def ea_case(tmp_path, **overrides):
    data = tmp_path / "data.fasta"
    data.write_text(">s0\nAAAA\n>s1\nAABA\n>s2\nABAB\n>s3\nBAAA\n>s4\nBBBB\n>s5\nBABB\n")
    settings = {
        "model_type": "eaDCA", "alphabet": "AB", "device": "cpu", "dtype": "float64",
        "n_chains": 80, "n_sweeps": 1, "max_epochs": 3, "target_pearson": 0.999999,
        "no_reweighting": True, "activation_steps": GSTEPS, "activation_fraction": 0.2,
        "ptt": PTTConfig(equilibration_rounds=2, initialization_rounds=2, swaps=1,
                         mixing_thermalization_rounds=2, mixing_chains=20),
        "checkpoint_interval": 1,
    }
    settings.update(overrides)
    return data, TrainingConfig(**settings)


def events_of(result):
    return [json.loads(line) for line in result.artifacts["events"].read_text().splitlines()]


def expected_block_position(step):
    """Graph number and position inside it after ``step`` accepted updates."""
    if step == 0:
        return 0, 0
    return (step - 1) // GSTEPS + 1, (step - 1) % GSTEPS + 1


def assert_block_positions(history):
    for step, structure, position in zip(history["Epochs"], history["Structure_steps"], history["steps_on_graph"]):
        assert (structure, position) == expected_block_position(step)
    for step, activated in zip(history["Epochs"], history["activated_entries"]):
        assert (activated > 0) == (step % GSTEPS == 1 if GSTEPS > 1 else step > 0)


# Configuration and selection


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_config_accepts_ptt_eadca(dtype):
    config = TrainingConfig(model_type="eaDCA", dtype=dtype, ptt=PTTConfig())
    assert config.limits.max_structure_steps == config.max_epochs
    assert config.limits.max_gradient_steps is None


def test_config_rejects_bfloat16_ptt_eadca():
    with pytest.raises(InputValidationError):
        TrainingConfig(model_type="eaDCA", dtype="bfloat16", ptt=PTTConfig())


def test_selector_picks_only_inactive_off_diagonal_entries():
    L, q = 3, 2
    Dkl = torch.zeros(L, q, L, q)
    Dkl[torch.arange(L), :, torch.arange(L), :] = 100.0  # diagonal blocks must never be chosen
    Dkl[0, 0, 1, 0] = Dkl[1, 0, 0, 0] = 50.0  # already active: must be skipped
    Dkl[0, 1, 2, 1] = Dkl[2, 1, 0, 1] = 5.0
    Dkl[1, 1, 2, 0] = Dkl[2, 0, 1, 1] = 4.0
    mask = torch.zeros(L, q, L, q, dtype=torch.bool)
    mask[0, 0, 1, 0] = mask[1, 0, 0, 0] = True
    # 3 * 4 = 12 unique entries, one active: 11 inactive, int(0.2 * 11) = 2.
    updated, requested, activated = select_inactive_elements(Dkl, mask, 0.2)
    assert (requested, activated) == (2, 2)
    new = updated & ~mask
    expected = torch.zeros_like(mask)
    expected[0, 1, 2, 1] = expected[2, 1, 0, 1] = True
    expected[1, 1, 2, 0] = expected[2, 0, 1, 1] = True
    assert torch.equal(new, expected)
    assert torch.equal(updated, updated.permute(2, 3, 0, 1))
    assert not updated[torch.arange(L), :, torch.arange(L), :].any()


def test_selector_guarantees_progress_and_stops_on_complete_graph():
    L, q = 2, 2
    mask = torch.zeros(L, q, L, q, dtype=torch.bool)
    Dkl = torch.rand(L, q, L, q)
    Dkl = Dkl + Dkl.permute(2, 3, 0, 1)
    updated, requested, activated = select_inactive_elements(Dkl, mask, 1e-9)
    assert requested == 0 and activated == 1
    full, _, activated = select_inactive_elements(Dkl, updated, 1.0)
    assert activated == 3
    assert torch.equal(full & get_mask_save(L, q, device="cpu"), get_mask_save(L, q, device="cpu"))
    again, _, activated = select_inactive_elements(Dkl, full, 1.0)
    assert activated == 0 and torch.equal(again, full)


def test_packed_elements_round_trip():
    L, q = 5, 3
    mask = torch.rand(L, q, L, q) < 0.3
    mask &= get_mask_save(L, q, device="cpu")
    mask = mask | mask.permute(2, 3, 0, 1)
    packed = _pack_elements(mask)
    assert packed["count"] == int(mask.sum()) // 2
    assert torch.equal(_unpack_elements(packed, mask), mask)
    with pytest.raises(InputValidationError):
        _unpack_elements({**packed, "count": packed["count"] + 1}, mask)


class _FakeSampler:
    def __init__(self, samples, params):
        self._samples, self._params = samples, params

    def endpoint_samples(self, copy=False):
        return self._samples

    def endpoint_params(self):
        return {key: value.clone() for key, value in self._params.items()}


def test_new_block_proposes_on_expanded_mask_and_kl_includes_it(monkeypatch):
    L, q = 3, 2
    generator = torch.Generator().manual_seed(0)
    samples = torch.nn.functional.one_hot(torch.randint(q, (200, L), generator=generator), q).double()
    params = {"bias": torch.zeros(L, q, dtype=torch.float64),
              "coupling_matrix": torch.zeros(L, q, L, q, dtype=torch.float64)}
    target = torch.nn.functional.one_hot(torch.tensor([[0, 0, 0], [1, 1, 1]] * 10), q).double()
    fi, fij = target.mean(0) * 0.9 + 0.05, torch.einsum("mia,mjb->iajb", target, target) / len(target) * 0.9 + 0.025
    pi, pij = samples.mean(0), torch.einsum("mia,mjb->iajb", samples, samples) / len(samples)
    config = TrainingConfig(model_type="eaDCA", dtype="float64", activation_fraction=0.5, activation_steps=2,
                            ptt=PTTConfig(trust_radius=1e-4))
    policy = _ElementActivationPolicy(config, torch.zeros(L, q, L, q, dtype=torch.bool), 0.05)
    monkeypatch.setattr(policy.optimizer, "_record_lags", lambda sampler: None)
    sampler = _FakeSampler(samples, params)
    candidate, mask = policy.propose(fi_target=fi, fij_target=fij, pi=pi, pij=pij, sampler=sampler)
    assert policy.pending["activated"] == 6 and int(mask.sum()) == 12
    diagnostics = policy.optimizer.trust_diagnostics
    assert diagnostics["linear_coupling"] > 0 and diagnostics["fisher_coupling"] > 0
    assert diagnostics["predicted_kl"] <= 1e-4 * (1 + 1e-9)
    couplings = candidate["coupling_matrix"]
    assert torch.count_nonzero(couplings[~mask]) == 0 and torch.count_nonzero(couplings[mask]) > 0
    # A rejected transition leaves the committed graph unchanged.
    policy.discard()
    assert not policy.mask.any() and policy.block_due
    # An accepted one commits mask and block position together.
    policy.propose(fi_target=fi, fij_target=fij, pi=pi, pij=pij, sampler=sampler)
    controller = TrainingController()
    policy.commit(mask, controller)
    assert controller.counters.structure_steps == 1 and policy.steps_on_graph == 1 and not policy.block_due


# Training


def test_ptt_eadca_blocks_masks_and_zero_inactive_couplings(tmp_path, monkeypatch):
    committed = []
    original = _ElementActivationPolicy.commit

    def recording_commit(self, candidate_mask, controller):
        original(self, candidate_mask, controller)
        committed.append(self.mask.clone())

    monkeypatch.setattr(_ElementActivationPolicy, "commit", recording_commit)
    data, config = ea_case(tmp_path)
    result = train_model(data, config=config, output_dir=tmp_path / "run")
    history = result.history
    assert result.stop_reason == "max_structure_steps"
    assert result.structure_steps == 3 and result.gradient_steps == 3 * GSTEPS
    assert_block_positions(history)
    activations = [event for event in events_of(result) if event["event"] == "ptt_activation"]
    assert [event["step"] for event in activations] == [1 + GSTEPS * k for k in range(3)]
    assert [event["structure_step"] for event in activations] == [1, 2, 3]
    for event in activations:
        assert event["gradient_step"] == event["step"] and event["density_after"] > event["density_before"]
    previous = torch.zeros_like(committed[0])
    for mask in committed:
        assert torch.equal(mask, mask.permute(2, 3, 0, 1))
        assert torch.equal(mask & previous, previous)
        assert not mask[torch.arange(4), :, torch.arange(4), :].any()
        previous = mask
    assert history["active_entries"][-1] == int(committed[-1].sum()) // 2
    couplings = result.model.params["coupling_matrix"]
    assert torch.count_nonzero(couplings[~committed[-1]]) == 0
    saved = PTTSampler.from_archive(result.artifacts["ptt_archive"], mode="resume")
    state = saved.training_state
    assert torch.equal(_unpack_elements(state["active_elements"], committed[-1]), committed[-1])
    assert (state["structure_steps"], state["steps_on_graph"], state["new_block_pending"]) == (3, GSTEPS, True)
    header = result.artifacts["history"].read_text().splitlines()[0].split(",")
    for column in ("structure_steps", "steps_on_graph", "activated_entries", "active_entries", "density",
                   "lr_bias", "lr_coupling", "kl"):
        assert column in header
    generated = sample_sequences(model=result.artifacts["ptt_archive"], ptt=True, n_sequences=11, n_sweeps=1000,
                                 reference_fasta=data, alphabet="AB", device="cpu")
    assert len(generated.sequences) == 11


def test_fixed_optimizer_keeps_rate_and_adaptive_respects_trust_radius(tmp_path):
    data, config = ea_case(tmp_path, max_epochs=2)
    fixed = train_model(data, config=replace(config, ptt=replace(config.ptt, optimizer="sgd")))
    assert set(fixed.history["bias_learning_rate"]) == {config.learning_rate}
    assert set(fixed.history["coupling_learning_rate"]) == {config.learning_rate}
    radius = 1e-3
    adaptive = train_model(data, config=replace(config, learning_rate=1.0,
                                                ptt=replace(config.ptt, trust_radius=radius)))
    kls = adaptive.history["ptt_predicted_kl"][1:]
    assert all(kl <= radius * (1 + 1e-9) for kl in kls)
    assert all(rate <= 1.0 for rate in adaptive.history["coupling_learning_rate"])
    assert min(adaptive.history["coupling_learning_rate"][1:]) < 1.0


def test_structure_and_gradient_limits_are_independent(tmp_path):
    data, config = ea_case(tmp_path, max_epochs=50)
    structural = train_model(data, config=replace(config, max_structure_steps=2))
    assert structural.stop_reason == "max_structure_steps"
    assert (structural.structure_steps, structural.gradient_steps) == (2, 2 * GSTEPS)
    gradient = train_model(data, config=replace(config, max_gradient_steps=3))
    assert gradient.stop_reason == "max_gradient_steps"
    assert (gradient.structure_steps, gradient.gradient_steps) == (2, 3)
    assert gradient.history["steps_on_graph"][-1] == 1


def test_complete_graph_stops_growth(tmp_path):
    data, config = ea_case(tmp_path, max_epochs=50, activation_fraction=1.0)
    result = train_model(data, config=config, output_dir=tmp_path / "complete")
    assert result.stop_reason == "max_structure_steps"
    assert result.structure_steps == 1 and result.gradient_steps == GSTEPS
    assert result.history["active_entries"][-1] == 6 * 4
    assert any(event["event"] == "ptt_graph_complete" for event in events_of(result))


def test_mid_block_interruption_resumes_bitwise(tmp_path):
    data, config = ea_case(tmp_path)
    complete = train_model(data, config=config)
    partial = train_model(data, config=replace(config, max_gradient_steps=3), output_dir=tmp_path / "partial")
    saved = PTTSampler.from_archive(partial.artifacts["ptt_archive"], mode="resume")
    assert (saved.training_state["steps_on_graph"], saved.training_state["new_block_pending"]) == (1, False)
    resumed = train_model(data, config=config, ptt_resume=partial.artifacts["ptt_archive"])
    for key in ("bias", "coupling_matrix"):
        torch.testing.assert_close(complete.model.params[key], resumed.model.params[key], rtol=0, atol=0)
    for key in ("Structure_steps", "steps_on_graph", "activated_entries", "active_entries", "Density",
                "bias_learning_rate", "coupling_learning_rate", "Pearson", "LL_train"):
        assert complete.history[key] == resumed.history[key]
    assert resumed.stop_reason == complete.stop_reason


def test_resume_rejects_changed_activation_settings(tmp_path):
    data, config = ea_case(tmp_path)
    partial = train_model(data, config=replace(config, max_gradient_steps=1), output_dir=tmp_path / "partial")
    for changed in ({"activation_fraction": 0.3}, {"activation_steps": 3}, {"pseudocount": 0.2},
                    {"learning_rate": 0.02}, {"l2_regularization": 0.1}):
        with pytest.raises(InputValidationError, match=next(iter(changed))):
            train_model(data, config=replace(config, **changed), ptt_resume=partial.artifacts["ptt_archive"])


def test_validation_plateau_uses_gradient_step_coordinates(tmp_path):
    data, config = ea_case(tmp_path, max_epochs=50)
    config = replace(config, ptt=replace(config.ptt, validation_stop=True,
                                         validation_window=1, validation_min_gain=10.0))
    result = train_model(data, validation_path=data, config=config, output_dir=tmp_path / "validation_stop")
    stop = next(event for event in events_of(result) if event["event"] == "ptt_validation_plateau")
    assert result.stop_reason == "validation_plateau"
    assert result.gradient_steps == stop["step"] == 2
    assert result.structure_steps == 1


def test_recovery_halves_rates_and_restores_block_position(tmp_path, monkeypatch):
    original = PTTSampler.transition_target
    calls = {"count": 0}

    def fail_once(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 4:
            self.last_failure = {
                "reason": "mixing_budget_exceeded", "model_version": self.model_version + 1,
                "mixing": {"tau_int": 1500.0, "tau_exp": 2000.0, "rounds": 100,
                           "required_rounds": 40000, "status": "budget_exceeded"},
                "events": [],
            }
            return False, 7
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PTTSampler, "transition_target", fail_once)
    data, config = ea_case(tmp_path, checkpoint_interval=2, ptt=replace(ea_case(tmp_path)[1].ptt, optimizer="sgd"))
    result = train_model(data, config=config, output_dir=tmp_path / "recovery")
    events = events_of(result)
    recovery = next(event for event in events if event["event"] == "ptt_learning_rate")
    restored = recovery["restored_step"]
    # Three updates were accepted (graph 2 began at the third). Saved every
    # second step, recovery rolls back across the block boundary to graph 1.
    assert restored == 2
    assert recovery["bias_learning_rate"] == recovery["coupling_learning_rate"] == config.learning_rate / 2
    history = result.history
    assert_block_positions(history)
    assert history["Epochs"] == list(range(result.gradient_steps + 1))
    assert set(history["bias_learning_rate"][restored + 1:]) == {config.learning_rate / 2}
    assert set(history["coupling_learning_rate"][restored + 1:]) == {config.learning_rate / 2}


def test_recovery_exhaustion_restores_empty_graph(tmp_path, monkeypatch):
    def reject(self, *args, **kwargs):
        self.last_failure = {
            "reason": "mixing_budget_exceeded", "model_version": self.model_version + 1,
            "mixing": {"tau_int": 1500.0, "tau_exp": 2000.0, "rounds": 100,
                       "required_rounds": 40000, "status": "budget_exceeded"},
            "events": [],
        }
        return False, 7

    monkeypatch.setattr(PTTSampler, "transition_target", reject)
    data, config = ea_case(tmp_path)
    with pytest.raises(ConvergenceError, match="replicas failed to mix again"):
        train_model(data, config=config, output_dir=tmp_path / "failure")
    saved = PTTSampler.from_archive(tmp_path / "failure/ptt.h5", mode="resume")
    state = saved.training_state
    assert (state["gradient_steps"], state["structure_steps"], state["steps_on_graph"]) == (0, 0, 0)
    assert state["active_elements"]["count"] == 0 and state["new_block_pending"]
    expected = config.learning_rate / 2 ** config.ptt.max_recoveries
    assert state["learning_rates"]["bias"] == pytest.approx(expected)
    assert state["learning_rates"]["coupling_matrix"] == pytest.approx(expected)


# Adaptive activation


def adaptive_case(tmp_path, **ptt_overrides):
    data, config = ea_case(tmp_path, activation_fraction=0.5)
    ptt = replace(config.ptt, **{"activation": "adaptive", "activation_significance": 0.0, **ptt_overrides})
    return data, replace(config, ptt=ptt)


def test_adaptive_activation_config():
    assert TrainingConfig(model_type="eaDCA", ptt=PTTConfig(activation="adaptive")).ptt.activation == "adaptive"
    with pytest.raises(InputValidationError, match="eaDCA only"):
        TrainingConfig(model_type="bmDCA", ptt=PTTConfig(activation="adaptive"))
    for bad in ({"activation": "greedy"}, {"activation_kl_share": 0.0}, {"activation_kl_share": 1.5},
                {"activation_significance": -1.0}):
        with pytest.raises(InputValidationError):
            PTTConfig(**bad)


def adaptive_policy(share=0.5, significance=0.0, fraction=1.0, trust_radius=0.01):
    L, q = 3, 2
    config = TrainingConfig(model_type="eaDCA", dtype="float64", activation_fraction=fraction,
                            ptt=PTTConfig(activation="adaptive", activation_kl_share=share,
                                          activation_significance=significance, trust_radius=trust_radius))
    policy = _ElementActivationPolicy(config, torch.zeros(L, q, L, q, dtype=torch.bool), 0.05, effective_size=20.0)
    generator = torch.Generator().manual_seed(1)
    samples = torch.nn.functional.one_hot(torch.randint(q, (300, L), generator=generator), q).double()
    target = torch.nn.functional.one_hot(torch.tensor([[0, 0, 0], [1, 1, 1]] * 10), q).double()
    fij = torch.einsum("mia,mjb->iajb", target, target) / len(target) * 0.9 + 0.025
    pij = torch.einsum("mia,mjb->iajb", samples, samples) / len(samples)
    return policy, samples, fij, pij


def test_first_step_kl_matches_direct_variance_of_each_prefix():
    policy, samples, _, _ = adaptive_policy()
    # Flat (L, q, L, q) = (3, 2, 3, 2) indices of J_01(0, 1), J_02(1, 1) and J_12(0, 1).
    indices = torch.tensor([0 * 12 + 0 * 6 + 1 * 2 + 1, 0 * 12 + 1 * 6 + 2 * 2 + 1, 1 * 12 + 0 * 6 + 2 * 2 + 1])
    weights = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)
    kls = policy._first_step_kl(samples, indices, weights, chunk=2)
    L, q = 3, 2
    states = samples.argmax(-1)
    for n in range(1, 4):
        score = torch.zeros(len(states), dtype=torch.float64)
        for index, weight in zip(indices[:n].tolist(), weights[:n].tolist()):
            i, a, j, b = index // (q * L * q), index // (L * q) % q, index // q % L, index % q
            score += weight * ((states[:, i] == a) & (states[:, j] == b)).double()
        assert kls[n - 1].item() == pytest.approx(0.5 * score.var(unbiased=False).item())


def test_adaptive_count_follows_kl_budget_and_significance():
    policy, samples, fij, pij = adaptive_policy(share=1.0, trust_radius=1.0)
    everything, details = policy._select_adaptive(fij, pij, pij, samples)
    assert details["activated"] == details["significant"] == details["candidates"] == 12
    assert int(everything.sum()) == 24
    tight = adaptive_policy(share=1e-3, trust_radius=1e-9)[0]
    tight_mask, tight_details = tight._select_adaptive(fij, pij, pij, samples)
    # The budget admits no candidate; one is activated to guarantee progress.
    assert tight_details["activated"] == 1 and int(tight_mask.sum()) == 2
    assert tight_details["predicted_kl"] > tight_details["kl_budget"]
    middle = adaptive_policy(share=1.0, trust_radius=0.5 * details["predicted_kl"])[0]
    _, middle_details = middle._select_adaptive(fij, pij, pij, samples)
    assert 1 <= middle_details["activated"] < 12
    assert middle_details["predicted_kl"] <= middle_details["kl_budget"] or middle_details["activated"] == 1
    strict = adaptive_policy(significance=1e6)[0]
    assert strict._select_adaptive(fij, pij, pij, samples)[0] is None
    capped = adaptive_policy(share=1.0, trust_radius=1.0, fraction=0.25)[0]
    assert capped._select_adaptive(fij, pij, pij, samples)[1]["activated"] == 3


def test_adaptive_training_resumes_bitwise_and_logs_selection(tmp_path):
    data, config = adaptive_case(tmp_path)
    complete = train_model(data, config=config, output_dir=tmp_path / "complete")
    activations = [event for event in events_of(complete) if event["event"] == "ptt_activation"]
    assert activations and all({"significant", "candidates", "kl_budget", "predicted_kl"} <= set(event)
                               for event in activations)
    assert_block_positions(complete.history)
    partial = train_model(data, config=replace(config, max_gradient_steps=3), output_dir=tmp_path / "partial")
    resumed = train_model(data, config=config, ptt_resume=partial.artifacts["ptt_archive"])
    for key in ("bias", "coupling_matrix"):
        torch.testing.assert_close(complete.model.params[key], resumed.model.params[key], rtol=0, atol=0)
    assert complete.history["active_entries"] == resumed.history["active_entries"]


def test_no_significant_candidate_stops_as_graph_converged(tmp_path):
    data, config = adaptive_case(tmp_path, activation_significance=1e6)
    result = train_model(data, config=config, output_dir=tmp_path / "converged")
    assert result.stop_reason == "graph_converged" and result.converged
    assert result.gradient_steps == 0 and result.structure_steps == 0
    assert any(event["event"] == "ptt_graph_converged" for event in events_of(result))


def test_recovery_halves_activation_kl_share(tmp_path, monkeypatch):
    def reject(self, *args, **kwargs):
        self.last_failure = {
            "reason": "mixing_budget_exceeded", "model_version": self.model_version + 1,
            "mixing": {"tau_int": 1500.0, "tau_exp": 2000.0, "rounds": 100,
                       "required_rounds": 40000, "status": "budget_exceeded"},
            "events": [],
        }
        return False, 7

    monkeypatch.setattr(PTTSampler, "transition_target", reject)
    data, config = adaptive_case(tmp_path)
    with pytest.raises(ConvergenceError):
        train_model(data, config=config, output_dir=tmp_path / "failure")
    state = PTTSampler.from_archive(tmp_path / "failure/ptt.h5", mode="resume").training_state
    expected = config.ptt.activation_kl_share / 2 ** config.ptt.max_recoveries
    assert state["activation_kl_share"] == pytest.approx(expected)
    assert state["recovery_events"][-1]["activation_kl_share"] == pytest.approx(expected)


def test_archive_alphabet_is_inherited_by_model_apis(tmp_path):
    """Model-loading APIs read a PTT archive's alphabet unless one is given explicitly."""
    from adabmDCA import compute_energies, inspect_model, predict_contacts, sample_sequences, scan_mutations

    data = tmp_path / "gapped.fasta"
    data.write_text(">s0\nA-BA\n>s1\nAAB-\n>s2\n-BAB\n>s3\nBA-A\n>s4\nBBBB\n>s5\nBA-B\n")
    config = TrainingConfig(
        model_type="bmDCA", alphabet="-AB", device="cpu", dtype="float64", n_chains=40, n_sweeps=1,
        max_epochs=1, target_pearson=0.999999, no_reweighting=True,
        ptt=PTTConfig(equilibration_rounds=2, initialization_rounds=2, swaps=1,
                      mixing_thermalization_rounds=2, mixing_chains=20),
    )
    archive = train_model(data, config=config, output_dir=tmp_path / "run").artifacts["ptt_archive"]
    assert inspect_model(archive, device="cpu").tokens == "-AB"
    assert compute_energies(["A-BA"], model=archive, device="cpu").shape == (1,)
    assert len(scan_mutations("A-BA", model=archive, device="cpu").mutations) == 4 * 2
    assert predict_contacts(model=archive, device="cpu").scores.shape == (4, 4)
    assert len(sample_sequences(model=archive, n_sequences=3, n_sweeps=2, device="cpu").sequences) == 3
    with pytest.raises(InputValidationError):
        inspect_model(archive, alphabet="protein", device="cpu")
