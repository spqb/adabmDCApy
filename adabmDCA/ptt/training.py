"""bmDCA, eaDCA and edgeDCA optimization using the shared PTT sampler."""

from __future__ import annotations

import hashlib
import statistics
import time

from adabmDCA.exceptions import ConvergenceError, InputValidationError
from adabmDCA.graph import compute_density
from adabmDCA.ptt.health import IMMOBILE_THRESHOLD
from adabmDCA.ptt.policies import _make_policy
from adabmDCA.serialization import to_jsonable
from adabmDCA.statmech import compute_entropy, compute_log_likelihood
from adabmDCA.stats import get_correlation_two_points, get_freq_single_point, get_freq_two_points
from adabmDCA.training_control import StopReason, TrainingMetrics
from adabmDCA.utils import get_mask_save


def _target_id(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        if tensor is not None:
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(b"None")
    return digest.hexdigest()


def _resume_settings(config_or_settings):
    """Return numerical PTT settings after removing runtime and token aliases."""
    settings = config_or_settings.as_dict() if hasattr(config_or_settings, "as_dict") else dict(config_or_settings)
    # slope_tolerance: removed setting, still present in archives written before 2.1.
    for name in ("max_epochs", "max_gradient_steps", "max_structure_steps", "checkpoint_interval", "use_wandb",
                 "alphabet", "slope_tolerance"):
        settings.pop(name, None)
    return settings


# Updates after a pause before another one may start.
_LAG_PAUSE_COOLDOWN = 10


def validation_window_gain(values, window):
    """Median validation log-likelihood of the last ``window`` updates minus that of the
    ``window`` before, or ``None`` before ``2 * window`` values are available.

    The logged validation log-likelihood carries the noise of the bridge log Z
    estimate: spikes at single updates and fluctuations correlated over tens of
    updates. Medians of whole windows ignore the spikes and average over the
    fluctuations.
    """
    values = [value for value in values if value is not None]
    if len(values) < 2 * window:
        return None
    return statistics.median(values[-window:]) - statistics.median(values[-2 * window:-window])


def validation_plateau(values, window, min_gain):
    """The :func:`validation_window_gain` when it is below ``min_gain``; otherwise ``None``."""
    gain = validation_window_gain(values, window)
    return gain if gain is not None and gain < min_gain else None


def train_ptt(*, ptt_sampler, fi_target, fij_target, mask, config, controller, fi_val=None, fij_val=None,
              fij_raw=None, edge_pseudocount=None, activation_pseudocount=None, effective_size=None):
    """Train with PTT snapshots, reservoirs and recovery based on mixing time.

    A failed mixing experiment restores the newest checkpoint that passes
    TRWA, then halves its learning rate.
    """
    return _PTTTrainer(
        sampler=ptt_sampler, fi_target=fi_target, fij_target=fij_target, mask=mask, config=config,
        controller=controller, fi_val=fi_val, fij_val=fij_val, fij_raw=fij_raw, edge_pseudocount=edge_pseudocount,
        activation_pseudocount=activation_pseudocount, effective_size=effective_size,
    ).run()



def _mixing_failure_summary(mixing, config, n_sweeps):
    """Why a mixing check or reservoir collection failed, and what to change, for error messages."""
    if not mixing:
        return "The replicas did not mix within the mixing-check budget."
    budget = config.mixing_max_rounds
    reservoir = mixing.get("reservoir_renewal")
    if reservoir is not None:
        # The record of the (passed) mixing check, extended by the failed collection.
        first = (f"At step {mixing.get('model_version')} a new reservoir could not be collected: the replica it "
                 f"is drawn from did not renew within {budget} exchange rounds")
        if reservoir.get("warmup_rounds") is None:
            first += " (its initial configurations were never all replaced)"
        else:
            first += (f" (its initial configurations were replaced after {reservoir['warmup_rounds']} rounds, "
                      "but not again)")
        immobile = reservoir.get("immobile") or ()
    else:
        first = (f"At step {mixing.get('model_version')} the replicas did not mix within "
                 f"{mixing.get('rounds')} exchange rounds")
        if mixing.get("method") == "renewal":
            trapped = mixing.get("trapped_fraction")
            if mixing.get("warmup_rounds") is None:
                first += " (the initial configurations were never all replaced"
            else:
                first += (f" (the initial configurations were replaced after {mixing['warmup_rounds']} rounds, "
                          "but the ladder did not renew again")
            if trapped:
                first += f"; {trapped:.1%} of the endpoint configurations were never replaced"
            first += ")"
        immobile = mixing.get("immobile") or ()
    sentences = [first + "."]
    acceptance = list(mixing.get("acceptance") or ())
    rates = ", ".join(f"{value:.3f}" for value in acceptance)
    blocking = [entry for entry in immobile if entry.get("blocking")]
    if blocking:
        entry = max(blocking, key=lambda item: item["immobile"])
        sentences.append(
            f"{entry['immobile']} of the {entry['old']} configurations still waiting to be renewed in replica "
            f"{entry['replica']} (counting from the bottom) practically never swap toward the bottom of the ladder "
            f"(acceptance under {IMMOBILE_THRESHOLD:g}): only local moves can free them, and "
            f"{budget} rounds of {n_sweeps} local sweeps were not enough.")
        if rates:
            sentences.append(f"Swap acceptance between adjacent replicas, bottom to top: {rates}.")
        sentences.append(f"Restart the training with more local sweeps per round, e.g. --nsweeps {10 * n_sweeps} "
                         "(each update then takes correspondingly longer); resuming this state does not help, "
                         "since its model already holds such configurations.")
        return " ".join(sentences)
    target = config.target_acceptance
    if acceptance:
        weakest = min(range(len(acceptance)), key=acceptance.__getitem__)
        sentences.append(f"Swap acceptance between adjacent replicas, bottom to top: {rates}.")
        if acceptance[weakest] < target:
            sentences.append(
                f"Configurations rarely cross link {weakest + 1} of {len(acceptance)} (acceptance "
                f"{acceptance[weakest]:.3f}, below the target {target:g}): the replicas are too far apart there. "
                "A smaller --lr, or a larger --ptt-target-acceptance for more closely spaced replicas, can help.")
        else:
            sentences.append(
                f"Every link accepts at least the target {target:g} of swaps, so the replicas relax too slowly "
                "between exchanges; more local sweeps per round (--nsweeps) can help.")
    return " ".join(sentences)


class _PTTTrainer:
    """One PTT training run: resume or initialization, the update loop, lag pauses and recovery.

    The model-specific part of each update belongs to the policy (see
    :mod:`adabmDCA.ptt.policies`); this class owns what every model shares.
    """

    def __init__(self, *, sampler, fi_target, fij_target, mask, config, controller, fi_val, fij_val,
                 fij_raw, edge_pseudocount, activation_pseudocount, effective_size):
        self.sampler, self.config, self.controller = sampler, config, controller
        self.fi_target, self.fij_target, self.fi_val, self.fij_val = fi_target, fij_target, fi_val, fij_val
        self.policy = _make_policy(config, mask, fij_raw=fij_raw, edge_pseudocount=edge_pseudocount,
                                   activation_pseudocount=activation_pseudocount, effective_size=effective_size)
        self.optimizer = self.policy.optimizer
        if sampler.config.validation_stop and (fi_val is None or fij_val is None):
            raise InputValidationError("Stopping on the validation log-likelihood requires a validation set.")
        self.stage_progress = controller.report_stage_progress if controller.stage_observer is not None else None
        self.start = time.monotonic()
        self.target_id = _target_id(fi_target, fij_target, fi_val, fij_val, mask)
        self.settings = _resume_settings(config)
        self.mask_save = get_mask_save(*fi_target.shape, device=fi_target.device)
        self.validation_stop = sampler.config.validation_stop
        self.recovery_events, self.elapsed = [], 0.0
        self.recovery_exhausted = False
        self.pauses = 0
        self.pi = self.pij = self.record = None
        self.plateau_gain = None

    # Setup

    def _resume(self):
        sampler, controller, policy = self.sampler, self.controller, self.policy
        state = sampler.training_state
        archived_settings = _resume_settings(state["settings"])
        target_mismatch = state["target_id"] != self.target_id
        differing_settings = sorted(
            key for key in set(archived_settings) | set(self.settings)
            if archived_settings.get(key) != self.settings.get(key)
        )
        if target_mismatch or differing_settings:
            reasons = []
            if target_mismatch:
                reasons.append("training or validation statistics")
            if differing_settings:
                reasons.append("settings: " + ", ".join(differing_settings))
            raise InputValidationError(
                "PTT resume targets or configuration differ from the archive in " + "; ".join(reasons)
                + " (only step budgets/checkpoint/logging may change).",
                details={"target_mismatch": target_mismatch, "differing_settings": differing_settings},
            )
        controller.history = state["history"]
        controller.counters.gradient_steps = state["gradient_steps"]
        policy.restore_graph(state, controller)
        controller.counters.sweeps = state["sweeps"]
        controller.history_changed()
        if hasattr(controller.checkpoint, "resumed"):
            controller.checkpoint.resumed(state["gradient_steps"])
        policy.restore_optimizer(sampler.optimizer_state)
        self.recovery_events = state.get("recovery_events", [])
        self.recovery_exhausted = state.get("recovery_exhausted", False)
        self.pauses = state.get("pauses", 0)
        self.elapsed = state["elapsed"]

    def _initialize(self):
        sampler, controller, stage_progress = self.sampler, self.controller, self.stage_progress
        controller.begin_stage("ptt_equilibration")
        if stage_progress is not None:
            stage_progress("ptt_equilibration", 0, sampler.config.initialization_rounds)
        before = sampler.local_sweeps
        sampler.equilibrate(
            rounds=sampler.config.initialization_rounds,
            local_sweeps=self.config.n_sweeps,
            is_cancelled=controller.is_cancelled,
            on_round=(None if stage_progress is None else
                      lambda done, total: stage_progress("ptt_equilibration", done, total)),
        )
        controller.counters.sweeps += sampler.local_sweeps - before

    def _initial_recovery_checkpoint(self):
        """Establish the first recovery point, whose ladder a mixing experiment has validated."""
        sampler, controller = self.sampler, self.controller
        controller.begin_stage("ptt_mixing", reason="initial_recovery_checkpoint")
        before = sampler.local_sweeps
        mixing = sampler.estimate_mixing_time(local_sweeps=self.config.n_sweeps, is_cancelled=controller.is_cancelled,
                                              on_progress=self.stage_progress)
        controller.counters.sweeps += sampler.local_sweeps - before
        self._log_events(sampler.events[-1:])
        self._replace_current_record()
        if not mixing.converged:
            controller.save_snapshot(self._snapshot())
            raise ConvergenceError("PTT initial mixing experiment exceeded its round budget.")
        self._remember(validated=True)
        controller.save_snapshot(self._snapshot())

    # State, metrics and records

    def _snapshot(self):
        sampler, controller, optimizer, policy = self.sampler, self.controller, self.optimizer, self.policy
        sampler.optimizer_state = optimizer.state_dict()
        sampler.training_state = {
            "target_id": self.target_id,
            "settings": self.settings,
            "history": controller.history,
            "gradient_steps": controller.counters.gradient_steps,
            **policy.graph_state(controller),
            "sweeps": controller.counters.sweeps,
            "learning_rate": optimizer.minimum_rate,
            "learning_rates": policy.rates(),
            "optimizer": policy.optimizer_summary(),
            "recovery_events": self.recovery_events,
            "recovery_exhausted": self.recovery_exhausted,
            "pauses": self.pauses,
            "elapsed": self.elapsed + time.monotonic() - self.start,
        }
        return {
            "params": sampler.models[-1],
            "chains": sampler.endpoint_samples(copy=False),
            "mask": policy.snapshot_mask(self.mask_save),
            "ptt_sampler": sampler,
        }

    def _metrics(self):
        sampler, optimizer = self.sampler, self.optimizer
        fi_target, fij_target, fi_val, fij_val = self.fi_target, self.fij_target, self.fi_val, self.fij_val
        params, chains = sampler.models[-1], sampler.endpoint_samples(copy=False)
        pi, pij = get_freq_single_point(chains), get_freq_two_points(chains)
        pearson, slope = get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)
        estimate = sampler.partition_estimate()
        if fi_val is not None and fij_val is not None:
            ll_val = compute_log_likelihood(fi_val, fij_val, params, estimate.log_z)
            p_val, s_val = get_correlation_two_points(fij=fij_val, pij=pij, fi=fi_val, pi=pi)
        else:
            ll_val = p_val = s_val = float("nan")
        ll_train = compute_log_likelihood(fi_target, fij_target, params, estimate.log_z)
        return (
            pi,
            pij,
            TrainingMetrics(
                pearson=pearson,
                slope=slope,
                ll_train=ll_train,
                ll_val=ll_val,
                pearson_val=p_val,
                slope_val=s_val,
                entropy=compute_entropy(chains, params, estimate.log_z),
                density=compute_density(self.policy.mask),
                elapsed_time=self.elapsed + time.monotonic() - self.start,
                # Mixing checks, ladder changes, pauses and recoveries are
                # events (events.jsonl); only quantities that change with
                # every update are part of the history.
                extra={
                    "logZ": estimate.log_z,
                    # Provenance of the normalizer, for progress events (not in history.csv).
                    "logZ_method": estimate.method,
                    "logZ_status": estimate.status,
                    "model_version": estimate.model_version,
                    "ladder_version": estimate.ladder_version,
                    "ptt_acceptance": min(sampler.acceptance),
                    "ptt_acceptance_rates": tuple(sampler.acceptance),
                    "ptt_replicas": sampler.n_active,
                    "ptt_total_models": sampler.total_models,
                    **self.policy.history_extra(),
                    **({} if optimizer.kind != "adaptive" else {
                        "ptt_predicted_kl": None if optimizer.trust_diagnostics is None else
                            optimizer.trust_diagnostics["predicted_kl"],
                        **{f"ptt_lag_{group}": value for group, value in (
                            optimizer._grouped_lags(optimizer.lag_diagnostics["lags"])
                            if optimizer.lag_diagnostics is not None
                            else dict.fromkeys(("drift", "drift_tail"))).items()},
                    }),
                },
            ),
        )

    def _log_events(self, events):
        for event in events:
            self.controller.begin_stage("ptt_" + event["kind"],
                                        **{key: value for key, value in event.items() if key != "kind"})

    def _remember(self, *, validated=False):
        sampler, controller = self.sampler, self.controller
        sampler.optimizer_state = self.optimizer.state_dict()
        point = {"step": controller.counters.gradient_steps, "learning_rate": self.optimizer.minimum_rate,
                 "learning_rates": self.policy.rates(),
                 "validated": validated, "state": sampler.capture_state(),
                 **self.policy.graph_state(controller)}
        if validated:
            sampler.recovery_points = [point]
        elif not sampler.recovery_points or sampler.recovery_points[-1]["step"] != point["step"]:
            sampler.recovery_points.append(point)

    def _replace_current_record(self):
        # Recovery may resample a previously recorded model. Replace that
        # row so archive normalizer provenance matches the saved chains.
        controller = self.controller
        self.pi, self.pij, self.record = self._metrics()
        counters = controller.counters
        row = {**self.record.as_record(counters.gradient_steps), "Sweeps": counters.sweeps,
               "Gradient_steps": counters.gradient_steps, "Structure_steps": counters.structure_steps,
               "Stage": counters.stage}
        for key, value in row.items():
            if key not in controller.history:
                controller.history[key] = [None] * len(controller.history["Epochs"])
            controller.history[key][-1] = value
        controller.history_changed()
        controller.report_stage_event("ptt_pearson_updated", pearson=self.record.pearson)

    def _plateau(self):
        if not self.validation_stop:
            return None
        config = self.sampler.config
        return validation_plateau(self.controller.history["LL_val"][1:], config.validation_window,
                                  config.validation_min_gain)

    def _recovery_exhaustion_message(self, *, restored=False, mixing=None):
        if mixing is None and self.recovery_events:
            mixing = self.recovery_events[-1].get("mixing")
        cause = _mixing_failure_summary(mixing, self.sampler.config, self.config.n_sweeps)
        saved = " The restored state was saved." if restored else ""
        if len(self.recovery_events) >= self.sampler.config.max_recoveries:
            return (f"PTT training stopped after {len(self.recovery_events)} recoveries: each one restored an "
                    f"earlier checkpoint and halved the learning rates, and the replicas failed to mix again. "
                    f"{cause}{saved}")
        return f"PTT training stopped: the learning rates reached their floor after repeated mixing failures. {cause}{saved}"

    # The update loop

    def _should_continue(self):
        return (not self.controller.gradient_limit_reached()
                and self.policy.stop_reason(self.controller) is None
                and self.plateau_gain is None
                and (self.validation_stop or not self.record.pearson >= self.config.target_pearson))

    def _lag_pause(self):
        # The endpoint chains trail the model: instead of stepping further,
        # give the endpoint a recent neighbour and let the ladder evolve at
        # fixed parameters until the lag has halved (or the budget runs out).
        sampler, controller, optimizer = self.sampler, self.controller, self.optimizer
        tolerance = sampler.config.lag_tolerance
        trigger = optimizer.current_lags(sampler)
        lag_before = max(value for value in trigger.values() if value is not None)
        event_start = len(sampler.events)
        healthy, rounds, work = sampler.equilibrate_for_lag(
            lambda trial: optimizer.current_lag(trial) <= 0.5 * tolerance,
            max_rounds=sampler.config.lag_pause_rounds, local_sweeps=self.config.n_sweeps,
            is_cancelled=controller.is_cancelled, on_progress=self.stage_progress,
        )
        controller.counters.sweeps += work
        self._log_events(sampler.events[event_start:] if healthy else sampler.last_failure["events"])
        optimizer.pause_cooldown = _LAG_PAUSE_COOLDOWN
        ladder_events = [event["kind"] for event in (sampler.events[event_start:] if healthy else [])
                         if event.get("reason") == "lag"]
        optimizer.last_pause = {
            "model_version": sampler.model_version, "rounds": rounds, "healthy": healthy,
            "lag_before": lag_before, "lag_after": optimizer.current_lag(sampler),
            "ladder_action": ("insert" if "snapshot_inserted" in ladder_events
                              else "replace" if "snapshot_replaced" in ladder_events else "none"),
            **{f"trigger_{group}": value for group, value in trigger.items()},
        }
        controller.begin_stage("ptt_lag_pause", **optimizer.last_pause)
        controller.begin_stage("ptt_optimization")
        self.pauses += 1
        if healthy:
            # The next gradient uses the equilibrated populations.
            self._replace_current_record()

    def _find_recovery_point(self):
        """Newest recovery point whose ladder passes a mixing experiment, restored into the sampler."""
        sampler, controller, stage_progress = self.sampler, self.controller, self.stage_progress
        # Reference recovery searches saved updates backwards. Keep the
        # numerical state and its matching historical statistics together.
        for point in reversed(sampler.recovery_points):
            probe = sampler.fork()
            probe.restore_state(point["state"])
            probe_step = point["step"]
            diagnostic = probe.estimate_mixing_time(
                local_sweeps=self.config.n_sweeps, is_cancelled=controller.is_cancelled,
                full_ladder=sampler.config.full_sampler,
                on_progress=(None if stage_progress is None else
                             lambda stage, current, total, probe_step=probe_step, **details: stage_progress(
                                 stage, current, total, probe_step=probe_step, **details)),
            )
            controller.counters.sweeps += diagnostic.local_sweeps
            if diagnostic.converged:
                sampler.restore_state(probe.capture_state())
                return point
        return None

    def _recover(self):
        """Roll back after a failed transition, halve the rates and save the restored state."""
        sampler, controller, optimizer, policy = self.sampler, self.controller, self.optimizer, self.policy
        failure = sampler.last_failure
        if failure is None or failure["reason"] != "mixing_budget_exceeded":
            raise RuntimeError("PTT rejected an update without a mixing failure.")
        policy.discard()
        self._log_events(failure["events"])
        controller.begin_stage("ptt_recovery", reason=failure["reason"], **failure["mixing"])
        restored = self._find_recovery_point()
        if restored is None:
            self.recovery_exhausted = True
            controller.save_snapshot(self._snapshot())
            raise ConvergenceError(
                "PTT training stopped: no saved checkpoint passed a mixing check after a mixing failure. "
                + _mixing_failure_summary(failure["mixing"], sampler.config, self.config.n_sweeps))
        old_rates = policy.rates()
        policy.restore_optimizer(sampler.optimizer_state)
        policy.restore_graph(restored, controller)
        controller.counters.gradient_steps = restored["step"]
        for key in controller.history:
            controller.history[key] = controller.history[key][:restored["step"] + 1]
        self.recovery_exhausted = (
            len(self.recovery_events) >= sampler.config.max_recoveries
            or optimizer.minimum_nominal_rate * 0.5 < sampler.config.min_learning_rate
        )
        if not self.recovery_exhausted:
            optimizer.halve()
            policy.on_recovery()
            self.recovery_events.append({
                "failed_step": failure["model_version"], "restored_step": restored["step"],
                "previous_learning_rates": old_rates,
                "learning_rates": policy.rates(),
                "learning_rate": optimizer.minimum_rate,
                **policy.recovery_details(),
                "reason": failure["reason"], "mixing": to_jsonable(failure["mixing"]),
            })
            controller.begin_stage("ptt_learning_rate", restored_step=restored["step"],
                                   learning_rate=optimizer.minimum_rate,
                                   **policy.rate_details(),
                                   reason="mixing_budget_exceeded")
            self._warm_up_after_recovery(restored["step"])
        self._replace_current_record()
        self._remember(validated=True)
        controller.save_snapshot(self._snapshot())
        if self.recovery_exhausted:
            raise ConvergenceError(self._recovery_exhaustion_message(restored=True, mixing=failure["mixing"]))
        controller.begin_stage("ptt_optimization")

    def _warm_up_after_recovery(self, restored_step):
        sampler, controller, stage_progress = self.sampler, self.controller, self.stage_progress
        before = sampler.local_sweeps
        if sampler.config.mixing_thermalization_rounds:
            total = sampler.config.mixing_thermalization_rounds
            if stage_progress is not None:
                stage_progress("recovery_warmup", 0, total, restored_step=restored_step)
            sampler.advance(
                rounds=total, local_sweeps=self.config.n_sweeps, is_cancelled=controller.is_cancelled,
                on_round=(None if stage_progress is None else
                          lambda done, total: stage_progress(
                              "recovery_warmup", done, total, restored_step=restored_step)),
            )
        controller.counters.sweeps += sampler.local_sweeps - before

    def _accept(self, candidate_mask, event_start, ladder_before):
        """Commit an accepted update: counters, graph, history, recovery points and checkpoints."""
        sampler, controller = self.sampler, self.controller
        self._log_events(sampler.events[event_start:])
        controller.add_gradient_steps(1)
        self.policy.commit(candidate_mask, controller)
        if controller.counters.stage != "ptt_optimization":
            controller.begin_stage("ptt_optimization")
        self.pi, self.pij, self.record = self._metrics()
        controller.record(self.record, epoch=controller.counters.gradient_steps)
        self.plateau_gain = self._plateau()
        if self.plateau_gain is not None:
            controller.begin_stage("ptt_validation_plateau", gain=self.plateau_gain,
                                   window=sampler.config.validation_window)
        validated = any(
            event["kind"] == "mixing" and event["status"] == "converged"
            and (not sampler.config.full_sampler or event.get("full_ladder", False))
            for event in sampler.events[event_start:]
        )
        checkpoint_due = controller.counters.gradient_steps % self.config.checkpoint_interval == 0
        if validated or checkpoint_due or ladder_before != sampler.ladder_version:
            self._remember(validated=validated)
        snapshot = self._snapshot()
        if controller.checkpoint is not None and (
            any(event["kind"] == "checkpoint_flagged" for event in sampler.events[event_start:])
            or controller.checkpoint.check(controller.counters.gradient_steps)
        ):
            controller.save_snapshot(snapshot)

    def run(self):
        sampler, controller, policy = self.sampler, self.controller, self.policy
        if sampler.training_state:
            self._resume()
        else:
            self._initialize()
        controller.begin_stage("ptt_optimization")
        self.pi, self.pij, self.record = self._metrics()
        if not controller.history["Epochs"]:
            # Append before saving, so the published archive contains this row.
            controller.record(self.record, epoch=0)
            controller.save_snapshot(self._snapshot())
        if not sampler.recovery_points:
            self._initial_recovery_checkpoint()
        if self.recovery_exhausted:
            raise ConvergenceError(self._recovery_exhaustion_message())
        self.plateau_gain = self._plateau()
        controller.begin_stage("ptt_optimization")
        while self._should_continue():
            controller.check_cancellation()
            if self.optimizer.wants_pause(sampler):
                self._lag_pause()
            candidate, candidate_mask = policy.propose(fi_target=self.fi_target, fij_target=self.fij_target,
                                                       pi=self.pi, pij=self.pij, sampler=sampler)
            if candidate is None:
                break
            event_start, ladder_before = len(sampler.events), sampler.ladder_version
            accepted, work = sampler.transition_target(
                candidate, local_sweeps=self.config.n_sweeps, is_cancelled=controller.is_cancelled,
                on_progress=self.stage_progress,
            )
            controller.counters.sweeps += work
            if accepted:
                self._accept(candidate_mask, event_start, ladder_before)
            else:
                self._recover()
        controller.set_stop_reason(
            StopReason.VALIDATION_PLATEAU if self.plateau_gain is not None
            else StopReason.TARGET_PEARSON
            if not self.validation_stop and self.record.pearson >= self.config.target_pearson
            else policy.stop_reason(controller) or StopReason.MAX_GRADIENT_STEPS
        )
        policy.finish(controller)
        controller.finalize(self._snapshot())
        return sampler.endpoint_samples(), sampler.endpoint_params(), controller.history
