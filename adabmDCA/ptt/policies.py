"""Model-specific PTT updates: the candidate mask and parameters, what an accepted
update commits, and the graph state that archives and recovery points carry."""

from __future__ import annotations

import base64
import binascii
import math

import numpy as np
import torch

from adabmDCA.exceptions import InputValidationError
from adabmDCA.graph import (
    activate_elements,
    compute_density,
    compute_Dkl_element_activation,
    rank_inactive_elements,
    select_inactive_elements,
)
from adabmDCA.ptt.optim import _PTTEdgeOptimizer, _PTTOptimizer
from adabmDCA.ptt.precision import accumulation_dtype, device_accumulation
from adabmDCA.stats import get_freq_two_points
from adabmDCA.training_control import StopReason
from adabmDCA.utils import get_mask_save


def _edge_pairs(mask):
    """Compact graph state for archive metadata and recovery points."""
    active = mask.any(dim=(1, 3)).triu(diagonal=1).nonzero(as_tuple=False)
    return [[int(i), int(j)] for i, j in active.tolist()]


def _mask_from_edges(edges, template):
    mask = torch.zeros_like(template, dtype=torch.bool)
    for i, j in edges:
        mask[i, :, j, :] = True
        mask[j, :, i, :] = True
    return mask



def _pack_elements(mask):
    """Compact eaDCA graph state: packed bits of the unique (i < j) coupling entries."""
    upper = mask[get_mask_save(*mask.shape[:2], device=mask.device)].cpu().numpy()
    return {"count": int(upper.sum()), "bits": base64.b64encode(np.packbits(upper).tobytes()).decode("ascii")}


def _unpack_elements(packed, template):
    upper = get_mask_save(*template.shape[:2], device=template.device)
    size = int(upper.sum())
    try:
        bits = np.unpackbits(np.frombuffer(base64.b64decode(packed["bits"], validate=True), dtype=np.uint8))
    except (binascii.Error, KeyError, TypeError, ValueError) as exc:
        raise InputValidationError("Invalid PTT eaDCA active-element state.") from exc
    if len(bits) != 8 * math.ceil(size / 8) or bits[size:].any() or int(bits.sum()) != int(packed["count"]):
        raise InputValidationError("Invalid PTT eaDCA active-element state.")
    mask = torch.zeros_like(template, dtype=torch.bool)
    mask[upper] = torch.from_numpy(bits[:size].astype(bool)).to(template.device)
    return mask | mask.permute(2, 3, 0, 1)


class _DensePolicy:
    """bmDCA: every update adjusts the fields and all couplings of a fixed graph.

    A policy owns the model-specific part of a PTT update: the candidate mask
    and parameters, what an accepted update commits, and the graph state that
    restart archives and recovery points must carry. The PTT transition,
    mixing, pauses, stopping and recovery stay shared in ``train_ptt``.
    """

    def __init__(self, config, mask):
        self.config = config
        self.mask = mask
        self.optimizer = _PTTOptimizer(config)

    def restore_optimizer(self, state):
        if state is None:
            raise InputValidationError("The PTT archive lacks the optimizer state needed to resume training.")
        self.optimizer.load_state_dict(state)

    def rates(self):
        return dict(self.optimizer.rates)

    def rate_details(self):
        return {"bias_learning_rate": self.optimizer.rates["bias"],
                "coupling_learning_rate": self.optimizer.rates["coupling_matrix"]}

    def recovery_details(self):
        return {}

    def optimizer_summary(self):
        return {"kind": self.optimizer.kind, "rates": dict(self.optimizer.rates)}

    def graph_state(self, controller):
        """Graph state stored with the restart state and every recovery point."""
        return {}

    def restore_graph(self, state, controller):
        pass

    def stop_reason(self, controller):
        """Structural reason to stop before the next update, if any."""

    def history_extra(self):
        return self.rate_details()

    def snapshot_mask(self, mask_save):
        return mask_save

    def propose(self, *, fi_target, fij_target, pi, pij, sampler):
        candidate = self.optimizer.step(
            fi=fi_target, fij=fij_target, pi=pi, pij=pij, params=sampler.endpoint_params(), mask=self.mask,
            l2_regularization=self.config.l2_regularization, sampler=sampler,
        )
        return candidate, self.mask

    def discard(self):
        """Forget a proposal that PTT rejected."""

    def on_recovery(self):
        """Make the next updates more conservative after a mixing recovery."""

    def finish(self, controller):
        """Report why the graph stopped growing, once training ends."""

    def commit(self, candidate_mask, controller):
        self.mask = candidate_mask


class _ElementActivationPolicy(_DensePolicy):
    """eaDCA: every ``activation_steps`` accepted updates begin on a grown graph.

    The first update of a block is proposed on the expanded mask, so the
    adaptive KL trust region already includes the new couplings. The mask is
    committed together with that update: a rejected transition leaves the
    graph as it was.

    With ``activation="adaptive"`` the fraction is only an upper limit: a block
    activates the significant candidates, in order of decreasing discrepancy,
    whose first update at the nominal coupling rate fits ``kl_share`` of the
    trust radius. Mixing recoveries halve that share with the learning rates.
    """

    def __init__(self, config, mask, pseudocount, effective_size=None):
        super().__init__(config, mask)
        self.pseudocount = float(pseudocount)
        self.adaptive = config.ptt.activation == "adaptive"
        if self.adaptive and effective_size is None:
            raise InputValidationError("PTT adaptive activation requires the effective size of the training data.")
        self.effective_size = None if effective_size is None else float(effective_size)
        self.kl_share = config.ptt.activation_kl_share
        self.upper = get_mask_save(*mask.shape[:2], device=mask.device)
        self.total_entries = int(self.upper.sum())
        self.steps_on_graph = 0
        self.last_activated = 0
        self.pending = None
        self.converged = None

    @property
    def block_due(self):
        return self.steps_on_graph == 0 or self.steps_on_graph >= self.config.activation_steps

    def active_entries(self):
        return int((self.mask & self.upper).sum())

    def graph_state(self, controller):
        return {
            "structure_steps": controller.counters.structure_steps,
            "steps_on_graph": self.steps_on_graph,
            "new_block_pending": self.block_due,
            "activated_entries": self.last_activated,
            "activation_kl_share": self.kl_share,
            "active_elements": _pack_elements(self.mask),
        }

    def restore_graph(self, state, controller):
        self.mask = _unpack_elements(state["active_elements"], self.mask)
        controller.counters.structure_steps = int(state["structure_steps"])
        self.steps_on_graph = int(state["steps_on_graph"])
        self.last_activated = int(state["activated_entries"])
        self.kl_share = float(state.get("activation_kl_share", self.config.ptt.activation_kl_share))
        self.pending = None

    def on_recovery(self):
        if self.adaptive:
            self.kl_share *= 0.5

    def recovery_details(self):
        return {"activation_kl_share": self.kl_share} if self.adaptive else {}

    def stop_reason(self, controller):
        if self.converged is not None:
            return StopReason.GRAPH_CONVERGED
        if self.block_due and (controller.structure_limit_reached() or self.active_entries() == self.total_entries):
            return StopReason.MAX_STRUCTURE_STEPS
        return None

    def history_extra(self):
        return {**self.rate_details(), "steps_on_graph": self.steps_on_graph,
                "activated_entries": self.last_activated, "active_entries": self.active_entries()}

    def snapshot_mask(self, mask_save):
        return torch.logical_and(self.mask, mask_save)

    def _first_step_kl(self, samples, indices, weights, chunk=4096):
        """Predicted KL, ½ Var(score), of the first update restricted to each prefix of ``indices``."""
        L, q = self.mask.shape[:2]
        states = samples.argmax(dim=-1)
        i, rest = indices // (q * L * q), indices % (q * L * q)
        a, rest = rest // (L * q), rest % (L * q)
        j, b = rest // q, rest % q
        running = torch.zeros(len(states), dtype=accumulation_dtype(states.device), device=states.device)
        kls = []
        for start in range(0, len(indices), chunk):
            part = slice(start, start + chunk)
            present = (states[:, i[part]] == a[part]) & (states[:, j[part]] == b[part])
            scores = running.unsqueeze(1) + (device_accumulation(present) * device_accumulation(weights[part])).cumsum(dim=1)
            kls.append(0.5 * scores.var(dim=0, unbiased=False))
            running = scores[:, -1]
        return torch.cat(kls)

    def _select_adaptive(self, fij_target, model, pij, samples):
        candidates = rank_inactive_elements(compute_Dkl_element_activation(fij_target, model), self.mask)
        inactive = len(candidates)
        requested = int(inactive * self.config.activation_fraction)
        candidates = candidates[:min(inactive, max(1, requested))]
        f, p = device_accumulation(fij_target.flatten()[candidates]), device_accumulation(model.flatten()[candidates])
        error = (f * (1 - f) / self.effective_size + p * (1 - p) / len(samples)).sqrt()
        significant = candidates[(f - p).abs() >= self.config.ptt.activation_significance * error]
        details = {"requested": requested, "candidates": len(candidates), "significant": len(significant),
                   "kl_budget": self.kl_share * self.config.ptt.trust_radius}
        if len(significant) == 0:
            return None, {**details, "activated": 0, "predicted_kl": 0.0}
        weights = self.optimizer.nominal_rates["coupling_matrix"] * (fij_target - pij).flatten()[significant]
        kls = self._first_step_kl(samples, significant, weights)
        over = (kls > details["kl_budget"]).nonzero()
        count = max(1, int(over[0]) if len(over) else len(significant))
        return activate_elements(self.mask, significant[:count]), {
            **details, "activated": count, "predicted_kl": float(kls[count - 1]),
        }

    def propose(self, *, fi_target, fij_target, pi, pij, sampler):
        mask, self.pending = self.mask, None
        if self.block_due:
            samples = sampler.endpoint_samples(copy=False)
            model = get_freq_two_points(samples, pseudo_count=self.pseudocount)
            if self.adaptive:
                mask, self.pending = self._select_adaptive(fij_target, model, pij, samples)
                if mask is None:
                    # No inactive coupling differs significantly from the data.
                    self.converged, self.pending = self.pending, None
                    return None, None
            else:
                mask, requested, activated = select_inactive_elements(
                    compute_Dkl_element_activation(fij_target, model), self.mask, self.config.activation_fraction,
                )
                self.pending = {"requested": requested, "activated": activated}
        candidate = self.optimizer.step(
            fi=fi_target, fij=fij_target, pi=pi, pij=pij, params=sampler.endpoint_params(), mask=mask,
            l2_regularization=self.config.l2_regularization, sampler=sampler,
        )
        return candidate, mask

    def discard(self):
        self.pending = None

    def finish(self, controller):
        if self.converged is not None:
            controller.begin_stage("ptt_graph_converged", active_entries=self.active_entries(), **self.converged)
        elif self.block_due and self.active_entries() == self.total_entries:
            controller.begin_stage("ptt_graph_complete", active_entries=self.total_entries)

    def commit(self, candidate_mask, controller):
        if self.pending is None:
            self.steps_on_graph += 1
            self.last_activated = 0
            return
        density_before = compute_density(self.mask)
        self.mask = candidate_mask
        self.steps_on_graph = 1
        self.last_activated = self.pending["activated"]
        controller.add_structure_step()
        controller.begin_stage(
            "ptt_activation", structure_step=controller.counters.structure_steps,
            gradient_step=controller.counters.gradient_steps, **self.pending,
            active_entries=self.active_entries(),
            density_before=density_before, density_after=compute_density(self.mask),
        )
        self.pending = None


class _EdgePolicy(_DensePolicy):
    """edgeDCA: one edge correction per update; new edges are structure steps."""

    def __init__(self, config, mask, pseudocount, fij_raw):
        self.config = config
        self.mask = mask
        self.fij_raw = fij_raw
        self.optimizer = _PTTEdgeOptimizer(config, pseudocount)

    def rates(self):
        return {"edge": self.optimizer.minimum_rate}

    def rate_details(self):
        return {"edge_pseudocount": self.optimizer.nominal_pseudocount}

    def recovery_details(self):
        return {"pseudocount": self.optimizer.nominal_pseudocount}

    def optimizer_summary(self):
        return {"kind": self.optimizer.kind, "pseudocount": self.optimizer.pseudocount}

    def graph_state(self, controller):
        return {"structure_steps": controller.counters.structure_steps, "active_edges": _edge_pairs(self.mask)}

    def restore_graph(self, state, controller):
        self.mask = _mask_from_edges(state["active_edges"], self.mask)
        controller.counters.structure_steps = int(state["structure_steps"])

    def stop_reason(self, controller):
        return StopReason.MAX_STRUCTURE_STEPS if controller.structure_limit_reached() else None

    def history_extra(self):
        optimizer = self.optimizer
        return {
            "edge_pseudocount": optimizer.pseudocount,
            "edge_i": None if optimizer.last_edge is None else optimizer.last_edge[0],
            "edge_j": None if optimizer.last_edge is None else optimizer.last_edge[1],
            "edge_new": optimizer.last_edge_new,
            "ptt_predicted_kl": None if optimizer.trust_diagnostics is None else
                optimizer.trust_diagnostics["predicted_kl"],
        }

    def snapshot_mask(self, mask_save):
        return torch.logical_and(self.mask, mask_save)

    def propose(self, *, fi_target, fij_target, pi, pij, sampler):
        return self.optimizer.step(
            fij_raw=self.fij_raw, pij_raw=pij, params=sampler.endpoint_params(), mask=self.mask, sampler=sampler,
        )

    def commit(self, candidate_mask, controller):
        self.mask = candidate_mask
        if self.optimizer.last_edge_new:
            controller.add_structure_step()


def _make_policy(config, mask, *, fij_raw, edge_pseudocount, activation_pseudocount, effective_size):
    if config.model_type == "edgeDCA":
        if fij_raw is None or edge_pseudocount is None:
            raise InputValidationError("PTT edgeDCA requires raw pair frequencies and a pseudocount.")
        return _EdgePolicy(config, mask, edge_pseudocount, fij_raw)
    if config.model_type == "eaDCA":
        if activation_pseudocount is None:
            raise InputValidationError("PTT eaDCA requires the pseudocount used to select new couplings.")
        return _ElementActivationPolicy(config, mask, activation_pseudocount, effective_size)
    return _DensePolicy(config, mask)


