"""Parameter updates of PTT training: bias/coupling gradient ascent and edge corrections."""

from __future__ import annotations

import math

import torch

from adabmDCA.exceptions import InputValidationError
from adabmDCA.graph import compute_Dkl_edge_activation
from adabmDCA.ptt.precision import accumulation_dtype, device_accumulation, diagnostic_double
from adabmDCA.statmech import compute_energy


class _PTTOptimizer:
    """Bias/coupling gradient ascent: fixed rates (``sgd``) or the ``adaptive`` trust region.

    ``adaptive`` scales the two group steps to maximize the predicted gain
    within a KL trust region, at most by the nominal rates. Before each
    update it also measures how far the endpoint chains lag the model; above
    ``lag_tolerance`` training pauses (see ``wants_pause``).
    """

    _keys = ("bias", "coupling_matrix")

    def __init__(self, config, state=None):
        ptt = config.ptt
        self.kind = ptt.optimizer
        self.trust_radius = ptt.trust_radius
        self.lag_tolerance = ptt.lag_tolerance
        self.pause_cooldown = 0
        self.last_pause = None
        self.rates = {key: config.learning_rate for key in self._keys}
        self.nominal_rates = dict(self.rates)
        self.trust_diagnostics = None
        self.lag_diagnostics = None
        if state is not None:
            self.load_state_dict(state)

    def state_dict(self):
        return {
            "kind": self.kind,
            "trust_radius": self.trust_radius,
            "rates": dict(self.rates),
            "nominal_rates": dict(self.nominal_rates),
            "trust_diagnostics": self.trust_diagnostics,
            "lag_diagnostics": self.lag_diagnostics,
            "pause_cooldown": self.pause_cooldown,
            "last_pause": self.last_pause,
        }

    def load_state_dict(self, state):
        if state["kind"] != self.kind or state["trust_radius"] != self.trust_radius:
            raise InputValidationError("PTT optimizer state is incompatible with the requested configuration.")
        self.rates = {key: float(state["rates"][key]) for key in self._keys}
        self.nominal_rates = {key: float(state["nominal_rates"][key]) for key in self._keys}
        self.trust_diagnostics = state["trust_diagnostics"]
        self.lag_diagnostics = state["lag_diagnostics"]
        self.pause_cooldown = int(state["pause_cooldown"])
        self.last_pause = state["last_pause"]

    @staticmethod
    def _directional_score_components(samples, bias_direction, coupling_direction, chunk_size=256):
        """Evaluate the two parameter-group scores without expanding pair features."""
        if samples.device.type == "mps":
            # PTT populations are already one-hot. GEMM avoids the N*L*L
            # categorical gather and uses Apple's optimized matrix kernels.
            flat = samples.flatten(1)
            bias = bias_direction.flatten()
            coupling = coupling_direction.reshape(flat.shape[1], flat.shape[1])
            # Bound the projected population to 32 MiB, independently of N.
            rows_per_chunk = max(1, (32 * 1024**2) // (flat.shape[1] * flat.element_size()))
            bias_scores, coupling_scores = [], []
            for batch in flat.split(rows_per_chunk):
                bias_scores.append(batch @ bias)
                projected = batch @ coupling
                coupling_scores.append(projected.mul_(batch).sum(dim=1).mul_(0.5))
            return torch.cat(bias_scores), torch.cat(coupling_scores)
        states = samples.argmax(dim=-1)
        length = states.shape[1]
        sites = torch.arange(length, device=states.device)
        bias_scores, coupling_scores = [], []
        for start in range(0, len(states), chunk_size):
            batch = states[start:start + chunk_size]
            bias_score = bias_direction[sites.unsqueeze(0), batch].sum(dim=1)
            left = sites.view(1, length, 1)
            right = sites.view(1, 1, length)
            coupling_score = coupling_direction[
                left, batch.unsqueeze(2), right, batch.unsqueeze(1)
            ].sum(dim=(1, 2)).mul_(0.5)
            bias_scores.append(bias_score)
            coupling_scores.append(coupling_score)
        return torch.cat(bias_scores), torch.cat(coupling_scores)

    @staticmethod
    def _solve_trust_region(matrix, linear, trust_radius):
        """Maximize the quadratic model in a two-dimensional box and KL ball."""
        a, cross, d = matrix
        bh, bj = linear
        largest = max(a, d, abs(cross), 1.0)
        smallest_eigenvalue = 0.5 * (a + d - math.hypot(a - d, 2.0 * cross))
        ridge = max(1e-12 * largest, -smallest_eigenvalue + 1e-12 * largest)
        a, d = a + ridge, d + ridge

        def quadratic(h, j):
            return 0.5 * (a * h * h + 2.0 * cross * h * j + d * j * j)

        def objective(h, j):
            return bh * h + bj * j - quadratic(h, j)

        def feasible(h, j):
            return -1e-12 <= h <= 1.0 + 1e-12 and -1e-12 <= j <= 1.0 + 1e-12

        def clipped(value):
            return min(1.0, max(0.0, value))

        determinant = a * d - cross * cross
        candidates = [(h, j) for h in (0.0, 1.0) for j in (0.0, 1.0)]
        if determinant > 0:
            candidates.append(((d * bh - cross * bj) / determinant,
                               (a * bj - cross * bh) / determinant))
        for h in (0.0, 1.0):
            candidates.append((h, clipped((bj - cross * h) / d)))
        for j in (0.0, 1.0):
            candidates.append((clipped((bh - cross * j) / a), j))
        box_candidates = [(clipped(h), clipped(j)) for h, j in candidates if feasible(h, j)]
        box_solution = max(box_candidates, key=lambda value: objective(*value))
        if quadratic(*box_solution) <= trust_radius * (1.0 + 1e-12):
            # Positive rates, as on the trust boundary below: a zero group rate
            # would be stored in recovery points, which archives reject.
            return (max(box_solution[0], 1e-12), max(box_solution[1], 1e-12)), ridge

        # On the active KL boundary the quadratic contribution is constant,
        # so the constrained optimum maximizes the linear gain.
        trust_candidates = [(0.0, 0.0)]
        if determinant > 0:
            direction = ((d * bh - cross * bj) / determinant,
                         (a * bj - cross * bh) / determinant)
            direction_q = quadratic(*direction)
            if direction_q > 0:
                factor = math.sqrt(trust_radius / direction_q)
                tangent = (factor * direction[0], factor * direction[1])
                if feasible(*tangent):
                    trust_candidates.append(tangent)

        def boundary_roots(fixed, fix_h):
            if fix_h:
                constant, coefficient, square = a * fixed * fixed, 2.0 * cross * fixed, d
            else:
                constant, coefficient, square = d * fixed * fixed, 2.0 * cross * fixed, a
            discriminant = coefficient * coefficient - 4.0 * square * (constant - 2.0 * trust_radius)
            if discriminant < 0:
                return []
            root = math.sqrt(max(0.0, discriminant))
            return [(-coefficient - root) / (2.0 * square),
                    (-coefficient + root) / (2.0 * square)]

        for h in (0.0, 1.0):
            trust_candidates.extend((h, j) for j in boundary_roots(h, True) if feasible(h, j))
        for j in (0.0, 1.0):
            trust_candidates.extend((h, j) for h in boundary_roots(j, False) if feasible(h, j))
        trust_candidates.extend(
            (h, j) for h in (0.0, 1.0) for j in (0.0, 1.0)
            if quadratic(h, j) <= trust_radius * (1.0 + 1e-12)
        )
        solution = max(trust_candidates, key=lambda value: bh * value[0] + bj * value[1])
        # Keep serialized learning rates positive even when the mathematical
        # box solution places one group exactly on its zero boundary.
        h, j = max(solution[0], 1e-12), max(solution[1], 1e-12)
        predicted_kl = quadratic(h, j)
        if predicted_kl > trust_radius:
            shrink = math.sqrt(trust_radius / predicted_kl)
            h, j = h * shrink, j * shrink
        return (h, j), ridge

    def _adapt(self, gradients, samples, l2_regularization):
        """Scale the two group rates within the KL trust region around the nominal rates."""
        directions = {key: self.nominal_rates[key] * gradients[key] for key in self._keys}
        bias_scores, coupling_scores = self._directional_score_components(
            samples, directions["bias"], directions["coupling_matrix"]
        )
        bias_scores, coupling_scores = diagnostic_double(bias_scores), diagnostic_double(coupling_scores)
        centered_bias = bias_scores - bias_scores.mean()
        centered_coupling = coupling_scores - coupling_scores.mean()
        variance_bias = float(centered_bias.square().mean())
        variance_coupling = float(centered_coupling.square().mean())
        covariance = float((centered_bias * centered_coupling).mean())
        regularization_curvature = float(
            0.5 * l2_regularization * device_accumulation(directions["coupling_matrix"]).square().sum()
        )
        linear_bias = float(
            (device_accumulation(gradients["bias"]) * device_accumulation(directions["bias"])).sum()
        )
        linear_coupling = float(
            0.5 * (
                device_accumulation(gradients["coupling_matrix"])
                * device_accumulation(directions["coupling_matrix"])
            ).sum()
        )
        scales, ridge = self._solve_trust_region(
            (variance_bias, covariance, variance_coupling + regularization_curvature),
            (linear_bias, linear_coupling), self.trust_radius,
        )
        bias_scale, coupling_scale = scales
        self.rates = {
            "bias": self.nominal_rates["bias"] * bias_scale,
            "coupling_matrix": self.nominal_rates["coupling_matrix"] * coupling_scale,
        }
        stabilized_bias = variance_bias + ridge
        stabilized_coupling = variance_coupling + regularization_curvature + ridge
        predicted_kl = 0.5 * (
            stabilized_bias * bias_scale * bias_scale
            + 2.0 * covariance * bias_scale * coupling_scale
            + stabilized_coupling * coupling_scale * coupling_scale
        )
        self.trust_diagnostics = {
            "bias_scale": bias_scale,
            "coupling_scale": coupling_scale,
            "predicted_kl": predicted_kl,
            "fisher_bias": stabilized_bias,
            "fisher_coupling": stabilized_coupling,
            "fisher_cross": covariance,
            "regularization_curvature": regularization_curvature,
            "ridge": ridge,
            "linear_bias": linear_bias,
            "linear_coupling": linear_coupling,
        }

    @staticmethod
    def _lag_observables(sampler, samples):
        """Names and standardized per-sample values of the observables whose lag is monitored.

        ``drift`` is the energy under the endpoint minus the energy under the
        replica below it: the direction the model moved since the last
        checkpoint. ``drift_low``/``drift_high`` indicate the configurations
        beyond two standard deviations of the drift: a mode forming along that
        direction is a small tail there, and its lag barely moves the mean.
        """
        endpoint, below = sampler.models[-1], sampler.models[-2]
        drift = device_accumulation(compute_energy(
            samples, {key: endpoint[key] - below[key] for key in ("bias", "coupling_matrix")}
        ))
        names, observables = ["drift"], [drift]
        drift_spread = drift.std(unbiased=False)
        if drift_spread > 1e-12 * drift.abs().max().clamp_min(1.0):
            standardized = (drift - drift.mean()) / drift_spread
            names += ["drift_low", "drift_high"]
            observables += [device_accumulation(standardized < -2.0), device_accumulation(standardized > 2.0)]
        values = torch.stack(observables)
        spread = values.std(1, unbiased=False)
        keep = spread > 1e-12 * values.abs().amax(1).clamp_min(1.0)
        names = [name for name, kept in zip(names, keep.tolist()) if kept]
        return names, (values[keep] - values[keep].mean(1, keepdim=True)) / spread[keep].unsqueeze(1)

    @staticmethod
    def _lags(sampler, z):
        """Current lag of each standardized observable: its covariance with the lag memory."""
        memory = sampler.lag_memory
        if memory is None or memory.shape != (len(sampler.chains), z.shape[1]):
            return torch.zeros(len(z), dtype=accumulation_dtype(z.device), device=z.device)
        return z @ memory[-1].to(z) / z.shape[1]

    @staticmethod
    def _grouped_lags(lags):
        """Largest absolute lag of the drift and of its tails."""
        groups = {"drift": [], "drift_tail": []}
        for name, value in lags.items():
            groups["drift_tail" if name.startswith("drift_") else name].append(abs(value))
        return {group: max(values) if values else None for group, values in groups.items()}

    def _measure_lags(self, sampler):
        names, z = self._lag_observables(sampler, sampler.endpoint_samples(copy=False))
        return dict(zip(names, self._lags(sampler, z).tolist()))

    def current_lags(self, sampler):
        """Current lag of the endpoint chains per signal (drift, drift tail)."""
        return self._grouped_lags(self._measure_lags(sampler))

    def current_lag(self, sampler):
        """Largest current lag of the endpoint chains over the monitored observables."""
        values = [value for value in self.current_lags(sampler).values() if value is not None]
        return max(values) if values else 0.0

    def wants_pause(self, sampler):
        """Whether training should pause and equilibrate before the next update."""
        if self.kind != "adaptive" or self.pause_cooldown > 0:
            return False
        return self.current_lag(sampler) > self.lag_tolerance

    def _record_lags(self, sampler):
        lags = self._measure_lags(sampler)
        observable = max(lags, key=lambda name: abs(lags[name])) if lags else None
        self.lag_diagnostics = {
            "lag": abs(lags[observable]) if lags else 0.0,
            "observable": observable,
            "lags": lags,
        }
        self.pause_cooldown = max(0, self.pause_cooldown - 1)

    def step(self, *, fi, fij, pi, pij, params, mask, l2_regularization, sampler=None):
        gradients = {
            "bias": fi - pi,
            "coupling_matrix": fij - pij - l2_regularization * params["coupling_matrix"],
        }
        gradients["coupling_matrix"] = gradients["coupling_matrix"] * mask
        if self.kind == "adaptive":
            if sampler is None:
                raise InputValidationError("The adaptive PTT optimizer requires its PTT sampler.")
            self._adapt(gradients, sampler.endpoint_samples(copy=False), l2_regularization)
            self._record_lags(sampler)
        updated = {key: value.clone() for key, value in params.items()}
        with torch.no_grad():
            for key in self._keys:
                updated[key] += self.rates[key] * gradients[key]
            updated["coupling_matrix"] *= mask
        return updated

    def halve(self):
        self.rates = {key: rate * 0.5 for key, rate in self.rates.items()}
        self.nominal_rates = {key: rate * 0.5 for key, rate in self.nominal_rates.items()}

    @property
    def minimum_rate(self):
        return min(self.rates.values())

    @property
    def minimum_nominal_rate(self):
        return min(self.nominal_rates.values())




class _PTTEdgeOptimizer(_PTTOptimizer):
    """One edge correction per step, with its pseudocount bounded by the PTT KL radius."""

    def __init__(self, config, nominal_pseudocount, state=None):
        self.kind = config.ptt.optimizer
        self.trust_radius = config.ptt.trust_radius
        self.lag_tolerance = config.ptt.lag_tolerance
        self.nominal_pseudocount = float(nominal_pseudocount)
        self.pseudocount = float(nominal_pseudocount)
        self.trust_diagnostics = None
        self.lag_diagnostics = None
        self.pause_cooldown = 0
        self.last_pause = None
        self.last_edge = None
        self.last_edge_new = None
        if state is not None:
            self.load_state_dict(state)

    def state_dict(self):
        return {
            "kind": self.kind, "trust_radius": self.trust_radius,
            "nominal_pseudocount": self.nominal_pseudocount,
            "pseudocount": self.pseudocount,
            "trust_diagnostics": self.trust_diagnostics,
            "lag_diagnostics": self.lag_diagnostics,
            "pause_cooldown": self.pause_cooldown,
            "last_pause": self.last_pause,
            "last_edge": self.last_edge,
            "last_edge_new": self.last_edge_new,
        }

    def load_state_dict(self, state):
        if state["kind"] != self.kind or state["trust_radius"] != self.trust_radius:
            raise InputValidationError("PTT edge optimizer state is incompatible with the requested configuration.")
        self.nominal_pseudocount = float(state["nominal_pseudocount"])
        self.pseudocount = float(state["pseudocount"])
        self.trust_diagnostics = state["trust_diagnostics"]
        self.lag_diagnostics = state["lag_diagnostics"]
        self.pause_cooldown = int(state["pause_cooldown"])
        self.last_pause = state["last_pause"]
        self.last_edge = state["last_edge"]
        self.last_edge_new = state["last_edge_new"]

    @property
    def minimum_rate(self):
        return 1.0 - self.pseudocount

    @property
    def minimum_nominal_rate(self):
        return 1.0 - self.nominal_pseudocount

    def halve(self):
        self.nominal_pseudocount = 1.0 - 0.5 * self.minimum_nominal_rate
        self.pseudocount = self.nominal_pseudocount

    @staticmethod
    def _smoothed(frequencies, alpha):
        q = frequencies.shape[1]
        return ((1.0 - alpha) * frequencies + alpha / (q * q)).clamp_min(
            torch.finfo(frequencies.dtype).tiny
        )

    @staticmethod
    def _candidate_kl(pair, correction):
        """Exact one-edge tilt KL given the old model's pair marginal."""
        pair = diagnostic_double(pair).clamp_min(1e-12)
        pair = pair / pair.sum()
        log_pair = pair.log()
        tilted_log = log_pair + diagnostic_double(correction)
        log_normalizer = torch.logsumexp(tilted_log.reshape(-1), dim=0)
        tilted = (tilted_log - log_normalizer).exp()
        return float((tilted * (diagnostic_double(correction) - log_normalizer)).sum())

    def step(self, *, fij_raw, pij_raw, params, mask, sampler=None):
        alpha = self.nominal_pseudocount
        if alpha <= 0.0 or alpha >= 1.0:
            raise InputValidationError("PTT edgeDCA requires a pseudocount strictly between 0 and 1.")
        target = self._smoothed(fij_raw, alpha)
        model = self._smoothed(pij_raw, alpha)
        score = compute_Dkl_edge_activation(target, model)
        index = int(score.argmax())
        length = score.shape[1]
        i, j = divmod(index, length)
        if not torch.isfinite(score[i, j]):
            raise InputValidationError("edgeDCA requires at least two sites for an edge update.")
        was_new = not bool(mask[i, 0, j, 0])
        pair_target = fij_raw[i, :, j, :]
        pair_model = pij_raw[i, :, j, :]

        def correction_at(value):
            return (self._smoothed(pair_target, value).log()
                    - self._smoothed(pair_model, value).log())

        correction = correction_at(alpha)
        kl = self._candidate_kl(pair_model, correction)
        if self.kind == "adaptive":
            # Recompute the actual log-ratio update after each increase in alpha.
            # Its dependence on alpha is nonlinear, so check each candidate.
            for _ in range(50):
                if kl <= self.trust_radius:
                    break
                alpha = 1.0 - 0.5 * (1.0 - alpha)
                correction = correction_at(alpha)
                kl = self._candidate_kl(pair_model, correction)
            else:
                raise InputValidationError("PTT edge update could not fit its KL trust radius.")
            if sampler is None:
                raise InputValidationError("The adaptive PTT edge optimizer requires its PTT sampler.")
            self._record_lags(sampler)
        updated = {key: value.clone() for key, value in params.items()}
        updated_mask = mask.clone()
        updated_mask[i, :, j, :] = True
        updated_mask[j, :, i, :] = True
        updated["coupling_matrix"][i, :, j, :] += correction
        updated["coupling_matrix"][j, :, i, :] += correction.T
        self.pseudocount = alpha
        self.last_edge = [i, j]
        self.last_edge_new = was_new
        self.trust_diagnostics = {"predicted_kl": kl}
        return updated, updated_mask

