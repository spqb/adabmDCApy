"""Parameter updates of PTT training: bias/coupling gradient ascent."""

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
        self.trust_radius = ptt.effective_trust_radius
        self.lag_tolerance = ptt.lag_tolerance
        self.centering = ptt.centering
        self.natural_gradient = ptt.natural_gradient
        self.ngd_damping = ptt.effective_ngd_damping
        self.ngd_cg_steps = ptt.ngd_cg_steps
        self.ngd_cg_tolerance = ptt.ngd_cg_tolerance
        self.ngd_diagnostics = None
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
            "ngd_diagnostics": self.ngd_diagnostics,
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
        self.ngd_diagnostics = state.get("ngd_diagnostics")
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

    def _adapt(self, gradients, samples, l2_regularization, steps=None, damping=0.0, offset=None):
        """Scale the two group rates within the KL trust region around the nominal rates.

        ``steps`` are the unscaled update directions (the gradients by default); ``damping`` adds
        ``damping * |step|^2`` to the curvature, as in the damped natural-gradient system. With the
        centring ``offset`` mu the norm is that of the centred coordinates, where the field step is
        dh + dJ mu.
        """
        steps = gradients if steps is None else steps
        directions = {key: self.nominal_rates[key] * steps[key] for key in self._keys}
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
        damping_bias = float(damping * device_accumulation(directions["bias"]).square().sum())
        damping_coupling = float(
            0.5 * damping * device_accumulation(directions["coupling_matrix"]).square().sum()
        )
        if offset is not None and damping:
            size = offset.numel()
            shift = directions["coupling_matrix"].reshape(size, size) @ offset
            covariance += float(damping * device_accumulation(directions["bias"].flatten() * shift).sum())
            damping_coupling += float(damping * device_accumulation(shift).square().sum())
        variance_bias += damping_bias
        variance_coupling += damping_coupling
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
            "damping_curvature": damping_bias + damping_coupling,
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

    @staticmethod
    def _centered(gradients, *, fi, pi, mask):
        """Centering trick: the gradients of the energy written in x - mu, with mu = (fi + pi) / 2, mapped back.

        Summed over the partner site, the raw coupling gradient is (L - 1) times the field gradient: the
        coupling step repeats the field step, and the two are almost collinear. Centred, the coupling
        gradient is the difference of connected correlations, Cij_data - Cij_model, which has no one-site
        part, and the field step absorbs the shift of the centred parametrization:
        dh_i(a) = (fi - pi)_i(a) - sum_{j,b} dJ_ij(a, b) mu_j(b).
        """
        offset = 0.5 * (fi + pi)
        coupling = (gradients["coupling_matrix"]
                    - (torch.einsum("ia,jb->iajb", fi, fi) - torch.einsum("ia,jb->iajb", pi, pi))) * mask
        bias = gradients["bias"] - torch.einsum("iajb,jb->ia", coupling, offset)
        return {"bias": bias, "coupling_matrix": coupling}

    def _natural_step(self, gradients, samples, mask, l2_regularization, offset=None):
        """Damped natural gradient: solve (F + l2 + damping) step = gradient by conjugate gradient.

        F is the Fisher matrix of the Potts model, the covariance over the endpoint chains of the
        features x_i(a) and x_i(a) x_j(b). Its products are two matrix products with the chains, so F
        is never stored. The couplings are stored twice (J_ij and J_ji), so their entries count with
        weight 1/2 in the inner products, as in the predicted gain and KL of the trust region. The
        damping is ``ngd_damping`` times the mean eigenvalue of F, tr(F) / (number of parameters):
        the chains span at most as many directions as there are chains, and the damping is the
        curvature assigned to the directions they do not see.

        With the centring ``offset`` mu the system is solved in the centred coordinates, whose pair
        features are (x - mu)_i(a) (x - mu)_j(b): ``gradients`` are then the centred ones (field
        gradient, connected-correlation coupling gradient), and the step is mapped back with
        dh = dh' - dJ mu. F + l2 alone is the same in both coordinates; the damping is not: in the
        centred ones, large damping tends to the centred gradient instead of the raw one.
        """
        x = samples.flatten(1).to(gradients["bias"].dtype)
        n, size = x.shape
        y = x if offset is None else x - offset.to(x.dtype)
        weights = mask.reshape(size, size).to(x.dtype)
        p = x.mean(0)
        trace = float(device_accumulation(p * (1 - p)).sum())
        pair = (y.T @ y) / n
        pair_square = pair if offset is None else (y * y).T @ (y * y) / n  # one-hot: x^2 = x
        trace += 0.5 * float(device_accumulation((pair_square - pair * pair) * weights).sum())
        parameters = size + 0.5 * float(device_accumulation(weights).sum())
        damping = self.ngd_damping * trace / parameters

        def product(h, j):
            score = x @ h + 0.5 * ((y @ j) * y).sum(1)
            score = score - score.mean()
            return (x.T @ score / n + damping * h,
                    ((y * score.unsqueeze(1)).T @ y / n) * weights + (damping + l2_regularization) * j)

        def dot(a, b):
            return float(device_accumulation(a[0] * b[0]).sum() + 0.5 * device_accumulation(a[1] * b[1]).sum())

        g = (gradients["bias"].reshape(size), gradients["coupling_matrix"].reshape(size, size) * weights)
        step = (torch.zeros_like(g[0]), torch.zeros_like(g[1]))
        residual = (g[0].clone(), g[1].clone())
        direction = (g[0].clone(), g[1].clone())
        rr = norm = dot(g, g)
        iterations = 0
        for _ in range(self.ngd_cg_steps):
            if rr <= self.ngd_cg_tolerance ** 2 * norm:
                break
            image = product(*direction)
            curvature = dot(direction, image)
            if not curvature > 0:
                break
            alpha = rr / curvature
            step = (step[0] + alpha * direction[0], step[1] + alpha * direction[1])
            residual = (residual[0] - alpha * image[0], residual[1] - alpha * image[1])
            rr, previous = dot(residual, residual), rr
            beta = rr / previous
            direction = (residual[0] + beta * direction[0], residual[1] + beta * direction[1])
            iterations += 1
        coupling = step[1].reshape(mask.shape)
        # Exactly symmetric: the float rounding of the products is not.
        coupling = 0.5 * (coupling + coupling.permute(2, 3, 0, 1)) * mask
        flat = (step[0], coupling.reshape(size, size))
        bias = step[0] if offset is None else step[0] - flat[1] @ offset.to(x.dtype)
        self.ngd_diagnostics = {
            "damping": damping,
            "cg_iterations": iterations,
            "cg_residual": math.sqrt(rr / norm) if norm > 0 else 0.0,
            # Angle between the natural and the plain gradient (in the coordinates of the solve)
            "cosine": dot(flat, g) / math.sqrt(max(dot(flat, flat) * norm, 1e-300)),
        }
        return {"bias": bias.reshape(gradients["bias"].shape), "coupling_matrix": coupling}, damping

    def step(self, *, fi, fij, pi, pij, params, mask, l2_regularization, sampler=None):
        gradients = {
            "bias": fi - pi,
            "coupling_matrix": fij - pij - l2_regularization * params["coupling_matrix"],
        }
        gradients["coupling_matrix"] = gradients["coupling_matrix"] * mask
        offset = None
        if self.centering and not self.natural_gradient:
            gradients = self._centered(gradients, fi=fi, pi=pi, mask=mask)
        steps, damping = gradients, 0.0
        if self.natural_gradient:
            if sampler is None:
                raise InputValidationError("The PTT natural gradient requires its PTT sampler.")
            system = gradients
            if self.centering:
                # Solved in the centred coordinates: field gradient, connected-correlation coupling gradient.
                offset = (0.5 * (fi + pi)).flatten()
                system = {"bias": gradients["bias"],
                          "coupling_matrix": self._centered(gradients, fi=fi, pi=pi, mask=mask)["coupling_matrix"]}
            samples = sampler.endpoint_samples(copy=False)
            if samples.shape[0] > 1:
                # The solve fits the step to its chains (F has at most as many directions): the
                # trust region measures it on the other half.
                solve_samples, held_out_samples = samples[0::2], samples[1::2]
            else:
                solve_samples = held_out_samples = samples
            steps, damping = self._natural_step(system, solve_samples, mask, l2_regularization, offset=offset)
        if self.kind == "adaptive":
            if sampler is None:
                raise InputValidationError("The adaptive PTT optimizer requires its PTT sampler.")
            trust_samples = held_out_samples if self.natural_gradient else sampler.endpoint_samples(copy=False)
            self._adapt(gradients, trust_samples, l2_regularization, steps=steps, damping=damping, offset=offset)
            self._record_lags(sampler)
        updated = {key: value.clone() for key, value in params.items()}
        with torch.no_grad():
            for key in self._keys:
                updated[key] += self.rates[key] * steps[key]
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
