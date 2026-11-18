"""Local Potts updates on hard support."""
from __future__ import annotations

import numpy as np
import torch

from adabmDCA.statmech import compute_energy

try:
    from numba import njit, prange
except ImportError:
    njit = None


if njit is not None:
    @njit(parallel=True, cache=True)
    def _masked_gibbs(states, fields, rows, increments, allowed, minimum, maximum, totals, sites, uniforms):
        length, q = fields.shape
        for chain in prange(len(states)):
            logits = np.empty(q, dtype=np.float64)
            for step, site in enumerate(sites):
                old = states[chain, site]
                maximum_logit = -np.inf
                for token in range(q):
                    valid = allowed[token]
                    for k in range(len(minimum)):
                        distance = totals[chain, k] + increments[site, token, k] - increments[site, old, k]
                        valid &= minimum[k] <= distance <= maximum[k]
                    logit = -np.inf
                    if valid:
                        logit = fields[site, token]
                        for other in range(length):
                            logit += rows[site, token, other, states[chain, other]]
                        maximum_logit = max(maximum_logit, logit)
                    logits[token] = logit
                total = 0.0
                for token in range(q):
                    total += np.exp(logits[token] - maximum_logit)
                    logits[token] = total
                chosen = np.searchsorted(logits, total * uniforms[chain, step], side="right")
                states[chain, site] = chosen
                for k in range(len(minimum)):
                    totals[chain, k] += increments[site, chosen, k] - increments[site, old, k]

    @njit(parallel=True, cache=True)
    def _energy(states, fields, rows):
        result = np.empty(len(states), dtype=np.float64)
        for chain in prange(len(states)):
            score = 0.0
            for site, token in enumerate(states[chain]):
                score += fields[site, token]
                for other in range(site):
                    score += rows[site, token, other, states[chain, other]]
            result[chain] = -score
        return result


class PottsKernel:
    def __init__(self, params, support):
        bias, coupling = params["bias"], params["coupling_matrix"]
        if bias.dtype not in (torch.float32, torch.float64):
            bias, coupling = bias.float(), coupling.float()
        # compute_energy sees only the symmetric part and the selected
        # diagonal token. Move its diagonal self energy into the fields.
        rows = (coupling + coupling.permute(2, 3, 0, 1)) * 0.5
        sites = torch.arange(len(bias), device=bias.device)
        diagonal = rows[sites, :, sites, :].diagonal(dim1=-1, dim2=-2)
        self.bias = (bias + 0.5 * diagonal).contiguous()
        rows[sites, :, sites, :] = 0
        self.rows = rows.contiguous()
        self.coupled = bool(torch.count_nonzero(self.rows))
        self.support = support
        self.device = bias.device
        self.cpu = self.device.type == "cpu" and njit is not None
        self.cuda = False
        if self.device.type == "cuda" and bias.dtype == torch.float32:
            from adabmDCA.cuda_kernels.sampling_triton import triton

            self.cuda = triton is not None
            if self.cuda:
                self.rows_transposed = self.rows.permute(0, 2, 3, 1).contiguous()
                self.bounds = (tuple(map(int, support.minimum)), tuple(map(int, support.maximum)))
        if self.cpu:
            self.bias_array = self.bias.detach().numpy()
            self.rows_array = self.rows.detach().numpy()
        else:
            self.increments = torch.as_tensor(support.increments, device=self.device)
            self.allowed = torch.as_tensor(support.allowed, device=self.device)
            self.minimum = torch.as_tensor(support.minimum, device=self.device)
            self.maximum = torch.as_tensor(support.maximum, device=self.device)

    def energy(self, states):
        if not self.coupled and self.cpu:
            return -self.bias_array[np.arange(self.support.length), states].sum(axis=1, dtype=np.float64)
        if self.cpu:
            return _energy(states, self.bias_array, self.rows_array)
        encoded = torch.as_tensor(states, device=self.device, dtype=torch.long)
        onehot = torch.nn.functional.one_hot(encoded, self.support.q).to(self.bias.dtype)
        return compute_energy(onehot, {"bias": self.bias, "coupling_matrix": self.rows}).cpu().double().numpy()

    def propose(self, states, beta, field_offsets, sites, uniforms):
        states = states.copy()
        support = self.support
        totals = support.distances(states)
        if self.cpu:
            fields = beta * self.bias_array - field_offsets
            _masked_gibbs(states, fields, self.rows_array if beta == 1 else self.scaled_rows(beta),
                          support.increments, support.allowed, support.minimum, support.maximum,
                          totals, sites, uniforms)
            return states
        return self._torch_proposal(states, beta, field_offsets, sites, totals, uniforms)

    def propose_groups(self, states, beta, field_offsets, groups, schedules, uniforms):
        if not self.cuda:
            return np.concatenate([
                self.propose(states[group], beta, field_offsets, sites, random)
                for group, sites, random in zip(groups, schedules, uniforms)
            ])
        from adabmDCA.cuda_kernels.sampling_triton import steered_gibbs_steps_triton

        def tensor(value, dtype):
            return torch.as_tensor(value, dtype=dtype, device=self.device).contiguous()

        samples = tensor(states, torch.int32)
        sites = tensor(np.stack(schedules, axis=1), torch.int32)
        draws = tensor(np.concatenate(uniforms).T, self.bias.dtype)
        offsets = tensor(field_offsets, self.bias.dtype)
        totals = tensor(self.support.distances(states), torch.int32)
        minimum, maximum = self.bounds
        steered_gibbs_steps_triton(samples, self.bias, self.rows_transposed, sites, draws, offsets,
                                  self.increments, totals, self.allowed, minimum, maximum, beta)
        return samples.cpu().numpy()

    def scaled_rows(self, beta):
        # Repeated rungs at one temperature share the same transformed couplings.
        if not hasattr(self, "_scaled"):
            self._scaled = {}
        if beta not in self._scaled:
            self._scaled[beta] = np.ascontiguousarray(self.rows_array * beta)
        return self._scaled[beta]

    def _torch_proposal(self, states, beta, field_offsets, sites, totals, uniforms):
        states = torch.as_tensor(states, dtype=torch.long, device=self.device)
        totals = torch.as_tensor(totals, device=self.device)
        uniforms = torch.as_tensor(uniforms.T.copy(), device=self.device, dtype=self.bias.dtype)
        fields = beta * self.bias - torch.as_tensor(field_offsets, device=self.device, dtype=self.bias.dtype)
        onehot = torch.nn.functional.one_hot(states, self.support.q).to(self.bias.dtype)
        flat = onehot.flatten(1)
        rows = self.rows.flatten(2)
        for step, site in enumerate(sites):
            proposed = totals[:, None] + self.increments[site] - self.increments[site, states[:, site], None]
            valid = ((proposed >= self.minimum) & (proposed <= self.maximum)).all(dim=-1) & self.allowed
            logits = fields[site] + beta * (flat @ rows[site].T)
            probabilities = logits.masked_fill(~valid, -torch.inf).softmax(-1)
            cdf = probabilities.cumsum(-1)
            chosen = (uniforms[step, :, None] * cdf[:, -1:] >= cdf).sum(dim=-1)
            totals = proposed[torch.arange(len(states), device=self.device), chosen]
            states[:, site] = chosen
            onehot[:, site] = torch.nn.functional.one_hot(chosen, self.support.q)
        return states.cpu().numpy().astype(np.int32)
