import unittest
from unittest.mock import patch

import torch

from adabmDCA.sampling import gibbs_sampling, metropolis_sampling, prepare_sampler


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.length = 4
        self.states = 3
        indices = torch.randint(self.states, (128, self.length))
        self.chains = torch.nn.functional.one_hot(indices, self.states).float()
        coupling = torch.randn(self.length, self.states, self.length, self.states) * 0.1
        coupling = 0.5 * (coupling + coupling.permute(2, 3, 0, 1))
        site = torch.arange(self.length)
        coupling[site, :, site, :] = 0
        self.params = {"bias": torch.randn(self.length, self.states) * 0.1, "coupling_matrix": coupling}

    def test_samplers_return_one_hot_without_mutating_input(self):
        original = self.chains.clone()
        for sampler in (gibbs_sampling, metropolis_sampling):
            sampled = sampler(self.chains, self.params, nsweeps=3, beta=0.7)
            torch.testing.assert_close(sampled.sum(dim=-1), torch.ones_like(sampled[..., 0]))
            self.assertTrue(torch.all((sampled == 0) | (sampled == 1)))
            torch.testing.assert_close(self.chains, original)

    def test_sampling_matches_original_implementation(self):
        from adabmDCA.sampling import gibbs_step_uniform_sites, metropolis_step_uniform_sites

        cases = ((gibbs_sampling, gibbs_step_uniform_sites), (metropolis_sampling, metropolis_step_uniform_sites))
        for sampler, step in cases:
            torch.manual_seed(23)
            expected = step(self.chains.clone(), self.params, beta=0.8)
            torch.manual_seed(23)
            actual = sampler(self.chains, self.params, nsweeps=1, beta=0.8)
            # sampler performs L updates; compare it against L baseline updates.
            torch.manual_seed(23)
            expected = self.chains.clone()
            for _ in range(self.length):
                expected = step(expected, self.params, beta=0.8)
            torch.testing.assert_close(actual, expected)

    def test_samplers_remain_torchscript_compatible(self):
        for sampler in (gibbs_sampling, metropolis_sampling):
            scripted = torch.jit.script(sampler)
            result = scripted(self.chains, self.params, 1, 1.0)
            self.assertEqual(result.shape, self.chains.shape)

    def test_prepare_sampler_uses_scripted_cpu_fallback(self):
        sampler = prepare_sampler("metropolis", torch.device("cpu"))
        result = sampler(self.chains, self.params, 1, 1.0)
        self.assertEqual(result.shape, self.chains.shape)

    def test_mixing_progress_closes_before_not_reached_message(self):
        from adabmDCA.resampling import compute_mixing_time

        events = []

        class ProgressBar:
            def set_description(self, description):
                events.append(("description", description))

            def update(self, amount):
                events.append(("update", amount))

            def close(self):
                events.append(("close", None))

        def record_message(message):
            events.append(("print", message))

        with (
            patch("adabmDCA.resampling.tqdm", return_value=ProgressBar()),
            patch("builtins.print", side_effect=record_message),
        ):
            compute_mixing_time(
                sampler=lambda **kwargs: kwargs["chains"],
                data=self.chains[:4],
                params=self.params,
                n_max_sweeps=1,
                beta=1.0,
            )

        self.assertEqual(events[-2][0], "close")
        self.assertEqual(events[-1], ("print", "Mixing time not reached within 0 sweeps."))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton sampling tests")
class TritonSamplingTests(unittest.TestCase):
    def test_public_independent_site_kernels_update_chains_in_place(self):
        from adabmDCA.sampling_triton import (
            gibbs_step_independent_sites_triton,
            is_triton_available,
            metropolis_step_independent_sites_triton,
        )

        if not is_triton_available():
            self.skipTest("Triton is unavailable")
        device = torch.device("cuda")
        num_chains, length, num_states = 64, 11, 4
        states = torch.randint(num_states, (num_chains, length), device=device)
        original = torch.nn.functional.one_hot(states, num_states).float()
        params = {
            "bias": torch.randn(length, num_states, device=device) * 0.1,
            "coupling_matrix": torch.randn(length, num_states, length, num_states, device=device) * 0.03,
        }
        for step in (gibbs_step_independent_sites_triton, metropolis_step_independent_sites_triton):
            chains = original.clone()
            pointer = chains.data_ptr()
            returned = step(chains, params, beta=0.8)
            self.assertEqual(returned.data_ptr(), pointer)
            torch.testing.assert_close(returned.sum(dim=-1), torch.ones_like(returned[..., 0]))

    def test_fused_independent_site_steps_match_controlled_pytorch_updates(self):
        from adabmDCA.sampling_triton import (
            _gibbs_step_independent_triton,
            _metropolis_step_independent_triton,
            is_triton_available,
        )

        if not is_triton_available():
            self.skipTest("Triton is unavailable")
        torch.manual_seed(37)
        device = torch.device("cuda")
        num_chains, length, num_states = 257, 17, 5
        initial = torch.randint(num_states, (num_chains, length), device=device, dtype=torch.int32)
        bias = torch.randn(length, num_states, device=device) * 0.1
        coupling = torch.randn(length, num_states, length, num_states, device=device) * 0.03
        params = {"bias": bias, "coupling_matrix": coupling}
        sites = torch.randint(length, (num_chains,), device=device, dtype=torch.int32)
        uniforms = torch.rand(num_chains, device=device)
        batch = torch.arange(num_chains, device=device)
        one_hot = torch.nn.functional.one_hot(initial.to(torch.int64), num_states).float()
        selected_couplings = coupling[sites.to(torch.int64)].reshape(num_chains, num_states, -1)
        fields = bias[sites.to(torch.int64)] + torch.bmm(
            selected_couplings,
            one_hot.reshape(num_chains, -1, 1),
        ).squeeze(-1)

        expected_gibbs = initial.clone()
        probabilities = torch.softmax(0.8 * fields, dim=-1)
        sampled = torch.sum(uniforms[:, None] > torch.cumsum(probabilities, dim=-1), dim=-1).to(torch.int32)
        expected_gibbs[batch, sites.to(torch.int64)] = sampled
        actual_gibbs = initial.clone()
        _gibbs_step_independent_triton(actual_gibbs, params, sites, uniforms, beta=0.8)
        torch.testing.assert_close(actual_gibbs, expected_gibbs, rtol=0, atol=0)

        proposals = torch.randint(num_states, (num_chains,), device=device, dtype=torch.int32)
        expected_metropolis = initial.clone()
        old = initial[batch, sites.to(torch.int64)]
        delta = fields[batch, old.to(torch.int64)] - fields[batch, proposals.to(torch.int64)]
        accepted = uniforms < torch.exp(-0.8 * delta)
        final = torch.where(accepted, proposals, old)
        expected_metropolis[batch, sites.to(torch.int64)] = final
        actual_metropolis = initial.clone()
        _metropolis_step_independent_triton(
            actual_metropolis,
            params,
            sites,
            proposals,
            uniforms,
            beta=0.8,
        )
        torch.testing.assert_close(actual_metropolis, expected_metropolis, rtol=0, atol=0)

    def test_fused_gibbs_steps_match_controlled_categorical_sampling(self):
        from adabmDCA.sampling_triton import _gibbs_steps_triton, is_triton_available

        if not is_triton_available():
            self.skipTest("Triton is unavailable")
        torch.manual_seed(29)
        device = torch.device("cuda")
        num_chains, length, num_states, num_steps = 257, 17, 5, 13
        initial = torch.randint(num_states, (num_chains, length), device=device, dtype=torch.int32)
        bias = torch.randn(length, num_states, device=device) * 0.1
        coupling = torch.randn(length, num_states, length, num_states, device=device) * 0.03
        params = {"bias": bias, "coupling_matrix": coupling}
        sites = torch.randint(length, (num_steps,), device=device, dtype=torch.int32)
        uniforms = torch.rand(num_steps, num_chains, device=device)

        expected = initial.clone()
        for step in range(num_steps):
            site = sites[step]
            one_hot = torch.nn.functional.one_hot(expected.to(torch.int64), num_states).float()
            field = bias[site] + one_hot.reshape(num_chains, -1) @ coupling[site].reshape(num_states, -1).T
            probabilities = torch.softmax(0.8 * field, dim=-1)
            expected[:, site] = torch.sum(
                uniforms[step, :, None] > torch.cumsum(probabilities, dim=-1),
                dim=-1,
            ).to(torch.int32)

        actual = initial.clone()
        _gibbs_steps_triton(actual, params, sites, uniforms, beta=0.8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_fused_steps_match_pytorch_for_controlled_random_numbers(self):
        from adabmDCA.sampling_triton import _metropolis_steps_triton, is_triton_available

        if not is_triton_available():
            self.skipTest("Triton is unavailable")
        torch.manual_seed(31)
        device = torch.device("cuda")
        num_chains, length, num_states, num_steps = 257, 17, 5, 13
        initial = torch.randint(num_states, (num_chains, length), device=device, dtype=torch.int32)
        bias = torch.randn(length, num_states, device=device) * 0.1
        coupling = torch.randn(length, num_states, length, num_states, device=device) * 0.03
        coupling = 0.5 * (coupling + coupling.permute(2, 3, 0, 1))
        diagonal = torch.arange(length, device=device)
        coupling[diagonal, :, diagonal, :] = 0
        params = {"bias": bias, "coupling_matrix": coupling}
        sites = torch.randint(length, (num_steps,), device=device, dtype=torch.int32)
        proposals = torch.randint(num_states, (num_steps, num_chains), device=device, dtype=torch.int32)
        uniforms = torch.rand(num_steps, num_chains, device=device)

        expected = initial.clone()
        batch = torch.arange(num_chains, device=device)
        for step in range(num_steps):
            site = sites[step]
            old = expected[:, site]
            proposed = proposals[step]
            one_hot = torch.nn.functional.one_hot(expected.to(torch.int64), num_states).float()
            field = bias[site] + one_hot.reshape(num_chains, -1) @ coupling[site].reshape(num_states, -1).T
            delta = field[batch, old] - field[batch, proposed]
            accepted = uniforms[step] < torch.exp(-0.8 * delta)
            expected[:, site] = torch.where(accepted, proposed, old)

        actual = initial.clone()
        _metropolis_steps_triton(actual, params, sites, proposals, uniforms, beta=0.8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
