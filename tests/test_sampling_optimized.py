"""Regression coverage against the frozen pre-optimization implementations."""

import unittest

import torch

from adabmDCA import sampling
from adabmDCA import sampling_triton as new
from benchmarks import sampling_triton_baseline as old
from benchmarks.benchmark_sampling import baseline_gibbs, baseline_metropolis


def fixture(device, dtype, n=33, length=17, q=5):
    torch.manual_seed(193)
    indices = torch.randint(q, (n, length), device=device, dtype=torch.int32)
    # Deliberately asymmetric, nonzero diagonal: do not rely on model conventions.
    params = {
        "bias": torch.randn(length, q, device=device, dtype=dtype) * 0.2,
        "coupling_matrix": torch.randn(length, q, length, q, device=device, dtype=dtype) * 0.1,
    }
    return indices, params


class TorchRegressionTests(unittest.TestCase):
    def test_exact_seeded_trajectories_and_rng_state(self):
        for dtype in (torch.float32, torch.float64):
            indices, params = fixture("cpu", dtype)
            chains = torch.nn.functional.one_hot(indices.long(), 5).to(dtype)
            original = chains.clone()
            for before, after in (
                (baseline_gibbs, sampling.gibbs_sampling),
                (baseline_metropolis, sampling.metropolis_sampling),
            ):
                for scripted in (False, True):
                    old_fn = torch.jit.script(before) if scripted else before
                    new_fn = torch.jit.script(after) if scripted else after
                    for sweeps in (0, 3):
                        with self.subTest(dtype=dtype, sampler=after.__name__, scripted=scripted, sweeps=sweeps):
                            torch.manual_seed(81)
                            expected = old_fn(chains, params, sweeps, 0.8)
                            rng = torch.get_rng_state()
                            torch.manual_seed(81)
                            actual = new_fn(chains, params, sweeps, 0.8)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                            torch.testing.assert_close(chains, original, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available() and new.is_triton_available(), "CUDA and Triton required")
class TritonRegressionTests(unittest.TestCase):
    def test_prepare_sampler_uses_new_kernels(self):
        self.assertIs(sampling.prepare_sampler("gibbs", torch.device("cuda")), new.gibbs_sampling_triton)
        self.assertIs(sampling.prepare_sampler("metropolis", torch.device("cuda")), new.metropolis_sampling_triton)

    def test_controlled_updates_across_layouts_chunks_and_dtypes(self):
        for dtype in (torch.float32, torch.float64):
            for length, q in ((1, 1), (17, 5), (33, 21)):
                initial, params = fixture("cuda", dtype, length=length, q=q)
                steps = 35  # multiple chunks plus a partial final chunk
                sites = torch.randint(length, (steps,), device="cuda", dtype=torch.int32)
                sites[1:4] = 0  # revisiting a site must observe its updated state
                uniforms = torch.rand(steps, initial.shape[0], device="cuda", dtype=dtype)
                proposals = torch.randint(q, uniforms.shape, device="cuda", dtype=torch.int32)
                expected_g, expected_m = initial.clone(), initial.clone()
                old._gibbs_steps_triton(expected_g, params, sites, uniforms, 0.8)
                old._metropolis_steps_triton(expected_m, params, sites, proposals, uniforms, 0.8)
                for chunk in (1, 4, 16, 64):
                    for transpose in (False, True):
                        with self.subTest(dtype=dtype, length=length, q=q, chunk=chunk, transpose=transpose):
                            actual = initial.clone()
                            new._gibbs_steps_triton(
                                actual,
                                params,
                                sites,
                                uniforms,
                                0.8,
                                steps_per_launch=chunk,
                                transpose_couplings=transpose,
                            )
                            torch.testing.assert_close(actual, expected_g, rtol=0, atol=0)
                    actual = initial.clone()
                    new._metropolis_steps_triton(
                        actual, params, sites, proposals, uniforms, 0.8, steps_per_launch=chunk
                    )
                    torch.testing.assert_close(actual, expected_m, rtol=0, atol=0)

    def test_public_seeded_trajectories_rng_and_input_preservation(self):
        for dtype in (torch.float32, torch.float64):
            initial, params = fixture("cuda", dtype)
            chains = torch.nn.functional.one_hot(initial.long(), 5).to(dtype)
            original = chains.clone()
            for before, after in (
                (old.gibbs_sampling_triton, new.gibbs_sampling_triton),
                (old.metropolis_sampling_triton, new.metropolis_sampling_triton),
            ):
                for sweeps in (0, 3):
                    torch.manual_seed(81)
                    expected = before(chains, params, sweeps, 0.8)
                    rng = torch.cuda.get_rng_state()
                    torch.manual_seed(81)
                    actual = after(chains, params, sweeps, 0.8)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(torch.cuda.get_rng_state(), rng, rtol=0, atol=0)
                    torch.testing.assert_close(chains, original, rtol=0, atol=0)
                    self.assertNotEqual(actual.data_ptr(), chains.data_ptr())

    def test_gibbs_zero_and_negative_beta_with_padding(self):
        initial, params = fixture("cuda", torch.float64)
        n = initial.shape[0]
        sites = torch.tensor([2, 2, 7], device="cuda", dtype=torch.int32)
        uniforms = torch.rand(3, n, device="cuda", dtype=torch.float64)
        uniforms[0, 0] = 0
        uniforms[0, 1] = 1 - torch.finfo(torch.float64).eps
        for beta in (0.0, -0.8):
            expected = initial.clone()
            for step, site in enumerate(sites):
                one_hot = torch.nn.functional.one_hot(expected.long(), 5).double()
                field = params["bias"][site] + one_hot.reshape(n, -1) @ params["coupling_matrix"][site].reshape(5, -1).T
                probabilities = torch.softmax(beta * field, -1)
                expected[:, site] = (uniforms[step, :, None] > probabilities.cumsum(-1)).sum(-1).clamp_max(4).int()
            for transpose in (False, True):
                actual = initial.clone()
                new._gibbs_steps_triton(actual, params, sites, uniforms, beta, transpose_couplings=transpose)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_independent_sites_exact_match(self):
        for dtype in (torch.float32, torch.float64):
            initial, params = fixture("cuda", dtype)
            for before, after in (
                (old.gibbs_step_independent_sites_triton, new.gibbs_step_independent_sites_triton),
                (old.metropolis_step_independent_sites_triton, new.metropolis_step_independent_sites_triton),
            ):
                chains = torch.nn.functional.one_hot(initial.long(), 5).to(dtype)
                torch.manual_seed(81)
                expected = before(chains.clone(), params, 0.8)
                torch.manual_seed(81)
                actual = after(chains, params, 0.8)
                self.assertEqual(actual.data_ptr(), chains.data_ptr())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_strided_inputs_and_empty_batches(self):
        initial, params = fixture("cuda", torch.float32)
        original = torch.nn.functional.one_hot(initial.long(), 5).float()
        strided = original.transpose(0, 1).contiguous().transpose(0, 1)
        self.assertFalse(strided.is_contiguous())
        for before, after, independent in (
            (old.gibbs_sampling_triton, new.gibbs_sampling_triton, False),
            (old.metropolis_sampling_triton, new.metropolis_sampling_triton, False),
            (old.gibbs_step_independent_sites_triton, new.gibbs_step_independent_sites_triton, True),
            (old.metropolis_step_independent_sites_triton, new.metropolis_step_independent_sites_triton, True),
        ):
            args = () if independent else (2,)
            torch.manual_seed(71)
            expected = before(original.clone(), params, *args)
            torch.manual_seed(71)
            actual = after(strided.clone(), params, *args)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            empty = after(original[:0], params, *args)
            self.assertEqual(empty.shape, original[:0].shape)

    def test_parameter_changes_are_not_cached(self):
        initial, params = fixture("cuda", torch.float32)
        chains = torch.nn.functional.one_hot(initial.long(), 5).float()
        new.gibbs_sampling_triton(chains, params, 1)
        params["coupling_matrix"].mul_(3)
        torch.manual_seed(71)
        expected = old.gibbs_sampling_triton(chains, params, 2)
        torch.manual_seed(71)
        actual = new.gibbs_sampling_triton(chains, params, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
