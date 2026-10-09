"""Edge cases of the Triton samplers: padded alphabets at beta <= 0, strided and empty inputs, parameter updates."""

import pytest
import torch

from adabmDCA import sampling_triton as triton_sampling

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def fixture(dtype, n=33, length=17, q=5):
    torch.manual_seed(193)
    indices = torch.randint(q, (n, length), device="cuda", dtype=torch.int32)
    # Deliberately asymmetric, nonzero diagonal: do not rely on model conventions.
    params = {
        "bias": torch.randn(length, q, device="cuda", dtype=dtype) * 0.2,
        "coupling_matrix": torch.randn(length, q, length, q, device="cuda", dtype=dtype) * 0.1,
    }
    return indices, params


@cuda
def test_gibbs_zero_and_negative_beta_with_padding():
    """q = 5 is padded to a power of two inside the kernel; padding must stay masked at beta <= 0."""
    initial, params = fixture(torch.float64)
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
            triton_sampling._gibbs_steps_triton(actual, params, sites, uniforms, beta, transpose_couplings=transpose)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("sampler, independent", [
    (triton_sampling.gibbs_sampling_triton, False),
    (triton_sampling.metropolis_sampling_triton, False),
    (triton_sampling.gibbs_step_independent_sites_triton, True),
    (triton_sampling.metropolis_step_independent_sites_triton, True),
])
def test_strided_inputs_match_contiguous_ones_and_empty_batches_work(sampler, independent):
    initial, params = fixture(torch.float32)
    contiguous = torch.nn.functional.one_hot(initial.long(), 5).float()
    strided = contiguous.transpose(0, 1).contiguous().transpose(0, 1)
    assert not strided.is_contiguous()
    args = () if independent else (2,)
    torch.manual_seed(71)
    expected = sampler(contiguous.clone(), params, *args)
    torch.manual_seed(71)
    actual = sampler(strided.clone(), params, *args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sampler(contiguous[:0], params, *args).shape == contiguous[:0].shape


@cuda
def test_parameter_changes_are_not_cached():
    """Couplings modified in place after a call must be used by the next call, not a stale copy."""
    initial, params = fixture(torch.float32)
    chains = torch.nn.functional.one_hot(initial.long(), 5).float()
    triton_sampling.gibbs_sampling_triton(chains, params, 1)
    params["coupling_matrix"].mul_(3)
    fresh = {key: value.clone() for key, value in params.items()}
    torch.manual_seed(71)
    expected = triton_sampling.gibbs_sampling_triton(chains, fresh, 2)
    torch.manual_seed(71)
    actual = triton_sampling.gibbs_sampling_triton(chains, params, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
