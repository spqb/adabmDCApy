"""BF16 sampling storage with FP32 training state and arithmetic."""

import argparse
from unittest.mock import patch

import pytest
import torch

from adabmDCA import DCAModel, sample_sequences, train_model
from adabmDCA import sampling_triton as kernels
from adabmDCA.alignment import Alignment
from adabmDCA.api.exceptions import InputValidationError
from adabmDCA.parser import add_args_sample, add_args_train
from adabmDCA.sampling import prepare_fixed_model_sampler, prepare_training_sampler
from adabmDCA.training_config import TrainingConfig

CUDA_BF16 = torch.cuda.is_available() and kernels.is_triton_available() and torch.cuda.get_device_capability()[0] >= 8
gpu = pytest.mark.skipif(not CUDA_BF16, reason="Ampere+ CUDA and Triton required")


def alignment():
    generator = torch.Generator().manual_seed(73)
    states = torch.randint(3, (64, 9), generator=generator)
    states[:, 1] = states[:, 0]
    return Alignment(
        names=tuple(f"s{i}" for i in range(64)),
        sequences=tuple("".join("ABC"[x] for x in row) for row in states.tolist()),
    )


def test_config_cli_and_unsupported_device():
    assert TrainingConfig(dtype="bfloat16").dtype == "bfloat16"
    parser = add_args_train(argparse.ArgumentParser())
    assert "bfloat16" in parser.format_help()
    with pytest.raises(InputValidationError, match="CUDA and Triton"):
        train_model(alignment(), dtype="bfloat16", device="cpu", alphabet="ABC")

    sample_parser = add_args_sample(argparse.ArgumentParser())
    assert sample_parser.parse_args(
        [
            "--path_params",
            "model.dat",
            "--output",
            "samples",
            "--ngen",
            "2",
            "--dtype",
            "bfloat16",
        ]
    ).dtype == "bfloat16"
    with pytest.raises(InputValidationError, match="CUDA and Triton"):
        sample_sequences(
            model=DCAModel(
                {"bias": torch.zeros(2, 3), "coupling_matrix": torch.zeros(2, 3, 2, 3)},
                alphabet="ABC",
            ),
            n_sequences=2,
            n_sweeps=1,
            dtype="bfloat16",
        )


@gpu
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("sampler", ["gibbs", "metropolis"])
def test_bf16_kernels_match_fp32_for_identically_rounded_parameters(independent, sampler):
    torch.manual_seed(41)
    n, length, q, steps = 257, 33, 21, 35
    states = torch.randint(q, (n, length), device="cuda", dtype=torch.int32)
    params = {
        "bias": (torch.randn(length, q, device="cuda") * 0.2).bfloat16(),
        "coupling_matrix": (torch.randn(length, q, length, q, device="cuda") * 0.1).bfloat16(),
    }
    reference_params = {key: value.float() for key, value in params.items()}
    sites = torch.randint(length, (n,) if independent else (steps,), device="cuda", dtype=torch.int32)
    uniforms = torch.rand((n,) if independent else (steps, n), device="cuda")
    proposals = torch.randint(q, uniforms.shape, device="cuda", dtype=torch.int32)
    fn = getattr(kernels, f"_{sampler}_step_independent_triton" if independent else f"_{sampler}_steps_triton")
    random = (sites, uniforms) if sampler == "gibbs" else (sites, proposals, uniforms)
    expected, actual = states.clone(), states.clone()
    fn(expected, reference_params, *random, beta=0.8)
    fn(actual, params, *random, beta=0.8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@gpu
@pytest.mark.parametrize("sampler", ["gibbs", "metropolis"])
def test_bf16_one_hot_uses_fp32_rng(sampler):
    torch.manual_seed(3)
    chains = torch.nn.functional.one_hot(torch.randint(5, (33, 17), device="cuda"), 5).float()
    params = {
        "bias": torch.randn(17, 5, device="cuda").bfloat16(),
        "coupling_matrix": (torch.randn(17, 5, 17, 5, device="cuda") * 0.1).bfloat16(),
    }
    fn = getattr(kernels, sampler + "_sampling_triton")
    torch.manual_seed(123)
    expected = fn(chains, {k: v.float() for k, v in params.items()}, 2)
    expected_rng = torch.cuda.get_rng_state()
    torch.manual_seed(123)
    actual = fn(chains.bfloat16(), params, 2)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), expected_rng, rtol=0, atol=0)


@gpu
@pytest.mark.parametrize("sampler", ["gibbs", "metropolis"])
def test_master_params_unchanged_and_quantization_refreshed(sampler):
    torch.manual_seed(19)
    chains = torch.nn.functional.one_hot(torch.randint(5, (33, 17), device="cuda"), 5).float()
    params = {
        "bias": torch.randn(17, 5, device="cuda"),
        "coupling_matrix": torch.randn(17, 5, 17, 5, device="cuda") * 0.1,
    }
    mixed = prepare_training_sampler(sampler, torch.device("cuda"), "bfloat16")
    fn = getattr(kernels, sampler + "_sampling_triton")
    for _ in range(2):
        params["coupling_matrix"].add_(0.03125)
        originals = {k: v.clone() for k, v in params.items()}
        rounded = {"bias": params["bias"], "coupling_matrix": params["coupling_matrix"].bfloat16().float()}
        torch.manual_seed(11)
        expected = fn(chains, rounded, 2)
        torch.manual_seed(11)
        actual = mixed(chains, params, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key, value in params.items():
            torch.testing.assert_close(value, originals[key], rtol=0, atol=0)


@gpu
@pytest.mark.parametrize("sampler", ["gibbs", "metropolis"])
def test_fixed_model_sampling_quantizes_couplings_once_and_keeps_fp32_state(sampler):
    torch.manual_seed(53)
    params = {
        "bias": torch.randn(7, 3, device="cuda"),
        "coupling_matrix": torch.randn(7, 3, 7, 3, device="cuda") * 0.1,
    }
    originals = {key: value.clone() for key, value in params.items()}
    _sampling_fn, sampling_params = prepare_fixed_model_sampler(
        sampler, torch.device("cuda"), "bfloat16", params
    )

    assert sampling_params["bias"] is params["bias"]
    assert sampling_params["coupling_matrix"].dtype == torch.bfloat16
    assert sampling_params["coupling_matrix"].data_ptr() != params["coupling_matrix"].data_ptr()
    for key, value in params.items():
        torch.testing.assert_close(value, originals[key], rtol=0, atol=0)

    result = sample_sequences(
        model=DCAModel(params, alphabet="ABC"),
        n_sequences=8,
        n_sweeps=2,
        sampler=sampler,
        dtype="bfloat16",
        seed=5,
    )
    assert result.sampling_dtype == "bfloat16"
    assert result.model.dtype == "float32"
    assert result.energies.dtype.name == "float32"


@gpu
@pytest.mark.parametrize("model_type", ["bmDCA", "eaDCA", "edDCA", "edgeDCA"])
@pytest.mark.parametrize("sampler", ["gibbs", "metropolis"])
def test_training_state_stays_fp32_and_sampler_uses_bf16(model_type, sampler, tmp_path):
    original = getattr(kernels, sampler + "_sampling_triton")
    observed = []

    def check(chains, params, nsweeps, beta=1.0, *, coupling_dtype=None):
        observed.append(coupling_dtype)
        assert chains.dtype == params["bias"].dtype == torch.float32
        return original(chains, params, nsweeps, beta, coupling_dtype=coupling_dtype)

    config = TrainingConfig(
        model_type=model_type,
        sampler=sampler,
        dtype="bfloat16",
        device="cuda",
        alphabet="ABC",
        n_chains=64,
        n_sweeps=1,
        max_epochs=2,
        max_gradient_steps=2,
        max_structure_steps=1,
        target_pearson=0.99999,
        no_reweighting=True,
        checkpoint_interval=1,
        inner_gradient_steps=2,
    )
    with patch.object(kernels, sampler + "_sampling_triton", check):
        result = train_model(alignment(), config=config, output_dir=tmp_path)
    assert observed and all(dtype == torch.bfloat16 for dtype in observed)
    assert result.chains.dtype == result.log_weights.dtype == torch.float32
    assert result.model.metadata.dtype == "float32"
    assert result.config.dtype == "bfloat16"
    for param in result.model.params.values():
        assert param.dtype == torch.float32 and torch.isfinite(param).all()
    assert result.artifacts["params"].exists()
    # Persistence uses full-precision master parameters, so existing loaders work.
    from adabmDCA import load_model

    loaded = load_model(result.artifacts["params"], alphabet="ABC", device="cuda")
    torch.testing.assert_close(loaded.params["bias"], result.model.params["bias"], atol=1e-5, rtol=1e-5)
