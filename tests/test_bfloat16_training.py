"""BF16 sampling storage with FP32 training state and arithmetic."""

import argparse
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from adabmDCA import DCAModel, sample_sequences, train_model
from adabmDCA import sampling_triton as kernels
from adabmDCA.alignment import Alignment
from adabmDCA.exceptions import InputValidationError
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
@pytest.mark.parametrize("sampler", ["gibbs", "metropolis", "metropolized_gibbs"])
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
@pytest.mark.parametrize("sampler", ["gibbs", "metropolis", "metropolized_gibbs"])
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
@pytest.mark.parametrize("sampler", ["gibbs", "metropolis", "metropolized_gibbs"])
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
@pytest.mark.parametrize("sampler", ["gibbs", "metropolis", "metropolized_gibbs"])
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
    assert result.chains.dtype == torch.float32
    assert result.model.metadata.dtype == "float32"
    assert result.config.dtype == "bfloat16"
    for param in result.model.params.values():
        assert param.dtype == torch.float32 and torch.isfinite(param).all()
    assert result.artifacts["params"].exists()
    # Persistence uses full-precision master parameters, so existing loaders work.
    from adabmDCA import load_model

    loaded = load_model(result.artifacts["params"], alphabet="ABC", device="cuda")
    torch.testing.assert_close(loaded.params["bias"], result.model.params["bias"], atol=1e-5, rtol=1e-5)


# PTT


def ptt_profile(length, q, device):
    torch.manual_seed(29)
    return {"bias": torch.randn(length, q, device=device) * 0.3,
            "coupling_matrix": torch.zeros(length, q, length, q, device=device)}


def couple(sampler, *, sparse=False, scale=0.1):
    """Give every replica above the profile the same random symmetric couplings.

    ``sparse`` couples only neighbouring sites, so the sparse kernels are used.
    """
    length, q = sampler.models[0]["bias"].shape
    device = sampler.models[0]["bias"].device
    couplings = torch.randn(length, q, length, q, device=device) * scale
    couplings = (couplings + couplings.permute(2, 3, 0, 1)) / 2
    sites = torch.arange(length, device=device)
    distance = (sites[:, None] - sites[None, :]).abs()
    keep = distance == 1 if sparse else distance > 0
    couplings *= keep[:, None, :, None]
    for model in sampler.models[1:]:
        model["coupling_matrix"].copy_(couplings)


def test_ptt_bfloat16_requires_cuda():
    from adabmDCA.ptt import PTTConfig, PTTSampler

    sampler = PTTSampler(ptt_profile(3, 2, "cpu"), tokens="AB", n_chains=4)
    with pytest.raises(InputValidationError, match="CUDA"):
        sampler.set_coupling_dtype(torch.bfloat16)
    assert TrainingConfig(dtype="bfloat16", ptt=PTTConfig()).dtype == "bfloat16"
    with pytest.raises(InputValidationError, match="CUDA"):
        train_model(alignment(), dtype="bfloat16", device="cpu", alphabet="ABC", ptt=PTTConfig())


@gpu
@pytest.mark.parametrize("name", ["gibbs", "metropolis", "metropolized_gibbs"])
@pytest.mark.parametrize("length", [9, 240])
@pytest.mark.parametrize("sparse", [False, True])
def test_ptt_bf16_kernels_match_fp32_for_rounded_couplings(name, length, sparse):
    from adabmDCA.ptt import PTTSampler

    sampler = PTTSampler(ptt_profile(length, 5, "cuda"), tokens="ABCDE", n_chains=33, sampler=name)
    sampler.models.append({key: value.clone() for key, value in sampler.models[-1].items()})
    sampler.chains.append(sampler.chains[-1].clone())
    couple(sampler, sparse=sparse)
    rounded = sampler.fork()
    for model in rounded.models:
        model["coupling_matrix"].copy_(model["coupling_matrix"].bfloat16().float())
    sampler.set_coupling_dtype(torch.bfloat16)
    copy = sampler.fork()
    assert copy.coupling_dtype == torch.bfloat16
    masters = [model["coupling_matrix"].clone() for model in sampler.models]
    for mode in ("train", "generate"):
        expected, actual = rounded.fork(seed=7), copy.fork(seed=7)
        for current in (expected, actual):
            current.mode = mode
        local = [1, 2]
        torch.manual_seed(5)
        if mode == "train" or expected._replica_kernel is None:
            reference = [expected._local_kernel(expected.chains[k], expected.models[k], 3) for k in local]
            torch.manual_seed(5)
            result = [actual._local_kernel(actual.chains[k], actual.models[k], 3) for k in local]
        else:
            stack = torch.stack([expected.chains[k] for k in local])
            reference = expected._replica_kernel(stack, *expected._stacked_replica_params(local), nsweeps=3)
            torch.manual_seed(5)
            biases, couplings = actual._stacked_replica_params(local)
            assert couplings.dtype == torch.bfloat16 and biases.dtype == torch.float32
            result = actual._replica_kernel(stack, biases, couplings, nsweeps=3)
        for left, right in zip(result, reference):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    for model, master in zip(sampler.models, masters):
        assert model["coupling_matrix"].dtype == torch.float32
        torch.testing.assert_close(model["coupling_matrix"], master, rtol=0, atol=0)


@gpu
def test_ptt_training_and_generation_use_bf16_couplings(tmp_path):
    from adabmDCA.ptt import PTTConfig, PTTSampler

    original = kernels.metropolized_gibbs_sampling_categorical_triton
    observed = []

    def check(states, params, nsweeps, beta=1.0, *, coupling_dtype=None, **kwargs):
        observed.append(coupling_dtype)
        assert params["bias"].dtype == params["coupling_matrix"].dtype == torch.float32
        return original(states, params, nsweeps, beta, coupling_dtype=coupling_dtype, **kwargs)

    config = TrainingConfig(
        dtype="bfloat16", device="cuda", alphabet="ABC", n_chains=64, n_sweeps=1, max_epochs=2,
        target_pearson=0.99999, no_reweighting=True, pseudocount=0.1, checkpoint_interval=1,
        ptt=PTTConfig(equilibration_rounds=2, initialization_rounds=2, swaps=1,
                      mixing_thermalization_rounds=2, mixing_chains=20),
    )
    with patch.object(kernels, "metropolized_gibbs_sampling_categorical_triton", check):
        partial = train_model(alignment(), config=replace(config, max_epochs=1), output_dir=tmp_path / "partial")
    assert observed and all(dtype == torch.bfloat16 for dtype in observed)
    assert partial.config.dtype == "bfloat16"
    for param in partial.model.params.values():
        assert param.dtype == torch.float32 and torch.isfinite(param).all()
    archive = partial.artifacts["ptt_archive"]
    # The archive keeps float32 parameters; BF16 is a runtime choice.
    assert PTTSampler.from_archive(archive, device="cuda", mode="resume").coupling_dtype is None
    resumed = train_model(alignment(), config=config, ptt_resume=archive, output_dir=tmp_path / "resumed")
    assert resumed.config.dtype == "bfloat16"

    reference = tmp_path / "data.fasta"
    alignment().write_fasta(reference)
    result = sample_sequences(model=archive, ptt=True, n_sequences=16, dtype="bfloat16", device="cuda",
                              alphabet="ABC", reference_fasta=reference, seed=3)
    assert len(result.sequences) == 16
