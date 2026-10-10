"""MPS BF16 coupling storage follows the CUDA mixed-precision contract."""

import pytest
import torch

from adabmDCA import DCAModel, load_model, sample_sequences, train_model
from adabmDCA.alignment import Alignment
from adabmDCA.mps_kernels import sampling as metal
from adabmDCA.mps_kernels.runtime import is_mps_available
from adabmDCA.sampling import prepare_fixed_model_sampler, prepare_training_sampler
from adabmDCA.training_config import TrainingConfig

pytestmark = pytest.mark.skipif(
    not is_mps_available() or not torch.backends.mps.is_macos_or_newer(14, 0),
    reason="BF16 Metal shaders unavailable",
)


def model(sparse=False, length=17, q=5):
    rng = torch.Generator().manual_seed(19)
    j = torch.randn(length, q, length, q, generator=rng) * .1
    j = (j + j.permute(2, 3, 0, 1)) / 2
    j[torch.arange(length), :, torch.arange(length), :] = 0
    if sparse:
        mask = torch.zeros(length, length, dtype=torch.bool)
        for i in range(length - 1):
            mask[i, i + 1] = mask[i + 1, i] = True
        j *= mask[:, None, :, None]
    return {"bias": torch.randn(length, q, generator=rng).to("mps"), "coupling_matrix": j.to("mps")}


def alignment():
    rows = torch.randint(3, (64, 9), generator=torch.Generator().manual_seed(73))
    rows[:, 1] = rows[:, 0]
    return Alignment(tuple(map(str, range(64))), tuple("".join("ABC"[x] for x in row) for row in rows.tolist()))


@pytest.mark.parametrize("method", metal.METHODS)
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("q", [5, 21])
def test_quantized_sampling_refreshes_without_changing_masters_or_rng(method, sparse, q):
    params = model(sparse, q=q)
    states = torch.zeros(37, 17, dtype=torch.int32, device="mps")
    chains = torch.nn.functional.one_hot(states.long(), q).float()
    mixed = prepare_training_sampler(method, torch.device("mps"), "bfloat16")
    for _ in range(2):
        params["coupling_matrix"].mul_(1.03125)
        originals = {k: v.clone() for k, v in params.items()}
        rounded = {"bias": params["bias"], "coupling_matrix": params["coupling_matrix"].bfloat16().float()}
        torch.manual_seed(11)
        expected = metal.sample_categorical(method, states, rounded, 3)
        after = torch.mps.get_rng_state()
        torch.manual_seed(11)
        actual = mixed(chains, params, 3)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual.argmax(-1).int(), expected, atol=0, rtol=0)
        assert torch.equal(after, torch.mps.get_rng_state())
        for key, value in params.items():
            torch.testing.assert_close(value, originals[key], atol=0, rtol=0)


@pytest.mark.parametrize("sparse", [False, True])
def test_cache_dtype_invalidation_and_inference(sparse):
    params = model(sparse)
    j = params["coupling_matrix"]
    def layout(dtype):
        return (metal.sparse_coupling_layout(j[None], source=j, coupling_dtype=dtype)[1]
                if sparse else metal._layout(j, coupling_dtype=dtype))
    initial = layout(torch.bfloat16)
    assert layout(torch.bfloat16) is initial
    assert layout(torch.float32).dtype == torch.float32
    j.mul_(1.5)
    updated = layout(torch.bfloat16)
    assert updated is not initial and updated.dtype == torch.bfloat16
    torch.testing.assert_close(updated.float(), layout(torch.float32).bfloat16().float(), atol=0, rtol=0)
    with torch.inference_mode():
        params = model(sparse)
        states = torch.zeros(7, 17, dtype=torch.int32, device="mps")
        assert metal.sample_categorical("gibbs", states, params, 2, coupling_dtype=torch.bfloat16).shape == states.shape


@pytest.mark.parametrize("method", metal.METHODS)
def test_fixed_model_api_keeps_fp32_energies_and_model(method):
    params = model(length=7, q=3)
    _, quantized = prepare_fixed_model_sampler(method, torch.device("mps"), "bfloat16", params)
    assert quantized["bias"] is params["bias"]
    assert quantized["coupling_matrix"].dtype == torch.bfloat16
    result = sample_sequences(model=DCAModel(params, alphabet="ABC"), device="mps", n_sequences=8,
                              n_sweeps=2, sampler=method, dtype="bfloat16", seed=5)
    assert result.sampling_dtype == "bfloat16"
    assert result.model.dtype == "float32"
    assert result.energies.dtype.name == "float32"


@pytest.mark.parametrize("model_type", ["bmDCA", "eaDCA", "edDCA", "edgeDCA"])
@pytest.mark.parametrize("method", metal.METHODS)
def test_training_and_saved_parameters_stay_fp32(model_type, method, tmp_path, monkeypatch):
    original = metal._run
    observed = []
    def check(*args, **kwargs):
        packed = kwargs.get("sparse")
        couplings = packed[1] if packed is not None else args[3]
        observed.append(couplings.dtype)
        assert args[2].dtype == torch.float32
        return original(*args, **kwargs)
    monkeypatch.setattr(metal, "_run", check)
    config = TrainingConfig(model_type=model_type, sampler=method, dtype="bfloat16", device="mps", alphabet="ABC",
                            n_chains=64, n_sweeps=1, max_epochs=2, max_gradient_steps=2, max_structure_steps=1,
                            target_pearson=.99999, no_reweighting=True, checkpoint_interval=1, inner_gradient_steps=2)
    result = train_model(alignment(), config=config, output_dir=tmp_path)
    assert observed and all(dtype == torch.bfloat16 for dtype in observed)
    assert result.chains.dtype == torch.float32
    assert result.model.metadata.dtype == "float32"
    assert result.config.dtype == "bfloat16"
    for value in result.model.params.values():
        assert value.dtype == torch.float32 and torch.isfinite(value).all()
    loaded = load_model(result.artifacts["params"], alphabet="ABC", device="mps")
    for key, value in result.model.params.items():
        torch.testing.assert_close(loaded.params[key], value, atol=1e-5, rtol=1e-5)


def test_bf16_rejects_disabled_backend_and_unsupported_dimensions(monkeypatch):
    monkeypatch.setenv("ADABMDCA_MPS", "0")
    with pytest.raises(ValueError, match="enabled MPS"):
        prepare_training_sampler("gibbs", torch.device("mps"), "bfloat16")
    monkeypatch.delenv("ADABMDCA_MPS")
    params = model(length=2, q=33)
    chains = torch.nn.functional.one_hot(torch.zeros(3, 2, dtype=torch.long, device="mps"), 33).float()
    with pytest.raises(ValueError, match="supported dimensions"):
        prepare_training_sampler("gibbs", torch.device("mps"), "bfloat16")(chains, params, 1)


def test_sparse_detection_uses_master_graph_and_refreshes_new_edges():
    j = torch.zeros(1, 17, 5, 17, 5, device="mps")
    j[0, 0, 0, 1, 0] = 1e-20  # A small nonzero master edge must retain its neighbour.
    neighbours, blocks, _ = metal.sparse_coupling_layout(j, coupling_dtype=torch.bfloat16)
    assert int(neighbours[0, 0, 0]) == 1
    assert blocks.dtype == torch.bfloat16
    j[0, 0, 0, 2, 0] = .125
    neighbours, blocks, width = metal.sparse_coupling_layout(j, coupling_dtype=torch.bfloat16)
    assert width == 2 and neighbours[0, 0].cpu().tolist() == [1, 2]
    assert float(blocks[0, 0, 1, 0, 0]) == .125
