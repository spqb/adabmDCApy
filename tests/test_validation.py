"""Regressions for invalid inputs that previously reached numerical code."""

from pathlib import Path

import numpy as np
import pytest
import torch

from adabmDCA import DCAModel, TrainingConfig, TrainingLimits, load_model
from adabmDCA.api.entropy import estimate_entropy
from adabmDCA.api.exceptions import InputValidationError, ModelLoadError
from adabmDCA.api.sampling import sample_sequences
from adabmDCA.api.splitting import split_alignment
from adabmDCA.io import load_params
from adabmDCA.training_config import ConfigurationError


@pytest.mark.parametrize("record", ["h 0 A {value}", "J 0 1 A B {value}"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e400"])
def test_parameter_files_reject_nonfinite_records(tmp_path: Path, record, value):
    path = tmp_path / "invalid.dat"
    path.write_text("h 0 A 0\nh 1 B 0\n" + record.format(value=value) + "\n")
    with pytest.raises(ValueError, match="value on line 3"):
        load_params(str(path), tokens="AB", device=torch.device("cpu"))
    with pytest.raises(ModelLoadError, match="value on line 3") as error:
        load_model(path, alphabet="AB", device="cpu")
    assert error.value.details["path"] == str(path)
    assert error.value.__cause__ is not None


@pytest.mark.parametrize("key", ["bias", "coupling_matrix"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_direct_models_reject_nonfinite_parameters(key, value):
    params = {"bias": torch.zeros(2, 2), "coupling_matrix": torch.zeros(2, 2, 2, 2)}
    params[key].reshape(-1)[-1] = value
    with pytest.raises(InputValidationError, match=f"'{key}'.*finite"):
        DCAModel(params, alphabet="AB")


def test_loader_rejects_overflow_during_symmetrization(tmp_path: Path):
    path = tmp_path / "overflow.dat"
    path.write_text("h 0 A 0\nh 1 B 0\nJ 0 1 A B 3e38\nJ 1 0 B A 3e38\n")
    with np.errstate(over="ignore"), pytest.raises(ModelLoadError, match="finite"):
        load_model(path, alphabet="AB", device="cpu")
    model = load_model(path, alphabet="AB", device="cpu", dtype="float64")
    assert torch.isfinite(model.params["coupling_matrix"]).all()


def test_loader_rejects_overflow_during_dtype_conversion(tmp_path: Path):
    path = tmp_path / "overflow.dat"
    path.write_text("h 0 A 1e100\nh 0 B 0\n")
    with np.errstate(over="ignore"), pytest.raises(ModelLoadError, match="finite"):
        load_model(path, alphabet="AB", device="cpu")
    assert np.isfinite(load_model(path, alphabet="AB", device="cpu", dtype="float64").compute_energies("A")).all()


def test_model_validation_checks_noncontiguous_parameters_without_copying():
    params = {"bias": torch.zeros(2, 2).T, "coupling_matrix": torch.zeros(2, 2, 2, 2).transpose(0, 1)}
    model = DCAModel(params, alphabet="AB")
    assert model.params["bias"] is params["bias"]
    assert model.params["coupling_matrix"] is params["coupling_matrix"]
    params["coupling_matrix"][1, 0, 1, 0] = float("nan")
    with pytest.raises(InputValidationError, match="finite"):
        DCAModel(params, alphabet="AB")


@pytest.mark.parametrize("field", [
    "n_chains", "n_sweeps", "max_epochs", "max_gradient_steps", "max_structure_steps",
    "checkpoint_interval", "inner_gradient_steps", "activation_steps",
])
@pytest.mark.parametrize("value", [1.5, float("nan"), float("inf"), True, "2", None, 0, -1])
def test_training_counts_are_positive_integers(field, value):
    if value is None and field in {"max_gradient_steps", "max_structure_steps", "checkpoint_interval"}:
        assert getattr(TrainingConfig(**{field: value}), field) is None
    else:
        with pytest.raises(ConfigurationError, match=field):
            TrainingConfig(**{field: value})


@pytest.mark.parametrize("field", ["max_gradient_steps", "max_structure_steps"])
@pytest.mark.parametrize("value", [1.5, float("nan"), True, 0])
def test_explicit_training_limits_validate_counts(field, value):
    with pytest.raises(InputValidationError, match=field):
        TrainingLimits(**{field: value})


@pytest.mark.parametrize("value", [False, 1.5, float("nan"), -(2**63) - 1, 2**64])
def test_training_seed_is_validated(value):
    with pytest.raises(ConfigurationError, match="seed"):
        TrainingConfig(seed=value)


def test_integer_types_and_seed_boundaries_are_preserved():
    assert TrainingConfig(n_chains=np.int64(2), seed=-(2**63)).n_chains == 2
    assert TrainingConfig(seed=2**64 - 1).seed == 2**64 - 1


@pytest.mark.parametrize("field", ["n_sequences", "n_sweeps", "n_measure", "mixing_multiplier", "seed"])
@pytest.mark.parametrize("value", [1.5, float("nan"), True])
def test_sampling_rejects_invalid_counts_before_loading(field, value):
    kwargs = {"model": "missing.dat", "n_sequences": 2, field: value}
    with pytest.raises(InputValidationError, match=field):
        sample_sequences(**kwargs)


@pytest.mark.parametrize("field", [
    "n_chains", "n_sweeps", "n_steps", "theta_sweeps", "zero_sweeps", "max_theta_iterations", "seed",
])
def test_entropy_rejects_invalid_counts_before_loading(field):
    with pytest.raises(InputValidationError, match=field):
        estimate_entropy(model="missing.dat", natural_alignment="missing.fa", target_alignment="missing.fa",
                         **{field: float("nan")})


@pytest.mark.parametrize("field", ["attempts", "max_train", "max_test", "seed"])
def test_splitting_rejects_invalid_counts_before_loading(field):
    with pytest.raises(InputValidationError, match=field):
        split_alignment("missing.fa", **{field: True})


def test_sampling_still_allows_zero_sweeps():
    model = DCAModel({"bias": torch.zeros(2, 2), "coupling_matrix": torch.zeros(2, 2, 2, 2)}, alphabet="AB")
    result = model.sample_sequences(2, n_sweeps=0)
    assert len(result.sequences) == 2
    assert result.num_sweeps == 0
    assert result.sampling_dtype == "float32"
