import json
from pathlib import Path

import numpy as np
import pytest

from adabmDCA import Alignment
from adabmDCA.api.exceptions import OutputSerializationError
from adabmDCA.api.results import (
    ContactMapResult,
    EnergyResult,
    ModelMetadata,
    ProfileSplitResult,
    SamplingResult,
)


@pytest.fixture
def metadata():
    return ModelMetadata(
        length=2,
        alphabet="AB-",
        tokens="AB-",
        device="cpu",
        dtype="float32",
    )


def test_energy_result_serializes_json_csv_and_fasta(tmp_path: Path, metadata):
    result = EnergyResult(
        sequences=("AB", "A-"),
        energies=np.asarray([1.0, np.nan]),
        model=metadata,
        names=("first", "second"),
    )

    artifacts = result.save_bundle(tmp_path / "nested", stem="scores")
    document = json.loads(artifacts["summary"].read_text(encoding="utf-8"))

    assert document["schema_version"] == "1.0"
    assert document["result_type"] == "energy"
    assert document["data"]["energies"] == [1.0, None]
    assert artifacts["csv"].read_text(encoding="utf-8").startswith("name,sequence,energy")
    assert artifacts["fasta"].read_text(encoding="utf-8").startswith(">first | DCAenergy:")


def test_contact_map_bundle_has_machine_readable_matrix_formats(tmp_path: Path):
    scores = np.asarray([[0.0, 1.5], [1.5, 0.0]])
    result = ContactMapResult(scores, "dca", "AB-", "AB-")

    artifacts = result.save_bundle(tmp_path, label="model")

    np.testing.assert_allclose(np.load(artifacts["npy"]), scores)
    assert artifacts["csv"].read_text(encoding="utf-8").startswith("position_i,position_j,score")
    assert not artifacts["matrix"].read_text(encoding="utf-8").startswith("position_i")


def test_sampling_bundle_owns_diagnostic_serialization(tmp_path: Path, metadata):
    result = SamplingResult(
        sequences=("AB",),
        energies=np.asarray([-2.0]),
        num_sweeps=3,
        sampler="gibbs",
        beta=1.0,
        seed=7,
        model=metadata,
        mixing_history={"sweep": [1, 2], "pearson": [0.2, 0.4]},
    )

    artifacts = result.save_bundle(tmp_path, label="trial")

    assert set(artifacts) == {"samples", "samples_csv", "summary", "mixing_log", "sampling_log"}
    assert artifacts["mixing_log"].read_text(encoding="utf-8").startswith("sweep,pearson")
    assert artifacts["sampling_log"].read_text(encoding="utf-8") == "\n"


def test_profile_split_bundle_uses_result_owned_outputs(tmp_path: Path):
    result = ProfileSplitResult(
        training=Alignment(("train",), ("AA",)),
        test=Alignment(("test",), ("BB",)),
        score=1,
        attempts=2,
        tokens="AB-",
        seed=3,
    )

    artifacts = result.save_bundle(tmp_path / "family")

    assert artifacts["training"].read_text(encoding="utf-8") == ">train\nAA\n"
    assert artifacts["test"].read_text(encoding="utf-8") == ">test\nBB\n"
    assert json.loads(artifacts["summary"].read_text(encoding="utf-8"))["result_type"] == "profile_split"


def test_save_rejects_unknown_format(tmp_path: Path, metadata):
    result = EnergyResult(("AB",), np.asarray([0.0]), metadata)

    with pytest.raises(OutputSerializationError) as context:
        result.save(tmp_path / "scores.unsupported")

    assert context.value.details["supported_formats"] == ["csv", "fasta", "json"]
