import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from adabmDCA import Alignment, TrainingConfig
from adabmDCA.api.exceptions import (
    AlignmentLengthError,
    AlignmentLoadError,
    ConvergenceError,
    InputValidationError,
    ModelLoadError,
    OperationCancelledError,
    OutputSerializationError,
)
from adabmDCA.api.model import DCAModel, load_model
from adabmDCA.cli import render_error


def test_error_documents_expose_transport_exit_codes():
    assert InputValidationError("invalid").to_dict()["exit_code"] == 2
    assert ModelLoadError("failed").to_dict()["exit_code"] == 1
    assert OutputSerializationError("failed").exit_code == 1
    assert ConvergenceError("failed").exit_code == 1
    assert OperationCancelledError("cancelled").exit_code == 130


def test_cli_renderer_uses_stable_code_message_and_details():
    rendered = render_error(
        InputValidationError(
            "Invalid threshold.",
            details={"name": "threshold", "value": -1},
        )
    )

    assert rendered.splitlines() == [
        "Error [input_validation_error]: Invalid threshold.",
        "  name: threshold",
        "  value: -1",
    ]


def test_missing_alignment_is_a_load_failure(tmp_path: Path):
    from adabmDCA import load_alignment

    with pytest.raises(AlignmentLoadError) as context:
        load_alignment(tmp_path / "missing.fasta")

    assert context.value.exit_code == 1
    assert context.value.details["path"].endswith("missing.fasta")


def test_structured_parse_errors_are_not_wrapped_again(tmp_path: Path):
    from adabmDCA import read_alignment

    path = tmp_path / "unequal.fasta"
    path.write_text(">first\nAA\n>second\nAAA\n", encoding="utf-8")

    with pytest.raises(AlignmentLengthError) as context:
        read_alignment(path)

    assert context.value.__cause__ is None
    assert context.value.details["lengths"] == [2, 3]


def test_expected_model_parse_errors_preserve_their_cause(tmp_path: Path):
    path = tmp_path / "bad.dat"
    path.write_text("not a model\n", encoding="utf-8")

    with pytest.raises(ModelLoadError) as context:
        load_model(path, alphabet="AB-", device="cpu")

    assert context.value.__cause__ is not None
    assert context.value.details == {"path": str(path)}


def test_unexpected_programming_errors_are_not_hidden_as_load_errors(tmp_path: Path):
    path = tmp_path / "params.dat"
    path.write_text("placeholder\n", encoding="utf-8")

    with (
        patch("adabmDCA.api.model.load_params", side_effect=AttributeError("bug")),
        pytest.raises(AttributeError, match="bug"),
    ):
        load_model(path, alphabet="AB-", device="cpu")


def test_configuration_errors_are_structured():
    with pytest.raises(InputValidationError) as context:
        TrainingConfig(n_chains=0)

    assert context.value.code == "input_validation_error"
    assert context.value.exit_code == 2

    with pytest.raises(InputValidationError) as nonfinite:
        TrainingConfig(learning_rate=math.nan)
    assert nonfinite.value.details["name"] == "learning_rate"


def test_invalid_sequence_container_is_translated_at_api_boundary():
    from adabmDCA.api.runtime import normalize_sequences

    with pytest.raises(InputValidationError) as context:
        normalize_sequences(None, tokens="AB-")

    assert context.value.__cause__ is not None


def test_high_level_contact_errors_precede_low_level_value_errors():
    from adabmDCA import predict_contacts

    model = DCAModel(
        {
            "bias": torch.zeros(2, 2),
            "coupling_matrix": torch.zeros(2, 2, 2, 2),
        },
        alphabet="AB",
    )

    with pytest.raises(InputValidationError) as context:
        predict_contacts(model=model)

    assert context.value.details["tokens"] == "AB"


def test_sampling_validates_mixing_budget_before_computation():
    from adabmDCA import sample_sequences

    model = DCAModel(
        {
            "bias": torch.zeros(2, 3),
            "coupling_matrix": torch.zeros(2, 3, 2, 3),
        },
        alphabet="AB-",
    )

    with pytest.raises(InputValidationError, match="estimating mixing time"):
        sample_sequences(
            model=model,
            n_sequences=2,
            n_sweeps=0,
            reference_fasta=Alignment(("a",), ("AB",)),
        )
