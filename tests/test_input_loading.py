import gzip
from pathlib import Path

import numpy as np
import pytest
import torch

from adabmDCA import Alignment
from adabmDCA.api.exceptions import (
    InputValidationError,
    ModelCompatibilityError,
    WeightLoadError,
)
from adabmDCA.api.input_loading import load_training_inputs
from adabmDCA.dataset import DatasetDCA
from adabmDCA.input_loading import (
    AlignmentLoadConfig,
    load_alignment,
    load_sequence_weights,
)
from adabmDCA.training_config import TrainingConfig


def test_loading_policy_reports_filtering_and_duplicates():
    alignment = Alignment(
        ("first", "invalid", "duplicate", "last"),
        ("AA", "AX", "AA", "BB"),
    )

    loaded = load_alignment(
        alignment,
        config=AlignmentLoadConfig(
            alphabet="AB-",
            invalid_sequences="drop",
            remove_duplicates=True,
        ),
    )

    assert loaded.alignment.names == ("first", "last")
    assert loaded.alignment.sequences == ("AA", "BB")
    assert loaded.retained_indices == (0, 3)
    assert loaded.dropped_indices == (1,)
    assert loaded.duplicate_indices == (2,)
    assert loaded.original_size == 4


def test_strict_loading_reports_unknown_tokens():
    alignment = Alignment(("valid", "bad"), ("AA", "AX"))

    with pytest.raises(InputValidationError) as context:
        load_alignment(
            alignment,
            config=AlignmentLoadConfig(alphabet="AB-"),
        )

    assert context.value.details["invalid_names"] == ["bad"]
    assert context.value.details["unexpected_tokens"] == ["X"]


def test_empty_filtered_alignment_suggests_correct_alphabet():
    alignment = Alignment(("first", "second"), ("MEK", "WVL"))

    with pytest.raises(InputValidationError) as context:
        load_alignment(
            alignment,
            config=AlignmentLoadConfig(alphabet="dna", invalid_sequences="drop"),
        )

    error = context.value
    assert "selected alphabet 'dna'" in error.message
    assert "--alphabet protein" in error.message
    assert error.details["original_sequences"] == 2
    assert error.details["selected_alphabet"] == "dna"
    assert error.details["allowed_tokens"] == "-ACGT"
    assert error.details["unexpected_tokens"] == ["E", "K", "L", "M", "V", "W"]


def test_compressed_fasta_and_stockholm_use_the_same_loader(tmp_path: Path):
    compressed = tmp_path / "alignment.fasta.gz"
    with gzip.open(compressed, "wt") as handle:
        handle.write(">s1\nAB\n>s2\nA-\n")
    stockholm = tmp_path / "alignment.sto"
    stockholm.write_text(
        "# STOCKHOLM 1.0\ns1 A.\ns2 AB\n//\n",
        encoding="utf-8",
    )

    loaded_gzip = load_alignment(
        compressed,
        config=AlignmentLoadConfig(alphabet="AB-"),
    )
    loaded_stockholm = load_alignment(
        stockholm,
        config=AlignmentLoadConfig(alphabet="AB-"),
    )

    assert loaded_gzip.alignment.sequences == ("AB", "A-")
    assert loaded_stockholm.alignment.sequences == ("A-", "AB")


def test_original_weights_follow_retained_indices():
    loaded = load_alignment(
        Alignment(("a", "bad", "duplicate", "b"), ("AA", "AX", "AA", "BB")),
        config=AlignmentLoadConfig(
            alphabet="AB-",
            invalid_sequences="drop",
            remove_duplicates=True,
        ),
    )

    weights = load_sequence_weights(
        [1.0, 2.0, 3.0, 4.0],
        loaded_alignment=loaded,
        no_reweighting=False,
        clustering_seqid=0.8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(weights, torch.tensor([1.0, 4.0]))


@pytest.mark.parametrize(
    "weights",
    ([1.0], [1.0, -1.0], [1.0, np.nan], [[1.0], [2.0]]),
)
def test_invalid_weights_raise_structured_errors(weights):
    loaded = load_alignment(
        Alignment(("a", "b"), ("AA", "BB")),
        config=AlignmentLoadConfig(alphabet="AB-"),
    )

    with pytest.raises(WeightLoadError):
        load_sequence_weights(
            weights,
            loaded_alignment=loaded,
            no_reweighting=False,
            clustering_seqid=0.8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_dataset_can_be_materialized_without_file_io(capsys):
    alignment = Alignment(("a", "b"), ("AA", "BB"))
    loaded = load_alignment(
        alignment,
        config=AlignmentLoadConfig(alphabet="AB-"),
    )

    dataset = DatasetDCA.from_loaded_alignment(
        loaded,
        weights=[1.0, 2.0],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert dataset.names.tolist() == ["a", "b"]
    assert dataset.data.tolist() == [[0, 0], [1, 1]]
    torch.testing.assert_close(dataset.weights, torch.tensor([1.0, 2.0]))
    assert capsys.readouterr().out == ""


def test_expected_length_is_checked_before_materialization():
    with pytest.raises(ModelCompatibilityError):
        load_alignment(
            Alignment(("a",), ("AAA",)),
            config=AlignmentLoadConfig(
                alphabet="AB-",
                expected_length=2,
            ),
        )


def test_training_loader_checks_validation_length():
    with pytest.raises(ModelCompatibilityError):
        load_training_inputs(
            Alignment(("train",), ("AA",)),
            validation=Alignment(("validation",), ("AAA",)),
            config=TrainingConfig(
                alphabet="AB-",
                no_reweighting=True,
                n_chains=2,
            ),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_training_loader_checks_initial_chain_dimensions(tmp_path: Path):
    chains = tmp_path / "chains.fasta"
    chains.write_text(">chain|log_weight=0\nAAA\n", encoding="utf-8")

    with pytest.raises(ModelCompatibilityError) as context:
        load_training_inputs(
            Alignment(("train",), ("AA",)),
            initial_chains_path=chains,
            config=TrainingConfig(
                alphabet="AB-",
                no_reweighting=True,
                n_chains=2,
            ),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert context.value.details["expected_suffix"] == (2, 3)


def test_training_loader_checks_initial_parameter_dimensions(tmp_path: Path):
    from adabmDCA.io import save_params

    params_path = tmp_path / "params.dat"
    save_params(
        str(params_path),
        {
            "bias": torch.zeros(3, 3),
            "coupling_matrix": torch.ones(3, 3, 3, 3),
        },
        tokens="AB-",
    )

    with pytest.raises(ModelCompatibilityError):
        load_training_inputs(
            Alignment(("train",), ("AA",)),
            initial_params_path=params_path,
            config=TrainingConfig(
                alphabet="AB-",
                no_reweighting=True,
                n_chains=2,
            ),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
