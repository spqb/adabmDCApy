from pathlib import Path
from unittest.mock import patch

import pytest

from adabmDCA import Alignment, train_model
from adabmDCA.dataset import DatasetDCA
from adabmDCA.input_loading import AlignmentLoadConfig
from adabmDCA.plot_training_log import create_plots, parse_training_log


def _training_kwargs(tmp_path: Path, *, model_type: str = "bmDCA"):
    return {
        "data_path": Alignment(("s1", "s2", "s3", "s4"), ("AA", "AB", "BA", "BB")),
        "validation_path": Alignment(("v1", "v2"), ("AA", "BB")),
        "output_dir": tmp_path,
        "label": model_type,
        "model_type": model_type,
        "alphabet": "AB",
        "device": "cpu",
        "no_reweighting": True,
        "n_chains": 8,
        "n_sweeps": 1,
        "max_epochs": 1,
        "target_pearson": 0.99,
        "seed": 3,
    }


def test_effective_size_preserves_fractional_weights():
    dataset = DatasetDCA.from_alignment(
        Alignment(("a", "b", "c"), ("AA", "AB", "BB")),
        weights=[0.25, 0.5, 0.75],
        load_config=AlignmentLoadConfig(alphabet="AB"),
    )
    assert dataset.get_effective_size() == pytest.approx(1.5)


def test_initialization_is_emitted_once_and_serialized(tmp_path: Path):
    initialized = []
    result = train_model(**_training_kwargs(tmp_path), on_initialized=initialized.append)

    assert len(initialized) == 1
    info = initialized[0]
    assert info.training.retained_sequences == 4
    assert info.training.sequence_length == 2
    assert info.training.num_states == 2
    assert info.training.effective_sequences == 4.0
    assert info.validation is not None
    assert info.validation.retained_sequences == 2
    assert result.initialization is info
    assert result.to_dict()["data"]["initialization"]["training"]["sequence_length"] == 2


def test_versioned_log_has_resolved_metadata_progress_and_end_summary(tmp_path: Path):
    result = train_model(**_training_kwargs(tmp_path, model_type="edgeDCA"), pseudocount=0.3)
    text = result.artifacts["log"].read_text(encoding="utf-8")

    assert text.startswith("adabmDCA training log\nformat_version: 2")
    assert "[TRAINING DATA]" in text
    assert "retained_sequences:   4" in text
    assert "effective_sequences:  4" in text
    assert "max_structure_steps:    1" in text
    assert "effective_pseudocount:  0.3" in text
    assert "learning_rate" not in text
    assert text.count("Step     Stage") == 1
    assert "Chain_ESS_frac" in text
    assert "[END]\nstatus: completed" in text
    assert "stop_reason: max_structure_steps" in text

    metadata, data = parse_training_log(str(result.artifacts["log"]))
    assert metadata["run.model"] == "edgeDCA"
    assert metadata["training data.sequence_length"] == "2"
    assert metadata["optimization.target_pearson"] == "0.99"
    assert data["Epochs"].tolist() == [1.0]
    assert data["Stage"].tolist() == ["activation"]
    assert data["Structure_steps"].tolist() == [1.0]
    assert data["Sweeps"].tolist() == [1.0]


def test_cli_configuration_contains_resolved_dataset_statistics(tmp_path: Path, capsys):
    from adabmDCA.scripts.train import create_parser, main

    fasta = tmp_path / "input.fa"
    fasta.write_text(">s1\nAA\n>s2\nAB\n>s3\nBA\n>s4\nBB\n", encoding="utf-8")
    args = create_parser().parse_args([
        "--data", str(fasta), "--output", str(tmp_path / "output"), "--alphabet", "AB",
        "--device", "cpu", "--nchains", "8", "--nsweeps", "1", "--nepochs", "1",
        "--target", "0.99", "--no_reweighting", "--no-progress",
    ])
    assert main(args) == 0
    output = capsys.readouterr().out
    assert "training sequences: 4" in output
    assert "sequence length: 2" in output
    assert "alphabet states: 2" in output
    assert "effective sequences (M_eff): 4" in output
    assert "device: cpu" in output


def test_failed_training_writes_terminal_status(tmp_path: Path):
    with (
        patch("adabmDCA.api.training.train_graph", side_effect=RuntimeError("numerical failure")),
        pytest.raises(RuntimeError, match="numerical failure"),
    ):
        train_model(**_training_kwargs(tmp_path))
    text = (tmp_path / "bmDCA.log").read_text(encoding="utf-8")
    assert "[END]\nstatus: failed" in text
    assert "error: RuntimeError: numerical failure" in text


def test_plotter_creates_plots_from_versioned_log(tmp_path: Path):
    result = train_model(**_training_kwargs(tmp_path))
    metadata, data = parse_training_log(str(result.artifacts["log"]))
    plots = tmp_path / "plots"
    create_plots(metadata, data, str(plots))
    assert {path.name for path in plots.glob("*.png")} == {
        "bmDCA_diagnostics.png", "bmDCA_entropy.png", "bmDCA_loglikelihood.png",
        "bmDCA_pearson.png", "bmDCA_slope.png",
    }
