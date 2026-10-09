import json
from pathlib import Path
from unittest.mock import patch

import pytest

from adabmDCA import Alignment, train_model
from adabmDCA.dataset import DatasetDCA
from adabmDCA.input_loading import AlignmentLoadConfig
from adabmDCA.plot_training_log import parse_training_log, plot_training_history, read_training_history


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

    assert text.startswith("adabmDCA training log\nformat_version: 4")
    assert "history: edgeDCA_history.csv\nevents: edgeDCA_events.jsonl" in text
    assert "[TRAINING DATA]" in text
    assert "retained_sequences:   4" in text
    assert "effective_sequences:  4" in text
    assert "max_structure_steps:    1" in text
    assert "effective_pseudocount:  0.3" in text
    assert "learning_rate" not in text
    assert text.count("[PROGRESS]") == 1
    progress = text.split("[PROGRESS]\n")[1].split("\n\n[END]")[0].splitlines()
    assert progress[0].startswith("»     0  activation: target_pearson 0.99")
    assert progress[1].split() == ["step", "grad", "sweeps", "time", "pearson", "slope", "val", "density"]
    assert progress[2].split()[:3] == ["1", "0", "1"]
    for removed in ("Chain_ESS_frac", "LL_train", "LL_val", "Entropy"):
        assert removed not in text
        assert removed not in result.history
    assert "logZ" not in result.history
    assert all("log_weight=" not in line for line in result.artifacts["chains"].read_text().splitlines())
    assert "[END]\nstatus: completed" in text
    assert "stop_reason: max_structure_steps" in text

    header, row = result.artifacts["history"].read_text(encoding="utf-8").splitlines()
    assert header == "step,gradient_steps,stage,sweeps,time_s,pearson,slope,pearson_val,slope_val,density"
    assert row.startswith("1,0,activation,1,")
    events = [json.loads(line) for line in result.artifacts["events"].read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["activation", "checkpoint_saved"]
    assert events[0]["target_pearson"] == 0.99 and events[1]["step"] == 1

    metadata, data = parse_training_log(str(result.artifacts["log"]))
    assert metadata["run.model"] == "edgeDCA"
    assert metadata["training data.sequence_length"] == "2"
    assert metadata["optimization.target_pearson"] == "0.99"
    assert data["Epochs"].tolist() == [1.0]
    assert data["Stage"].tolist() == ["activation"]
    assert data["Gradient_steps"].tolist() == [0.0]
    assert data["Sweeps"].tolist() == [1.0]


def test_history_is_written_during_training(tmp_path: Path):
    rows = []

    def observe(event):
        rows.append(len((tmp_path / "bmDCA_history.csv").read_text(encoding="utf-8").splitlines()) - 1)

    with patch("adabmDCA.training.get_correlation_two_points", return_value=(0.5, 1.0)):
        result = train_model(**{**_training_kwargs(tmp_path), "max_epochs": 3}, progress=observe)
    assert rows == [1, 2, 3]
    assert "history" not in result.save_bundle(tmp_path, label="bmDCA")
    header = result.artifacts["history"].read_text(encoding="utf-8").splitlines()[0]
    assert header == "step,sweeps,time_s,pearson,slope,pearson_val,slope_val"
    elsewhere = result.to_csv(tmp_path / "copy.csv")
    assert elsewhere.read_text(encoding="utf-8") == result.artifacts["history"].read_text(encoding="utf-8")


def test_cli_configuration_contains_resolved_dataset_statistics(tmp_path: Path, capsys):
    from adabmDCA.scripts.train import create_parser, main

    fasta = tmp_path / "input.fa"
    fasta.write_text(">s1\nAA\n>s2\nAB\n>s3\nBA\n>s4\nBB\n", encoding="utf-8")
    args = create_parser().parse_args([
        "--data", str(fasta), "--output", str(tmp_path / "output"), "--alphabet", "AB",
        "--strategy", "pcd",
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


def test_plotter_draws_the_recorded_quantities_from_the_history_table(tmp_path: Path):
    with patch("adabmDCA.training.get_correlation_two_points", return_value=(0.5, 1.0)):
        result = train_model(**{**_training_kwargs(tmp_path), "max_epochs": 3})
    written = plot_training_history(result.artifacts["history"], tmp_path / "plots")
    assert set(written) == {"pearson", "slope", "overview"}
    assert all(path.is_file() and path.parent == tmp_path / "plots" for path in written.values())
    # The log points to its history table; the default folder is <label>_plots next to it.
    assert set(plot_training_history(result.artifacts["log"])) == set(written)
    assert (tmp_path / "bmDCA_plots" / "bmDCA_overview.png").is_file()
    history, context = read_training_history(result.artifacts["history"])
    assert list(history["step"]) == result.history["Epochs"]
    assert context["label"] == "bmDCA" and context["model"] == "bmDCA" and context["target_pearson"] == 0.99


def test_plotter_reads_tables_written_before_format_4(tmp_path: Path):
    table = tmp_path / "history.csv"
    table.write_text("Epochs,Pearson,Slope,Pearson_val,Slope_val,Density,Time\n1,0.5,0.4,,,1,0.1\n2,0.6,0.5,,,1,0.2\n")
    history, context = read_training_history(table)
    assert list(history.columns[:3]) == ["step", "pearson", "slope"] and context["label"] == "training"
    assert set(plot_training_history(table)) == {"pearson", "slope", "overview"}


def test_plotter_reads_logs_written_before_format_4(tmp_path: Path):
    log = tmp_path / "old.log"
    log.write_text(
        "adabmDCA training log\nformat_version: 3\n\n[RUN]\nlabel:  old\n\n[STAGE OPTIMIZATION]\n\n"
        "Step     Stage          Grad_steps   Struct_steps  Sweeps       Pearson     Slope       Pearson_val   "
        "Slope_val    Density      Elapsed_s\n"
        "1        optimization   1            0             10           0.5         0.4         nan           "
        "nan          1            0.1\n",
        encoding="utf-8",
    )
    metadata, data = parse_training_log(str(log))
    assert metadata["run.label"] == "old"
    assert data["Pearson"].tolist() == [0.5] and data["Stage"].tolist() == ["optimization"]
