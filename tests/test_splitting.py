"""Behavioral tests for the cluster-preserving split."""

import json
import subprocess
import sys

import pytest

from adabmDCA.alignment import Alignment
from adabmDCA.exceptions import InputValidationError
from adabmDCA.api.splitting import split_alignment


def test_clustering_keeps_whole_groups_and_all_records():
    alignment = Alignment(
        ("a0", "a1", "a2", "b0", "b1", "c0", "c1", "d0", "d1"),
        ("AAAAAAAAAA", "AAAAAAAAAA", "AAAAAAAATA",
         "CCCCCCCCCC", "CCCCCCCCCT", "GGGGGGGGGG", "GGGGGGGGGT",
         "TTTTTTTTTT", "TTTTTTTTTA"),
    )
    result = split_alignment(alignment, alphabet="ACGT", identity=0.8, train_fraction=0.7, seed=4, device="cpu")
    train = set(result.training.names)
    test = set(result.test.names)
    assert train.isdisjoint(test)
    assert train | test == set(alignment.names)
    for group in ({"a0", "a1", "a2"}, {"b0", "b1"}, {"c0", "c1"}, {"d0", "d1"}):
        assert group <= train or group <= test
    assert len(train) == 5 or len(train) == 6 or len(train) == 7
    assert result.method == "clustering"
    assert result.to_dict()["data"]["identity"] == 0.8


def test_clustering_reproducible_and_rejects_unsplittable_alignment():
    alignment = Alignment(("a", "b", "c", "d"), ("AAAA", "CCCC", "GGGG", "TTTT"))
    first = split_alignment(alignment, alphabet="ACGT", seed=19, device="cpu")
    second = split_alignment(alignment, alphabet="ACGT", seed=19, device="cpu")
    assert first.training.names == second.training.names
    with pytest.raises(InputValidationError, match="fewer than two clusters"):
        split_alignment(Alignment(("a", "b"), ("AAAA", "AAAA")), alphabet="ACGT", device="cpu")


@pytest.mark.parametrize("kwargs", [{"identity": 0}, {"identity": 1.1}, {"train_fraction": 0},
                                    {"train_fraction": 1}, {"method": "unknown"}])
def test_invalid_split_options(kwargs):
    with pytest.raises(InputValidationError):
        split_alignment("missing.fasta", **kwargs)


def test_split_data_cli_defaults_and_cobalt_option(tmp_path):
    source = tmp_path / "input.fasta"
    source.write_text(">a\nAAAAAAAAAA\n>b\nCCCCCCCCCC\n>c\nGGGGGGGGGG\n>d\nTTTTTTTTTT\n")
    prefix = tmp_path / "split"
    command = [sys.executable, "-m", "adabmDCA.cli", "split-data", str(prefix), str(source),
               "--alphabet", "ACGT", "--device", "cpu"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    document = json.loads((tmp_path / "split.split.json").read_text())
    assert document["data"]["method"] == "clustering"
    assert sum((tmp_path / name).read_text().count(">") for name in ("split.train.fasta", "split.test.fasta")) == 4
    help_result = subprocess.run([sys.executable, "-m", "adabmDCA.cli", "split-data", "--help"],
                                 text=True, capture_output=True, check=False)
    assert "Split an alignment into training and test sets with reduced homology correlation" in " ".join(help_result.stdout.split())
    assert "--method {clustering,cobalt}" in help_result.stdout
    assert "--identity" in help_result.stdout
    assert "clustering options:" in help_result.stdout
    assert "cobalt options:" in help_result.stdout
    assert "MMseqs2-like sequence clustering" in help_result.stdout
    assert "Petti & Eddy, PLoS Comput Biol 18(3):e1009492, 2022" in help_result.stdout
    assert help_result.stdout.index("clustering options:") < help_result.stdout.index("cobalt options:")


def test_cobalt_remains_available():
    alignment = Alignment(("a", "b", "c", "d"), ("AAAA", "CCCC", "GGGG", "TTTT"))
    result = split_alignment(alignment, method="cobalt", alphabet="ACGT", seed=2, device="cpu", attempts=3)
    assert result.method == "cobalt"
    assert result.training.num_sequences > 0
    assert result.test.num_sequences > 0
