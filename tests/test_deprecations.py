import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from adabmDCA.dataset import DatasetDCA
from adabmDCA.fasta import import_from_fasta, write_fasta
from adabmDCA.input_loading import AlignmentLoadConfig
from adabmDCA.io import load_chains, save_chains


def test_legacy_fasta_import_warns_and_preserves_filter_mask(tmp_path: Path):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(
        ">first\nAA\n>invalid\nAX\n>duplicate\nAA\n>last\nBB\n",
        encoding="utf-8",
    )

    with pytest.warns(DeprecationWarning, match="import_from_fasta is deprecated"):
        names, sequences, mask = import_from_fasta(
            str(fasta),
            tokens="AB-",
            filter_sequences=True,
            remove_duplicates=True,
            return_mask=True,
        )

    assert names.tolist() == ["first", "last"]
    assert sequences.tolist() == [[0, 0], [1, 1]]
    assert mask.tolist() == [True, False, False, True]


def test_legacy_fasta_writer_warns_and_delegates_to_alignment_io(tmp_path: Path):
    fasta = tmp_path / "output.fasta"

    with pytest.warns(DeprecationWarning, match="write_fasta is deprecated"):
        write_fasta(
            str(fasta),
            headers=["first", "second"],
            sequences=np.asarray([[0, 1], [1, 2]]),
            tokens="AB-",
        )

    assert fasta.read_text(encoding="utf-8") == ">first\nAB\n>second\nB-\n"


def test_legacy_dataset_construction_warns_but_factories_do_not(tmp_path: Path):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">first\nAA\n>second\nBB\n", encoding="utf-8")

    with pytest.warns(DeprecationWarning, match=r"DatasetDCA\(path_data="):
        legacy = DatasetDCA(
            fasta,
            alphabet="AB-",
            no_reweighting=True,
            message=False,
        )
    assert len(legacy) == 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        current = DatasetDCA.from_alignment(
            fasta,
            load_config=AlignmentLoadConfig(alphabet="AB-"),
            no_reweighting=True,
        )
    assert len(current) == 2
    assert not [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]


def test_internal_chain_io_does_not_call_deprecated_fasta_helpers(tmp_path: Path):
    fasta = tmp_path / "chains.fasta"
    chains = torch.tensor([[0, 1], [1, 2]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save_chains(str(fasta), chains, tokens="AB-")
        loaded = load_chains(str(fasta), tokens="AB-")[0]

    assert loaded.argmax(dim=-1).tolist() == chains.tolist()
    assert not [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]
