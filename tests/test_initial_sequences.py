"""Plain MCMC started from given sequences, and runs of a fixed number of sweeps."""

import pandas as pd
import pytest
import torch

from adabmDCA import sample_sequences
from adabmDCA.alignment import Alignment
from adabmDCA.api.model import DCAModel
from adabmDCA.exceptions import InputValidationError, ModelCompatibilityError

SEQUENCES = ("AAB", "ABA", "BAA", "ABB", "BBA", "B-A")


def _model():
    return DCAModel({"bias": torch.zeros(3, 3), "coupling_matrix": torch.zeros(3, 3, 3, 3)}, alphabet="AB-")


def _fasta(path, sequences=SEQUENCES):
    path.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(sequences)))
    return path


def _sample(initial, **kwargs):
    return sample_sequences(model=_model(), initial_sequences=initial, n_sweeps=0, device="cpu", **kwargs)


@pytest.mark.parametrize("form", ["strings", "alignment", "path", "result", "indices", "onehot"])
def test_zero_sweeps_return_the_initial_sequences_in_every_input_form(form, tmp_path):
    indices = torch.tensor([["AB-".index(c) for c in s] for s in SEQUENCES])
    initial = {
        "strings": list(SEQUENCES),
        "alignment": Alignment(names=tuple(map(str, range(len(SEQUENCES)))), sequences=SEQUENCES),
        "path": _fasta(tmp_path / "init.fasta"),
        "result": _sample(list(SEQUENCES)),
        "indices": indices.numpy(),
        "onehot": torch.nn.functional.one_hot(indices, 3).float(),
    }[form]

    result = _sample(initial)

    assert result.sequences == SEQUENCES
    assert result.num_sweeps == 0


def test_smaller_n_sequences_starts_from_a_seeded_random_subset():
    first = _sample(list(SEQUENCES), n_sequences=3, seed=4)
    again = _sample(list(SEQUENCES), n_sequences=3, seed=4)

    assert first.sequences == again.sequences
    assert len(set(first.sequences)) == 3 and set(first.sequences) <= set(SEQUENCES)
    with pytest.raises(InputValidationError, match="exceeds"):
        _sample(list(SEQUENCES), n_sequences=7)


def test_invalid_initial_sequences_are_rejected():
    with pytest.raises(ModelCompatibilityError):
        _sample(["AABA"])
    with pytest.raises(InputValidationError):
        _sample(["AXB"])
    with pytest.raises(InputValidationError, match="token indices"):
        _sample(torch.tensor([[0, 1, 3]]))
    with pytest.raises(InputValidationError, match="integer"):
        _sample(torch.zeros(2, 3))
    with pytest.raises(InputValidationError, match="n_sequences is required"):
        sample_sequences(model=_model(), n_sweeps=0, device="cpu")
    with pytest.raises(InputValidationError, match="plain MCMC only"):
        sample_sequences(model="unused.h5", ptt=True, n_sequences=2, initial_sequences=list(SEQUENCES))


def test_fixed_sweeps_skip_the_mixing_time_and_record_from_sweep_zero(tmp_path):
    reference = _fasta(tmp_path / "train.fasta")

    result = sample_sequences(model=_model(), initial_sequences=list(SEQUENCES), n_sweeps=12,
                              reference_fasta=reference, measure_mixing_time=False, record_every=5, device="cpu")

    assert result.num_sweeps == 12
    assert result.mixing_history == {}
    assert result.sampling_history["nsweeps"] == [0, 5, 10, 12]
    assert all(len(values) == 4 for values in result.sampling_history.values())
    # A zero model gives every sequence energy 0.
    assert result.sampling_history["mean_energy"] == [0.0] * 4


def test_cli_starts_plain_mcmc_from_a_fasta_for_a_fixed_number_of_sweeps(tmp_path):
    from adabmDCA.io import save_params
    from adabmDCA.scripts.sample import create_parser, run

    params = tmp_path / "params.dat"
    save_params(str(params), {"bias": torch.full((3, 3), 0.1), "coupling_matrix": torch.zeros(3, 3, 3, 3)},
                tokens="AB-")
    reference = _fasta(tmp_path / "train.fasta")
    args = create_parser().parse_args([
        "-p", str(params), "-o", str(tmp_path / "out"), "--strategy", "pcd", "--alphabet", "AB-",
        "-d", str(reference), "--init-sequences", str(reference), "--nsweeps", "3", "--record-every", "2",
        "--device", "cpu",
    ])

    result = run(args)
    artifacts = result.save_bundle(tmp_path / "out")

    assert len(result.sequences) == len(SEQUENCES)
    assert result.num_sweeps == 3
    log = pd.read_csv(artifacts["sampling_log"])
    assert list(log["nsweeps"]) == [0, 2, 3]
    assert {"pearson", "slope", "mean_energy"} <= set(log.columns)
