import subprocess
import sys
import warnings

import pytest
import torch

from adabmDCA import Alignment
from adabmDCA.api.entropy import estimate_entropy
from adabmDCA.api.exceptions import InputValidationError
from adabmDCA.io import save_params


@pytest.fixture
def entropy_inputs(tmp_path):
    model = tmp_path / "params.dat"
    save_params(str(model), {
        "bias": torch.tensor([[0.2, -0.1], [0.1, -0.2]]),
        "coupling_matrix": torch.zeros(2, 2, 2, 2),
    }, tokens="AB")
    natural = tmp_path / "natural.fa"
    natural.write_text(">a\nAA\n>b\nBB\n")
    return dict(
        model=model, natural_alignment=natural, n_chains=16, n_sweeps=1,
        n_steps=2, theta_sweeps=1, zero_sweeps=1, max_theta_iterations=2,
        alphabet="AB", device="cpu", seed=4,
    )


@pytest.mark.parametrize("sequences", [("BB", "AA"), ("AA", "BB"), ("BB", "BB"), ("XX", "BB", "AA")])
def test_entropy_warns_and_matches_first_valid_target(entropy_inputs, sequences):
    names = tuple(f"target{i}" for i in range(len(sequences)))
    first = next(i for i, sequence in enumerate(sequences) if "X" not in sequence)
    with pytest.warns(UserWarning, match=rf"using only the first \('target{first}'\)"):
        actual = estimate_entropy(target_alignment=Alignment(names, sequences), **entropy_inputs)
    with warnings.catch_warnings(record=True) as caught:
        expected = estimate_entropy(
            target_alignment=Alignment((names[first],), (sequences[first],)), **entropy_inputs,
        )
    assert not any("using only the first" in str(item.message) for item in caught)
    assert actual.entropy == expected.entropy
    assert actual.free_energy == expected.free_energy
    assert actual.history["mean_sequence_identity"] == expected.history["mean_sequence_identity"]


def test_entropy_still_rejects_targets_without_valid_sequences(entropy_inputs):
    with pytest.raises(InputValidationError, match="All sequences were removed"):
        estimate_entropy(target_alignment=Alignment(("bad",), ("XX",)), **entropy_inputs)


def test_entropy_cli_completes_and_warns_for_multiple_targets(entropy_inputs, tmp_path):
    target = tmp_path / "targets.fa"
    target.write_text(">first\nBB\n>second\nAA\n")
    output = tmp_path / "entropy"
    completed = subprocess.run([
        sys.executable, "-m", "adabmDCA.cli", "entropy",
        "-p", str(entropy_inputs["model"]), "-d", str(entropy_inputs["natural_alignment"]),
        "-t", str(target), "-o", str(output), "--alphabet", "AB", "--device", "cpu",
        "--nchains", "16", "--nsteps", "2", "--nsweeps", "1",
        "--nsweeps_theta", "1", "--nsweeps_zero", "1", "--seed", "4",
    ], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "using only the first ('first')" in completed.stderr
    assert "completed successfully" in completed.stdout
    assert (output / "entropy.log").is_file()
