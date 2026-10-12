from pathlib import Path

import numpy as np
import torch

from adabmDCA import Alignment, TrainingConfig
from adabmDCA.api.entropy import estimate_entropy
from adabmDCA.api.reintegration import reintegrate_model
from adabmDCA.io import save_params


def test_reintegration_calls_training_without_cli_recursion():
    result = reintegrate_model(
        Alignment(("n1", "n2", "n3", "n4"), ("AA", "AB", "BA", "BB")),
        Alignment(("e1", "e2"), ("A-", "B-")),
        [-1.0, 1.0],
        lambda_value=1.0,
        config=TrainingConfig(
            alphabet="AB-",
            no_reweighting=True,
            n_chains=8,
            n_sweeps=1,
            max_epochs=1,
            target_pearson=0.1,
            device="cpu",
        ),
    )

    assert result.alignment.num_sequences == 6
    np.testing.assert_allclose(result.weights[-2:], [-2.0, 2.0])
    assert result.training.model.metadata.length == 2
    assert result.training.config.no_reweighting is False


def test_reintegration_trains_and_resumes_with_ptt(tmp_path: Path):
    from dataclasses import replace

    from adabmDCA.ptt import PTTConfig, PTTSampler

    natural = Alignment(tuple(f"n{i}" for i in range(6)), ("AAA", "AAB", "ABA", "BAA", "BBB", "BAB"))
    experimental = Alignment(("e1", "e2", "e3"), ("AB-", "BBA", "A-A"))
    config = TrainingConfig(
        alphabet="AB-", n_chains=40, n_sweeps=1, max_epochs=2, target_pearson=0.999999, device="cpu",
        dtype="float64", pseudocount=0.1, checkpoint_interval=1,
        ptt=PTTConfig(equilibration_rounds=2, initialization_rounds=2, swaps=1,
                      mixing_thermalization_rounds=2, mixing_chains=20),
    )
    partial = reintegrate_model(natural, experimental, [1.0, -1.0, 0.5], config=replace(config, max_epochs=1),
                                output_dir=tmp_path / "partial", label="ptt")
    assert partial.training.config.ptt is not None
    assert (partial.weights[-3:] < 0).sum() == 1
    archive = partial.training.artifacts["ptt_archive"]
    assert PTTSampler.from_archive(archive, mode="resume").training_state

    resumed = reintegrate_model(natural, experimental, [1.0, -1.0, 0.5], config=config, ptt_resume=archive,
                                output_dir=tmp_path / "resumed", label="ptt")
    complete = reintegrate_model(natural, experimental, [1.0, -1.0, 0.5], config=config)
    for key in ("bias", "coupling_matrix"):
        torch.testing.assert_close(resumed.training.model.params[key], complete.training.model.params[key],
                                   rtol=0, atol=0)


def test_reintegrate_cli_defaults_to_pcd_and_accepts_ptt():
    import pytest

    from adabmDCA.exceptions import InputValidationError
    from adabmDCA.scripts.reintegrate import _config_from_args, create_parser

    required = ["-d", "natural.fasta", "--reint", "tested.fasta", "--adj", "adj.txt", "-o", "out"]
    parser = create_parser()
    assert "Training strategy (default: pcd)." in parser.format_help()
    assert _config_from_args(parser.parse_args(required)).ptt is None
    config = _config_from_args(parser.parse_args([*required, "--strategy", "ptt", "--ptt-swaps-factor", "3"]))
    assert config.ptt is not None and config.ptt.swaps == 3
    with pytest.raises(InputValidationError):
        _config_from_args(parser.parse_args([*required, "--ptt-swaps-factor", "3"]))


def test_entropy_api_is_bounded_observable_and_serializable(tmp_path: Path):
    model_path = tmp_path / "params.dat"
    save_params(
        str(model_path),
        {
            "bias": torch.zeros(2, 2),
            "coupling_matrix": torch.ones(2, 2, 2, 2),
        },
        tokens="AB",
    )
    events = []

    result = estimate_entropy(
        model=model_path,
        natural_alignment=Alignment(("a", "b"), ("AA", "BB")),
        target_alignment=Alignment(("target",), ("AA",)),
        n_chains=16,
        n_sweeps=1,
        n_steps=2,
        theta_max=5.0,
        theta_sweeps=1,
        zero_sweeps=1,
        max_theta_iterations=2,
        alphabet="AB",
        seed=4,
        device="cpu",
        output_dir=tmp_path / "entropy",
        progress=events.append,
    )

    assert len(result.history["step"]) == 2
    assert events[-1].stage == "integration"
    assert events[-1].completed == 2
    assert set(result.artifacts) == {"log", "csv", "summary"}
    assert all(path.is_file() for path in result.artifacts.values())
