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
