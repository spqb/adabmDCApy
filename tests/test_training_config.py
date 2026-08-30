import inspect

import pytest

from adabmDCA.api.training import train_model
from adabmDCA.parser import add_args_train
from adabmDCA.training_config import (
    DEFAULT_ACTIVATION_FRACTION,
    DEFAULT_ACTIVATION_STEPS,
    DEFAULT_ALPHABET,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_CLUSTERING_SEQID,
    DEFAULT_DECIMATION_RATE,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_L2_REGULARIZATION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MODEL_TYPE,
    DEFAULT_N_CHAINS,
    DEFAULT_N_SWEEPS,
    DEFAULT_SAMPLER,
    DEFAULT_SEED,
    DEFAULT_TARGET_DENSITY,
    DEFAULT_TARGET_PEARSON,
    EDGE_DEFAULT_PSEUDOCOUNT,
    EDGE_EMPIRICAL_PSEUDOCOUNT,
    SPARSE_CHECKPOINT_INTERVAL,
    ConfigurationError,
    TrainingConfig,
)


def test_api_and_cli_defaults_share_canonical_values():
    assert DEFAULT_SAMPLER == "metropolis"
    expected = {
        "model_type": DEFAULT_MODEL_TYPE,
        "alphabet": DEFAULT_ALPHABET,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "n_sweeps": DEFAULT_N_SWEEPS,
        "sampler": DEFAULT_SAMPLER,
        "n_chains": DEFAULT_N_CHAINS,
        "target_pearson": DEFAULT_TARGET_PEARSON,
        "max_epochs": DEFAULT_MAX_EPOCHS,
        "l2_regularization": DEFAULT_L2_REGULARIZATION,
        "seed": DEFAULT_SEED,
        "clustering_seqid": DEFAULT_CLUSTERING_SEQID,
        "activation_steps": DEFAULT_ACTIVATION_STEPS,
        "activation_fraction": DEFAULT_ACTIVATION_FRACTION,
        "target_density": DEFAULT_TARGET_DENSITY,
        "decimation_rate": DEFAULT_DECIMATION_RATE,
        "device": DEFAULT_DEVICE,
        "dtype": DEFAULT_DTYPE,
    }
    signature = inspect.signature(train_model)
    for name, value in expected.items():
        assert signature.parameters[name].default == value

    import argparse

    parser = add_args_train(argparse.ArgumentParser())
    assert parser.get_default("model") == DEFAULT_MODEL_TYPE
    assert parser.get_default("lr") == DEFAULT_LEARNING_RATE
    assert parser.get_default("nepochs") == DEFAULT_MAX_EPOCHS
    assert parser.get_default("gsteps") == DEFAULT_ACTIVATION_STEPS
    assert parser.get_default("density") == DEFAULT_TARGET_DENSITY
    assert parser.get_default("sampler") == "metropolis"

    from adabmDCA.api.entropy import estimate_entropy
    from adabmDCA.api.model import DCAModel
    from adabmDCA.api.sampling import sample_sequences

    assert inspect.signature(sample_sequences).parameters["sampler"].default == "metropolis"
    assert inspect.signature(estimate_entropy).parameters["sampler"].default == "metropolis"
    assert inspect.signature(DCAModel.sample).parameters["sampler"].default == "metropolis"


def test_model_aware_limits_and_checkpoint_intervals():
    dense = TrainingConfig(max_epochs=12)
    assert dense.limits.max_gradient_steps == 12
    assert dense.limits.max_structure_steps is None
    assert dense.resolved_checkpoint_interval == DEFAULT_CHECKPOINT_INTERVAL

    sparse = TrainingConfig(model_type="eaDCA", max_epochs=7)
    assert sparse.limits.max_gradient_steps is None
    assert sparse.limits.max_structure_steps == 7
    assert sparse.resolved_checkpoint_interval == SPARSE_CHECKPOINT_INTERVAL

    explicit = TrainingConfig(
        model_type="edDCA",
        max_epochs=20,
        max_gradient_steps=30,
        max_structure_steps=4,
        checkpoint_interval=3,
    )
    assert explicit.limits.max_gradient_steps == 30
    assert explicit.limits.max_structure_steps == 4
    assert explicit.resolved_checkpoint_interval == 3


def test_model_aware_pseudocount_resolution():
    dense = TrainingConfig()
    edge = TrainingConfig(model_type="edgeDCA")

    assert dense.resolve_pseudocount(25.0) == pytest.approx(0.04)
    assert dense.empirical_pseudocount is None
    assert edge.resolve_pseudocount(25.0) == EDGE_DEFAULT_PSEUDOCOUNT
    assert edge.empirical_pseudocount == EDGE_EMPIRICAL_PSEUDOCOUNT


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_type": "unknown"}, "Unsupported model_type"),
        ({"n_chains": 0}, "n_chains must be positive"),
        ({"target_pearson": 1.0}, "target_pearson"),
        ({"pseudocount": 1.1}, "pseudocount"),
        ({"model_type": "eaDCA", "activation_fraction": 0.0}, "activation_fraction"),
        ({"model_type": "edDCA", "decimation_rate": 0.0}, "decimation_rate"),
        ({"checkpoint_interval": 0}, "checkpoint_interval"),
        ({"inner_gradient_steps": 0}, "inner_gradient_steps"),
    ],
)
def test_invalid_configuration_fails_at_construction(kwargs, message):
    with pytest.raises(ConfigurationError, match=message):
        TrainingConfig(**kwargs)


def test_advanced_runtime_values_are_serialized():
    config = TrainingConfig(
        checkpoint_interval=3,
        inner_gradient_steps=17,
        slope_tolerance=0.05,
        edge_logz_chain_fraction=0.3,
    )

    serialized = config.as_dict()

    assert serialized["checkpoint_interval"] == 3
    assert serialized["inner_gradient_steps"] == 17
    assert serialized["slope_tolerance"] == 0.05
    assert serialized["edge_logz_chain_fraction"] == 0.3
