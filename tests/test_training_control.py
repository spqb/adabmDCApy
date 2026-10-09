from unittest.mock import patch

import pytest
import torch

from adabmDCA.training import train_eaDCA, train_edDCA
from adabmDCA.training_control import (
    StopReason,
    TrainingCancelled,
    TrainingController,
    TrainingLimits,
    TrainingMetrics,
)


def _metrics() -> TrainingMetrics:
    return TrainingMetrics(
        pearson=0.2,
        slope=0.8,
        pearson_val=float("nan"),
        slope_val=float("nan"),
        density=0.1,
        elapsed_time=0.01,
    )


def _tiny_state():
    fi = torch.full((2, 2), 0.5)
    fij = torch.full((2, 2, 2, 2), 0.25)
    params = {
        "bias": torch.zeros(2, 2),
        "coupling_matrix": torch.zeros(2, 2, 2, 2),
    }
    mask = torch.zeros_like(params["coupling_matrix"], dtype=torch.bool)
    chains = torch.nn.functional.one_hot(torch.zeros((4, 2), dtype=torch.int64), num_classes=2).float()
    return fi, fij, params, mask, chains


def test_controller_owns_history_progress_and_checkpoint_schedule():
    class Store:
        def __init__(self):
            self.logs = []
            self.saves = []
            self.stages = []

        def begin_stage(self, stage, metadata):
            self.stages.append((stage, metadata))

        def log(self, record):
            self.logs.append(record)

        def check(self, updates):
            return updates == 2

        def save(self, **snapshot):
            self.saves.append(snapshot)

    store = Store()
    observed = []
    controller = TrainingController(
        checkpoint=store,
        observer=lambda record, counters: observed.append((record["Epochs"], counters.gradient_steps)),
    )
    snapshot = {"params": {}, "mask": None, "chains": None}
    controller.begin_stage("optimization", target_pearson=0.95)

    for step in (1, 2):
        controller.add_gradient_steps(1, sweeps_per_step=3)
        controller.record(_metrics(), epoch=step, snapshot=snapshot)
    controller.finalize(snapshot)
    controller.finalize(snapshot)

    assert controller.history["Epochs"] == [1, 2]
    assert observed == [(1, 1), (2, 2)]
    assert len(store.logs) == 2
    # The scheduled checkpoint at the last step already holds the final state.
    assert len(store.saves) == 1
    assert store.stages == [("optimization", {"target_pearson": 0.95})]
    assert controller.counters.stage == "optimization"
    assert controller.counters.sweeps == 6


def test_stage_updates_do_not_append_metric_rows():
    stages = []
    metrics = []
    controller = TrainingController(stage_observer=stages.append,
                                    observer=lambda row, counters: metrics.append(row))
    controller.begin_stage("ptt_mixing", reason="overlap")
    controller.report_stage_progress("mixing_warmup", 25, 100, chains=20)
    controller.report_stage_progress("mixing_warmup", 100, 100, chains=20)
    assert [(event.stage, event.kind, event.current, event.total) for event in stages] == [
        ("ptt_mixing", "start", None, None),
        ("mixing_warmup", "progress", 25, 100),
        ("mixing_warmup", "progress", 100, 100),
    ]
    assert stages[1].details == {"chains": 20}
    assert controller.history["Epochs"] == [] and metrics == []
    controller.record(_metrics(), epoch=0)
    assert len(metrics) == len(controller.history["Epochs"]) == 1


def test_controller_reports_cancellation():
    controller = TrainingController(is_cancelled=lambda: True)

    with pytest.raises(TrainingCancelled):
        controller.check_cancellation()

    assert controller.stop_reason is StopReason.CANCELLED


def test_eadca_tracks_nested_gradient_and_structure_steps():
    fi, fij, params, mask, chains = _tiny_state()
    controller = TrainingController(limits=TrainingLimits(max_structure_steps=2))

    def inner_training(**kwargs):
        return kwargs["chains"], kwargs["params"], {"Epochs": [1, 2, 3]}

    with (
        patch("adabmDCA.training.train_graph", side_effect=inner_training),
        patch("adabmDCA.training.activate_graph_elements", side_effect=lambda **kwargs: kwargs["mask"]),
        patch("adabmDCA.training.get_freq_single_point", return_value=fi),
        patch("adabmDCA.training.get_freq_two_points", return_value=fij),
        patch("adabmDCA.training.get_correlation_two_points", return_value=(0.0, 0.0)),
        patch("adabmDCA.training.compute_density", return_value=0.0),
    ):
        _, _, history = train_eaDCA(
            sampler=lambda **kwargs: kwargs["chains"],
            fi_target=fi,
            fij_target=fij,
            params=params,
            mask=mask,
            chains=chains,
            target_pearson=0.95,
            nsweeps=2,
            max_epochs=2,
            pseudo_count=0.1,
            lr=0.01,
            factivate=0.1,
            gsteps=3,
            controller=controller,
        )

    assert history["Epochs"] == [1, 2]
    assert controller.counters.gradient_steps == 6
    assert controller.counters.structure_steps == 2
    assert controller.counters.sweeps == 12
    assert controller.stop_reason is StopReason.MAX_STRUCTURE_STEPS


def test_eddca_respects_structure_budget_and_tracks_inner_steps(capsys):
    fi, fij, params, mask, chains = _tiny_state()
    mask.fill_(True)
    controller = TrainingController(limits=TrainingLimits(max_structure_steps=2))

    def inner_training(**kwargs):
        return kwargs["chains"], kwargs["params"], {"Epochs": [1, 2]}

    with (
        patch("adabmDCA.training.train_graph", side_effect=inner_training),
        patch("adabmDCA.training.decimate_graph", side_effect=lambda **kwargs: (kwargs["params"], kwargs["mask"])),
        patch("adabmDCA.training.get_freq_single_point", return_value=fi),
        patch("adabmDCA.training.get_freq_two_points", return_value=fij),
        patch("adabmDCA.training.get_correlation_two_points", return_value=(0.9, 1.0)),
        patch("adabmDCA.training.compute_density", return_value=1.0),
    ):
        _, _, history = train_edDCA(
            sampler=lambda **kwargs: kwargs["chains"],
            chains=chains,
            fi_target=fi,
            fij_target=fij,
            params=params,
            mask=mask,
            lr=0.01,
            nsweeps=2,
            target_pearson=0.8,
            target_density=0.1,
            drate=0.1,
            max_epochs=2,
            controller=controller,
        )

    assert history["Epochs"] == [1, 2]
    assert controller.counters.gradient_steps == 4
    assert controller.counters.structure_steps == 2
    assert controller.counters.sweeps == 12
    assert controller.counters.stage == "decimation"
    assert controller.stop_reason is StopReason.MAX_STRUCTURE_STEPS
    assert capsys.readouterr().out == ""
