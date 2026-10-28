"""Custom-model tests cover create, snapshot round-trip, background runs and
read-only detail pages.
"""

from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from core import datasets
from core.metrics import metrics_for
from experiments.models import Experiment
from experiments.services import run as run_service
from experiments.services import snapshot as snapshot_adapter

from tests.ui.custom_models.conftest import VALID_MODEL_SRC
from tests.conftest import post_new_experiment


def _demo_dataset_path():
    """`_demo_dataset_path` names iris since the models here only classify.
    """
    return datasets.demo_datasets()["iris"]


def _custom_snapshot(name="cm", dataset_path=None):
    return {
        "version": "0",
        "name": name,
        "model_name": "My Custom Model",
        "model_path": "",
        "optimizer_name": "Random Search",
        "optimizer_params": {},
        "primary_metric": None,
        "original_metric": None,
        "metric_names": metrics_for("classification"),
        "seed": 0,
        "dataset_path": dataset_path or _demo_dataset_path(),
        "result": None,
    }


def _model_file():
    return SimpleUploadedFile("m.py", VALID_MODEL_SRC.encode("utf-8"))


# create view


def test_create_view_stores_custom_model(client):
    """The new-experiment flow saves an uploaded model file and its resolved
    name."""
    resp = post_new_experiment(client, {
        "name": "cm-create",
        "model_name": "",
        "optimizer_name": "Random Search",
        "demo_dataset": _demo_dataset_path(), "task": "classification",
        "seed": "0",
        "model_file": _model_file(),
    })
    assert resp.status_code == 302
    exp = Experiment.objects.get(data__name="cm-create")
    assert exp.data.model_name == "My Custom Model"
    assert exp.data.model_file  # file stored
    assert Path(exp.data.model_file.path).is_file()


# snapshot adapter


def test_adapter_adopts_uploaded_model_file():
    """`experiment_from_snapshot` adopts the upload under MEDIA."""
    exp = snapshot_adapter.experiment_from_snapshot(
        _custom_snapshot(), model_file=_model_file(),
    )
    assert exp.data.model_file and Path(exp.data.model_file.path).is_file()


def test_snapshot_emits_absolute_loadable_model_path():
    """The run engine cannot resolve a relative model path."""
    exp = snapshot_adapter.experiment_from_snapshot(
        _custom_snapshot(), model_file=_model_file(),
    )
    out = snapshot_adapter.snapshot_from_experiment(exp)
    assert Path(out["model"]["path"]).is_absolute()
    assert Path(out["model"]["path"]).is_file()


# running


def test_execute_run_optimizes_a_custom_model():
    """`execute_run` rebuilds the experiment from its row."""
    exp = snapshot_adapter.experiment_from_snapshot(
        _custom_snapshot(name="cm-run"), model_file=_model_file(), adopt_paths=True,
    )
    run = run_service.create_run(exp, {"max_trials": 3}, "accuracy")
    run_service.execute_run(run.id)

    run.refresh_from_db()
    exp.refresh_from_db()
    assert run.status == "done", run.error
    assert exp.data.result is not None
    assert len(exp.data.result["data"]) == 3


# read-only, gating


def test_custom_model_with_file_is_runnable(client):
    exp = snapshot_adapter.experiment_from_snapshot(
        _custom_snapshot(name="cm-ok"), model_file=_model_file(), adopt_paths=True,
    )
    resp = client.get(reverse("ui:experiment_detail", args=[exp.pk]))
    assert resp.context["can_run"] is True


def test_custom_model_without_file_is_readonly(client):
    """`can_run` is False without the model file, as for a missing dataset."""
    exp = snapshot_adapter.experiment_from_snapshot(_custom_snapshot(name="cm-missing"))
    assert not exp.data.model_file
    resp = client.get(reverse("ui:experiment_detail", args=[exp.pk]))
    assert resp.context["can_run"] is False


def test_custom_model_readonly_when_flag_off(client, settings):
    """`ALLOW_CUSTOM_MODELS=False` blocks runs even with the file present."""
    exp = snapshot_adapter.experiment_from_snapshot(
        _custom_snapshot(name="cm-flagoff"), model_file=_model_file(),
    )
    settings.ALLOW_CUSTOM_MODELS = False
    resp = client.get(reverse("ui:experiment_detail", args=[exp.pk]))
    assert resp.context["can_run"] is False


def test_run_view_refuses_unavailable_custom_model(client):
    """`experiment_run` redirects without creating a run."""
    exp = snapshot_adapter.experiment_from_snapshot(_custom_snapshot(name="cm-norun"))
    resp = client.post(reverse("ui:experiment_run", args=[exp.pk]),
                       {"max_trials": "3", "optimize_metric": "accuracy"})
    assert resp.status_code == 302
    assert exp.runs.count() == 0
