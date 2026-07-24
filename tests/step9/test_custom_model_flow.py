"""Step 9: create, persist, run, and import/export a custom-model experiment.

End-to-end coverage of the web layer around an uploaded model .py:
* the create view stores the file and the resolved model name;
* the snapshot adapter round-trips the model file (adopt in, absolute path out
  so a run can actually load it);
* the background engine runs a custom-model experiment;
* the detail page treats a custom model whose file is missing (or whose feature
  flag is off) as read-only, exactly like a missing dataset.
"""

from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from core import io
from web.models import Experiment
from web.services import run as run_service
from web.services import snapshot as snapshot_adapter

from tests.step9.conftest import VALID_MODEL_SRC


def _demo_dataset_path():
    return next(iter(io.demo_datasets().values()))


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
        "metric_names": list(run_service.METRICS),
        "seed": 0,
        "dataset_path": dataset_path or _demo_dataset_path(),
        "result": None,
    }


def _model_file():
    return SimpleUploadedFile("m.py", VALID_MODEL_SRC.encode("utf-8"))


# --- create view --------------------------------------------------------------


def test_create_view_stores_custom_model(client):
    """POSTing the new-experiment form with a model .py persists the file and
    the resolved model name, then redirects to the detail page."""
    resp = client.post(reverse("web:new_experiment"), {
        "name": "cm-create",
        "model_name": "",
        "optimizer_name": "Random Search",
        "demo_dataset": _demo_dataset_path(),
        "seed": "0",
        "model_file": _model_file(),
    })
    assert resp.status_code == 302
    exp = Experiment.objects.get(name="cm-create")
    assert exp.model_name == "My Custom Model"
    assert exp.model_file  # a file was stored
    assert Path(exp.model_file.path).is_file()


# --- snapshot adapter ---------------------------------------------------------


def test_adapter_adopts_uploaded_model_file():
    """experiment_from_snapshot stores an uploaded model file under MEDIA."""
    exp = snapshot_adapter.experiment_from_snapshot(
        _custom_snapshot(), model_file=_model_file(),
    )
    assert exp.model_file and Path(exp.model_file.path).is_file()


def test_snapshot_emits_absolute_loadable_model_path():
    """snapshot_from_experiment must emit an absolute, on-disk model_path so the
    run engine can load it (a relative name would not resolve)."""
    exp = snapshot_adapter.experiment_from_snapshot(
        _custom_snapshot(), model_file=_model_file(),
    )
    out = snapshot_adapter.snapshot_from_experiment(exp)
    assert Path(out["model_path"]).is_absolute()
    assert Path(out["model_path"]).is_file()


# --- running ------------------------------------------------------------------


def test_execute_run_optimizes_a_custom_model():
    """The background engine rebuilds a custom-model experiment from its row and
    runs it to completion, producing trials."""
    exp = snapshot_adapter.experiment_from_snapshot(
        _custom_snapshot(name="cm-run"), model_file=_model_file(),
    )
    run = run_service.create_run(exp, n_trials=3, optimize_metric="accuracy")
    run_service.execute_run(run.id)

    run.refresh_from_db()
    exp.refresh_from_db()
    assert run.status == "done", run.error
    assert exp.result is not None
    assert len(exp.result["data"]) == 3


# --- read-only / gating -------------------------------------------------------


def test_custom_model_with_file_is_runnable(client):
    exp = snapshot_adapter.experiment_from_snapshot(
        _custom_snapshot(name="cm-ok"), model_file=_model_file(),
    )
    resp = client.get(reverse("web:experiment_detail", args=[exp.pk]))
    assert resp.context["can_run"] is True


def test_custom_model_without_file_is_readonly(client):
    """An experiment naming a non-registry model with no stored file cannot run
    (its model is unavailable) — same treatment as a missing dataset."""
    exp = snapshot_adapter.experiment_from_snapshot(_custom_snapshot(name="cm-missing"))
    assert not exp.model_file
    resp = client.get(reverse("web:experiment_detail", args=[exp.pk]))
    assert resp.context["can_run"] is False


def test_custom_model_readonly_when_flag_off(client, settings):
    """With the feature disabled, an existing custom-model experiment is not
    runnable even though its file is present."""
    exp = snapshot_adapter.experiment_from_snapshot(
        _custom_snapshot(name="cm-flagoff"), model_file=_model_file(),
    )
    settings.ALLOW_CUSTOM_MODELS = False
    resp = client.get(reverse("web:experiment_detail", args=[exp.pk]))
    assert resp.context["can_run"] is False


def test_run_view_refuses_unavailable_custom_model(client):
    """POSTing Run for a custom model whose file is missing does not start a
    run — it redirects without creating one."""
    exp = snapshot_adapter.experiment_from_snapshot(_custom_snapshot(name="cm-norun"))
    resp = client.post(reverse("web:experiment_run", args=[exp.pk]),
                       {"n_trials": "3", "optimize_metric": "accuracy"})
    assert resp.status_code == 302
    assert exp.runs.count() == 0
