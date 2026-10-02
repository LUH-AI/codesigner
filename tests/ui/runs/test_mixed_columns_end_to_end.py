"""A dataset of text, dates and gaps, from the create form to an exported model.

`tests/fixtures/flats.csv` is uploaded as it is — a city with missing values,
a yes/no column, dates, a timestamp — and run with each built-in model. Runs
execute inline in tests (see `runs_execute_synchronously`), so posting the Run
form leaves a finished result behind.
"""

import ast
from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

FLATS = Path(__file__).parent.parent.parent / "fixtures" / "flats.csv"


def _create(client, name, model_name):
    upload = SimpleUploadedFile("flats.csv", FLATS.read_bytes(), content_type="text/csv")
    return client.post(reverse("ui:new_experiment"), {
        "name": name, "model_name": model_name, "optimizer_name": "Random Search",
        "dataset_file": upload, "task": "classification", "evaluation_scheme": "kfold", "evaluation_value": 3,
        "seed": 0,
    })


@pytest.mark.django_db
@pytest.mark.parametrize("model_name", ["Random Forest", "SVM"])
def test_a_mixed_file_runs_every_trial_and_exports_its_encoding(client, model_name):
    """What: an uploaded file with text, dates and missing values is a
    classification experiment whose trials all measure something, and the
    exported model carries the encoding to read such a CSV itself.
    How: creates the experiment from the raw upload, runs four cross-validated
    random-search trials, checks none of them failed, then reads `COLUMNS`
    and `PROCESSING` out of the exported file."""
    from ui.models import Experiment
    from ui.views import _rebuild_result

    assert _create(client, f"flats {model_name}", model_name).status_code == 302
    exp = Experiment.objects.get(data__name=f"flats {model_name}")
    assert exp.data.task == "classification"
    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 4, "optimize_metric": "accuracy"})
    exp.refresh_from_db()

    result = _rebuild_result(exp)
    assert len(result.trials) == 4
    assert not any(t.failed for t in result.trials), [t.run_info for t in result.trials]
    assert all(t.scores["accuracy"] > 0.5 for t in result.trials)
    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 200

    body = client.get(reverse("ui:experiment_export_model", args=[exp.pk])).content.decode()
    assigned = {n.targets[0].id: n.value for n in ast.parse(body).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    columns = ast.literal_eval(assigned["COLUMNS"])["columns"]
    kinds = {c["name"]: c["kind"] for c in columns}
    assert kinds["listed"] == "datetime" and kinds["city"] == "categorical"
    assert set(ast.literal_eval(assigned["PROCESSING"])) == {"missing", "scale", "labels"}
