"""`tests/fixtures/flats.csv`, with text, dates and gaps, runs from the create
form to an exported model.
"""

import ast
from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from tests.conftest import post_new_experiment

FLATS = Path(__file__).parent.parent.parent / "fixtures" / "flats.csv"


def _create(client, name, model_name):
    upload = SimpleUploadedFile("flats.csv", FLATS.read_bytes(), content_type="text/csv")
    return post_new_experiment(client, {
        "name": name, "model_name": model_name, "optimizer_name": "Random Search",
        "dataset_file": upload, "task": "classification", "evaluation_scheme": "kfold", "evaluation_value": 3,
        "seed": 0,
    })


@pytest.mark.django_db
@pytest.mark.parametrize("model_name", ["Random Forest", "SVM"])
def test_a_mixed_file_runs_every_trial_and_exports_its_encoding(client, model_name):
    """The exported model carries `COLUMNS` and `PROCESSING` to encode such a
    CSV itself."""
    from experiments.models import Experiment
    from experiments.services.snapshot import rebuild_result

    assert _create(client, f"flats {model_name}", model_name).status_code == 302
    exp = Experiment.objects.get(data__name=f"flats {model_name}")
    assert exp.data.task == "classification"
    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 4, "optimize_metric": "accuracy"})
    exp.refresh_from_db()

    result = rebuild_result(exp)
    assert result is not None
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
