"""Step 3 (updated in Step 6): the new-experiment page *creates* an experiment.

Since Step 6 separated create from run, submitting this form saves the
experiment (no result yet) and redirects to its detail page; running happens
there, in the background (see tests/ui/runs/). These tests cover creation and its
validation; the run flow and results rendering live in step6/step4.

DB access + isolated media come from tests/ui/conftest.py.
"""

import io as _io

from django.urls import reverse

from tests.conftest import DATASETS_DIR


def _valid_post(**overrides):
    """A valid create-experiment POST: Random Forest + Random Search on iris."""
    data = {
        "name": "my-exp",
        "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
        "seed": 0,
    }
    data.update(overrides)
    return data


def test_get_shows_the_form(client):
    """GET renders the setup form: model, optimizer, dataset, seed.

    The metric and trial-count fields are gone from creation — they are chosen
    per run on the detail page.
    """
    response = client.get(reverse("ui:new_experiment"))
    assert response.status_code == 200
    html = response.content.decode()
    for field in ("name", "model_name", "optimizer_name", "demo_dataset",
                  "dataset_file", "seed"):
        assert field in html


def test_post_creates_experiment_and_redirects(client):
    """A valid submission saves an experiment with no result and redirects to detail.

    Expect: 302 to the experiment's detail page, and an Experiment row with no
    result and no committed metric (creating does not run).
    """
    from ui.models import Experiment

    resp = client.post(reverse("ui:new_experiment"), _valid_post(name="created"))
    exp = Experiment.objects.get(name="created")

    assert resp.status_code == 302
    assert resp["Location"] == reverse("ui:experiment_detail", args=[exp.pk])
    assert exp.result is None
    assert exp.current_metric is None


def test_post_with_uploaded_csv_stores_the_dataset(client):
    """Creating with an uploaded CSV stores that dataset on the experiment."""
    from ui.models import Experiment

    with open(DATASETS_DIR / "iris.csv", "rb") as f:
        upload = _io.BytesIO(f.read())
    upload.name = "iris.csv"
    data = _valid_post(name="uploaded", demo_dataset="")
    data["dataset_file"] = upload

    resp = client.post(reverse("ui:new_experiment"), data)
    assert resp.status_code == 302
    assert Experiment.objects.get(name="uploaded").dataset


def test_missing_dataset_is_rejected(client):
    """Submitting with neither a demo dataset nor an upload re-renders with an error."""
    from ui.models import Experiment

    resp = client.post(reverse("ui:new_experiment"), _valid_post(demo_dataset=""))
    assert resp.status_code == 200
    assert "demo dataset or upload" in resp.content.decode()
    assert Experiment.objects.count() == 0


def test_missing_name_is_rejected(client):
    """A blank name re-renders the form and creates nothing."""
    from ui.models import Experiment

    resp = client.post(reverse("ui:new_experiment"), _valid_post(name=""))
    assert resp.status_code == 200
    assert Experiment.objects.count() == 0


def test_five_fold_cross_validation_is_the_default(client):
    """A single 80/20 split on a small table is noisy enough that a search can
    spend its budget chasing the split rather than the model, and the tables
    this is pointed at are small. Holdout stays one selection away."""
    from ui.models import Experiment

    html = client.get(reverse("ui:new_experiment")).content.decode()
    client.post(reverse("ui:new_experiment"), _valid_post(name="folded"))

    assert '<option value="5" selected>' in html
    assert Experiment.objects.get(name="folded").cv_folds == 5


def test_a_choice_of_holdout_is_still_honoured(client):
    from ui.models import Experiment

    client.post(reverse("ui:new_experiment"),
                _valid_post(name="split", cv_folds="0"))

    assert Experiment.objects.get(name="split").cv_folds == 0
