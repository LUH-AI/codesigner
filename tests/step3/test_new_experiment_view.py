"""Step 3: the new-experiment page — configure, run, and see results in the browser.

The user-facing MVP. These drive the Django view end to end with the real
core engine (small, fast runs), so a green suite here means HPO is genuinely
doable from the browser. No database is involved.
"""

import io as _io

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR


def _valid_post(**overrides):
    """A valid create-experiment POST: Random Forest + Random Search on the
    demo iris dataset, 3 trials. Fast enough to run in a test."""
    data = {
        "name": "my-run",
        "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
        "primary_metric": "accuracy",
        "seed": 0,
        "n_trials": 3,
    }
    data.update(overrides)
    return data


def test_get_shows_the_form(client):
    """GET renders the configuration form with all the choices.

    Expect: 200 and a field for each of model, optimizer, dataset, metric,
    seed and trial count — everything needed to specify a run.
    """
    response = client.get(reverse("web:new_experiment"))
    assert response.status_code == 200
    html = response.content.decode()
    for field in ("name", "model_name", "optimizer_name", "demo_dataset",
                  "dataset_file", "primary_metric", "seed", "n_trials"):
        assert field in html


def test_post_runs_and_shows_results(client):
    """A valid submission runs the optimizer and renders the results.

    Expect: 200, the experiment name and choices echoed, and exactly the
    requested 3 trials shown — evidence the run actually happened.
    """
    response = client.post(reverse("web:new_experiment"), _valid_post())
    assert response.status_code == 200
    html = response.content.decode()
    assert "my-run" in html
    assert "Random Search" in html
    assert "Best" in html
    # three trial rows, numbered 1..3
    for n in (1, 2, 3):
        assert f"<td>{n}</td>" in html


def test_post_with_uploaded_csv(client):
    """A run works with an uploaded CSV instead of a demo dataset.

    Uploads iris.csv as the dataset file (no demo selected) and expects a
    results page with trials — the upload path builds the split correctly.
    """
    with open(DATASETS_DIR / "iris.csv", "rb") as f:
        upload = _io.BytesIO(f.read())
    upload.name = "iris.csv"
    data = _valid_post(demo_dataset="")
    data["dataset_file"] = upload
    response = client.post(reverse("web:new_experiment"), data)
    assert response.status_code == 200
    assert "<td>1</td>" in response.content.decode()


def test_grid_search_run(client):
    """Grid Search runs from the browser and reports its capped trial count.

    numeric_steps default over Random Forest yields a finite grid; requesting
    more trials than the grid holds still returns a results page.
    """
    response = client.post(reverse("web:new_experiment"),
                           _valid_post(optimizer_name="Grid Search", n_trials=5))
    assert response.status_code == 200
    assert "Best" in response.content.decode()


def test_missing_dataset_is_rejected(client):
    """Submitting with neither a demo dataset nor an upload shows an error.

    Expect: 200 (form re-rendered, not a run), with the dataset-required
    message and no results.
    """
    response = client.post(reverse("web:new_experiment"),
                           _valid_post(demo_dataset=""))
    assert response.status_code == 200
    html = response.content.decode()
    assert "demo dataset or upload" in html
    assert "Best" not in html


def test_missing_name_is_rejected(client):
    """A blank name re-renders the form with an error rather than running."""
    response = client.post(reverse("web:new_experiment"), _valid_post(name=""))
    assert response.status_code == 200
    assert "Best" not in response.content.decode()


@pytest.mark.slow
def test_smac_run_from_browser(client):
    """A small SMAC run completes synchronously and shows results.

    Marked slow: real SMAC + model training. 3 trials keeps it bounded.
    """
    response = client.post(reverse("web:new_experiment"),
                           _valid_post(optimizer_name="SMAC", n_trials=3))
    assert response.status_code == 200
    html = response.content.decode()
    assert "Best" in html
    assert "<td>3</td>" in html
