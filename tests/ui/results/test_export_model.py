"""The Export model button: one trial's configuration, in a runnable script.

Uses the bundled test2.ihpo fixture (Random Forest, Random Search, 30 trials):
index 8 is the best trial by accuracy.
"""

import ast

import pytest
from django.core.files.base import ContentFile
from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


def _experiment():
    from ui.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def _url(exp, **params):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return reverse("ui:experiment_export_model", args=[exp.pk]) + (f"?{query}" if query else "")


def _config(body):
    tree = ast.parse(body)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "CONFIG")
    return ast.literal_eval(node.value)


def test_the_selected_trial_is_exported(client):
    """What: the file is an attachment named for the model and trial, holding
    that trial's configuration. How: exports index 3 and compares CONFIG with
    the stored trial."""
    from ui.views import _rebuild_result

    exp = _experiment()
    resp = client.get(_url(exp, metric="accuracy", idx=3))

    assert resp.status_code == 200
    trial = _rebuild_result(exp).trials[3]
    assert f'filename="random-forest-trial-{trial.trial}.py"' in resp["Content-Disposition"]
    assert _config(resp.content.decode()) == dict(trial.config)


def test_without_a_selection_the_best_trial_is_exported(client):
    """What: with no parameters the file holds the best trial on the current
    metric. How: compares with index 8, the fixture's best by accuracy."""
    from ui.views import _rebuild_result

    exp = _experiment()
    resp = client.get(_url(exp))

    assert resp.status_code == 200
    assert _config(resp.content.decode()) == dict(_rebuild_result(exp).trials[8].config)


@pytest.mark.parametrize("params", [{"metric": "nope", "idx": 0},
                                    {"metric": "accuracy", "idx": 999},
                                    {"metric": "accuracy", "idx": "x"}])
def test_a_bad_selection_is_refused(client, params):
    """What: an unknown metric or an index outside the trials is a 400.
    How: asks for each."""
    assert client.get(_url(_experiment(), **params)).status_code == 400


def test_an_imported_run_without_a_model_has_nothing_to_export(client):
    """What: a model the registry does not know and no file behind it cannot be
    exported, and the page says why rather than offering a broken link.
    How: renames the model, then fetches the export and the page."""
    exp = _experiment()
    exp.data.model_name = "Somebody Else's Model"
    exp.data.save()

    assert client.get(_url(exp)).status_code == 404
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert 'id="export-model"' not in page
    assert "imported without its model" in page


def test_a_custom_model_exports_from_its_file(client):
    """What: an uploaded model is exported from its own source.
    How: attaches a model file and looks for its class in the download."""
    exp = _experiment()
    exp.data.model_name = "Mine"
    exp.data.model_file.save("mine.py", ContentFile(
        b"from codesigner_model import BaseModel\n\n\n"
        b"class Mine(BaseModel):\n    name = \"Mine\"\n"
        b"    def get_config_space(self, seed=0): return None\n"
        b"    def fit_predict(self, config, X_train, y_train, X_val, seed=0): return []\n"),
        save=False)
    exp.data.save()

    body = client.get(_url(exp)).content.decode()
    assert "class Mine(BaseModel)" in body
    assert "MODEL_CLASS = Mine" in body


def test_the_page_offers_the_export(client):
    """What: the experiment page has the button beside the .ihpo export.
    How: looks for its link in the page."""
    exp = _experiment()
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert 'id="export-model"' in page
    assert reverse("ui:experiment_export_model", args=[exp.pk]) in page
