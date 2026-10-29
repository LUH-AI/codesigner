"""The Export model button writes one trial's configuration as a runnable
script.
"""

import ast

import pytest
from django.core.files.base import ContentFile
from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


def _experiment():
    from experiments.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def _url(exp, **params):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return reverse("ui:experiment_export_model", args=[exp.pk]) + (f"?{query}" if query else "")


def _config(body):
    tree = ast.parse(body)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "HYPERPARAMETERS")
    return ast.literal_eval(node.value)


def test_the_selected_trial_is_exported(client):
    """The export is an attachment named for model and trial, containing the
    trial's configuration."""
    from experiments.services.snapshot import rebuild_result

    exp = _experiment()
    resp = client.get(_url(exp, metric="accuracy", idx=3))

    assert resp.status_code == 200
    result = rebuild_result(exp)
    assert result is not None
    trial = result.trials[3]
    assert f'filename="random-forest-trial-{trial.trial}.py"' in resp["Content-Disposition"]
    assert _config(resp.content.decode()) == dict(trial.config)


def test_without_a_selection_the_best_trial_is_exported(client):
    from experiments.services.snapshot import rebuild_result

    exp = _experiment()
    resp = client.get(_url(exp))

    assert resp.status_code == 200
    result = rebuild_result(exp)
    assert result is not None
    assert _config(resp.content.decode()) == dict(result.trials[8].config)


@pytest.mark.parametrize("params", [{"metric": "nope", "idx": 0},
                                    {"metric": "accuracy", "idx": 999},
                                    {"metric": "accuracy", "idx": "x"}])
def test_a_bad_selection_is_refused(client, params):
    """The export route answers an unknown metric or out-of-range index with
    400."""
    assert client.get(_url(_experiment(), **params)).status_code == 400


def test_an_imported_run_without_a_model_has_nothing_to_export(client):
    """An unknown model without a file returns 404, and the page explains why
    instead of offering the button."""
    exp = _experiment()
    exp.data.model_name = "Somebody Else's Model"
    exp.data.save()

    assert client.get(_url(exp)).status_code == 404
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert 'id="export-model"' not in page
    assert "imported without its model" in page


def test_a_custom_model_exports_from_its_file(client):
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
    assert "model = Mine()" in body and "get_config_space" not in body


def test_the_page_offers_the_export(client):
    """The button sits beside the .ihpo export."""
    exp = _experiment()
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert 'id="export-model"' in page
    assert reverse("ui:experiment_export_model", args=[exp.pk]) in page
