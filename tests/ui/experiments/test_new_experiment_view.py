"""The new-experiment form saves an experiment without a result and redirects to
its detail page, where runs happen.
"""

import io as _io

from django.urls import reverse

from tests.conftest import DATASETS_DIR, get_new_experiment, post_new_experiment


def _valid_post(**overrides):
    data = {
        "name": "my-exp",
        "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
        "seed": 0,
    }
    data.update(overrides)
    return data


def test_get_shows_the_form(client):
    """Metric and stopping rule are picked per run on the detail page, and the
    optimizer has its own step.
    """
    response = get_new_experiment(client)
    assert response.status_code == 200
    html = response.content.decode()
    for field in ("name", "model_name", "demo_dataset", "dataset_file", "seed"):
        assert field in html


def test_post_creates_experiment_and_redirects(client):
    """Creating leaves the result and committed metric empty.
    """
    from experiments.models import Experiment

    resp = post_new_experiment(client, _valid_post(name="created"))
    exp = Experiment.objects.get(data__name="created")

    assert resp.status_code == 302
    assert resp["Location"] == reverse("ui:experiment_detail", args=[exp.pk])
    assert exp.data.result is None
    assert exp.data.current_metric is None


def test_post_with_uploaded_csv_stores_the_dataset(client):
    from experiments.models import Experiment

    with open(DATASETS_DIR / "iris.csv", "rb") as f:
        upload = _io.BytesIO(f.read())
    upload.name = "iris.csv"
    data = _valid_post(name="uploaded", demo_dataset="")
    data["dataset_file"] = upload

    resp = post_new_experiment(client, data)
    assert resp.status_code == 302
    assert Experiment.objects.get(data__name="uploaded").data.dataset


def test_missing_dataset_is_rejected(client):
    from experiments.models import Experiment

    resp = post_new_experiment(client, _valid_post(demo_dataset=""))
    assert resp.status_code == 200
    assert "demo dataset or upload" in resp.content.decode()
    assert not Experiment.objects.filter(draft=False).exists()


def test_a_name_is_optional(client):
    from experiments.models import Experiment

    post_new_experiment(client, _valid_post(name=""))
    exp = Experiment.objects.get(draft=False)
    html = client.get(f"/experiments/{exp.pk}/").content.decode()

    assert exp.name == "" and exp.title == "Untitled Experiment"
    assert "Experiment: Untitled Experiment" in html


def test_five_fold_cross_validation_is_the_default(client):
    """A single 80/20 split on a small table is noisy enough that a search
    chases the split instead of the model.
    """
    from experiments.models import Experiment

    html = get_new_experiment(client).content.decode()
    post_new_experiment(client, _valid_post(name="folded"))

    assert '<option value="kfold" selected>' in html
    assert 'value="5"\n                       aria-describedby="evaluation_value_helptext" data-evaluation-value' in html
    assert Experiment.objects.get(data__name="folded").data.cv_folds == 5


def test_a_choice_of_holdout_is_still_honoured(client):
    """A holdout choice saves no folds and the typed share."""
    from experiments.models import Experiment

    post_new_experiment(client,
                _valid_post(name="split", evaluation_scheme="holdout",
                            evaluation_value="0.3"))

    exp = Experiment.objects.get(data__name="split")
    assert exp.data.cv_folds == 0
    assert exp.data.test_size == 0.3


def test_the_number_beside_the_scheme_is_read_as_that_scheme_asks(client):
    """One field holds folds under k-fold, a share under holdout."""
    from experiments.models import Experiment

    post_new_experiment(client,
                _valid_post(name="sevenfold", evaluation_scheme="kfold",
                            evaluation_value="7"))

    exp = Experiment.objects.get(data__name="sevenfold")
    assert exp.data.cv_folds == 7
    assert exp.data.test_size == 0.2


def test_the_scheme_and_its_number_sit_in_one_row(client):
    """The number reads as part of the scheme decision."""
    html = get_new_experiment(client).content.decode()
    row = html.split('class="field-row" data-evaluation', 1)[1].split("</div>\n        </div>", 1)[0]

    assert 'name="evaluation_scheme"' in row
    assert 'name="evaluation_value"' in row


def test_the_page_carries_the_bounds_the_selector_switches_between(client):
    """The page ships the form's `EVALUATION_SCHEMES` as JSON, so the script
    keeps no copy that could drift."""
    import json

    from ui.forms import EVALUATION_BACKTEST, EVALUATION_HOLDOUT, EVALUATION_KFOLD

    html = get_new_experiment(client).content.decode()
    body = html.split('id="evaluation-schemes"', 1)[1].split(">", 1)[1]
    schemes = json.loads(body.split("</script>", 1)[0])

    assert set(schemes) == {EVALUATION_KFOLD, EVALUATION_HOLDOUT, EVALUATION_BACKTEST}
    assert schemes[EVALUATION_KFOLD]["default"] == 5
    assert schemes[EVALUATION_HOLDOUT]["default"] == 0.2
    assert schemes[EVALUATION_KFOLD]["label"] != schemes[EVALUATION_HOLDOUT]["label"]
    assert schemes[EVALUATION_KFOLD]["max"] > schemes[EVALUATION_HOLDOUT]["max"]


def test_an_out_of_range_number_is_clamped_rather_than_refused(client):
    """Any in-range value is plausible, so the form clamps other numbers into
    range."""
    from experiments.models import Experiment

    post_new_experiment(client,
                _valid_post(name="greedy", evaluation_scheme="kfold",
                            evaluation_value="40"))
    post_new_experiment(client,
                _valid_post(name="mostly-held-out", evaluation_scheme="holdout",
                            evaluation_value="0.9"))

    assert Experiment.objects.get(data__name="greedy").data.cv_folds == 20
    assert Experiment.objects.get(data__name="mostly-held-out").data.test_size == 0.5
