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
    this is pointed at are small. Holdout stays one selection away.

    The page is asked for the scheme selector's own state and the number beside
    it, and the form is submitted with neither, since leaving both alone is how
    someone gets the default.
    """
    from ui.models import Experiment

    html = client.get(reverse("ui:new_experiment")).content.decode()
    client.post(reverse("ui:new_experiment"), _valid_post(name="folded"))

    assert '<option value="kfold" selected>' in html
    assert 'value="5"\n                       aria-describedby="evaluation_value_helptext" data-evaluation-value' in html
    assert Experiment.objects.get(name="folded").cv_folds == 5


def test_a_choice_of_holdout_is_still_honoured(client):
    """Picking the other scheme stores no folds and the share that was typed —
    the number beside the selector means something different under each, and
    only the one the scheme asked for is kept."""
    from ui.models import Experiment

    client.post(reverse("ui:new_experiment"),
                _valid_post(name="split", evaluation_scheme="holdout",
                            evaluation_value="0.3"))

    exp = Experiment.objects.get(name="split")
    assert exp.cv_folds == 0
    assert exp.test_size == 0.3


def test_the_number_beside_the_scheme_is_read_as_that_scheme_asks(client):
    """One field carries folds under cross-validation and a held-out share
    under a split, so the same number has to land in a different column."""
    from ui.models import Experiment

    client.post(reverse("ui:new_experiment"),
                _valid_post(name="sevenfold", evaluation_scheme="kfold",
                            evaluation_value="7"))

    exp = Experiment.objects.get(name="sevenfold")
    assert exp.cv_folds == 7
    assert exp.test_size == 0.2


def test_the_scheme_and_its_number_sit_in_one_row(client):
    """They are one decision — how the data is divided, and how finely — so the
    number has to read as the selector's and not as a setting of its own.
    Stacked, the two look like unrelated fields that happen to be adjacent."""
    html = client.get(reverse("ui:new_experiment")).content.decode()
    row = html.split('class="field-row" data-evaluation', 1)[1].split("</div>\n        </div>", 1)[0]

    assert 'name="evaluation_scheme"' in row
    assert 'name="evaluation_value"' in row


def test_the_page_carries_the_bounds_the_selector_switches_between(client):
    """The number's label, range and default follow the scheme, and the script
    that swaps them reads the form's own EVALUATION_SCHEMES rather than a copy.
    This asserts the page actually ships that table, since a restated copy in
    the template is exactly the drift it exists to prevent."""
    import json

    from ui.forms import EVALUATION_HOLDOUT, EVALUATION_KFOLD

    html = client.get(reverse("ui:new_experiment")).content.decode()
    body = html.split('id="evaluation-schemes"', 1)[1].split(">", 1)[1]
    schemes = json.loads(body.split("</script>", 1)[0])

    assert set(schemes) == {EVALUATION_KFOLD, EVALUATION_HOLDOUT}
    assert schemes[EVALUATION_KFOLD]["default"] == 5
    assert schemes[EVALUATION_HOLDOUT]["default"] == 0.2
    assert schemes[EVALUATION_KFOLD]["label"] != schemes[EVALUATION_HOLDOUT]["label"]
    assert schemes[EVALUATION_KFOLD]["max"] > schemes[EVALUATION_HOLDOUT]["max"]


def test_an_out_of_range_number_is_clamped_rather_than_refused(client):
    """The two ends of the range are both meaningful and everything between
    them is too, so there is no typo here worth failing a form over: 40 folds
    becomes the most that is offered, and the experiment is still created."""
    from ui.models import Experiment

    client.post(reverse("ui:new_experiment"),
                _valid_post(name="greedy", evaluation_scheme="kfold",
                            evaluation_value="40"))
    client.post(reverse("ui:new_experiment"),
                _valid_post(name="mostly-held-out", evaluation_scheme="holdout",
                            evaluation_value="0.9"))

    assert Experiment.objects.get(name="greedy").cv_folds == 20
    assert Experiment.objects.get(name="mostly-held-out").test_size == 0.5
