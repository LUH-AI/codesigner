"""With the config space hoisted out of the file, the trial-based figures draw
for an imported SMAC run with no model or dataset.
"""

import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from core import io
from ui.figures.catalog import FIGURES

from tests.conftest import FIXTURES_DIR

RUN_DIR = FIXTURES_DIR / "smac_run"
COST = "smac:cost"

pytestmark = pytest.mark.django_db


def _uploads(names=None):
    """Django only keeps an upload's basename, so the importer matches on
    basenames.
    """
    return [SimpleUploadedFile(p.name, p.read_bytes(), "application/json")
            for p in sorted(RUN_DIR.glob("*.json"))
            if names is None or p.name in names]


def _import(client, uploads=None):
    response = client.post(reverse("ui:import_experiment"),
                           {"smac_dir": uploads if uploads is not None else _uploads()})
    return response


def _imported():
    """`_imported` builds the row through the importer directly."""
    from core.optimizers.smac.importer import snapshot_from_smac
    from experiments.services import snapshot as adapter

    files = {p.name: json.loads(p.read_text()) for p in RUN_DIR.glob("*.json")}
    return adapter.experiment_from_snapshot(
        io.parse(io.to_bytes(snapshot_from_smac(files))))


# Upload

def test_the_import_page_offers_a_directory_picker(client):
    body = client.get(reverse("ui:import_experiment")).content.decode()

    assert "webkitdirectory" in body
    assert 'name="smac_dir"' in body


def test_uploading_a_run_creates_an_experiment(client):
    from experiments.models import Experiment

    response = _import(client)

    exp = Experiment.objects.get()
    assert response.status_code == 302
    assert response["Location"] == reverse("ui:experiment_detail", args=[exp.pk])
    assert exp.name == "demo-run"
    assert exp.data.metric_names == [COST]
    assert exp.data.result is not None
    assert len(exp.data.result["data"]) == 25


def test_two_runs_at_once_are_refused_rather_than_blended(client):
    """A runhistory read against another run's config space never happened, so
    the import page shows an error for two runs and creates no experiment.
    """
    from experiments.models import Experiment

    response = _import(client, _uploads() + _uploads(["runhistory.json"]))

    assert response.status_code == 200
    assert "more than one run" in response.content.decode()
    assert not Experiment.objects.exists()


def test_something_that_is_not_a_run_is_refused_with_a_reason(client):
    from experiments.models import Experiment

    junk = SimpleUploadedFile("runhistory.json", b'{"nope": true}',
                              "application/json")
    response = _import(client, [junk])

    assert response.status_code == 200
    assert "not a readable SMAC run" in response.content.decode()
    assert not Experiment.objects.exists()


# Detail page

def test_the_detail_page_renders_and_offers_no_run(client):
    """A read-only run has no dataset and no model."""
    exp = _imported()

    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'name="optimize_metric"' not in body


def test_the_page_does_not_invent_an_evaluation_scheme():
    """The columns contain defaults, and reporting them as fact would claim a 20%
    holdout.
    """
    from experiments.services.model_export import evaluation_label

    assert str(evaluation_label(_imported())) == "Not recorded"


def test_every_trial_based_figure_draws():
    from experiments.services.snapshot import config_space_for, rebuild_experiment

    built = rebuild_experiment(_imported())
    assert built is not None
    space = config_space_for(built)
    assert space is not None

    drew = {}
    for figure in FIGURES:
        extra = {"config_space": space} if figure.needs_config_space else {}
        metric = COST if figure.per_metric else None
        if figure.views:
            drew[figure.key] = all(
                figure.plot(built["result"], metric, view=v, **extra) is not None
                for v in figure.views)
        else:
            drew[figure.key] = figure.plot(built["result"], metric, **extra) is not None

    for key in ("performance_over_time", "configuration_cube",
                "parallel_coordinates", "trial_duration"):
        assert drew[key], key


def test_the_analytics_can_be_computed_afterwards(client):
    """`experiment_compute_analytics` only needs the trials and the config
    space.
    """
    exp = _imported()
    assert exp.data.result is not None
    assert not exp.data.result.get("hyperparameter_importance")

    client.post(reverse("ui:experiment_compute_analytics", args=[exp.pk]))
    exp.refresh_from_db()

    importance = exp.data.result["hyperparameter_importance"][COST]
    assert set(importance) == {"depth", "kind", "rate"}
    assert sum(importance.values()) > 0


def test_the_explanation_is_oriented_to_the_cost():
    """Read as a gain, a cost would invert every sign.
    """
    from experiments.services.snapshot import rebuild_experiment
    from ui.views.analytics import _local_ablation_data

    built = rebuild_experiment(_imported())
    assert built is not None
    best = built["result"].best_index(COST)

    _figure, warning, effects = _local_ablation_data(built, COST, best)

    assert warning is None
    assert effects["depth"] > 0


# Export

def test_an_imported_run_exports_and_re_imports(client):
    """A recomputed `kind: registry` would stop the rebuild, and a missing
    `space` leaves the copy nothing to draw.
    """
    from experiments.services import snapshot as adapter

    exp = _imported()
    body = client.post(reverse("ui:experiment_export", args=[exp.pk]),
                       {"timestamps": "keep", "tracebacks": "keep"}).content

    again = io.parse(body)
    assert again["model"]["kind"] == "external"
    assert "space" in again

    copy = adapter.experiment_from_snapshot(again)
    assert client.get(reverse("ui:experiment_detail", args=[copy.pk])).status_code == 200
