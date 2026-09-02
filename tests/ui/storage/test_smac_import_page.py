"""A SMAC run, imported, browsed, and exported again.

*What:* the end of the import path — the upload form, the row it creates, what
the detail page does with it, and the round trip back out to an `.ihpo`. The
claim being pinned is the one that made the work worth doing: with the config
space hoisted out of the file, **every figure draws** for a run that has no
model and no dataset behind it.

*How:* posts `tests/fixtures/smac_run`'s five files at the import page the way a
directory picker would, then asks each figure in the catalog to draw and each
deferred computation to run.
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
    """The fixture run's files as a directory picker would post them.

    Django keeps only the basename of an upload, so that is what the browser's
    relative paths arrive as — which is why the importer matches on basenames.
    """
    return [SimpleUploadedFile(p.name, p.read_bytes(), "application/json")
            for p in sorted(RUN_DIR.glob("*.json"))
            if names is None or p.name in names]


def _import(client, uploads=None):
    response = client.post(reverse("ui:import_experiment"),
                           {"smac_dir": uploads if uploads is not None else _uploads()})
    return response


def _imported():
    """The row, built through the importer without going via the page."""
    from core.smac_import import snapshot_from_smac
    from ui.services import snapshot as adapter

    files = {p.name: json.loads(p.read_text()) for p in RUN_DIR.glob("*.json")}
    return adapter.experiment_from_snapshot(
        io.parse(io.to_bytes(snapshot_from_smac(files))))


# ── the upload ───────────────────────────────────────────────────────────────

def test_the_import_page_offers_a_directory_picker(client):
    body = client.get(reverse("ui:import_experiment")).content.decode()

    assert "webkitdirectory" in body
    assert 'name="smac_dir"' in body


def test_uploading_a_run_creates_an_experiment(client):
    from ui.models import Experiment

    response = _import(client)

    exp = Experiment.objects.get()
    assert response.status_code == 302
    assert response["Location"] == reverse("ui:experiment_detail", args=[exp.pk])
    assert exp.name == "demo-run"
    assert exp.metric_names == [COST]
    assert len(exp.result["data"]) == 25


def test_two_runs_at_once_are_refused_rather_than_blended(client):
    """Django keeps only basenames, so two runs cannot be told apart.

    A runhistory read against another run's config space is not a run that
    happened, so this is refused rather than resolved by picking one.
    """
    from ui.models import Experiment

    response = _import(client, _uploads() + _uploads(["runhistory.json"]))

    assert response.status_code == 200
    assert "more than one run" in response.content.decode()
    assert not Experiment.objects.exists()


def test_something_that_is_not_a_run_is_refused_with_a_reason(client):
    from ui.models import Experiment

    junk = SimpleUploadedFile("runhistory.json", b'{"nope": true}',
                              "application/json")
    response = _import(client, [junk])

    assert response.status_code == 200
    assert "not a readable SMAC run" in response.content.decode()
    assert not Experiment.objects.exists()


# ── what the page does with it ───────────────────────────────────────────────

def test_the_detail_page_renders_and_offers_no_run(client):
    """Read-only: there is no dataset, no model, and nothing to verify one with."""
    exp = _imported()

    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'name="optimize_metric"' not in body


def test_the_page_does_not_invent_an_evaluation_scheme():
    """A SMAC scenario records no split, so the box says so.

    The columns still hold their defaults — they have to hold something — and
    reporting those as fact would put a 20% holdout on the page for a run that
    may have used no such thing.
    """
    from ui.views import _evaluation_label

    assert str(_evaluation_label(_imported())) == "Not recorded"


def test_every_trial_based_figure_draws():
    """The claim the config-space hoist was for."""
    from ui.views import _config_space_for, _rebuild_experiment

    built = _rebuild_experiment(_imported())
    space = _config_space_for(built)
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
    """An imported run's games start empty, like a cancelled run's, and fill in.

    Nothing new was needed for this: `experiment_compute_analytics` already
    recomputes from the trials, and the trials plus the config space are
    everything HyperSHAP needs — no dataset, no model.
    """
    exp = _imported()
    assert not exp.result.get("hyperparameter_importance")

    client.post(reverse("ui:experiment_compute_analytics", args=[exp.pk]))
    exp.refresh_from_db()

    importance = exp.result["hyperparameter_importance"][COST]
    assert set(importance) == {"depth", "kind", "rate"}
    assert sum(importance.values()) > 0


def test_the_explanation_is_oriented_to_the_cost():
    """The incumbent's own hyperparameters helped, and the figure says so.

    Read against a cost as though bigger were better, every sign here inverts
    and the best configuration's best hyperparameter reads as the one that hurt
    it most.
    """
    from ui.views import _local_ablation_data, _rebuild_experiment

    built = _rebuild_experiment(_imported())
    best = built["result"].best_index(COST)

    _figure, warning, effects = _local_ablation_data(built, COST, best)

    assert warning is None
    assert effects["depth"] > 0


# ── back out again ───────────────────────────────────────────────────────────

def test_an_imported_run_exports_and_re_imports(client):
    """Including the two things that are easy to lose on the way through.

    `kind: external` — recomputed as "registry" it names a model the registry
    cannot resolve, and the experiment stops rebuilding at all — and `space`,
    without which the re-imported copy has no search space to draw from.
    """
    from ui.services import snapshot as adapter

    exp = _imported()
    body = client.post(reverse("ui:experiment_export", args=[exp.pk]),
                       {"timestamps": "keep", "tracebacks": "keep"}).content

    again = io.parse(body)
    assert again["model"]["kind"] == "external"
    assert "space" in again

    copy = adapter.experiment_from_snapshot(again)
    assert client.get(reverse("ui:experiment_detail", args=[copy.pk])).status_code == 200
