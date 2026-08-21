"""The experiment page's inner sidebar: three sections, and they stay put.

Everything in the sidebar is a decision the figures are an answer to — which
experiment, which metric, which trial. A control that scrolls out of view while
you are using it is one you have to go and find, so the sidebar is fixed and the
figures scroll past it.

The first section is the sidebar's own (base.html) and shows on every page; the
experiment page adds the other two through `sidebar_sections`.
"""

import re

from django.urls import reverse

from tests.conftest import DATASETS_DIR

SECTIONS = ["Experiment Selection", "Experiment Evaluation",
            "Explanation Game", "Selected Configuration"]


_RESULT = {
    "stats": {"submitted": 1, "finished": 1, "running": 0},
    "data": [{"config_id": 1, "cost": 0.2, "time": 2.5,
              "scores": {"accuracy": 0.8, "f1": 0.7},
              "incumbent_score": 0.8, "incumbent_config_id": 1}],
    "configs": {"1": {"n_estimators": 100}},
    "config_origins": {"1": "Random Search"}, "optimizer_state": {},
    "primary_metric": "accuracy", "best_score": 0.8, "best_config_id": "1",
    "hyperparameter_importance": {"accuracy": {"n_estimators": 1.0}},
    "hyperparameter_importance_warning": {}, "trials_limit": None,
}


def _experiment(**overrides):
    """A runnable experiment with a result, so every section has something to
    show — a dataset it can actually reach is what puts the run form there."""
    from ui.services import snapshot as adapter

    fields = dict(
        version="0.1.0", name="sided", model_name="Random Forest",
        model_path="", optimizer_name="Random Search", optimizer_params={},
        primary_metric="accuracy", original_metric="accuracy",
        metric_names=["accuracy", "f1"], seed=0,
        dataset_path=str(DATASETS_DIR / "iris.csv"), result=_RESULT,
    )
    fields.update(overrides)
    exp = adapter.experiment_from_snapshot(fields, adopt_paths=True)
    exp.current_metric = "accuracy"
    exp.save()
    return exp


def _sidebar(client, exp):
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    return html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]


def test_the_sidebar_has_its_sections_in_order(client):
    """Which experiment, then what it is scored on, then what is selected —
    each one narrowing the last."""
    sidebar = _sidebar(client, _experiment())
    headings = re.findall(r"<h2>([^<]+)</h2>", sidebar)

    assert headings == SECTIONS


def test_other_pages_keep_the_first_section_alone(client):
    """The sections come from the page, not from the sidebar: a page with only
    one decision to offer shows one."""
    _experiment()
    html = client.get(reverse("ui:home")).content.decode()
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]

    assert re.findall(r"<h2>([^<]+)</h2>", sidebar) == SECTIONS[:1]


def test_a_read_only_experiment_still_gets_them_all(client):
    """No dataset means no run to configure, but the metric still drives every
    figure and a trial can still be selected — so the sections are there with
    the run form left out, rather than the sections disappearing."""
    sidebar = _sidebar(client, _experiment(name="frozen", dataset_path=""))

    assert re.findall(r"<h2>([^<]+)</h2>", sidebar) == SECTIONS
    assert 'name="max_trials"' not in sidebar
    assert 'id="metric-select"' in sidebar


def test_what_bounds_the_next_run_is_folded_away(client):
    """It is set once; the metric above it is chosen constantly. So the metric
    is the section and the rest is behind a fold — closed, since a page that
    opens with a form open is a page asking to be filled in."""
    sidebar = _sidebar(client, _experiment())
    details = sidebar.split("<details class=", 1)[1]

    assert "open" not in details.split(">", 1)[0]
    assert 'name="max_trials"' in details
    assert 'name="optimize_metric"' not in details, "the metric stays outside the fold"


def test_a_run_in_flight_replaces_the_fold(client):
    """There is nothing to configure until it finishes, and the alternative is
    a form offering to start a run you cannot have."""
    exp = _experiment(name="busy")
    exp.runs.create(primary_metric="accuracy", status="running", stopping={})

    sidebar = _sidebar(client, exp)

    assert 'id="run-status"' in sidebar
    assert "<details class=" not in sidebar
    assert 'id="metric-select"' in sidebar, "the figures still need it"
