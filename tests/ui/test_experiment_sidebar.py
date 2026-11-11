"""The figures scroll past a fixed sidebar, since its controls choose the
experiment and trial. The experiment page adds sections to the base
section through `sidebar_sections`.
"""

import re

from django.urls import reverse

from tests.conftest import DATASETS_DIR

SECTIONS = ["Experiment Selection", "HyperSHAP", "Selected Configuration"]


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
    """A reachable dataset puts the run form on the page."""
    from experiments.services import snapshot as adapter

    fields = dict(
        version="0.1.0", name="sided", model_name="Random Forest",
        model_path="", optimizer_name="Random Search", optimizer_params={},
        primary_metric="accuracy", original_metric="accuracy",
        metric_names=["accuracy", "f1"], seed=0,
        dataset_path=str(DATASETS_DIR / "iris.csv"), result=_RESULT,
    )
    fields.update(overrides)
    exp = adapter.experiment_from_snapshot(fields, adopt_paths=True)
    exp.data.current_metric = "accuracy"
    exp.data.save()
    return exp


def _sidebar(client, exp):
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    return html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]


def test_the_sidebar_has_its_sections_in_order(client):
    """Each section narrows the last."""
    sidebar = _sidebar(client, _experiment())
    headings = re.findall(r"<h2>([^<]+)</h2>", sidebar)

    assert headings == SECTIONS


def test_other_pages_keep_the_first_section_alone(client):
    """A page with one decision to offer shows one section."""
    _experiment()
    html = client.get(reverse("ui:home")).content.decode()
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]

    assert re.findall(r"<h2>([^<]+)</h2>", sidebar) == SECTIONS[:1]


def _card(client, exp):
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    return html.split('data-figure="evaluation"', 1)[1].split("</section>", 1)[0]


def test_a_read_only_experiment_still_gets_them_all(client):
    """What: a read-only experiment keeps every sidebar section, and its
    Evaluation card keeps the metric selector without the run form.
    How: an experiment without a dataset."""
    exp = _experiment(name="frozen", dataset_path="")
    sidebar = _sidebar(client, exp)
    card = _card(client, exp)

    assert re.findall(r"<h2>([^<]+)</h2>", sidebar) == SECTIONS
    assert 'name="max_trials"' not in card and 'id="metric-select"' in card


def test_the_evaluation_card_lays_out_its_columns(client):
    """What: the metric sits in the card's header, inside the run form; the
    first column is what ends the run, and the last holds the failure limits
    then Run, and nothing folds away. How: the card of an experiment with a
    result."""
    card = _card(client, _experiment())
    head = card.split('class="card-head evaluation-head"', 1)[1].split('class="evaluation-columns"', 1)[0]
    columns = card.split('class="evaluation-column"')[1:]

    assert 'name="optimize_metric"' in head and ">Evaluation<" in head
    assert card.index("<form") < card.index('name="optimize_metric"')
    assert 'name="max_trials"' in columns[0]
    last = columns[-1]
    assert last.index('name="max_failures"') < last.index('data-assist="run"')
    assert "<details" not in card


def test_a_run_in_flight_replaces_the_form(client):
    """What: a running run shows its status in the card instead of the form,
    and the metric selector stays, since the figures read it. How: an
    experiment with a running run."""
    exp = _experiment(name="busy")
    exp.runs.create(primary_metric="accuracy", status="running", stopping={})

    card = _card(client, exp)

    assert 'id="run-status"' in card and 'name="max_trials"' not in card
    assert 'id="metric-select"' in card


# Experiment views

def _views(html):
    """View labels under the open experiment, and the current one."""
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]
    if 'class="tab-list experiment-views"' not in sidebar:
        return [], []
    block = sidebar.split('class="tab-list experiment-views"', 1)[1]
    top = re.sub(r'<div class="tab-list data-sections">.*?</div>', "", block, flags=re.S)
    labels = [t.strip() for t in re.findall(r'<a class="tab[^"]*"[^>]*>([^<]+)</a>', top)][:3]
    current = [t.strip() for t in re.findall(r'aria-current="page">([^<]+)</a>', top)]
    return labels, current


def test_the_open_experiment_lists_its_three_views(client):
    exp = _experiment()
    pages = {
        reverse("ui:experiment_detail", args=[exp.pk]): "Dashboard",
        reverse("ui:experiment_settings", args=[exp.pk]): "Dashboard",
        reverse("ui:data_overview", args=[exp.pk]): "Data Handling",
        reverse("ui:experiment_timeline", args=[exp.pk]): "Timeline",
    }
    for url, current in pages.items():
        labels, marked = _views(client.get(url).content.decode())
        assert labels == ["Dashboard", "Data Handling", "Timeline"], url
        assert marked == [current], url


def test_only_the_open_experiment_shows_its_views(client):
    """Every other experiment stays one row linking to its dashboard."""
    first = _experiment(name="first")
    second = _experiment(name="second")
    html = client.get(reverse("ui:experiment_timeline", args=[first.pk])).content.decode()
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]

    assert sidebar.count('class="tab-list experiment-views"') == 1
    assert f'href="{reverse("ui:experiment_detail", args=[second.pk])}"' in sidebar
    assert f'href="{reverse("ui:experiment_timeline", args=[second.pk])}"' not in sidebar


def test_pages_of_no_experiment_show_no_views(client):
    _experiment()
    for url in (reverse("ui:home"), reverse("ui:experiment_list")):
        assert _views(client.get(url).content.decode()) == ([], []), url


def test_an_open_experiment_beyond_the_recent_ones_is_still_listed(client):
    """An open experiment keeps its row beyond the sidebar's limit."""
    from ui.context_processors import SIDEBAR_LIMIT

    oldest = _experiment(name="oldest")
    for n in range(SIDEBAR_LIMIT):
        _experiment(name=f"newer {n}")
    html = client.get(reverse("ui:experiment_detail", args=[oldest.pk])).content.decode()

    assert _views(html)[0] == ["Dashboard", "Data Handling", "Timeline"]
    assert "oldest" in html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]
