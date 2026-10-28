"""The page's selection bus needs no per-figure knowledge, because each figure
declaring `selects_trials` says where its trials are in
`layout.meta["selection"]`.
"""

from django.urls import reverse

from ui.figures import FIGURES, FIGURES_BY_KEY
from experiments.models import Experiment

def _experiment():
    """`_experiment` holds three finished trials, so a highlight can move."""
    def trial(n, estimators, accuracy):
        return {"config_id": n, "cost": 1 - accuracy, "time": 2.5,
                "scores": {"accuracy": accuracy, "f1": accuracy - 0.05},
                "incumbent_score": accuracy, "incumbent_config_id": n}

    return Experiment.objects.create(
        name="selectable", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy", "f1"], current_metric="accuracy",
        original_metric="accuracy", seed=0,
        result={
            "stats": {"submitted": 3, "finished": 3, "running": 0},
            "data": [trial(1, 50, 0.7), trial(2, 100, 0.9), trial(3, 150, 0.8)],
            "configs": {"1": {"n_estimators": 50}, "2": {"n_estimators": 100},
                        "3": {"n_estimators": 150}},
            "config_origins": {"1": "Random Search", "2": "Random Search",
                               "3": "Random Search"},
            "optimizer_state": {},
            "primary_metric": "accuracy", "best_score": 0.9, "best_config_id": "2",
            "hyperparameter_importance": {"accuracy": {"n_estimators": 1.0}},
            "hyperparameter_importance_warning": {}, "trials_limit": None,
        },
    )


def _page(client, exp):
    return client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()


def _trials_body(html):
    """Trials table rows, anchored past the page's other tables."""
    return html.split('data-figure="trials"', 1)[1] \
               .split("<tbody>", 1)[1].split("</tbody>", 1)[0]


def test_the_figures_that_draw_every_trial_are_the_ones_that_select_one():
    """Only the figures that draw each trial can select one; summaries have no
    trial points."""
    selectable = {f.key for f in FIGURES if f.selects_trials}

    assert selectable == {
        "performance_over_time", "configuration_cube", "parallel_coordinates",
        "trial_duration", "local_effects", "trials",
    }
    assert not FIGURES_BY_KEY["selected_configuration"].selects_trials, \
        "it displays the selection rather than making one"


def test_the_page_ships_the_declaration_rather_than_a_list_of_names(client):
    """Declaring `selects_trials` alone makes a figure selectable."""
    html = _page(client, _experiment())
    shipped = html.split('id="selectable-figures-data"', 1)[1].split(">", 1)[1]
    shipped = shipped.split("</script>", 1)[0]

    for key in ("performance_over_time", "configuration_cube", "trials"):
        assert key in shipped
    assert "hyperparameter_importance" not in shipped


def test_the_highlight_colors_come_from_the_builders(client):
    """The page restyles the highlight itself, so the server sends the plot
    colours instead of the script restating them."""
    from ui.figures import MARKER_COLOR, SELECTION_COLOR

    html = _page(client, _experiment())
    shipped = html.split('id="selection-colors-data"', 1)[1].split(">", 1)[1]
    shipped = shipped.split("</script>", 1)[0]

    assert MARKER_COLOR in shipped
    assert SELECTION_COLOR in shipped


def test_every_trials_row_carries_the_trial_it_stands_for(client):
    """The table keeps each row's trial in `data-trial-idx`, because a sort
    reorders the rows."""
    import re

    exp = _experiment()
    html = _page(client, exp)
    body = _trials_body(html)

    indices = [int(m) for m in re.findall(r'data-trial-idx="(\d+)"', body)]
    assert exp.data.result is not None
    assert indices == list(range(len(exp.data.result["data"])))


def test_the_rows_can_be_reached_without_a_mouse(client):
    html = _page(client, _experiment())
    body = _trials_body(html)

    assert body.count("data-trial-idx=") == 3
    assert body.count("tabindex=") == body.count("data-trial-idx=")
