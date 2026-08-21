"""One trial is selected at a time, and every figure showing it says so.

Clicking a point used to mean something on exactly one figure — Trial
performance — and be seen by exactly one other, the Selected configuration
panel. Every other figure that draws all the trials (the cube, parallel
coordinates, trial duration, local effects, the trials table) drew a point you
could see and had no way to ask about.

Now a figure declares `selects_trials` and its plot carries a
`layout.meta["selection"]` saying where its trials are, so the page's selection
bus needs no per-figure knowledge. These cover the declaration and the page
plumbing; the bus itself is JavaScript, which this suite does not run.
"""

from django.urls import reverse

from ui.figures import FIGURES, FIGURES_BY_KEY
from ui.models import Experiment

def _experiment():
    """An experiment with three finished trials, so a highlight has somewhere
    to move to and the table has rows to tell apart."""
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
    """The trials table's rows — anchored on the figure, since the page has
    other tables (the configuration panels, the importance view)."""
    return html.split('data-figure="trials"', 1)[1] \
               .split("<tbody>", 1)[1].split("</tbody>", 1)[0]


def test_the_figures_that_draw_every_trial_are_the_ones_that_select_one():
    """The declaration, stated as the rule behind it: a figure can offer a
    trial to click and a trial to highlight exactly when it draws the trials
    themselves. A summary over them — importance, interactions, partial
    dependence — has no point that is a trial."""
    selectable = {f.key for f in FIGURES if f.selects_trials}

    assert selectable == {
        "performance_over_time", "configuration_cube", "parallel_coordinates",
        "trial_duration", "local_effects", "trials",
    }
    assert not FIGURES_BY_KEY["selected_configuration"].selects_trials, \
        "it displays the selection rather than making one"


def test_the_page_ships_the_declaration_rather_than_a_list_of_names(client):
    """The bus iterates what the catalog says, so a figure that starts drawing
    trials becomes selectable by declaring it and nothing else."""
    html = _page(client, _experiment())
    shipped = html.split('id="selectable-figures-data"', 1)[1].split(">", 1)[1]
    shipped = shipped.split("</script>", 1)[0]

    for key in ("performance_over_time", "configuration_cube", "trials"):
        assert key in shipped
    assert "hyperparameter_importance" not in shipped


def test_the_highlight_colors_come_from_the_builders(client):
    """The page draws the highlight itself — a restyle, not a rebuilt figure —
    so it needs the colours the plots are already drawn in. Restating them in
    the script is one place for the two to disagree about what selected looks
    like."""
    from ui.figures import MARKER_COLOR, SELECTION_COLOR

    html = _page(client, _experiment())
    shipped = html.split('id="selection-colors-data"', 1)[1].split(">", 1)[1]
    shipped = shipped.split("</script>", 1)[0]

    assert MARKER_COLOR in shipped
    assert SELECTION_COLOR in shipped


def test_every_trials_row_carries_the_trial_it_stands_for(client):
    """The table's identity has to be an attribute, not a position: its outcome
    columns sort by reordering these very rows, so after one click on a header
    the nth row is not the nth trial."""
    import re

    exp = _experiment()
    html = _page(client, exp)
    body = _trials_body(html)

    indices = [int(m) for m in re.findall(r'data-trial-idx="(\d+)"', body)]
    assert indices == list(range(len(exp.result["data"])))


def test_the_rows_can_be_reached_without_a_mouse(client):
    """Every row selects a trial, so every row is a control."""
    html = _page(client, _experiment())
    body = _trials_body(html)

    assert body.count("data-trial-idx=") == 3
    assert body.count("tabindex=") == body.count("data-trial-idx=")
