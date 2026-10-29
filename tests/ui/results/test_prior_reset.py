"""Reset restores the prior the last finished run searched under, or uniform
when none exists. `Run.priors` copies the statement at run creation, because
`Experiment.priors` moves with every drag.
"""

import json

import pytest
from django.urls import reverse

from experiments.models import Experiment, Run

pytestmark = pytest.mark.django_db

#: The run's prior, in `save_prior`'s shape.
SEARCHED_UNDER = {
    "n_estimators": {
        "kind": "normal",
        "params": {"mu": 0.2, "sigma": 0.05},
        "decay": {"shape": "linear", "beta_ratio": 0.1},
        "at_trial": 0,
        "delay_decay": True,
    }
}

#: The reader's later drag.
SINCE_DRAGGED = {
    "n_estimators": {
        "kind": "beta",
        "params": {"alpha": 9.0, "beta": 2.0},
        "decay": {"shape": "none"},
        "at_trial": 7,
        "delay_decay": False,
    }
}


def _experiment(priors=None):
    return Experiment.objects.create(
        name="resettable", model_name="Random Forest", optimizer_name="SMAC",
        metric_names=["accuracy"], seed=0, priors=priors or {})


def _run(exp, priors, status="done"):
    return Run.objects.create(experiment=exp, primary_metric="accuracy",
                              status=status, priors=priors or {})


def _reset(client, exp, hp="n_estimators"):
    response = client.post(reverse("ui:reset_prior", args=[exp.pk]), {"hp": hp})
    return response, (json.loads(response.content)
                      if response.status_code == 200 else None)


# run record

def test_a_run_records_what_it_searched_under():
    """The run's copy keeps its value while the experiment's statement moves."""
    from experiments.services.run import create_run

    exp = _experiment(SEARCHED_UNDER)
    run = create_run(exp, {}, "accuracy")

    exp.data.priors = SINCE_DRAGGED
    exp.data.save(update_fields=["priors"])
    run.refresh_from_db()

    assert run.priors == SEARCHED_UNDER


# reset

def test_it_goes_back_to_the_last_run(client):
    exp = _experiment(SINCE_DRAGGED)
    _run(exp, SEARCHED_UNDER)

    response, data = _reset(client, exp)
    assert data is not None
    exp.refresh_from_db()

    assert response.status_code == 200
    assert data["restored"] is True
    assert exp.data.priors["n_estimators"] == SEARCHED_UNDER["n_estimators"]


def test_it_sends_back_the_curve_of_what_it_restored(client):
    """The response carries the restored curve, so the panel's knots land on the
    line."""
    from core.priors import prior_densities

    exp = _experiment(SINCE_DRAGGED)
    _run(exp, SEARCHED_UNDER)
    meta = {"positions": [0.0, 0.2, 0.5, 1.0], "span": [0.0, 1.0],
            "cloud": {"positions": [0.1]}}

    response = client.post(reverse("ui:reset_prior", args=[exp.pk]),
                           {"hp": "n_estimators", "meta": json.dumps(meta)})
    data = json.loads(response.content)

    assert data["density"] == prior_densities(SEARCHED_UNDER["n_estimators"], meta)
    assert data["density"] != prior_densities(SINCE_DRAGGED["n_estimators"], meta)


def test_without_positions_it_sends_no_curve(client):
    """Without positions the endpoint computes no curve on a guessed grid."""
    exp = _experiment(SINCE_DRAGGED)
    _run(exp, SEARCHED_UNDER)

    _, data = _reset(client, exp)
    assert data is not None

    assert data["density"] is None


def test_with_no_run_it_goes_to_uniform(client):
    """Uniform cannot change a ranking, so the endpoint stores it as absence."""
    exp = _experiment(SINCE_DRAGGED)

    response, data = _reset(client, exp)
    assert data is not None
    exp.refresh_from_db()

    assert data["restored"] is False
    assert "n_estimators" not in exp.data.priors


def test_a_run_that_stated_nothing_resets_to_uniform(client):
    exp = _experiment(SINCE_DRAGGED)
    _run(exp, {})

    _, data = _reset(client, exp)
    assert data is not None
    exp.refresh_from_db()

    assert data["restored"] is False
    assert "n_estimators" not in exp.data.priors


def test_a_pending_run_is_not_what_it_goes_back_to(client):
    """A pending run's copy contains the very edits reset should undo."""
    exp = _experiment(SINCE_DRAGGED)
    _run(exp, SEARCHED_UNDER)
    _run(exp, SINCE_DRAGGED, status="pending")

    _, data = _reset(client, exp)
    assert data is not None
    exp.refresh_from_db()

    assert exp.data.priors["n_estimators"] == SEARCHED_UNDER["n_estimators"]
    assert data["restored"] is True


def test_the_most_recent_finished_run_wins(client):
    exp = _experiment(SINCE_DRAGGED)
    older = {"n_estimators": dict(SEARCHED_UNDER["n_estimators"],
                                  params={"mu": 0.9, "sigma": 0.2})}
    _run(exp, older)
    _run(exp, SEARCHED_UNDER)

    _reset(client, exp)
    exp.refresh_from_db()

    assert exp.data.priors["n_estimators"]["params"] == {"mu": 0.2, "sigma": 0.05}


def test_other_hyperparameters_are_left_alone(client):
    exp = _experiment({**SINCE_DRAGGED,
                       "max_depth": {"kind": "beta", "params": {"alpha": 2.0,
                                                                "beta": 2.0},
                                     "decay": {"shape": "none"}, "at_trial": 0}})
    _run(exp, SEARCHED_UNDER)

    _reset(client, exp)
    exp.refresh_from_db()

    assert exp.data.priors["max_depth"]["kind"] == "beta"


def test_it_is_a_post(client):
    response = client.get(reverse("ui:reset_prior", args=[_experiment().pk]))

    assert response.status_code == 405


def test_it_needs_a_hyperparameter(client):
    response = client.post(reverse("ui:reset_prior", args=[_experiment().pk]), {})

    assert response.status_code == 400


# the button

def test_the_button_is_on_the_prior_row(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="acq-reset"' in html
    assert "data-reset-url=" in html
    assert "data-reset-to-uniform=" in html, "the wording belongs to the template"
