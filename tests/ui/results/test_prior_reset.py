"""Putting a belief back to what the last run searched under.

*What:* the prior panel saves as it is dragged, so there is no other way back
to the statement the trials on screen actually came out of. Reset is that way
back — and to uniform when there is nothing to go back to, which is the same
thing as never having stated one.

It needs a record that did not exist: the per-run `events` say what a prior
*did* (applied, skipped, with which decay) and never what it *was*. `Run.priors`
is a copy taken when the run is created, because `Experiment.priors` is the live
statement and moves the moment somebody drags the curve.

*How:* through the endpoint, against runs built directly — a real optimization
is not needed to test what is restored from a record, and using one would make
this a test of SMAC.
"""

import json

import pytest
from django.urls import reverse

from ui.models import Experiment, Run

pytestmark = pytest.mark.django_db

#: What a run searched under, in the shape `save_prior` stores.
SEARCHED_UNDER = {
    "n_estimators": {
        "kind": "normal",
        "params": {"mu": 0.2, "sigma": 0.05},
        "decay": {"shape": "linear", "beta_ratio": 0.1},
        "at_trial": 0,
        "delay_decay": True,
    }
}

#: And what the reader has since dragged it to.
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


# ── what a run remembers ─────────────────────────────────────────────────────

def test_a_run_records_what_it_searched_under():
    """Copied at creation, not referenced: the experiment's own statement goes
    on moving and this has to stay where it was."""
    from ui.services.run import create_run

    exp = _experiment(SEARCHED_UNDER)
    run = create_run(exp, {}, "accuracy")

    exp.data.priors = SINCE_DRAGGED
    exp.data.save(update_fields=["priors"])
    run.refresh_from_db()

    assert run.priors == SEARCHED_UNDER


# ── and what reset does with it ──────────────────────────────────────────────

def test_it_goes_back_to_the_last_run(client):
    exp = _experiment(SINCE_DRAGGED)
    _run(exp, SEARCHED_UNDER)

    response, data = _reset(client, exp)
    exp.refresh_from_db()

    assert response.status_code == 200
    assert data["restored"] is True
    assert exp.data.priors["n_estimators"] == SEARCHED_UNDER["n_estimators"]


def test_it_sends_back_the_curve_of_what_it_restored(client):
    """The knots come back in `prior`, and the panel draws them over whatever
    curve it holds. Without the restored belief's own curve alongside them they
    landed off the line — the curve still being the dragged one's."""
    from ui.views import _prior_densities

    exp = _experiment(SINCE_DRAGGED)
    _run(exp, SEARCHED_UNDER)
    meta = {"positions": [0.0, 0.2, 0.5, 1.0], "span": [0.0, 1.0],
            "cloud": {"positions": [0.1]}}

    response = client.post(reverse("ui:reset_prior", args=[exp.pk]),
                           {"hp": "n_estimators", "meta": json.dumps(meta)})
    data = json.loads(response.content)

    assert data["density"] == _prior_densities(SEARCHED_UNDER["n_estimators"], meta)
    assert data["density"] != _prior_densities(SINCE_DRAGGED["n_estimators"], meta)


def test_without_positions_it_sends_no_curve(client):
    """The page always says where it wants one. Anything else asking gets the
    prior and nothing computed on a guessed grid."""
    exp = _experiment(SINCE_DRAGGED)
    _run(exp, SEARCHED_UNDER)

    _, data = _reset(client, exp)

    assert data["density"] is None


def test_with_no_run_it_goes_to_uniform(client):
    """Which is stored as absence. Uniform multiplies the acquisition by a
    constant and cannot change a ranking, so it says exactly what stating
    nothing says — and a flat table the optimizer would multiply by for no
    reason is worth not storing."""
    exp = _experiment(SINCE_DRAGGED)

    response, data = _reset(client, exp)
    exp.refresh_from_db()

    assert data["restored"] is False
    assert "n_estimators" not in exp.data.priors


def test_a_run_that_stated_nothing_resets_to_uniform(client):
    """"Nothing to go back to" covers both: no run at all, and a run that
    searched under no belief for this hyperparameter."""
    exp = _experiment(SINCE_DRAGGED)
    _run(exp, {})

    _, data = _reset(client, exp)
    exp.refresh_from_db()

    assert data["restored"] is False
    assert "n_estimators" not in exp.data.priors


def test_a_pending_run_is_not_what_it_goes_back_to(client):
    """A run that has not started has searched under nothing yet, and its copy
    is of the very edits this is meant to undo."""
    exp = _experiment(SINCE_DRAGGED)
    _run(exp, SEARCHED_UNDER)
    _run(exp, SINCE_DRAGGED, status="pending")

    _, data = _reset(client, exp)
    exp.refresh_from_db()

    assert exp.data.priors["n_estimators"] == SEARCHED_UNDER["n_estimators"]
    assert data["restored"] is True


def test_the_most_recent_finished_run_wins(client):
    """Two runs, two statements. "What it was at the end of the last run" is
    the last one."""
    exp = _experiment(SINCE_DRAGGED)
    older = {"n_estimators": dict(SEARCHED_UNDER["n_estimators"],
                                  params={"mu": 0.9, "sigma": 0.2})}
    _run(exp, older)
    _run(exp, SEARCHED_UNDER)

    _reset(client, exp)
    exp.refresh_from_db()

    assert exp.data.priors["n_estimators"]["params"] == {"mu": 0.2, "sigma": 0.05}


def test_other_hyperparameters_are_left_alone(client):
    """It resets one belief, not the statement as a whole."""
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


# ── the button ───────────────────────────────────────────────────────────────

def test_the_button_is_on_the_prior_row(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="acq-reset"' in html
    assert "data-reset-url=" in html
    assert "data-reset-to-uniform=" in html, "the wording belongs to the template"
