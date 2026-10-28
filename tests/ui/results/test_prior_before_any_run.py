"""An unrun experiment shows the prior card alone, because a prior describes the
search space and the space exists before any trial.
"""

import json

import pytest
from django.urls import reverse

from experiments.models import Experiment

pytestmark = pytest.mark.django_db


def _unrun():
    """`_unrun` creates a never-run experiment, a space without trials."""
    return Experiment.objects.create(
        name="fresh", model_name="Random Forest", optimizer_name="SMAC",
        metric_names=["accuracy"], seed=0)


def _page(client, exp):
    return client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()


def _hp(exp):
    from experiments.services.snapshot import config_space_for, rebuild_experiment

    space = config_space_for(rebuild_experiment(exp))
    assert space is not None
    return list(space.keys())[0]


# the page

def test_the_card_is_there_before_any_run(client):
    html = _page(client, _unrun())

    assert 'data-figure="prior"' in html
    assert 'data-figure="acquisition_slice"' not in html
    assert 'id="acq-plot-prior"' in html, "the panel it exists for"


def test_only_the_prior_panel_is(client):
    """The surrogate panels would be blank without a fitted model."""
    html = _page(client, _unrun())

    assert 'id="acq-plot-acquisition"' not in html
    assert 'id="acq-show-slice"' not in html, "nothing to toggle"
    assert 'id="acq-requery"' not in html, "nobody to ask what runs next"


def test_the_card_says_it_wants_the_prior_alone(client):
    """The script reads `data-prior-only` to skip the full slice the server
    would refuse."""
    html = _page(client, _unrun())

    assert 'data-prior-only="1"' in html


def test_it_carries_a_metric_of_its_own(client):
    """With no metric switcher, the card carries its own metric."""
    exp = _unrun()
    html = _page(client, exp)

    assert 'data-metric="accuracy"' in html


def test_the_hyperparameters_come_from_the_space(client):
    """With no trials, the hyperparameter options come from the space."""
    exp = _unrun()
    html = _page(client, exp)

    assert f'<option value="{_hp(exp)}">' in html


def test_a_page_with_results_still_has_the_whole_card(client):
    from core import io
    from tests.conftest import FIXTURES_DIR
    from experiments.services import snapshot as adapter

    exp = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    html = _page(client, exp)

    assert 'id="acq-plot-acquisition"' in html
    assert 'id="acq-requery"' in html
    assert 'data-prior-only=""' in html, "and does not ask for the prior alone"


# the endpoint

def test_the_density_is_served_without_a_run(client):
    exp = _unrun()
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": "accuracy", "hp": _hp(exp), "prior_only": "1"})

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data["figures"]["prior"], "a panel to draw on"
    assert data["insideDesign"] is True, "nothing has run, so nothing is past it"


def test_the_full_slice_is_still_refused_without_a_run(client):
    """Without a surrogate the full slice is still refused."""
    exp = _unrun()
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": "accuracy", "hp": _hp(exp)})

    assert response.status_code == 400


def test_a_hyperparameter_outside_the_space_is_still_refused(client):
    exp = _unrun()
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": "accuracy", "hp": "not_a_hyperparameter",
                           "prior_only": "1"})

    assert response.status_code == 400


def test_a_belief_stated_now_has_not_started_fading(client):
    """Before any trial, a new belief has zero decay steps and lies inside the
    initial design."""
    exp = _unrun()
    hp = _hp(exp)
    client.post(reverse("ui:save_prior", args=[exp.pk]),
                data=json.dumps({"hp": hp, "delay_decay": True,
                                 "prior": {"kind": "normal",
                                           "params": {"mu": 0.5, "sigma": 0.1},
                                           "decay": {"shape": "linear"},
                                           "delay_decay": True}}),
                content_type="application/json")
    exp.refresh_from_db()

    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": "accuracy", "hp": hp, "prior_only": "1"})
    prior = json.loads(response.content)["prior"]

    assert exp.data.priors[hp]["delay_decay"] is True
    assert prior["exponent"] == 1.0
    assert prior["steps"] == 0
