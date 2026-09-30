"""Stating a belief before the first trial.

*What:* a prior is a statement about the **search space**, and the space exists
before the search does. Before the first run is exactly when somebody has
something to say about where the optimum might be — and until now the page gave
them nowhere to say it, because the whole figure grid waits on a result.

So one figure comes through that early return: the acquisition card, carrying
its prior panel alone. The acquisition curve, the candidates and the incumbent
are all the surrogate's, and there is no surrogate to ask.

*How:* at the page for what is rendered, and at the endpoint for what it will
answer — the two have to agree, since the card immediately asks for a density
and a card that asked for something refused would show a failure instead of the
panel it exists for.
"""

import json

import pytest
from django.urls import reverse

from ui.models import Experiment

pytestmark = pytest.mark.django_db


def _unrun():
    """An experiment created and never run. No result, no trials, a space."""
    return Experiment.objects.create(
        name="fresh", model_name="Random Forest", optimizer_name="SMAC",
        metric_names=["accuracy"], seed=0)


def _page(client, exp):
    return client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()


def _hp(exp):
    from ui.views import _config_space_for, _rebuild_experiment

    return list(_config_space_for(_rebuild_experiment(exp)).keys())[0]


# ── the page ─────────────────────────────────────────────────────────────────

def test_the_card_is_there_before_any_run(client):
    html = _page(client, _unrun())

    assert 'data-figure="acquisition_slice"' in html
    assert 'id="acq-plot-prior"' in html, "the panel it exists for"


def test_only_the_prior_panel_is(client):
    """The other two are the surrogate's. Rendered empty they would be two
    blank rectangles claiming a model nobody fitted."""
    html = _page(client, _unrun())

    assert 'id="acq-plot-acquisition"' not in html
    assert 'id="acq-show-slice"' not in html, "nothing to toggle"
    assert 'id="acq-rewalk"' not in html, "nobody to ask what runs next"


def test_the_card_says_it_wants_the_prior_alone(client):
    """The script reads this to decide which request to make. Without it the
    card would ask for a full slice, the server would refuse for want of a
    surrogate, and the reader would get the refusal."""
    html = _page(client, _unrun())

    assert 'data-prior-only="1"' in html


def test_it_carries_a_metric_of_its_own(client):
    """There is no metric switcher on a page with no results, so the card has
    to carry one or its request goes out without a metric and is refused."""
    exp = _unrun()
    html = _page(client, exp)

    assert 'data-metric="accuracy"' in html


def test_the_hyperparameters_come_from_the_space(client):
    """Not from a trial's configuration, which is where they normally come
    from. There are no trials — that is the entire situation."""
    exp = _unrun()
    html = _page(client, exp)

    assert f'<option value="{_hp(exp)}">' in html


def test_a_page_with_results_still_has_the_whole_card(client):
    """The pre-run case is an addition, not a replacement."""
    from core import io
    from tests.conftest import FIXTURES_DIR
    from ui.services import snapshot as adapter

    exp = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    html = _page(client, exp)

    assert 'id="acq-plot-acquisition"' in html
    assert 'id="acq-rewalk"' in html
    assert 'data-prior-only=""' in html, "and does not ask for the prior alone"


# ── the endpoint it will ask ─────────────────────────────────────────────────

def test_the_density_is_served_without_a_run(client):
    exp = _unrun()
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": "accuracy", "hp": _hp(exp), "prior_only": "1"})

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data["figures"]["prior"], "a panel to draw on"
    assert data["insideDesign"] is True, "nothing has run, so nothing is past it"


def test_the_full_slice_is_still_refused_without_a_run(client):
    """There is no surrogate. Refusing is right; what changed is that the prior
    no longer needs one."""
    exp = _unrun()
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": "accuracy", "hp": _hp(exp)})

    assert response.status_code == 400


def test_a_hyperparameter_outside_the_space_is_still_refused(client):
    """The check moved from the trials to the space; it did not go away."""
    exp = _unrun()
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": "accuracy", "hp": "not_a_hyperparameter",
                           "prior_only": "1"})

    assert response.status_code == 400


def test_a_belief_stated_now_has_not_started_fading(client):
    """Zero steps, and inside the initial design by definition — so the wait
    the checkbox offers is both available and, by default, on."""
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
