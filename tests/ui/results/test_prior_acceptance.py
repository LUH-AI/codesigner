"""Asking the model whether a stated belief is worth acting on.

*What:* DynaBO's safeguard, reachable from the figure. A belief that points
somewhere the surrogate already thinks is bad costs trials, and the model
usually has an opinion about the region it names — so "Evaluate Prior" offers
the belief to SMAC's `IncumbentComparisonPolicy` and reports what came back.

The rule the whole feature turns on: a refusal is not an instruction. The
reader may know something the model does not, which is the reason for stating a
belief in the first place, so nothing is written either way and a rejected
prior is left exactly where it was while the page asks what to do about it.

*How:* through the endpoint for the contract it answers with, at the page for
the controls, and at the source for the two things that cannot be reached from
a test without a model fitted by the branch SMAC — that the verdict is SMAC's
and that only a refusal asks a question.
"""

import json
from pathlib import Path

import pytest
from django.urls import reverse

from tests.conftest import FIXTURES_DIR
from ui.models import Experiment
from ui.services.settings import SETTING_BOUNDS, SETTING_DEFAULTS, resolve_settings

pytestmark = pytest.mark.django_db

#: The three answers, and the only three. `unjudged` is not a failure: the
#: policy is required to accept when there is nothing to judge on, and
#: reporting that as approval would claim a check that never ran.
VERDICTS = {"accepted", "rejected", "unjudged"}


def _experiment():
    from core import io
    from ui.services import snapshot as adapter

    return adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def _bare_experiment():
    """One that has never been run, so there is no model to judge against."""
    return Experiment.objects.create(
        name="unrun", model_name="Random Forest", optimizer_name="SMAC",
        metric_names=["accuracy"], seed=0)


def _script():
    return (Path(__file__).parents[3] / "ui" / "static" / "ui"
            / "acquisition.js").read_text()


def _evaluate(client, exp, hp=""):
    response = client.post(reverse("ui:evaluate_prior", args=[exp.pk]), {"hp": hp})
    return response, (json.loads(response.content)
                      if response.status_code == 200 else None)


# ── what it answers with ─────────────────────────────────────────────────────

def test_an_unrun_experiment_is_not_judged(client):
    """And says so rather than approving. Before the first trial the policy has
    no model and no incumbent, so it is contractually bound to accept — which
    is the one case where "accepted" would be a lie."""
    response, data = _evaluate(client, _bare_experiment())

    assert response.status_code == 200
    assert data["verdict"] == "unjudged"
    assert data["message"], "an answer nobody can read is not an answer"


def test_every_answer_is_one_of_the_three(client):
    """Whatever the installed SMAC can do. Without the acquisition weight layer
    the verdict is `unjudged`; with it, the model decides — and the page has to
    handle exactly this set either way."""
    exp = _experiment()
    response, data = _evaluate(client, exp)

    assert response.status_code == 200
    assert data["verdict"] in VERDICTS
    assert data["message"]
    assert data["tolerance"] == resolve_settings(exp)["prior_acceptance_tolerance"]


def test_judging_writes_nothing(client):
    """The point of asking rather than enforcing. Whatever the model thinks, the
    belief is still the reader's until they withdraw it themselves."""
    exp = _experiment()
    exp.data.priors = {"n_estimators": {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1},
                                   "decay": {"shape": "none"}, "at_trial": 0}}
    exp.data.save(update_fields=["priors"])
    before = json.dumps(exp.data.priors, sort_keys=True)

    _evaluate(client, exp, "n_estimators")
    exp.refresh_from_db()

    assert json.dumps(exp.data.priors, sort_keys=True) == before


def test_it_is_a_post(client):
    """It costs a model fit and two hundred acquisition evaluations. A GET that
    expensive is one a link, a prefetch or a crawler can spend for you."""
    response = client.get(reverse("ui:evaluate_prior", args=[_bare_experiment().pk]))

    assert response.status_code == 405


# ── the setting behind it ────────────────────────────────────────────────────

def test_the_tolerance_is_the_readers_to_set():
    """It is in the objective's own units — an accuracy and an RMSE do not
    disagree by the same numbers — so no default can be right for every run."""
    assert SETTING_DEFAULTS["prior_acceptance_tolerance"] == 0.15
    assert SETTING_BOUNDS["prior_acceptance_tolerance"][0] == 0.0


def test_the_tolerance_keeps_its_decimals(client):
    """The settings form reads each value as the type of its default, and this
    is the first one that is not a whole number. Read as an integer it would
    truncate to zero and silently start refusing every prior."""
    from django.test import RequestFactory

    from ui.views import _posted_settings

    request = RequestFactory().post("/", {"prior_acceptance_tolerance": "0.4"})
    assert _posted_settings(request)["prior_acceptance_tolerance"] == pytest.approx(0.4)


def test_the_tolerance_is_offered_on_the_settings_page(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_settings", args=[exp.pk])).content.decode()

    assert 'name="prior_acceptance_tolerance"' in html


# ── the page ─────────────────────────────────────────────────────────────────

def test_the_figure_offers_the_question_and_the_answer(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="acq-evaluate"' in html, "somewhere to ask from"
    assert 'id="acq-verdict"' in html, "somewhere to say what came back"
    assert "<dialog" in html and 'id="acq-verdict-dialog"' in html
    assert 'id="acq-verdict-keep"' in html and 'id="acq-verdict-discard"' in html


def test_only_a_refusal_asks_a_question():
    """Approval and "could not be judged" are things to read, not things to
    answer — a dialog for either would be a dialog people learn to dismiss."""
    source = _script()

    assert 'data.verdict === "rejected" && verdictDialog' in source


def test_the_verdict_is_smacs():
    """Nothing here re-decides it. The server offers the belief to `add_prior`
    behind an acceptance policy and reports whether it was registered, which is
    where SMAC makes the judgement — a second implementation would be a second
    opinion to keep in step forever."""
    optimizer = (Path(__file__).parents[3] / "core" / "optimizers"
                 / "smac_optimizer.py").read_text()

    assert "acceptance_policy=acceptance" in optimizer
    assert "IncumbentComparisonPolicy(threshold=-abs(tolerance))" in optimizer


def test_a_changed_belief_drops_its_verdict():
    """A stale approval is the one outcome that could mislead: it would sit
    under a curve it no longer describes and suggest there is nothing to ask."""
    assert "setVerdict(null);" in _script()
