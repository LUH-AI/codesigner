""""Evaluate Prior" offers a stated belief to SMAC's `ClimbingComparisonPolicy`
and reports the verdict. A refusal writes nothing, because the reader may know
something the model lacks.
"""

import json
import random
import re
from pathlib import Path
from typing import Any

import pytest
from django.urls import reverse

from experiments.models import Experiment
from experiments.services.settings import SETTING_BOUNDS
from ui.experiment_settings import SETTING_DEFAULTS, resolve_settings

pytestmark = pytest.mark.django_db

#: `unjudged` means the policy had nothing to judge on, which differs from
#: approval.
VERDICTS = {"accepted", "rejected", "unjudged"}


#: Unit-axis position of the `min_samples_split` optimum.
OPTIMUM = 0.04


def _accuracy(config):
    """A belief at the far end of `min_samples_split` costs about 0.4 accuracy."""
    return (0.95 - 3.0 * (config["min_samples_split"] - 0.03) ** 2
            - 0.2 * (config["max_features"] - 0.8) ** 2)


def _experiment(strategy="gp", n=30):
    """`_experiment` builds a SMAC experiment with *n* trials of `_accuracy`
    under *strategy*."""
    from core import io, provenance
    from core.models import RandomForestModel
    from core.optimizers import SMACOptimizer
    from core.optimizers.base import OptimizationResult, TrialResult
    from experiments.services import snapshot as adapter

    space = RandomForestModel().get_config_space()
    space.seed(3)
    noise = random.Random(3)
    trials = []
    for i, config in enumerate(space.sample_configuration(n)):
        values = dict(config)
        score = _accuracy(values) + noise.gauss(0, 0.005)
        trials.append(TrialResult(trial=i + 1, config=values, scores={"accuracy": score},
                                  score=score, incumbent_score=score, incumbent_config=values))
    best = max(trials, key=lambda t: t.score)
    result = OptimizationResult(
        trials=trials, primary_metric="accuracy", best_config=best.config,
        best_score=best.score, hyperparameter_importance={},
        hyperparameter_importance_warning={})
    optimizer = SMACOptimizer(search_strategy=strategy)
    return adapter.experiment_from_snapshot(io.parse(io.to_bytes({
        "format": io.SNAPSHOT_FORMAT, "version": "", "name": "prior check", "seed": 0,
        "dataset": {"filename": "", "path": ""},
        "model": {"kind": "registry", "name": RandomForestModel().name, "path": ""},
        "evaluation": provenance.evaluation(0),
        "metrics": {"names": ["accuracy"], "current": "accuracy", "original": "accuracy"},
        "optimizer": {"name": optimizer.name, "params": optimizer.get_params()},
        "result": optimizer.serialize_result(result)})))


def _believing(exp, mu):
    """`_believing` centres `min_samples_split` on *mu*."""
    exp.data.priors = {"min_samples_split": {
        "kind": "normal", "params": {"mu": mu, "sigma": 0.05},
        "decay": {"shape": "none"}, "at_trial": 0}}
    exp.data.save(update_fields=["priors"])
    return exp


def _bare_experiment():
    """`_bare_experiment` has no run and no model to judge."""
    return Experiment.objects.create(
        name="unrun", model_name="Random Forest", optimizer_name="SMAC",
        metric_names=["accuracy"], seed=0)


def _script():
    return (Path(__file__).parents[3] / "ui" / "static" / "ui"
            / "acquisition.js").read_text()


def _evaluate(client, exp, hp="") -> tuple[Any, Any]:
    response = client.post(reverse("ui:evaluate_prior", args=[exp.pk]), {"hp": hp})
    return response, (json.loads(response.content)
                      if response.status_code == 200 else None)


# the answer

def test_an_unrun_experiment_is_not_judged(client):
    """Without a model the policy must accept, so the endpoint reports
    `unjudged` instead."""
    response, data = _evaluate(client, _bare_experiment())

    assert response.status_code == 200
    assert data["verdict"] == "unjudged"
    assert data["message"], "an answer nobody can read is not an answer"


def test_a_belief_at_the_optimum_is_accepted(client):
    response, data = _evaluate(client, _believing(_experiment(), OPTIMUM),
                               "min_samples_split")

    assert response.status_code == 200
    assert data["verdict"] == "accepted"


def test_a_belief_far_from_the_optimum_is_refused(client):
    """The refusal also reports the allowed margin."""
    response, data = _evaluate(client, _believing(_experiment(), 0.95),
                               "min_samples_split")

    assert data["verdict"] == "rejected"
    assert data["margin"] == pytest.approx(0.15), "an accuracy's scale is one"
    assert "0.15" in data["message"]


def test_every_answer_is_one_of_the_three(client):
    """The page must handle the three `VERDICTS`, with or without SMAC's
    acquisition weight layer."""
    exp = _experiment()
    response, data = _evaluate(client, exp)

    assert response.status_code == 200
    assert data["verdict"] in VERDICTS
    assert data["message"]
    assert data["tolerance"] == resolve_settings(exp)["prior_acceptance_tolerance"]


def test_judging_writes_nothing(client):
    """The belief stays the reader's until withdrawn."""
    exp = _experiment()
    exp.data.priors = {"n_estimators": {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1},
                                   "decay": {"shape": "none"}, "at_trial": 0}}
    exp.data.save(update_fields=["priors"])
    before = json.dumps(exp.data.priors, sort_keys=True)

    _evaluate(client, exp, "n_estimators")
    exp.refresh_from_db()

    assert json.dumps(exp.data.priors, sort_keys=True) == before


def test_it_is_a_post(client):
    """A model fit and many acquisition evaluations are too costly for a GET
    that prefetchers could trigger."""
    response = client.get(reverse("ui:evaluate_prior", args=[_bare_experiment().pk]))

    assert response.status_code == 405


# the setting

def test_the_tolerance_is_a_share_of_the_metrics_scale():
    """DynaBO's 0.15 is the default share."""
    assert SETTING_DEFAULTS["prior_acceptance_tolerance"] == 0.15
    assert SETTING_BOUNDS["prior_acceptance_tolerance"] == (0.0, 1.0)


def test_the_tolerance_keeps_its_decimals(client):
    """Read as an integer, the tolerance would truncate to zero and reject every
    prior."""
    from django.test import RequestFactory

    from ui.views.preferences import _posted_settings

    request = RequestFactory().post("/", {"prior_acceptance_tolerance": "0.4"})
    assert _posted_settings(request)["prior_acceptance_tolerance"] == pytest.approx(0.4)


def test_the_tolerance_is_offered_on_the_settings_page(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_settings", args=[exp.pk])).content.decode()

    assert 'name="prior_acceptance_tolerance"' in html
    assert 'name="prior_acceptance_top_k"' in html


def test_top_k_defaults_to_smacs():
    """The default matches `ClimbingComparisonPolicy`'s top k, and the bounds
    forbid zero climbs."""
    assert SETTING_DEFAULTS["prior_acceptance_top_k"] == 10
    assert SETTING_BOUNDS["prior_acceptance_top_k"] == (1, 100)


@pytest.mark.parametrize("posted, used", [
    ({}, 10), ({"top_k": "3"}, 3), ({"top_k": "0"}, 1), ({"top_k": "5000"}, 100),
    ({"top_k": "many"}, 10)])
def test_the_posted_top_k_is_clamped_or_falls_back(client, monkeypatch, posted, used):
    """A missing or non-numeric top k falls back to the setting."""
    from core.optimizers.smac import SMACOptimizer

    seen = {}

    def _record(self, *args, top_k=None, **kwargs):
        seen["top_k"] = top_k
        return {"verdict": "accepted", "reason": ""}

    monkeypatch.setattr(SMACOptimizer, "evaluate_prior", _record)
    exp = _experiment()
    response = client.post(reverse("ui:evaluate_prior", args=[exp.pk]),
                           {"hp": "", **posted})

    assert response.status_code == 200
    assert seen["top_k"] == used
    assert json.loads(response.content)["top_k"] == used


# the page

def test_the_figure_offers_the_question_and_the_answer(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="acq-evaluate"' in html, "somewhere to ask from"
    assert 'id="acq-verdict"' in html, "somewhere to say what came back"
    assert "<dialog" in html and 'id="acq-verdict-dialog"' in html
    assert 'id="acq-verdict-keep"' in html and 'id="acq-verdict-discard"' in html


# availability

def _evaluate_button(html):
    found = re.search(r'<button[^>]*id="acq-evaluate"[^>]*>', html)
    assert found is not None
    return found.group(0)


def _top_k_field(html):
    found = re.search(r'<input[^>]*id="acq-top-k"[^>]*>', html)
    assert found is not None
    return found.group(0)


def test_only_the_gaussian_process_judges_priors():
    """Under the random forest every prior passes, and a model-free optimizer
    has nothing to judge with."""
    from core.optimizers.random_optimizer import RandomOptimizer
    from core.optimizers.smac import SMACOptimizer

    smac = SMACOptimizer()
    assert smac.judges_priors({"search_strategy": "gp"})
    assert not smac.judges_priors({"search_strategy": "rf"})
    assert smac.judges_priors({})
    assert smac.judges_priors({"search_strategy": "withdrawn"})
    assert not RandomOptimizer().judges_priors({})


def test_the_button_is_disabled_under_the_random_forest(client):
    """Under the random forest the button is greyed out with a tooltip, and the
    script learns it must stay disabled."""
    url = reverse("ui:experiment_detail", args=[_experiment().pk])
    rf_url = reverse("ui:experiment_detail", args=[_experiment("rf").pk])

    html = client.get(url).content.decode()
    assert "disabled" not in _evaluate_button(html)
    assert 'data-judges-priors="1"' in html
    field = _top_k_field(html)
    assert "disabled" not in field and 'value="10"' in field

    html = client.get(rf_url).content.decode()
    button = _evaluate_button(html)
    assert "disabled" in button and "title=" in button
    assert 'data-judges-priors=""' in html
    assert "disabled" in _top_k_field(html)


def test_the_random_forest_is_not_judged_even_if_asked(client, monkeypatch):
    """A stale page's live button gets `unjudged`, never a random forest
    verdict."""
    from core.optimizers.smac import SMACOptimizer

    def _reached(*args, **kwargs):
        raise AssertionError("the random forest was asked to judge a prior")

    monkeypatch.setattr(SMACOptimizer, "evaluate_prior", _reached)
    response, data = _evaluate(client, _believing(_experiment("rf"), 0.95),
                               "min_samples_split")

    assert response.status_code == 200
    assert data["verdict"] == "unjudged"
    assert data["message"]


def test_the_script_never_re_enables_a_button_it_cannot_use():
    """The script's two enabling paths both defer to the server's flag."""
    source = _script()

    assert 'evaluateBtn.disabled = !judgesPriors;' in source
    assert 'evaluateBtn.disabled = kind === "uniform" || !judgesPriors;' in source


def test_only_a_refusal_asks_a_question():
    """A dialog for approval would teach people to dismiss it."""
    source = _script()

    assert 'data.verdict === "rejected" && verdictDialog' in source


def test_the_verdict_is_smacs():
    """The server relies on SMAC's `add_prior` acceptance policy, keeping one
    implementation of the judgement."""
    smac = Path(__file__).parents[3] / "core" / "optimizers" / "smac"
    optimizer = (smac / "optimizer.py").read_text() + (smac / "priors.py").read_text()

    assert "acceptance_policy=acceptance" in optimizer
    assert "class _RecordingPolicy(ClimbingComparisonPolicy)" in optimizer
    assert "threshold=-abs(tolerance), top_k=int(top_k)" in optimizer


def test_a_changed_belief_drops_its_verdict():
    """A stale approval would mislead under a curve it no longer describes."""
    assert "setVerdict(null);" in _script()


# tolerance scale

def _diabetes(client):
    """`_diabetes` runs a three-trial diabetes regression scored by RMSE."""
    from tests.conftest import DATASETS_DIR, post_new_experiment

    post_new_experiment(client, {
        "name": "diabetes", "model_name": "Random Forest", "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "diabetes.csv"), "task": "regression",
        "evaluation_scheme": "holdout", "evaluation_value": 0.2, "seed": 0})
    exp = Experiment.objects.get(data__name="diabetes")
    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 3, "optimize_metric": "rmse"})
    exp.refresh_from_db()
    return exp


def test_an_unbounded_metric_is_scaled_by_what_knowing_nothing_scores(client):
    """An RMSE has no worst end, so its scale is the spread of the validation
    targets."""
    import numpy as np

    from core import datasets
    from tests.conftest import DATASETS_DIR
    from experiments.services.snapshot import rebuild_result
    from experiments.services.priors import metric_scale

    exp = _diabetes(client)
    _, _, _, y_val = datasets.load_splits(DATASETS_DIR / "diabetes.csv", 0, task="regression")

    assert metric_scale(exp, "rmse", rebuild_result(exp)) == pytest.approx(float(np.std(y_val)))


def test_without_its_data_an_unbounded_metric_is_scaled_by_its_trials(client, monkeypatch):
    """An imported experiment without its dataset is scaled by the spread of
    trial scores."""
    from core import io
    from experiments.services.snapshot import rebuild_result
    from experiments.services.priors import metric_scale

    exp = _diabetes(client)
    result = rebuild_result(exp)
    assert result is not None

    def _missing(*args, **kwargs):
        raise ValueError("dataset not found")

    monkeypatch.setattr(io, "build_experiment", _missing)
    scores = [t.scores["rmse"] for t in result.trials]

    assert metric_scale(exp, "rmse", result) == pytest.approx(max(scores) - min(scores))


def test_smac_is_handed_the_tolerance_in_the_metrics_units(client, monkeypatch):
    """SMAC receives the setting's share times the metric's scale."""
    from core.optimizers.smac import SMACOptimizer

    seen = {}

    def _record(self, *args, tolerance=None, **kwargs):
        seen["tolerance"] = tolerance
        return {"verdict": "accepted", "reason": ""}

    monkeypatch.setattr(SMACOptimizer, "evaluate_prior", _record)
    exp = _experiment()
    exp.use_default_settings = False
    exp.settings = {"prior_acceptance_tolerance": 0.2}
    exp.save()
    _response, data = _evaluate(client, exp)

    assert seen["tolerance"] == pytest.approx(0.2)
    assert data["tolerance"] == 0.2 and data["margin"] == pytest.approx(0.2)
