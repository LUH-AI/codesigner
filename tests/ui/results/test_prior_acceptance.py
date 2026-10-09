"""Asking the model whether a stated belief is worth acting on.

*What:* DynaBO's safeguard, reachable from the figure. A belief that points
somewhere the surrogate already thinks is bad costs trials, and the model
usually has an opinion about where it leads — so "Evaluate Prior" offers the
belief to SMAC's `ClimbingComparisonPolicy` and reports what came back.

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
import random
import re
from pathlib import Path

import pytest
from django.urls import reverse

from ui.models import Experiment
from ui.services.settings import SETTING_BOUNDS, SETTING_DEFAULTS, resolve_settings

pytestmark = pytest.mark.django_db

#: The three answers, and the only three. `unjudged` is not a failure: the
#: policy is required to accept when there is nothing to judge on, and
#: reporting that as approval would claim a check that never ran.
VERDICTS = {"accepted", "rejected", "unjudged"}


#: Where the made-up accuracy below peaks along `min_samples_split`, the
#: hyperparameter the priors here are about, as a position on its unit axis.
OPTIMUM = 0.04


def _accuracy(config):
    """Peaks at `min_samples_split` = 0.03 and `max_features` = 0.8, and falls
    off steeply enough along the first that a belief at its far end costs
    about 0.4 — far past the default tolerance."""
    return (0.95 - 3.0 * (config["min_samples_split"] - 0.03) ** 2
            - 0.2 * (config["max_features"] - 0.8) ** 2)


def _experiment(strategy="gp", n=30):
    """A Random Forest experiment searched by SMAC under *strategy*, with *n*
    trials of `_accuracy`, written and read back as an `.ihpo` the way this
    build writes one."""
    from core import io
    from core.metrics import METRICS
    from core.models import RandomForestModel
    from core.optimizers import SMACOptimizer
    from core.optimizers.base import OptimizationResult, TrialResult
    from ui.services import snapshot as adapter

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
    return adapter.experiment_from_snapshot(io.parse(io.save("prior check", {
        "seed": 0, "dataset_path": "", "model_name": RandomForestModel().name,
        "metrics": {"accuracy": METRICS["accuracy"]}, "current_metric": "accuracy",
        "original_metric": "accuracy", "optimizer": SMACOptimizer(search_strategy=strategy),
        "result": result})))


def _believing(exp, mu):
    """*exp* with a Normal belief about `min_samples_split` at *mu*."""
    exp.data.priors = {"min_samples_split": {
        "kind": "normal", "params": {"mu": mu, "sigma": 0.05},
        "decay": {"shape": "none"}, "at_trial": 0}}
    exp.data.save(update_fields=["priors"])
    return exp


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


def test_a_belief_at_the_optimum_is_accepted(client):
    """*What:* the whole path, page request to SMAC and back, accepts a belief
    pointing where the objective peaks.

    *How:* posted for an experiment whose made-up accuracy peaks where the
    belief points."""
    response, data = _evaluate(client, _believing(_experiment(), OPTIMUM),
                               "min_samples_split")

    assert response.status_code == 200
    assert data["verdict"] == "accepted"


def test_a_belief_far_from_the_optimum_is_refused(client):
    """*What:* the whole path refuses a belief pointing where the objective is
    far worse, and says by how much it was allowed to be.

    *How:* the same experiment, believing in the far end of the axis."""
    response, data = _evaluate(client, _believing(_experiment(), 0.95),
                               "min_samples_split")

    assert data["verdict"] == "rejected"
    assert data["margin"] == pytest.approx(0.15), "an accuracy's scale is one"
    assert "0.15" in data["message"]


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
    """It costs a model fit, thousands of acquisition evaluations and a local
    search per climb. A GET that expensive is one a link, a prefetch or a
    crawler can spend for you."""
    response = client.get(reverse("ui:evaluate_prior", args=[_bare_experiment().pk]))

    assert response.status_code == 405


# ── the setting behind it ────────────────────────────────────────────────────

def test_the_tolerance_is_a_share_of_the_metrics_scale():
    """*What:* the tolerance is a share, DynaBO's 0.15 by default, so it can
    run from refusing anything worse to accepting all but the worthless.

    *How:* read from the settings tables."""
    assert SETTING_DEFAULTS["prior_acceptance_tolerance"] == 0.15
    assert SETTING_BOUNDS["prior_acceptance_tolerance"] == (0.0, 1.0)


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
    assert 'name="prior_acceptance_top_k"' in html


def test_top_k_defaults_to_smacs():
    """*What:* the setting starts at `ClimbingComparisonPolicy`'s own top k and
    can never ask for no climbs at all.

    *How:* read from the settings tables."""
    assert SETTING_DEFAULTS["prior_acceptance_top_k"] == 10
    assert SETTING_BOUNDS["prior_acceptance_top_k"] == (1, 100)


@pytest.mark.parametrize("posted, used", [
    ({}, 10), ({"top_k": "3"}, 3), ({"top_k": "0"}, 1), ({"top_k": "5000"}, 100),
    ({"top_k": "many"}, 10)])
def test_the_posted_top_k_is_clamped_or_falls_back(client, monkeypatch, posted, used):
    """*What:* the top k from the field beside the button is used, clamped to
    the setting's bounds, and the setting's value when it is missing or not a
    number.

    *How:* SMAC's judgement replaced by one that records the top k it was
    handed, for each kind of posted value."""
    from core.optimizers.smac_optimizer import SMACOptimizer

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


# ── the page ─────────────────────────────────────────────────────────────────

def test_the_figure_offers_the_question_and_the_answer(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="acq-evaluate"' in html, "somewhere to ask from"
    assert 'id="acq-verdict"' in html, "somewhere to say what came back"
    assert "<dialog" in html and 'id="acq-verdict-dialog"' in html
    assert 'id="acq-verdict-keep"' in html and 'id="acq-verdict-discard"' in html


# ── whether it is offered ────────────────────────────────────────────────────

def _evaluate_button(html):
    return re.search(r'<button[^>]*id="acq-evaluate"[^>]*>', html).group(0)


def _top_k_field(html):
    return re.search(r'<input[^>]*id="acq-top-k"[^>]*>', html).group(0)


def test_only_the_gaussian_process_judges_priors():
    """*What:* which optimizer settings can judge a prior. Under the random
    forest the bound the judgement scores with is so optimistic that every prior
    passes, and an optimizer with no model has nothing to judge with.

    *How:* asked of the optimizers directly, for both strategies, for none
    stated, and for a strategy that no longer exists, which SMAC reads as the
    Gaussian process."""
    from core.optimizers.random_optimizer import RandomOptimizer
    from core.optimizers.smac_optimizer import SMACOptimizer

    smac = SMACOptimizer()
    assert smac.judges_priors({"search_strategy": "gp"})
    assert not smac.judges_priors({"search_strategy": "rf"})
    assert smac.judges_priors({})
    assert smac.judges_priors({"search_strategy": "withdrawn"})
    assert not RandomOptimizer().judges_priors({})


def test_the_button_is_disabled_under_the_random_forest(client):
    """*What:* "Evaluate Prior" is greyed out, with the reason as its tooltip,
    when the experiment's optimizer cannot judge a prior, and the script is told
    so it never re-enables it.

    *How:* the dashboards of two experiments that differ only in strategy."""
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
    """*What:* a stale page whose button is still live gets "unjudged" rather
    than a verdict every prior would pass.

    *How:* the endpoint posted to under the random forest, with SMAC's judgement
    replaced by one that fails the test if it is reached."""
    from core.optimizers.smac_optimizer import SMACOptimizer

    def _reached(*args, **kwargs):
        raise AssertionError("the random forest was asked to judge a prior")

    monkeypatch.setattr(SMACOptimizer, "evaluate_prior", _reached)
    response, data = _evaluate(client, _believing(_experiment("rf"), 0.95),
                               "min_samples_split")

    assert response.status_code == 200
    assert data["verdict"] == "unjudged"
    assert data["message"]


def test_the_script_never_re_enables_a_button_it_cannot_use():
    """*What:* the script's two paths that enable the button, after an answer
    and on a change of shape, both defer to the server's flag.

    *How:* read from the source, since no test here runs the script."""
    source = _script()

    assert 'evaluateBtn.disabled = !judgesPriors;' in source
    assert 'evaluateBtn.disabled = kind === "uniform" || !judgesPriors;' in source


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
    assert "class _RecordingPolicy(ClimbingComparisonPolicy)" in optimizer
    assert "threshold=-abs(tolerance), top_k=int(top_k)" in optimizer


def test_a_changed_belief_drops_its_verdict():
    """A stale approval is the one outcome that could mislead: it would sit
    under a curve it no longer describes and suggest there is nothing to ask."""
    assert "setVerdict(null);" in _script()


# ── the tolerance's scale ────────────────────────────────────────────────────

def _diabetes(client):
    """A regression on diabetes, run for three trials, scored as an RMSE."""
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
    """*What:* an RMSE has no worst end, so its scale is what predicting the
    mean scores: the spread of the validation targets.

    *How:* a real diabetes run, against that spread worked out from the same
    holdout split."""
    import numpy as np

    from core import io
    from tests.conftest import DATASETS_DIR
    from ui.views import _metric_scale, _rebuild_result

    exp = _diabetes(client)
    _, _, _, y_val = io._load_splits(DATASETS_DIR / "diabetes.csv", 0, task="regression")

    assert _metric_scale(exp, "rmse", _rebuild_result(exp)) == pytest.approx(float(np.std(y_val)))


def test_without_its_data_an_unbounded_metric_is_scaled_by_its_trials(client, monkeypatch):
    """*What:* an experiment whose dataset is not here — an imported one — is
    scaled by the spread of the scores its trials reached instead.

    *How:* the same run, with rebuilding it from its data made to fail."""
    from core import io
    from ui.views import _metric_scale, _rebuild_result

    exp = _diabetes(client)
    result = _rebuild_result(exp)

    def _missing(*args, **kwargs):
        raise ValueError("dataset not found")

    monkeypatch.setattr(io, "build_experiment", _missing)
    scores = [t.scores["rmse"] for t in result.trials]

    assert _metric_scale(exp, "rmse", result) == pytest.approx(max(scores) - min(scores))


def test_smac_is_handed_the_tolerance_in_the_metrics_units(client, monkeypatch):
    """*What:* the share from the settings, times the metric's scale, is the
    threshold SMAC compares against.

    *How:* SMAC's judgement replaced by one that records the tolerance it was
    handed, for a tolerance of 0.2 on an accuracy."""
    from core.optimizers.smac_optimizer import SMACOptimizer

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
