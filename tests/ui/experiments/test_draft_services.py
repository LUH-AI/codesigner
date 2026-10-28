"""Setting a draft up without a page: ui/drafts.py and its neighbours.

The setup pages are thin over these functions, and anything else that sets a
draft up — the assistant to come — calls them with plain values: who is acting
comes from a request, everything else from mappings, and what there is to tell
the person comes back as notices.
"""

import pytest
from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory

from tests.conftest import DATASETS_DIR
from ui import drafts, run_form
from experiments.services import priors
from ui import setup_steps as steps

pytestmark = pytest.mark.django_db


def _request():
    """Somebody acting on an instance without accounts."""
    request = RequestFactory().post("/")
    request.user = AnonymousUser()
    return request


def _setup(**overrides):
    data = {"name": "from a service", "task": "classification", "model_name": "Random Forest",
            "demo_dataset": str(DATASETS_DIR / "iris.csv"),
            "evaluation_scheme": "holdout", "evaluation_value": 0.2, "seed": 0}
    data.update(overrides)
    return data


def _saved_through_step_one():
    request = _request()
    exp = drafts.new(request)
    assert drafts.save_step(request, exp, steps.SETUP, _setup()).saved
    exp.refresh_from_db()
    return request, exp


def test_a_draft_is_made_and_its_first_step_saved_from_a_dict():
    """What: a new draft, then step 1 from a plain dict, as the page would post it.
    How: `drafts.new`, `drafts.save_step` and the stored row."""
    request = _request()
    exp = drafts.new(request)
    outcome = drafts.save_step(request, exp, steps.SETUP, _setup())
    exp.refresh_from_db()

    assert exp.draft and outcome.saved and not outcome.gone
    assert exp.data.name == "from a service" and exp.data.model_name == "Random Forest"
    assert exp.data.demo_dataset == "iris"
    assert steps.saved(exp, steps.SETUP)


def test_an_invalid_first_step_says_why_on_its_form():
    """What: step 1 without a dataset is not saved, and its form carries the error.
    How: saves a dict with no dataset and reads the outcome."""
    request = _request()
    exp = drafts.new(request)
    outcome = drafts.save_step(request, exp, steps.SETUP, _setup(demo_dataset=""))

    assert not outcome.saved and outcome.form.errors
    assert not steps.saved(exp, steps.SETUP)


def test_a_later_step_waits_for_the_first():
    """What: the Optimizer step is refused while step 1 is unsaved.
    How: saves step 2 on a fresh draft."""
    request = _request()
    outcome = drafts.save_step(request, drafts.new(request), steps.OPTIMIZER,
                               {"optimizer_name": "Random Search"})

    assert not outcome.saved and outcome.refusal


def test_a_partial_optimizer_save_keeps_what_it_does_not_mention():
    """What: changing one SMAC setting leaves the others as they were, ticked
    boxes included — where a whole form leaves an unticked box absent, so False.
    How: saves SMAC on its defaults, then only a trial cap, partially, as a number."""
    request, exp = _saved_through_step_one()
    drafts.save_step(request, exp, steps.OPTIMIZER, {"optimizer_name": "SMAC"}, partial=True)
    exp.refresh_from_db()
    assert exp.data.optimizer_params["use_share_cap"] is True

    outcome = drafts.save_step(request, exp, steps.OPTIMIZER, {"opt_trial_cap": 7}, partial=True)
    exp.refresh_from_db()

    assert outcome.saved and steps.saved(exp, steps.OPTIMIZER)
    assert exp.data.optimizer_name == "SMAC"
    assert exp.data.optimizer_params["trial_cap"] == 7
    assert exp.data.optimizer_params["use_share_cap"] is True


def test_the_data_step_saves_only_what_it_is_given():
    """What: the Data step stores the choices it names and keeps the rest.
    How: saves one processing choice and reads all three back."""
    request, exp = _saved_through_step_one()
    before = dict(exp.data.processing or {})
    outcome = drafts.save_step(request, exp, steps.DATA, {"scale": "standardize"})
    exp.refresh_from_db()

    assert outcome.saved and steps.saved(exp, steps.DATA)
    assert exp.data.processing["scale"] == "standardize"
    for step in ("missing", "labels"):
        assert exp.data.processing.get(step, "auto") == before.get(step, "auto")


def test_create_waits_for_the_first_three_steps_then_begins_the_history():
    """What: Create is refused until steps 1 to 3 are saved; then the draft is an
    experiment whose history begins with how it was set up.
    How: `drafts.create` before and after saving steps 2 and 3."""
    request, exp = _saved_through_step_one()
    assert drafts.create(request, exp)

    drafts.save_step(request, exp, steps.OPTIMIZER, {"optimizer_name": "Random Search"})
    drafts.save_step(request, exp, steps.DATA, {})
    assert drafts.create(request, exp) is None
    exp.refresh_from_db()

    assert not exp.draft
    assert exp.data.history.filter(kind="created").exists()


def test_a_prior_is_stated_and_taken_back_without_a_page():
    """What: a belief about one hyperparameter is stored, then reset to nothing,
    since no run searched under one.
    How: `priors.state` and `priors.reset` on a draft."""
    request, exp = _saved_through_step_one()
    stated = priors.state(exp, "max_depth",
                          {"kind": "normal", "params": {"mu": 0.3, "sigma": 0.1},
                           "decay": {"shape": "none"}}, user=request.user)
    assert stated["max_depth"]["kind"] == "normal"

    assert priors.reset(exp, "max_depth", user=request.user) is False
    exp.refresh_from_db()
    assert "max_depth" not in (exp.data.priors or {})


def test_the_run_form_reads_numbers_as_well_as_text():
    """What: stopping criteria given as numbers read as a browser's text would,
    and a 0 is a value, not a blank.
    How: `run_form.stopping_from` and `trial_timeout_from` over typed values."""
    stopping = run_form.stopping_from({"max_trials": 25, "max_failures": "3",
                                       "target_score": 0}, "accuracy")
    timeout = run_form.trial_timeout_from({"trial_timeout_mode": "fixed",
                                           "trial_timeout_seconds": 0})

    assert stopping == {"max_trials": 25, "max_failures": 3, "target_score": 0.0}
    assert timeout["seconds"] == 0.0


def test_settling_storage_answers_with_notices():
    """What: an experiment with no group has no limit, so settling keeps it and
    has nothing to say.
    How: `drafts.settle_storage` on a draft saved without accounts."""
    _request_, exp = _saved_through_step_one()

    assert drafts.settle_storage(exp) == (True, [])


# ── a dataset before step 1 ──────────────────────────────────────────────────

def _csv(name="rows.csv", text="a,b,y\n1,2,0\n3,4,1\n5,6,0\n"):
    from django.core.files.uploadedfile import SimpleUploadedFile

    return SimpleUploadedFile(name, text.encode(), content_type="text/csv")


def test_a_demo_dataset_is_staged_and_step_one_keeps_it():
    """What: a demo put on a fresh draft is its dataset, and step 1 saved without
    naming one keeps it.
    How: `stage_dataset` by the demo's name, then step 1 with no dataset field."""
    request = _request()
    exp = drafts.new(request)
    staged = drafts.stage_dataset(request, exp, demo="iris")
    exp.refresh_from_db()

    assert staged.saved and exp.data.demo_dataset == "iris"
    assert not steps.saved(exp, steps.SETUP)

    outcome = drafts.save_step(request, exp, steps.SETUP, _setup(demo_dataset=""))
    exp.refresh_from_db()
    assert outcome.saved and exp.data.demo_dataset == "iris"


def test_an_uploaded_dataset_is_staged_and_replaces_the_last():
    """What: an uploaded CSV is stored on the draft; a demo staged after it takes
    its place and the file goes; a chosen target goes with the old data.
    How: stages an upload, sets a target, then stages a demo."""
    request = _request()
    exp = drafts.new(request)
    assert drafts.stage_dataset(request, exp, upload=_csv()).saved
    exp.refresh_from_db()
    stored = exp.data.dataset.path
    exp.data.target_column = "a"
    exp.data.save(update_fields=["target_column"])

    assert drafts.stage_dataset(request, exp, demo="wine").saved
    exp.refresh_from_db()

    from pathlib import Path

    assert exp.data.demo_dataset == "wine" and not exp.data.dataset
    assert not Path(stored).exists()
    assert exp.data.target_column == ""


def test_staging_refuses_what_step_one_would():
    """What: a file that is not a CSV, a demo that does not exist, nothing at
    all, and a draft whose step 1 is saved are each refused with a reason.
    How: `stage_dataset` in each of those cases."""
    request = _request()
    exp = drafts.new(request)

    assert drafts.stage_dataset(request, exp, upload=_csv("rows.txt")).refusal
    assert drafts.stage_dataset(request, exp, demo="nothing-like-it").refusal
    assert drafts.stage_dataset(request, exp).refusal

    _request_, saved = _saved_through_step_one()
    assert drafts.stage_dataset(request, saved, demo="iris").refusal
