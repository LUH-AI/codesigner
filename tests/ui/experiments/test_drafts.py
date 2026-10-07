"""New makes a draft, set up in four steps before it becomes an experiment.

See ui/services/setup.py: step 1 is what the experiment is, step 2 its
optimizer, step 3 how its columns are processed, step 4 what is believed
about the optimum. Steps 1 to 3 must each be saved before the draft can be
created.
"""

import json
import re

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR, start_draft
from ui.models import Experiment

pytestmark = pytest.mark.django_db


def _setup(**overrides):
    data = {"name": "set up", "task": "classification", "model_name": "Random Forest",
            "optimizer_name": "Random Search", "demo_dataset": str(DATASETS_DIR / "iris.csv"),
            "evaluation_scheme": "holdout", "evaluation_value": 0.2, "seed": 0}
    data.update(overrides)
    return data


def _save(client, exp, step, data=None):
    names = {"setup": "ui:setup", "optimizer": "ui:setup_optimizer", "data": "ui:setup_data"}
    if step == "optimizer" and data is None:
        data = {"optimizer_name": "Random Search"}
    return client.post(reverse(names[step], args=[exp.pk]), data or {})


def _create(client, exp):
    return client.post(reverse("ui:setup_create", args=[exp.pk]))


# ── New, and what a draft is ────────────────────────────────────────────────

def test_new_makes_a_draft_and_opens_its_first_step(client):
    """What: pressing New makes a draft at once and opens step 1.
    How: posts New and reads the row and the redirect."""
    response = client.post(reverse("ui:new_experiment"))
    exp = Experiment.objects.get()

    assert exp.draft and exp.name == "" and exp.title == "Untitled Experiment"
    assert response["Location"] == reverse("ui:setup", args=[exp.pk])


def test_a_draft_is_in_the_sidebar_as_a_draft(client):
    """What: the draft is listed straight away, marked Draft, linking to its
    setup, with none of an experiment's views under it.
    How: makes a draft and reads the sidebar on its setup page."""
    exp = start_draft(client)
    html = client.get(reverse("ui:setup", args=[exp.pk])).content.decode()
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]

    assert 'class="draft-badge"' in sidebar
    assert f'href="{reverse("ui:setup_resume", args=[exp.pk])}"' in sidebar
    assert "experiment-views" not in sidebar


def test_a_draft_cannot_be_reached_as_an_experiment(client):
    """What: a draft has no dashboard, timeline or Data Handling page — a
    look at one is sent to the setup — and cannot be run or exported.
    How: asks for each, and posts a run."""
    exp = start_draft(client)
    resume = reverse("ui:setup_resume", args=[exp.pk])
    for name in ("ui:experiment_detail", "ui:experiment_timeline", "ui:experiment_export"):
        response = client.get(reverse(name, args=[exp.pk]))
        assert response.status_code == 302 and response["Location"] == resume, name
    assert client.get(reverse("datahandling:data_overview", args=[exp.pk]))["Location"] == resume
    assert client.post(reverse("ui:experiment_run", args=[exp.pk]),
                       {"max_trials": 3}).status_code == 404


def test_another_persons_draft_is_not_there(client, settings):
    """What: a draft is its owner's alone — not their group's, not their
    lead's. How: two people in one group; the second asks for the first's
    draft and its setup."""
    from access.models import Group, Membership
    from django.contrib.auth import get_user_model

    settings.REQUIRE_LOGIN = True
    settings.EXPERIMENT_POLICY = "access.policy.GroupPolicy"
    User = get_user_model()
    group = Group.objects.create(name="lab", user_limit=5)
    ana, vera = User.objects.create_user("ana", password="x"), User.objects.create_user("vera", password="x")
    Membership.objects.create(user=ana, group=group, role=Membership.MEMBER)
    Membership.objects.create(user=vera, group=group, role=Membership.PRIMARY_LEAD)
    client.force_login(ana)
    exp = start_draft(client)

    client.force_login(vera)
    assert client.get(reverse("ui:setup", args=[exp.pk])).status_code == 404
    assert "Untitled Experiment" not in client.get(reverse("ui:home")).content.decode()


# ── the steps ───────────────────────────────────────────────────────────────

def test_the_later_steps_wait_for_step_1(client):
    """What: there is nothing to search, process or believe about before there
    is a dataset and a model. How: asks for steps 2 to 4 of a fresh draft."""
    exp = start_draft(client)
    step1 = reverse("ui:setup", args=[exp.pk])
    assert client.get(reverse("ui:setup_optimizer", args=[exp.pk]))["Location"] == step1
    assert client.get(reverse("ui:setup_data", args=[exp.pk]))["Location"] == step1
    assert client.get(reverse("ui:setup_priors", args=[exp.pk]))["Location"] == step1


def test_saving_each_step_moves_on(client):
    """What: Save on each step goes to the next, and each step is then marked
    saved. How: saves the three and reads the draft."""
    exp = start_draft(client)
    assert _save(client, exp, "setup", _setup())["Location"] == \
        reverse("ui:setup_optimizer", args=[exp.pk])
    assert _save(client, exp, "optimizer")["Location"] == reverse("ui:setup_data", args=[exp.pk])
    assert _save(client, exp, "data", {"scale": "standardize"})["Location"] == \
        reverse("ui:setup_priors", args=[exp.pk])
    exp.refresh_from_db()

    assert set(exp.setup_saved) == {"setup", "optimizer", "data"}
    assert exp.data.name == "set up" and exp.data.processing["scale"] == "standardize"
    assert exp.data.optimizer_name == "Random Search"


def test_create_waits_for_every_required_save(client):
    """What: Create is refused until steps 1 to 3 have all been saved, and
    the button says why. How: tries after step 1, after step 2, then after
    all three."""
    exp = start_draft(client)
    _save(client, exp, "setup", _setup())
    page = client.get(reverse("ui:setup_priors", args=[exp.pk])).content.decode()
    assert "Save step 2, Optimizer, first." in page
    _save(client, exp, "optimizer")
    page = client.get(reverse("ui:setup_priors", args=[exp.pk])).content.decode()
    assert "Save step 3, Data, first." in page
    _create(client, exp)
    exp.refresh_from_db()
    assert exp.draft

    _save(client, exp, "data")
    assert _create(client, exp)["Location"] == reverse("ui:experiment_detail", args=[exp.pk])
    exp.refresh_from_db()
    assert not exp.draft


def test_creating_records_how_it_was_set_up(client):
    """What: the experiment's history begins at creation: how it was set up
    (the whole configuration, its Data Handling choices included), then each
    prior as an entry of its own. A choice made in the setup is not recorded
    as a change — nothing ran under anything else — and nothing done while
    it was a draft is recorded as it happened. How: sets processing and a
    prior, creates, reads the history."""
    exp = start_draft(client)
    _save(client, exp, "setup", _setup())
    _save(client, exp, "optimizer")
    _save(client, exp, "data", {"scale": "standardize"})
    client.post(reverse("ui:save_prior", args=[exp.pk]),
                json.dumps({"hp": "max_depth", "prior": {"kind": "normal",
                                                         "params": {"mu": 0.5, "sigma": 0.1}}}),
                content_type="application/json")
    _create(client, exp)

    created, prior = Experiment.objects.get(pk=exp.pk).data.history.order_by("id")
    assert created.kind == "created"
    assert created.payload["configuration"]["dataset"]["processing"]["scale"] == "standardize"
    assert (prior.kind, prior.payload["hyperparameter"]) == ("prior_stated", "max_depth")


def test_a_different_dataset_asks_for_step_2_again(client):
    """What: step 1 saved again with another dataset clears step 2's save —
    what the processing comes to depends on the data — and keeps the choices.
    How: saves both steps, then step 1 with wine."""
    exp = start_draft(client)
    _save(client, exp, "setup", _setup())
    _save(client, exp, "optimizer")
    _save(client, exp, "data", {"scale": "standardize"})
    _save(client, exp, "setup", _setup(demo_dataset=str(DATASETS_DIR / "wine.csv")))
    exp.refresh_from_db()

    assert set(exp.setup_saved) == {"setup", "optimizer"}
    assert exp.data.processing["scale"] == "standardize"


def test_a_different_model_withdraws_the_priors(client):
    """What: priors are about a model's search space, so another model
    withdraws them; another name keeps everything.
    How: states a prior, renames, then changes the model."""
    exp = start_draft(client)
    _save(client, exp, "setup", _setup())
    _save(client, exp, "optimizer")
    _save(client, exp, "data")
    client.post(reverse("ui:save_prior", args=[exp.pk]),
                json.dumps({"hp": "max_depth", "prior": {"kind": "normal",
                                                         "params": {"mu": 0.5, "sigma": 0.1}}}),
                content_type="application/json")
    _save(client, exp, "setup", _setup(name="renamed"))
    exp.refresh_from_db()
    assert exp.data.priors and set(exp.setup_saved) == {"setup", "optimizer", "data"}

    _save(client, exp, "setup", _setup(model_name="Extra Trees"))
    exp.refresh_from_db()
    assert not exp.data.priors and set(exp.setup_saved) == {"setup", "optimizer"}


def test_step_1_is_filled_from_the_draft_and_keeps_its_upload(client):
    """What: a draft saved before reopens step 1 with its answers, and an
    uploaded dataset is kept when nothing new is chosen.
    How: saves with an uploaded CSV, reopens the page, saves again without one."""
    from django.core.files.uploadedfile import SimpleUploadedFile

    exp = start_draft(client)
    upload = SimpleUploadedFile("flowers.csv", (DATASETS_DIR / "iris.csv").read_bytes())
    _save(client, exp, "setup", {**_setup(demo_dataset=""), "dataset_file": upload})
    html = client.get(reverse("ui:setup", args=[exp.pk])).content.decode()
    assert 'value="set up"' in html and "Currently flowers.csv" in html

    assert _save(client, exp, "setup", _setup(demo_dataset="", name="again")).status_code == 302
    exp.refresh_from_db()
    assert exp.data.name == "again" and exp.data.dataset.name.endswith(".csv")


def test_a_draft_can_be_thrown_away(client):
    """What: Discard deletes the draft outright — no bin — and its files.
    How: saves an upload, discards, and looks for the row and the file."""
    from django.core.files.uploadedfile import SimpleUploadedFile

    exp = start_draft(client)
    upload = SimpleUploadedFile("gone.csv", (DATASETS_DIR / "iris.csv").read_bytes())
    _save(client, exp, "setup", {**_setup(demo_dataset=""), "dataset_file": upload})
    exp.refresh_from_db()
    path = exp.data.dataset_path()

    client.post(reverse("ui:setup_discard", args=[exp.pk]))
    assert not Experiment.objects.filter(pk=exp.pk).exists()
    assert not path.exists()


def test_a_created_experiment_has_no_setup(client):
    """What: once created, the setup is gone: its pages go to the dashboard.
    How: creates one and asks for step 1."""
    exp = start_draft(client)
    _save(client, exp, "setup", _setup())
    _save(client, exp, "optimizer")
    _save(client, exp, "data")
    _create(client, exp)
    response = client.get(reverse("ui:setup", args=[exp.pk]))
    assert response["Location"] == reverse("ui:experiment_detail", args=[exp.pk])


def test_the_steps_are_shown_as_steps(client):
    """What: the setup's steps are numbered, in order, each saying whether it
    has been saved — not the dashboard's tabs.
    How: reads the step bar on step 2 after step 1 is saved."""
    exp = start_draft(client)
    _save(client, exp, "setup", _setup())
    html = client.get(reverse("ui:setup_optimizer", args=[exp.pk])).content.decode()
    bar = html.split('class="setup-steps"', 1)[1].split("</nav>", 1)[0]

    steps = re.findall(r'<li class="setup-step([^"]*)"', bar)
    assert len(steps) == 4
    assert "is-saved" in steps[0] and "is-current" in steps[1]
    assert 'aria-current="step"' in bar
