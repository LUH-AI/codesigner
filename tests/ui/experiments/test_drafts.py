"""`ui:new_experiment` makes a draft with the four setup steps of
`ui/setup_steps.py`. `ui:setup_create` needs steps 1 to 3 saved.
"""

import json
import re

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR, start_draft
from experiments.models import Experiment

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


# what a draft is

def test_new_makes_a_draft_and_opens_its_first_step(client):
    response = client.post(reverse("ui:new_experiment"))
    exp = Experiment.objects.get()

    assert exp.draft and exp.name == "" and exp.title == "Untitled Experiment"
    assert response["Location"] == reverse("ui:setup", args=[exp.pk])


def test_a_draft_is_in_the_sidebar_as_a_draft(client):
    """The sidebar lists the draft at once, marked Draft and linking to its
    setup."""
    exp = start_draft(client)
    html = client.get(reverse("ui:setup", args=[exp.pk])).content.decode()
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]

    assert 'class="draft-badge"' in sidebar
    assert f'href="{reverse("ui:setup_resume", args=[exp.pk])}"' in sidebar
    assert "experiment-views" not in sidebar


def test_a_draft_cannot_be_reached_as_an_experiment(client):
    """Each experiment page of a draft redirects to its setup, and a run POST
    gets 404."""
    exp = start_draft(client)
    resume = reverse("ui:setup_resume", args=[exp.pk])
    for name in ("ui:experiment_detail", "ui:experiment_timeline", "ui:experiment_export"):
        response = client.get(reverse(name, args=[exp.pk]))
        assert response.status_code == 302 and response["Location"] == resume, name
    assert client.get(reverse("ui:data_overview", args=[exp.pk]))["Location"] == resume
    assert client.post(reverse("ui:experiment_run", args=[exp.pk]),
                       {"max_trials": 3}).status_code == 404


def test_another_persons_draft_is_not_there(client, settings):
    """Only the owner sees a draft; a lead in the group gets 404."""
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


# the steps

def test_the_later_steps_wait_for_step_1(client):
    exp = start_draft(client)
    step1 = reverse("ui:setup", args=[exp.pk])
    assert client.get(reverse("ui:setup_optimizer", args=[exp.pk]))["Location"] == step1
    assert client.get(reverse("ui:setup_data", args=[exp.pk]))["Location"] == step1
    assert client.get(reverse("ui:setup_priors", args=[exp.pk]))["Location"] == step1


def test_saving_each_step_moves_on(client):
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
    """`ui:setup_create` waits until steps 1 to 3 are saved, and the page
    names the missing step."""
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
    """`created` opens the history with the full configuration, then each prior
    gets an entry; setup choices record no changes."""
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
    """Another dataset clears step 3's save, because processing depends on the
    data, but keeps the choices."""
    exp = start_draft(client)
    _save(client, exp, "setup", _setup())
    _save(client, exp, "optimizer")
    _save(client, exp, "data", {"scale": "standardize"})
    _save(client, exp, "setup", _setup(demo_dataset=str(DATASETS_DIR / "wine.csv")))
    exp.refresh_from_db()

    assert set(exp.setup_saved) == {"setup", "optimizer"}
    assert exp.data.processing["scale"] == "standardize"


def test_a_different_model_withdraws_the_priors(client):
    """A new model withdraws the priors, which belong to its search space; a
    rename keeps them."""
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
    """Step 1 reopens filled in and keeps the upload when no new file comes."""
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
    """Discard deletes the row and its files, with no bin."""
    from django.core.files.uploadedfile import SimpleUploadedFile

    exp = start_draft(client)
    upload = SimpleUploadedFile("gone.csv", (DATASETS_DIR / "iris.csv").read_bytes())
    _save(client, exp, "setup", {**_setup(demo_dataset=""), "dataset_file": upload})
    exp.refresh_from_db()
    path = exp.data.dataset_path()
    assert path is not None

    client.post(reverse("ui:setup_discard", args=[exp.pk]))
    assert not Experiment.objects.filter(pk=exp.pk).exists()
    assert not path.exists()


def test_a_created_experiment_has_no_setup(client):
    """A created experiment's setup pages redirect to the dashboard."""
    exp = start_draft(client)
    _save(client, exp, "setup", _setup())
    _save(client, exp, "optimizer")
    _save(client, exp, "data")
    _create(client, exp)
    response = client.get(reverse("ui:setup", args=[exp.pk]))
    assert response["Location"] == reverse("ui:experiment_detail", args=[exp.pk])


def test_the_steps_are_shown_as_steps(client):
    """The `setup-steps` bar shows four numbered steps with saved and current
    marks."""
    exp = start_draft(client)
    _save(client, exp, "setup", _setup())
    html = client.get(reverse("ui:setup_optimizer", args=[exp.pk])).content.decode()
    bar = html.split('class="setup-steps"', 1)[1].split("</nav>", 1)[0]

    steps = re.findall(r'<li class="setup-step([^"]*)"', bar)
    assert len(steps) == 4
    assert "is-saved" in steps[0] and "is-current" in steps[1]
    assert 'aria-current="step"' in bar
