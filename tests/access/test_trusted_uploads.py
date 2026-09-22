"""Who may bring code onto a hosted instance, and who may change everyone's defaults.

`ALLOW_CUSTOM_MODELS` is instance-wide: on a hosted instance it means every
account or none, and "every account" is wrong for a capability that is arbitrary
code execution. So a per-account permission sits on top of it, and the flag stays
the floor — off means off, for everyone, including staff.

The check that matters is the one in `execute_run`. A gate on the upload form is
advice: the form can be bypassed, an experiment can change hands, and a run is
started by a task rather than by a request. The worker is the process that would
actually execute the code, so that is where the decision is made — and the tests
below go straight at it rather than through the page.
"""

import textwrap

import pytest
from django.contrib.auth.models import Permission
from django.core.files.base import ContentFile
from django.urls import reverse

from tests.conftest import DATASETS_DIR
from ui.models import Experiment, Run
from ui.services import run as run_service

pytestmark = pytest.mark.django_db

MODEL_SOURCE = textwrap.dedent('''
    # /// script
    # dependencies = ["ConfigSpace"]
    # ///
    from codesigner_model import BaseModel

    class Mine(BaseModel):
        name = "Mine"
        def get_config_space(self, seed=0): return None
        def fit_predict(self, config, X_train, y_train, X_val, seed=0): return []
''').encode()


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True
    # The floor, raised. Every test here is about the per-account permission
    # that sits on top of the instance-wide flag, and the flag is off by default
    # now — so without this there is nothing for the permission to sit on. The
    # one test about the floor itself takes `settings` and lowers it again.
    settings.ALLOW_CUSTOM_MODELS = True


def _trust(user):
    user.user_permissions.add(
        Permission.objects.get(codename="use_custom_models"))
    return type(user).objects.get(pk=user.pk)   # drop the permission cache


@pytest.fixture
def ana(django_user_model):
    return django_user_model.objects.create_user(username="ana", password="pw")


@pytest.fixture
def trusted(django_user_model):
    return _trust(django_user_model.objects.create_user(username="tru", password="pw"))


def _custom_experiment(owner=None) -> Experiment:
    exp = Experiment(name="custom", model_name="Mine",
                     optimizer_name="Random Search", metric_names=["accuracy"],
                     seed=0, owner=owner, env_status=Experiment.ENV_LEGACY)
    exp.model_file.save("mine.py", ContentFile(MODEL_SOURCE), save=False)
    exp.dataset.save("iris.csv", ContentFile((DATASETS_DIR / "iris.csv").read_bytes()),
                     save=False)
    exp.save()
    return exp


# ── the worker: the check that actually decides ──────────────────────────────

def test_the_worker_refuses_an_untrusted_owners_model(hosted, ana):
    """Reached by a task, not a request, so no form gate applies here."""
    exp = _custom_experiment(owner=ana)
    run = run_service.create_run(exp, {"max_trials": 3}, "accuracy", started_by=ana)

    run_service.execute_run(run.id)
    run.refresh_from_db()

    assert run.status == "error"
    assert "not allowed to run custom models" in run.error


def test_the_worker_names_who_was_refused(hosted, ana):
    """A worker log entry that just says "denied" cannot be acted on."""
    exp = _custom_experiment(owner=ana)
    run = run_service.create_run(exp, {"max_trials": 3}, "accuracy", started_by=ana)

    run_service.execute_run(run.id)
    run.refresh_from_db()

    assert "ana" in run.error


def test_who_pressed_run_is_what_is_checked_not_only_who_owns_it(hosted, ana, trusted):
    """An unowned experiment has no owner to check, so the run records who
    started it. Without that, an experiment predating accounts could never run a
    custom model, or could always run one — both wrong."""
    exp = _custom_experiment(owner=None)

    refused = run_service.create_run(exp, {"max_trials": 3}, "accuracy", started_by=ana)
    run_service.execute_run(refused.id)
    refused.refresh_from_db()
    assert refused.status == "error"

    # The trusted user gets past the permission and stops at the next gate
    # instead — this experiment predates environments, and a hosted instance
    # will not import a model into its own process. Two independent refusals,
    # and this is the one being pinned.
    allowed = run_service.create_run(exp, {"max_trials": 3}, "accuracy", started_by=trusted)
    run_service.execute_run(allowed.id)
    allowed.refresh_from_db()
    assert "not allowed to run custom models" not in allowed.error
    assert "cannot run here" in allowed.error


def test_a_registry_model_is_never_gated(hosted, ana):
    """The permission is about code the instance did not ship."""
    exp = Experiment.objects.create(
        name="builtin", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=ana)
    exp.dataset.save("iris.csv", ContentFile((DATASETS_DIR / "iris.csv").read_bytes()))
    run = run_service.create_run(exp, {"max_trials": 2}, "accuracy", started_by=ana)

    run_service.execute_run(run.id)
    run.refresh_from_db()

    assert run.status == "done"


def test_nothing_is_gated_without_accounts(settings, ana):
    settings.REQUIRE_LOGIN = False
    exp = _custom_experiment(owner=ana)

    from ui.permissions import policy
    assert policy().custom_model_refusal(exp, ana) == ""


# ── the pages ────────────────────────────────────────────────────────────────

def test_an_untrusted_user_is_not_offered_the_upload_field(client, hosted, ana):
    client.force_login(ana)

    html = client.get(reverse("ui:new_experiment")).content.decode()

    assert 'name="model_file"' not in html


def test_a_trusted_user_is(client, hosted, trusted):
    client.force_login(trusted)

    html = client.get(reverse("ui:new_experiment")).content.decode()

    assert 'name="model_file"' in html


def test_posting_an_upload_anyway_does_not_create_a_custom_model(client, hosted, ana):
    """The field being absent from the form is what makes the POST fail, rather
    than the upload being quietly accepted and ignored."""
    from django.core.files.uploadedfile import SimpleUploadedFile

    client.force_login(ana)
    client.post(reverse("ui:new_experiment"), {
        "name": "sneaky", "model_name": "", "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "seed": "0",
        "model_file": SimpleUploadedFile("mine.py", MODEL_SOURCE,
                                         content_type="text/x-python"),
    })

    assert not Experiment.objects.filter(name="sneaky").exists()


def test_an_untrusted_user_cannot_attach_a_model_when_importing(client, hosted, ana):
    from django.core.files.uploadedfile import SimpleUploadedFile

    client.force_login(ana)
    snapshot = ('{"version": "0.1.0", "name": "imported", "model_name": "Mine",'
                ' "model_path": "", "optimizer_name": "Random Search",'
                ' "optimizer_params": {}, "primary_metric": null,'
                ' "original_metric": null, "metric_names": ["accuracy"],'
                ' "seed": 0, "dataset_path": "", "result": null}')

    client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", snapshot.encode()),
        "model": SimpleUploadedFile("mine.py", MODEL_SOURCE),
    })

    assert not Experiment.objects.get(name="imported").model_file


def test_the_page_says_why_it_cannot_be_run(client, hosted, ana):
    """Rather than a Run form that is simply missing, with no explanation."""
    exp = _custom_experiment(owner=ana)
    client.force_login(ana)

    resp = client.get(reverse("ui:experiment_detail", args=[exp.pk]))

    assert resp.context["can_run"] is False
    assert "not allowed to run custom models" in resp.content.decode()


def test_the_instance_flag_still_outranks_the_permission(client, settings, hosted, trusted):
    """Off means off. The permission grants a subset of what the flag allows,
    never more — an operator turning custom models off must not have to audit
    who holds what."""
    settings.ALLOW_CUSTOM_MODELS = False
    client.force_login(trusted)

    html = client.get(reverse("ui:new_experiment")).content.decode()

    assert 'name="model_file"' not in html


# ── the defaults everyone follows ────────────────────────────────────────────

def test_changing_the_defaults_needs_the_permission_for_it(client, hosted, ana,
                                                           django_user_model):
    """One person changing these changes what every inheriting experiment on the
    instance draws — so it is its own grant rather than something that arrives
    bundled with being able to open /admin/.
    """
    from django.contrib.auth.models import Permission

    client.force_login(ana)
    assert client.get(reverse("ui:default_experiment_settings")).status_code == 403

    # `is_staff` alone is deliberately not enough: it is Django's flag for
    # reaching the admin site, and it used to carry this by accident.
    clerk = django_user_model.objects.create_user(
        username="deskclerk", password="pw", is_staff=True)
    client.force_login(clerk)
    assert client.get(reverse("ui:default_experiment_settings")).status_code == 403

    ana.user_permissions.add(Permission.objects.get(
        content_type__app_label="access", codename="change_defaults"))
    client.force_login(django_user_model.objects.get(pk=ana.pk))
    assert client.get(reverse("ui:default_experiment_settings")).status_code == 200


def test_an_ordinary_user_is_not_shown_the_defaults_link(client, hosted, ana):
    client.force_login(ana)

    html = client.get(reverse("ui:appearance")).content.decode()

    assert reverse("ui:default_experiment_settings") not in html


def test_promoting_your_settings_to_the_defaults_needs_it_too(client, hosted, ana):
    """The other way into the same global state, from an experiment's own
    settings page."""
    exp = Experiment.objects.create(
        name="mine", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=ana)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_settings", args=[exp.pk]),
                       {"save_as_default": "1", "show_trials": "on"})

    assert resp.status_code == 403


def test_the_defaults_stay_open_without_accounts(client, settings):
    settings.REQUIRE_LOGIN = False
    assert client.get(reverse("ui:default_experiment_settings")).status_code == 200
