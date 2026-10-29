"""`ALLOW_CUSTOM_MODELS` is the instance-wide floor beneath the per-account
`use_custom_models` permission. `execute_run` makes the decisive check because
the worker executes the code.
"""

import textwrap

import pytest
from django.contrib.auth.models import Permission
from django.core.files.base import ContentFile
from django.urls import reverse

from tests.conftest import DATASETS_DIR, get_new_experiment, post_new_experiment
from experiments.models import Experiment, Run
from experiments.services import run as run_service

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
    # `ALLOW_CUSTOM_MODELS` defaults to off, and the permission needs the floor
    # raised.
    settings.ALLOW_CUSTOM_MODELS = True


def _trust(user):
    user.user_permissions.add(
        Permission.objects.get(codename="use_custom_models"))
    return type(user).objects.get(pk=user.pk)


@pytest.fixture
def ana(django_user_model):
    return django_user_model.objects.create_user(username="ana", password="pw")


@pytest.fixture
def trusted(django_user_model):
    return _trust(django_user_model.objects.create_user(username="tru", password="pw"))


def _custom_experiment(owner=None) -> Experiment:
    exp = Experiment.objects.create(
        name="custom", model_name="Mine", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner,
        env_status=Experiment.ENV_LEGACY)
    exp.data.model_file.save("mine.py", ContentFile(MODEL_SOURCE), save=False)
    exp.data.dataset.save("iris.csv", ContentFile((DATASETS_DIR / "iris.csv").read_bytes()),
                          save=False)
    exp.data.save()
    return exp


def test_the_worker_refuses_an_untrusted_owners_model(hosted, ana):
    """No form gate applies to a task."""
    exp = _custom_experiment(owner=ana)
    run = run_service.create_run(exp, {"max_trials": 3}, "accuracy", started_by=ana)

    run_service.execute_run(run.id)
    run.refresh_from_db()

    assert run.status == "error"
    assert "not allowed to run custom models" in run.error


def test_the_worker_names_who_was_refused(hosted, ana):
    """A bare "denied" log entry cannot be acted on."""
    exp = _custom_experiment(owner=ana)
    run = run_service.create_run(exp, {"max_trials": 3}, "accuracy", started_by=ana)

    run_service.execute_run(run.id)
    run.refresh_from_db()

    assert "ana" in run.error


def test_who_pressed_run_is_what_is_checked_not_only_who_owns_it(hosted, ana, trusted):
    """`Run.started_by` names the person to check, since an ownerless experiment
    has no owner."""
    exp = _custom_experiment(owner=None)

    refused = run_service.create_run(exp, {"max_trials": 3}, "accuracy", started_by=ana)
    run_service.execute_run(refused.id)
    refused.refresh_from_db()
    assert refused.status == "error"

    # The trusted user passes the permission and stops at the environment gate,
    # since `.resolve_runner` returns an error for an in-process model on a
    # hosted instance.
    allowed = run_service.create_run(exp, {"max_trials": 3}, "accuracy", started_by=trusted)
    run_service.execute_run(allowed.id)
    allowed.refresh_from_db()
    assert "not allowed to run custom models" not in allowed.error
    assert "cannot run here" in allowed.error


def test_a_registry_model_is_never_gated(hosted, ana):
    """The permission only covers uploaded code."""
    exp = Experiment.objects.create(
        name="builtin", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=ana)
    exp.data.dataset.save("iris.csv", ContentFile((DATASETS_DIR / "iris.csv").read_bytes()))
    run = run_service.create_run(exp, {"max_trials": 2}, "accuracy", started_by=ana)

    run_service.execute_run(run.id)
    run.refresh_from_db()

    assert run.status == "done"


def test_nothing_is_gated_without_accounts(settings, ana):
    settings.REQUIRE_LOGIN = False
    exp = _custom_experiment(owner=ana)

    from access.policy import policy
    assert policy().custom_model_refusal(exp, ana) == ""


def test_an_untrusted_user_is_not_offered_the_upload_field(client, hosted, ana):
    client.force_login(ana)

    html = get_new_experiment(client).content.decode()

    assert 'name="model_file"' not in html


def test_a_trusted_user_is(client, hosted, trusted):
    client.force_login(trusted)

    html = get_new_experiment(client).content.decode()

    assert 'name="model_file"' in html


def test_posting_an_upload_anyway_does_not_create_a_custom_model(client, hosted, ana):
    """The POST fails because the form lacks the field."""
    from django.core.files.uploadedfile import SimpleUploadedFile

    client.force_login(ana)
    post_new_experiment(client, {
        "name": "sneaky", "model_name": "", "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification", "seed": "0",
        "model_file": SimpleUploadedFile("mine.py", MODEL_SOURCE,
                                         content_type="text/x-python"),
    })

    assert not Experiment.objects.filter(data__name="sneaky").exists()


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

    assert not Experiment.objects.get(data__name="imported").data.model_file


def test_the_page_says_why_it_cannot_be_run(client, hosted, ana):
    """The page explains the missing Run form."""
    exp = _custom_experiment(owner=ana)
    client.force_login(ana)

    resp = client.get(reverse("ui:experiment_detail", args=[exp.pk]))

    assert resp.context["can_run"] is False
    assert "not allowed to run custom models" in resp.content.decode()


def test_the_instance_flag_still_outranks_the_permission(client, settings, hosted, trusted):
    """An operator turning custom models off needs no audit, since the
    permission grants a subset of the flag."""
    settings.ALLOW_CUSTOM_MODELS = False
    client.force_login(trusted)

    html = get_new_experiment(client).content.decode()

    assert 'name="model_file"' not in html


def test_changing_the_defaults_needs_the_permission_for_it(client, hosted, ana,
                                                           django_user_model):
    """The `change_defaults` permission protects defaults that reach every
    inheriting experiment.
    """
    from django.contrib.auth.models import Permission

    client.force_login(ana)
    assert client.get(reverse("ui:default_experiment_settings")).status_code == 403

    # `is_staff` only reaches the admin site.
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
    """`save_as_default` changes global state and needs the permission too."""
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
