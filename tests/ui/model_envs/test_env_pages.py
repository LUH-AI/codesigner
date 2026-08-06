"""What an experiment page says while its model's environment is built.

An environment takes minutes to resolve, so creating an experiment cannot wait
for it. The page reports progress with the same polling the run box uses, and
what it offers depends on how things ended: retry a failure, or build a real
environment for a model that is currently being imported in-process.
"""

import textwrap

from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from tests.conftest import DATASETS_DIR
from ui.models import Experiment, Run
from ui.services import modelenv

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


def _experiment(**overrides) -> Experiment:
    fields = dict(
        name="custom", model_name="Mine", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)
    fields.update(overrides)
    exp = Experiment(**fields)
    exp.model_file.save("mine.py", ContentFile(MODEL_SOURCE), save=False)
    exp.dataset.save("iris.csv", ContentFile((DATASETS_DIR / "iris.csv").read_bytes()),
                     save=False)
    exp.save()
    return exp


def _page(client, exp) -> str:
    return client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()


# ── creating one queues the build ────────────────────────────────────────────

def test_uploading_a_model_queues_its_environment(client, fake_uv):
    """Creation returns immediately; the environment is built behind it."""
    upload = SimpleUploadedFile("mine.py", MODEL_SOURCE, content_type="text/x-python")

    resp = client.post(reverse("ui:new_experiment"), {
        "name": "queued", "model_name": "", "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "seed": "0",
        "model_file": upload,
    })

    assert resp.status_code == 302
    exp = Experiment.objects.get(name="queued")
    # the fake uv resolves instantly, and tests run tasks inline
    assert exp.env_status == Experiment.ENV_READY
    # what the source declared was recorded before anything was built
    assert exp.env_meta["dependencies"] == ["ConfigSpace"]


def test_a_registry_model_needs_no_environment(client):
    client.post(reverse("ui:new_experiment"), {
        "name": "plain", "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "seed": "0",
    })

    assert Experiment.objects.get(name="plain").env_status == Experiment.ENV_NONE


# ── while it is building ─────────────────────────────────────────────────────

def test_the_page_polls_while_preparing(client):
    exp = _experiment(env_status=Experiment.ENV_PREPARING,
                      env_meta={"dependencies": ["scikit-learn"]})
    html = _page(client, exp)

    assert reverse("ui:env_status", args=[exp.pk]) in html
    assert "hx-trigger" in html
    assert "scikit-learn" in html          # says what it is installing
    assert 'name="n_trials"' not in html   # and nothing can be run yet


def test_the_poll_target_reports_progress_then_asks_for_a_reload(client):
    exp = _experiment(env_status=Experiment.ENV_PREPARING)
    url = reverse("ui:env_status", args=[exp.pk])

    assert "env-status" in client.get(url).content.decode()

    Experiment.objects.filter(pk=exp.pk).update(env_status=Experiment.ENV_READY)
    settled = client.get(url)

    assert settled["HX-Refresh"] == "true"
    assert settled.content == b""


def test_a_preparing_experiment_cannot_be_run(client):
    exp = _experiment(env_status=Experiment.ENV_PREPARING)
    resp = client.get(reverse("ui:experiment_detail", args=[exp.pk]))
    assert resp.context["can_run"] is False


def test_the_sidebar_marks_an_experiment_whose_environment_is_building(client):
    _experiment(env_status=Experiment.ENV_PENDING)
    assert "⏳" in client.get(reverse("ui:home")).content.decode()


# ── when it fails ────────────────────────────────────────────────────────────

def test_a_failure_shows_the_tool_output_and_offers_a_retry(client):
    exp = _experiment(env_status=Experiment.ENV_FAILED,
                      env_error="x No solution found: no such package `nope`")
    html = _page(client, exp)

    assert "No solution found" in html
    assert reverse("ui:prepare_env", args=[exp.pk]) in html


def test_retrying_queues_another_build(client, fake_uv):
    exp = _experiment(env_status=Experiment.ENV_FAILED, env_error="it went wrong")

    resp = client.post(reverse("ui:prepare_env", args=[exp.pk]))
    exp.refresh_from_db()

    assert resp.status_code == 302
    assert exp.env_status == Experiment.ENV_READY
    assert exp.env_error == ""


def test_retrying_requires_a_post(client):
    exp = _experiment(env_status=Experiment.ENV_FAILED)
    assert client.get(reverse("ui:prepare_env", args=[exp.pk])).status_code == 405


def test_a_run_in_flight_blocks_a_rebuild(client, fake_uv):
    """Rebuilding under a running optimization would change what it is using."""
    exp = _experiment(env_status=Experiment.ENV_FAILED)
    Run.objects.create(experiment=exp, n_trials=3, primary_metric="accuracy",
                       status="running")

    client.post(reverse("ui:prepare_env", args=[exp.pk]))
    exp.refresh_from_db()

    assert exp.env_status == Experiment.ENV_FAILED


# ── when it is running in-process instead ────────────────────────────────────

def test_an_in_process_model_says_so_and_offers_to_build_one(client, settings):
    settings.REQUIRE_LOGIN = False
    exp = _experiment(env_status=Experiment.ENV_SKIPPED,
                      env_error="uv is not installed, so this model runs …")
    html = _page(client, exp)

    assert "not one of its own" in html
    assert reverse("ui:prepare_env", args=[exp.pk]) in html


def test_an_in_process_model_can_still_be_run_locally(client, settings):
    settings.REQUIRE_LOGIN = False
    exp = _experiment(env_status=Experiment.ENV_LEGACY)
    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).context["can_run"] is True


def test_an_in_process_model_cannot_be_run_on_a_hosted_instance(client, settings,
                                                                django_user_model):
    """The whole point of the refusal: one user's code would run in the process
    holding everyone's data.

    Signed in, because `REQUIRE_LOGIN` also raises the login wall — being signed
    in is what makes this a hosted instance's *user* rather than a stranger, and
    the refusal has to hold for them too."""
    settings.REQUIRE_LOGIN = True
    django_user_model.objects.create_user(username="someone", password="pw")
    client.login(username="someone", password="pw")
    exp = _experiment(env_status=Experiment.ENV_LEGACY)

    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).context["can_run"] is False


# ── when it is ready ─────────────────────────────────────────────────────────

def test_a_ready_experiment_shows_the_run_form_and_its_environment(client):
    exp = _experiment(env_status=Experiment.ENV_READY,
                      env_meta={"dependencies": ["scikit-learn"], "python": "3.12.4"})
    html = _page(client, exp)

    assert 'name="n_trials"' in html
    assert "scikit-learn" in html
    assert "Python 3.12.4" in html


# ── recovering from an interrupted worker ────────────────────────────────────

def test_a_build_interrupted_by_a_restart_is_swept(client):
    """Not re-queued: a build that killed the worker would restart-loop, so the
    user decides whether to try again."""
    exp = _experiment(env_status=Experiment.ENV_PREPARING)

    assert modelenv.sweep_stale_environments() == 1
    exp.refresh_from_db()

    assert exp.env_status == Experiment.ENV_FAILED
    assert "Interrupted by a restart" in exp.env_error
    assert reverse("ui:prepare_env", args=[exp.pk]) in _page(client, exp)
