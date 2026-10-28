"""`test_modelenv` pins the uv argv and runs the lifecycle against a fake uv.
"""

import textwrap

from django.core.files.base import ContentFile

from core.model_source import inspect_model_source
from experiments.models import Experiment
from experiments.services import modelenv

MODEL_SOURCE = textwrap.dedent('''
    # /// script
    # requires-python = ">=3.11"
    # dependencies = ["scikit-learn", "ConfigSpace"]
    # ///
    from codesigner_model import BaseModel

    class Mine(BaseModel):
        name = "Mine"
        def get_config_space(self, seed=0): return None
        def fit_predict(self, config, X_train, y_train, X_val, seed=0): return []
''').encode()


def _experiment(source: bytes = MODEL_SOURCE, **overrides) -> Experiment:
    fields = dict(
        name="custom", model_name="Mine", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, env_status=Experiment.ENV_PENDING)
    fields.update(overrides)
    exp = Experiment.objects.create(**fields)
    exp.data.model_file.save("mine.py", ContentFile(source), save=False)
    exp.data.save()
    return exp


def test_lock_command_targets_the_runner():
    exp = _experiment()
    argv = modelenv.lock_command("/bin/uv", modelenv.runner_path(exp))

    assert argv[:3] == ["/bin/uv", "lock", "--script"]
    assert argv[3].endswith("mine_runner.py")


def test_run_command_is_locked_and_layers_the_contract():
    """`--locked` stops drift from the pin; `--with` keeps the per-checkout SDK
    path out of the lock."""
    exp = _experiment()
    argv = modelenv.run_command("/bin/uv", modelenv.runner_path(exp))

    assert argv[:3] == ["/bin/uv", "run", "--locked"]
    assert "--with" in argv
    assert argv[argv.index("--with") + 1] == modelenv.sdk_requirement()
    assert argv[argv.index("--script") + 1].endswith("mine_runner.py")


def test_offline_and_no_downloads_are_passed_through(settings):
    """Air-gapped deployments need both flags."""
    settings.MODEL_ENV_OFFLINE = True
    settings.MODEL_ENV_PYTHON_DOWNLOADS = False
    exp = _experiment()

    for argv in (modelenv.lock_command("uv", modelenv.runner_path(exp)),
                 modelenv.run_command("uv", modelenv.runner_path(exp))):
        assert "--offline" in argv
        assert "--no-python-downloads" in argv


def test_the_child_environment_does_not_carry_the_secret_key(settings, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "not-for-models")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")

    env = modelenv.child_env()

    assert "SECRET_KEY" not in env
    assert "DATABASE_URL" not in env
    # A denylist strips secrets and lets `HTTPS_PROXY` through.
    assert env["HTTPS_PROXY"] == "http://proxy.invalid"


def test_the_runner_declares_the_model_dependencies_and_numpy():
    """uv reads the runner's header; numpy is the harness's own need."""
    exp = _experiment()
    info, _ = inspect_model_source(MODEL_SOURCE)

    runner = modelenv.write_runner(exp, info)
    text = runner.read_text()

    assert text.startswith(modelenv.BLOCK_OPEN)
    assert 'requires-python = ">=3.11"' in text
    assert '"scikit-learn"' in text and '"ConfigSpace"' in text
    assert '"numpy"' in text


def test_the_runner_hands_off_to_the_harness_with_the_model():
    exp = _experiment()
    text = modelenv.write_runner(exp, inspect_model_source(MODEL_SOURCE)[0]).read_text()

    assert "runpy.run_path" in text
    assert "harness.py" in text
    assert str(modelenv.script_path(exp)) in text


def test_the_runner_forwards_extra_arguments():
    """`--describe` and `--seed` pass through."""
    exp = _experiment()
    text = modelenv.write_runner(exp, inspect_model_source(MODEL_SOURCE)[0]).read_text()
    assert "*sys.argv[1:]" in text


def test_uv_present_means_environments(fake_uv):
    assert modelenv.decide_mode()[0] == "subprocess"


def test_no_uv_locally_falls_back_with_an_explanation(no_uv, settings):
    settings.REQUIRE_LOGIN = False
    mode, explanation = modelenv.decide_mode()

    assert mode == "in_process"
    assert "uv is not installed" in explanation


def test_no_uv_on_a_hosted_instance_refuses(no_uv, settings):
    """A fallback would put user models beside everyone's data."""
    settings.REQUIRE_LOGIN = True
    mode, explanation = modelenv.decide_mode()

    assert mode == "refused"
    assert "cannot run here" in explanation


def test_a_path_hit_that_does_not_run_is_not_uv(tmp_path, settings):
    broken = tmp_path / "uv"
    broken.write_text("#!/bin/sh\nexit 1\n")
    broken.chmod(0o755)
    settings.UV_BIN = str(broken)

    assert modelenv.uv_path() is None


def test_preparing_locks_the_environment_and_records_what_it_built(fake_uv):
    exp = _experiment()

    modelenv.prepare_environment(exp.pk)
    exp.refresh_from_db()

    assert exp.env_status == Experiment.ENV_READY
    assert exp.env_error == ""
    assert exp.env_prepared_at is not None
    assert modelenv.lock_path(exp).is_file()
    assert exp.env_meta["model_class"] == "FakeModel"
    assert exp.env_meta["python"] == "3.12.0"
    assert exp.env_meta["lock_sha256"]


def test_preparing_corrects_the_name_the_source_only_claimed(fake_uv):
    """`prepare_environment` replaces the name the form read from a source
    literal with the built class's name."""
    exp = _experiment(model_name="Mine")

    modelenv.prepare_environment(exp.pk)
    exp.refresh_from_db()

    assert exp.data.model_name == "Fake Model"
    assert exp.env_meta["name_from_source"] == "Mine"


def test_a_failed_resolve_is_reported_verbatim(failing_uv):
    """uv's diagnosis reads well enough to show verbatim."""
    exp = _experiment()

    modelenv.prepare_environment(exp.pk)
    exp.refresh_from_db()

    assert exp.env_status == Experiment.ENV_FAILED
    assert "No solution found" in exp.env_error


def test_preparing_without_uv_settles_as_skipped(no_uv, settings):
    settings.REQUIRE_LOGIN = False
    exp = _experiment()

    modelenv.prepare_environment(exp.pk)
    exp.refresh_from_db()

    assert exp.env_status == Experiment.ENV_SKIPPED
    assert "uv is not installed" in exp.env_error


def test_preparing_without_uv_on_a_hosted_instance_fails(no_uv, settings):
    settings.REQUIRE_LOGIN = True
    exp = _experiment()

    modelenv.prepare_environment(exp.pk)
    exp.refresh_from_db()

    assert exp.env_status == Experiment.ENV_FAILED


def test_a_prepared_experiment_runs_through_uv(fake_uv):
    exp = _experiment()
    modelenv.prepare_environment(exp.pk)
    exp.refresh_from_db()

    launch, refusal = modelenv.resolve_runner(exp)

    assert refusal == ""
    assert launch is not None
    assert launch[:3] == [str(fake_uv), "run", "--locked"]
    # The lock follows the runner; the model file has none.
    assert launch[launch.index("--script") + 1] == str(modelenv.runner_path(exp))


def test_a_changed_lock_is_refused_rather_than_re_resolved(fake_uv):
    """A drifted dependency set fails without explanation."""
    exp = _experiment()
    modelenv.prepare_environment(exp.pk)
    exp.refresh_from_db()
    modelenv.lock_path(exp).write_text("# tampered\n")

    launch, refusal = modelenv.resolve_runner(exp)

    assert launch is None
    assert "no longer pinned" in refusal or "has changed" in refusal


def test_a_missing_lock_is_refused(fake_uv):
    exp = _experiment()
    modelenv.prepare_environment(exp.pk)
    exp.refresh_from_db()
    modelenv.lock_path(exp).unlink()

    launch, refusal = modelenv.resolve_runner(exp)

    assert launch is None
    assert "no longer pinned" in refusal


def test_uv_disappearing_after_preparation_is_refused(fake_uv, settings, tmp_path):
    exp = _experiment()
    modelenv.prepare_environment(exp.pk)
    exp.refresh_from_db()
    settings.UV_BIN = str(tmp_path / "gone")

    launch, refusal = modelenv.resolve_runner(exp)

    assert launch is None
    assert "no longer available" in refusal


def test_an_experiment_that_predates_environments_runs_in_process(settings):
    settings.REQUIRE_LOGIN = False
    exp = _experiment(env_status=Experiment.ENV_LEGACY)

    launch, refusal = modelenv.resolve_runner(exp)

    assert launch is None and refusal == ""


def test_an_unprepared_experiment_runs_in_process(settings):
    """Imported rows know nothing of environments and run as before."""
    settings.REQUIRE_LOGIN = False
    exp = _experiment(env_status=Experiment.ENV_NONE)

    assert modelenv.resolve_runner(exp) == (None, "")


def test_in_process_is_refused_on_a_hosted_instance(settings):
    settings.REQUIRE_LOGIN = True
    exp = _experiment(env_status=Experiment.ENV_LEGACY)

    launch, refusal = modelenv.resolve_runner(exp)

    assert launch is None
    assert "cannot run here" in refusal


def test_a_run_waits_rather_than_starting_while_preparing():
    exp = _experiment(env_status=Experiment.ENV_PREPARING)

    launch, refusal = modelenv.resolve_runner(exp)

    assert launch is None
    assert "still being prepared" in refusal


def test_a_failed_environment_explains_itself_at_run_time():
    exp = _experiment(env_status=Experiment.ENV_FAILED, env_error="it exploded")

    assert modelenv.resolve_runner(exp) == (None, "it exploded")
