"""`prepare_environment` builds a custom model's uv environment once, at
creation, so every run uses the same pinned dependencies. The `modelenv` module
holds every uv call, and the rest of the application reads
`Experiment.env_status`.

The environment isolates dependencies only, and the model runs with the
application's user, filesystem and network.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from core.modelhost import deadline

from core.modelhost import describe

#: Named so nothing reformats them.
BLOCK_OPEN = "# /// script"
BLOCK_CLOSE = "# ///"

#: Seconds before uv counts as unusable.
_PROBE_TIMEOUT = 15.0


def script_path(exp) -> Path:
    return Path(exp.data.model_file.path)


def runner_path(exp) -> Path:
    """uv builds the environment from the header of the script it runs, so the
    runner declares the model's dependencies and hands off to the harness. The
    stable path lets uv reuse its cached environment.
    """
    script = script_path(exp)
    return script.with_name(script.stem + "_runner.py")


def lock_path(exp) -> Path:
    """`uv lock --script` writes it here."""
    runner = runner_path(exp)
    return runner.with_name(runner.name + ".lock")


#: Numpy is the one package every model environment gets unasked.
_RUNNER_REQUIREMENTS = ("numpy",)


def write_runner(exp, info=None) -> Path:
    """Without *info*, the dependencies recorded in `env_meta` are reused.
    """
    from core.modelhost import HARNESS

    meta = exp.env_meta or {}
    declared = list(info.dependencies if info else meta.get("dependencies") or [])
    requires = (info.requires_python if info else meta.get("requires_python")) or ""

    header = [BLOCK_OPEN]
    if requires:
        header.append(f'# requires-python = "{requires}"')
    header.append("# dependencies = [")
    header.extend(f'#     "{dep}",' for dep in [*declared, *_RUNNER_REQUIREMENTS])
    header.append("# ]")
    header.append(BLOCK_CLOSE)

    body = [
        '"""Generated. Runs Codesigner\'s harness inside this model\'s environment.',
        "",
        "uv builds an environment from the header above; this then hands over to",
        "the harness, which imports the model and speaks the protocol on stdio.",
        '"""',
        "",
        "import runpy",
        "import sys",
        "",
        f"sys.argv = [\"harness\", \"--model-file\", {str(script_path(exp))!r}, *sys.argv[1:]]",
        f"runpy.run_path({str(HARNESS)!r}, run_name=\"__main__\")",
        "",
    ]

    path = runner_path(exp)
    path.write_text("\n".join([*header, "", *body]), encoding="utf-8")
    return path


def sdk_requirement() -> str:
    """`sdk_requirement` prefers a built wheel over the source tree, which makes
    uv fetch a build backend.
    """
    wheels = Path(settings.MODEL_SDK_WHEEL)
    if wheels.is_dir():
        found = sorted(wheels.glob("codesigner_model-*.whl"))
        if found:
            return str(found[-1])
    elif wheels.is_file():
        return str(wheels)
    return settings.MODEL_SDK_PATH


def uv_path() -> str | None:
    """`uv_path` asks uv its version, because being on PATH proves nothing.
    Pages read `env_status` instead of calling `uv_path`.
    """
    found = shutil.which(settings.UV_BIN)
    if not found:
        return None
    try:
        probe = subprocess.run([found, "--version"], capture_output=True,
                               timeout=_PROBE_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return found if probe.returncode == 0 else None


#: Refusal text for a hosted instance without uv.
HOSTED_WITHOUT_UV = (
    "This server runs uploaded models in isolated environments, which requires "
    "uv. uv is not installed, so custom models cannot run here. Ask the operator "
    "to install uv, or to turn ALLOW_CUSTOM_MODELS off."
)


def decide_mode() -> tuple[str, str]:
    """Without uv, `decide_mode` falls back to the application process on an
    instance without accounts. An instance with accounts refuses, since the
    model would run beside everyone's data.
    """
    if uv_path():
        return "subprocess", ""
    if settings.REQUIRE_LOGIN:
        return "refused", HOSTED_WITHOUT_UV
    return "in_process", (
        "uv is not installed, so this model runs inside the application's own "
        "Python environment. The dependencies it declares are ignored — anything "
        "it imports must already be installed here."
    )


def lock_command(uv: str, runner: Path) -> list[str]:
    argv = [uv, "lock", "--script", str(runner)]
    if settings.MODEL_ENV_OFFLINE:
        argv.append("--offline")
    if not settings.MODEL_ENV_PYTHON_DOWNLOADS:
        argv.append("--no-python-downloads")
    return argv


def run_command(uv: str, runner: Path) -> list[str]:
    """`--locked` reports a stale lock instead of re-resolving. The SDK arrives
    through `--with`, outside the lock, because its path differs between a
    checkout and the image.
    """
    argv = [uv, "run", "--locked", "--with", sdk_requirement(), "--script", str(runner)]
    if settings.MODEL_ENV_OFFLINE:
        argv.append("--offline")
    if not settings.MODEL_ENV_PYTHON_DOWNLOADS:
        argv.append("--no-python-downloads")
    return argv


def child_env() -> dict:
    """The environment uses a denylist because proxy, certificate and index
    settings vary by operator.
    """
    env = {k: v for k, v in os.environ.items() if k not in settings.MODEL_ENV_DENYLIST}
    env["CODESIGNER_MAX_FILE_BYTES"] = str(settings.MODEL_MAX_FILE_BYTES)
    return env


def prepare_environment(experiment_id) -> None:
    """`prepare_environment` runs in the worker and finishes by importing and
    instantiating the model in its own environment, which reports failures
    without holding a request open.
    """
    from experiments.models import Experiment

    exp = Experiment.objects.filter(pk=experiment_id).first()
    if exp is None or not exp.data.model_file:
        return

    mode, explanation = decide_mode()
    if mode == "in_process":
        _settle(exp, Experiment.ENV_SKIPPED, error=explanation)
        return
    if mode == "refused":
        _settle(exp, Experiment.ENV_FAILED, error=explanation)
        return

    Experiment.objects.filter(pk=exp.pk).update(
        env_status=Experiment.ENV_PREPARING, env_error="")

    uv = uv_path()
    try:
        runner = write_runner(exp, _source_info(exp))
        _run_uv(lock_command(uv, runner), "resolving this model's dependencies")  # pyright: ignore[reportArgumentType]
        hello = describe(run_command(uv, runner), env=child_env(), task=exp.data.task,  # pyright: ignore[reportArgumentType]
                         start_timeout=settings.MODEL_ENV_PREPARE_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 — whatever went wrong, the page shows it
        _settle(exp, Experiment.ENV_FAILED, error=str(exc))
        return

    meta = dict(exp.env_meta)
    locked_before = meta.get("lock_sha256")
    meta.update({
        "model_class": hello.get("model_class"),
        "python": hello.get("python"),
        "lock_sha256": _digest(lock_path(exp)),
    })
    # A rebuild that resolved differently can change scores, so the history
    # records it.
    if locked_before and locked_before != meta["lock_sha256"]:
        from . import history

        history.record(exp, "environment_rebuilt", old=locked_before,
                       new=meta["lock_sha256"])
    # The built class's own name overrides the name read from source.
    declared = hello.get("name")
    if declared and declared != exp.data.model_name:
        meta["name_from_source"] = exp.data.model_name
        exp.data.model_name = declared
        # The real class name travels in the file; build details describe the
        # local machine only.
        exp.data.save(update_fields=["model_name"])

    exp.env_status = Experiment.ENV_READY
    exp.env_error = ""
    exp.env_meta = meta
    exp.env_prepared_at = timezone.now()
    exp.save(update_fields=["env_status", "env_error", "env_meta",
                            "env_prepared_at"])


def _source_info(exp):
    """`_source_info` re-reads the file so a retry picks up edits and the
    runner's dependencies match the source.
    """
    from core.model_source import inspect_model_source

    info, _ = inspect_model_source(script_path(exp).read_bytes())
    return info


def _run_uv(argv: list[str], what: str) -> None:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, env=child_env(),
                              timeout=settings.MODEL_ENV_PREPARE_TIMEOUT, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out {what} after {settings.MODEL_ENV_PREPARE_TIMEOUT:.0f}s.") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not run uv: {exc}") from exc
    if done.returncode != 0:
        # uv leads with its diagnosis; tool output stays untranslated.
        detail = (done.stderr or done.stdout or "").strip()[:4000]
        raise RuntimeError(f"Failed {what}:\n{detail}")


def _settle(exp, status: str, *, error: str = "") -> None:
    from experiments.models import Experiment

    Experiment.objects.filter(pk=exp.pk).update(
        env_status=status, env_error=error, env_prepared_at=timezone.now())


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def resolve_runner(exp) -> tuple[list[str] | None, str]:
    """`resolve_runner` returns a `uv run` prefix, None to import in-process, or
    an error that stops the run.
    """
    from experiments.models import Experiment

    if exp.env_status == Experiment.ENV_READY:
        uv = uv_path()
        if not uv:
            # A fallback would run the model against dependencies other than the
            # pinned ones.
            return None, ("This model's environment was built with uv, which is no "
                          "longer available on this server.")
        if not lock_path(exp).is_file():
            return None, ("This model's environment is no longer pinned. Prepare it "
                          "again from the experiment page.")
        if exp.env_meta.get("lock_sha256") and \
                _digest(lock_path(exp)) != exp.env_meta["lock_sha256"]:
            return None, ("This model's lock has changed since it was prepared. "
                          "Prepare it again from the experiment page.")
        if not runner_path(exp).is_file():
            return None, ("This model's runner is missing. Prepare its environment "
                          "again from the experiment page.")
        # The runner owns the header and the lock.
        return run_command(uv, runner_path(exp)), ""

    if exp.env_status == Experiment.ENV_FAILED:
        return None, exp.env_error or "This model's environment could not be built."

    if exp.env_pending:
        return None, "This model's environment is still being prepared."

    # An instance without accounts loads skipped, legacy and unprepared models
    # in-process.
    if settings.REQUIRE_LOGIN:
        return None, HOSTED_WITHOUT_UV
    return None, ""


def start_preparation(exp, info=None) -> None:
    """The page shows the declared dependencies while the build runs.
    """
    from experiments.models import Experiment
    from ..tasks import prepare_model_env_task
    from .dispatch import enqueue

    if not (exp.data.model_file and settings.ALLOW_CUSTOM_MODELS):
        return

    meta = dict(exp.env_meta or {})
    if info is not None:
        meta.update({"dependencies": list(info.dependencies),
                     "requires_python": info.requires_python,
                     "class_name": info.class_name})
    Experiment.objects.filter(pk=exp.pk).update(
        env_status=Experiment.ENV_PENDING, env_error="", env_meta=meta)

    enqueue(prepare_model_env_task, exp.pk)


def sweep_stale_environments() -> int:
    """The container entrypoint runs `sweep_stale_environments` at startup. A
    re-queued build that killed the worker would loop, so the user retries
    instead.
    """
    from experiments.models import Experiment

    return Experiment.objects.filter(env_status=Experiment.ENV_PREPARING).update(
        env_status=Experiment.ENV_FAILED,
        env_error="Interrupted by a restart. Try preparing it again.")


def session_kwargs(trial_timeout=None, seed: int = 0) -> dict:
    """Each run gets a fresh empty working directory, and only the application
    side knows the deployment's default timeout.
    """
    if not trial_timeout:
        policy = deadline.FixedDeadline(settings.MODEL_TRIAL_TIMEOUT)
    else:
        policy = deadline.as_deadline({"seed": seed, **trial_timeout})
    return {
        "env": child_env(),
        "trial_timeout": policy,
        "cwd": tempfile.mkdtemp(prefix="codesigner-run-"),
    }
