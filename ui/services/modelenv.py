"""Building and pinning the environment a custom model runs in.

A model declares its dependencies in a PEP 723 header and uv builds an
environment from them. That happens **once**, when the experiment is created:
resolving hits the network and can take minutes, and pinning it means every run
of that experiment uses the same dependencies rather than whatever resolves that
day. Runs then only have to start a process.

Everything that shells out to uv lives here, so the rest of the application asks
`Experiment.env_status` — a column — rather than the filesystem.

This buys **dependency isolation, not a security boundary**: the model runs as
the same user with the same filesystem and network access. Real sandboxing is a
separate piece of work, and the README must not imply otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from core.modelhost import describe

#: The PEP 723 block delimiters, named so nothing reformats them by accident.
BLOCK_OPEN = "# /// script"
BLOCK_CLOSE = "# ///"

#: What `uv --version` must answer within before we treat uv as unusable.
_PROBE_TIMEOUT = 15.0

# ── where things are ─────────────────────────────────────────────────────────


def script_path(exp) -> Path:
    """The model file uv operates on."""
    return Path(exp.model_file.path)


def runner_path(exp) -> Path:
    """The script uv actually runs, generated beside the model.

    uv builds an environment from the PEP 723 header of *the script it runs*, so
    pointing it at the model file would run the model rather than our harness.
    The runner declares the model's dependencies and then hands off to the
    harness — which is how the harness comes to execute inside the model's
    environment.

    Written beside the model in MEDIA rather than in a temp directory: uv keys a
    script's cached environment off its path, so a stable path means the
    environment is built once and reused by every run instead of being rebuilt.
    """
    script = script_path(exp)
    return script.with_name(script.stem + "_runner.py")


def lock_path(exp) -> Path:
    """Where uv keeps the runner's lock (`uv lock --script` writes it here)."""
    runner = runner_path(exp)
    return runner.with_name(runner.name + ".lock")


#: The harness needs numpy to read the dataset arrays it is handed. That is the
#: runner's requirement rather than the model's, and it is the one package a
#: model environment gets without asking — worth saying out loud.
_RUNNER_REQUIREMENTS = ("numpy",)


def write_runner(exp, info=None) -> Path:
    """(Re)generate the runner for *exp* and return its path.

    *info* is the `ModelSourceInfo` read from the model's source. Without it the
    dependencies already recorded in `env_meta` are used, so retrying does not
    have to re-read the file.
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
    """What to install the contract from.

    A built wheel if the image made one, otherwise the source tree — which
    works locally but makes uv fetch a build backend, so it is the fallback.
    """
    wheels = Path(settings.MODEL_SDK_WHEEL)
    if wheels.is_dir():
        found = sorted(wheels.glob("codesigner_model-*.whl"))
        if found:
            return str(found[-1])
    elif wheels.is_file():
        return str(wheels)
    return settings.MODEL_SDK_PATH


# ── is uv usable, and what does that mean ────────────────────────────────────


def uv_path() -> str | None:
    """uv's path if it is installed and runs, else None.

    Being on PATH is not proof it works, so it is asked its version. Called from
    the worker and from `execute_run`, never from a page render — a request reads
    `env_status` instead.
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


#: Why a hosted instance will not fall back to importing the model here.
HOSTED_WITHOUT_UV = (
    "This server runs uploaded models in isolated environments, which requires "
    "uv. uv is not installed, so custom models cannot run here. Ask the operator "
    "to install uv, or to turn ALLOW_CUSTOM_MODELS off."
)


def decide_mode() -> tuple[str, str]:
    """How custom models run on this instance: ``(mode, explanation)``.

    ``subprocess`` — uv is here; environments get built.
    ``in_process``  — no uv, and this instance has no accounts, so a model is
                      imported the way it was before environments existed.
    ``refused``     — no uv on an instance with accounts. Falling back would run
                      one user's code in the process holding everyone's data.
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


# ── the commands ─────────────────────────────────────────────────────────────


def lock_command(uv: str, runner: Path) -> list[str]:
    """Resolve the runner's dependencies and write its lock."""
    argv = [uv, "lock", "--script", str(runner)]
    if settings.MODEL_ENV_OFFLINE:
        argv.append("--offline")
    if not settings.MODEL_ENV_PYTHON_DOWNLOADS:
        argv.append("--no-python-downloads")
    return argv


def run_command(uv: str, runner: Path) -> list[str]:
    """Argv that runs the harness in the model's locked environment.

    ``--locked`` refuses to re-resolve: a lock that no longer matches its script
    is a mismatch to report, not something to paper over silently. The contract
    arrives with ``--with`` rather than being declared in the header, so it is
    not part of what was locked — the SDK's path differs between a local checkout
    and the image, and that should not invalidate a lock.
    """
    argv = [uv, "run", "--locked", "--with", sdk_requirement(), "--script", str(runner)]
    if settings.MODEL_ENV_OFFLINE:
        argv.append("--offline")
    if not settings.MODEL_ENV_PYTHON_DOWNLOADS:
        argv.append("--no-python-downloads")
    return argv


def child_env() -> dict:
    """The environment a model's process inherits.

    A denylist rather than an allowlist: proxy, certificate and index settings
    are numerous and operator-specific, and removing the few that matter is
    both safer and less likely to break a deployment than guessing at a
    complete list of what to keep.
    """
    env = {k: v for k, v in os.environ.items() if k not in settings.MODEL_ENV_DENYLIST}
    env["CODESIGNER_MAX_FILE_BYTES"] = str(settings.MODEL_MAX_FILE_BYTES)
    return env


# ── preparing ────────────────────────────────────────────────────────────────


def prepare_environment(experiment_id) -> None:
    """Resolve, build and check one experiment's model environment.

    Runs in the worker. Three steps, and the third is the one that makes the
    wait worth it: the model is imported and instantiated in its own
    environment, so everything the create form stopped checking is checked here
    — where a failure can be reported without holding a request open.
    """
    from ..models import Experiment

    exp = Experiment.objects.filter(pk=experiment_id).first()
    if exp is None or not exp.model_file:
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
        _run_uv(lock_command(uv, runner), "resolving this model's dependencies")
        # Building the environment and checking the model are the same step: uv
        # materialises the environment to run this, and answering means the model
        # imported and instantiated. Everything the create form stopped doing is
        # done here, where a failure need not hold a request open.
        hello = describe(run_command(uv, runner), env=child_env(),
                         start_timeout=settings.MODEL_ENV_PREPARE_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 — whatever went wrong, the page shows it
        _settle(exp, Experiment.ENV_FAILED, error=str(exc))
        return

    meta = dict(exp.env_meta)
    meta.update({
        "model_class": hello.get("model_class"),
        "python": hello.get("python"),
        "lock_sha256": _digest(lock_path(exp)),
    })
    # The name in the source is what the form read before anything ran. Now that
    # the class has actually been built, its own name is the authority.
    declared = hello.get("name")
    if declared and declared != exp.model_name:
        meta["name_from_source"] = exp.model_name
        exp.model_name = declared

    exp.env_status = Experiment.ENV_READY
    exp.env_error = ""
    exp.env_meta = meta
    exp.env_prepared_at = timezone.now()
    exp.save(update_fields=["model_name", "env_status", "env_error", "env_meta",
                            "env_prepared_at"])


def _source_info(exp):
    """What the model file declares, re-read from disk.

    Read again rather than trusted from `env_meta` so a retry picks up an edited
    file, and so the dependencies the runner declares always match the source
    that is actually there.
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
        # uv leads with a readable diagnosis, so the head of its output is the
        # useful part. Not translated — it is a tool's output, like the model
        # errors beside it.
        detail = (done.stderr or done.stdout or "").strip()[:4000]
        raise RuntimeError(f"Failed {what}:\n{detail}")


def _settle(exp, status: str, *, error: str = "") -> None:
    from ..models import Experiment

    Experiment.objects.filter(pk=exp.pk).update(
        env_status=status, env_error=error, env_prepared_at=timezone.now())


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


# ── using it, at run time ────────────────────────────────────────────────────


def resolve_runner(exp) -> tuple[list[str] | None, str]:
    """How to start this experiment's model: ``(argv_prefix, error)``.

    ``argv_prefix`` is what to put before the harness — `uv run …` for a
    prepared environment, or None to mean "import it here". An error string
    means the run cannot proceed.
    """
    from ..models import Experiment

    if exp.env_status == Experiment.ENV_READY:
        uv = uv_path()
        if not uv:
            # Falling back here would run the model against a completely
            # different dependency set than the one it was pinned to, and the
            # failures from that are arbitrarily strange.
            return None, ("This model's environment was built with uv, which is no "
                          "longer available on this server.")
        if not lock_path(exp).is_file():
            return None, ("This model's environment is no longer pinned. Prepare it "
                          "again from the experiment page.")
        if exp.env_meta.get("lock_sha256") and \
                _digest(lock_path(exp)) != exp.env_meta["lock_sha256"]:
            return None, ("This model's lock has changed since it was prepared. "
                          "Prepare it again from the experiment page.")
        return run_command(uv, script_path(exp)), ""

    if exp.env_status == Experiment.ENV_FAILED:
        return None, exp.env_error or "This model's environment could not be built."

    if exp.env_pending:
        return None, "This model's environment is still being prepared."

    # Everything else — skipped, legacy, or never prepared at all (a row created
    # by the import command, or before any of this existed) — is imported here,
    # exactly as it was. Except on an instance with accounts, where doing that
    # would run one user's code in the process holding everyone's data.
    if settings.REQUIRE_LOGIN:
        return None, HOSTED_WITHOUT_UV
    return None, ""


def session_kwargs() -> dict:
    """How `core.modelhost.model_session` should be configured here.

    A fresh empty working directory per run, so a model that writes
    `./output.csv` does not scribble in the application's own directory.
    """
    return {
        "env": child_env(),
        "trial_timeout": settings.MODEL_TRIAL_TIMEOUT,
        "cwd": tempfile.mkdtemp(prefix="codesigner-run-"),
    }


def prepared_summary(exp) -> str:
    """A short line about the environment, for the experiment page."""
    meta = exp.env_meta or {}
    parts = []
    deps = meta.get("dependencies") or []
    if deps:
        parts.append(", ".join(deps))
    if meta.get("python"):
        parts.append(f"Python {meta['python']}")
    return " · ".join(parts)


def as_json(exp) -> str:
    """The environment's metadata, for debugging and the admin."""
    return json.dumps(exp.env_meta or {}, indent=2, sort_keys=True)
