"""Three switches decide whether the web process and the consumer can be apart.

*What:* the queue, the database and the upload store each default to a file on
local disk — which is one machine — and each can be pointed at a network service
instead. That is the whole of what separates the deployment configurations, so
what is pinned here is that the defaults are unchanged and that naming a service
actually reaches it, rather than silently falling back to the file.

*How:* the same fresh-subprocess probe `test_settings.py` uses and for the same
reason: these are derived at import time (`if HUEY_CLASS.endswith(...)`), so
pytest-django's `settings` fixture would patch an already-loaded value without
re-running the branch that set it.
"""

import json
import os
import subprocess
import sys

import pytest

_PROBE = (
    "import django; django.setup(); "
    "from django.conf import settings; import json; "
    "print(json.dumps({"
    "'huey_class': settings.HUEY['huey_class'], "
    "'huey_url': settings.HUEY.get('url'), "
    "'huey_filename': settings.HUEY.get('filename'), "
    "'csrf': list(settings.CSRF_TRUSTED_ORIGINS), "
    "'storage': settings.STORAGES['default']['BACKEND'], "
    "'engine': settings.DATABASES['default']['ENGINE'], "
    "}))"
)


def _settings_with(**env_overrides) -> dict:
    """Settings as loaded under *env_overrides*. `None` means **unset**.

    Unset rather than empty, because the two are different and the difference
    matters: `HUEY_CLASS=` is an empty class name, which huey refuses at
    startup, and that refusal is wanted — a blank broker should stop the
    process, not quietly resolve to something. So a test that means "the
    operator set nothing" has to remove the variable.
    """
    env = {**os.environ, "DJANGO_SETTINGS_MODULE": "config.settings",
           "SECRET_KEY": "probe-only-not-a-real-secret",
           # The subprocess re-reads `.env`, which `tests/conftest.py` cannot
           # reach — so the developer's own file would otherwise decide what
           # these probes see. Immediate mode in particular: huey keeps its
           # queue in memory there, and without it naming `RedisHuey` makes
           # djhuey *instantiate* one and fail for want of the driver. What is
           # being checked is which broker is configured, not that one answers.
           "HUEY_IMMEDIATE": "True",
           "REQUIRE_LOGIN": "False",
           "RUN_BACKEND": "local"}
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"settings did not load:\n{result.stderr}")
    return json.loads(result.stdout)


# ── A: all-in-one, which is what every existing install already is ──────────

def test_the_defaults_are_one_machine():
    """Unset means files: a queue on disk, SQLite, and local uploads.

    Stated explicitly rather than assumed, because it is the configuration the
    entire test suite and every development install run under. A change that
    moved any of these would be invisible until something tried to share them.
    """
    found = _settings_with(HUEY_CLASS=None, HUEY_FILENAME=None, REDIS_URL=None,
                           DEFAULT_FILE_STORAGE=None)

    assert found["huey_class"] == "huey.SqliteHuey"
    assert found["huey_filename"].endswith("huey.sqlite3")
    assert found["huey_url"] is None, "a file broker has no URL to be reached at"
    assert found["storage"] == "django.core.files.storage.FileSystemStorage"
    assert "sqlite" in found["engine"]


# ── C: the same two processes over network services ─────────────────────────

def test_naming_redis_gives_the_broker_a_url_and_no_file():
    """The two are mutually exclusive on purpose.

    Leaving a filename set alongside a URL is how a misconfigured instance ends
    up quietly enqueuing into a local file the other machine cannot see — a
    queue that accepts work and never runs it, which looks like a hung worker
    rather than a wrong setting.
    """
    found = _settings_with(HUEY_CLASS="huey.RedisHuey",
                           REDIS_URL="redis://broker:6379/2")

    assert found["huey_class"] == "huey.RedisHuey"
    assert found["huey_url"] == "redis://broker:6379/2"
    assert found["huey_filename"] is None


def test_redis_without_a_url_still_does_not_fall_back_to_a_file():
    """It defaults to localhost, which fails loudly if nothing is there."""
    found = _settings_with(HUEY_CLASS="huey.RedisHuey", REDIS_URL=None)

    assert found["huey_url"].startswith("redis://")
    assert found["huey_filename"] is None


def test_the_database_switch_needs_no_code():
    """`DATABASE_URL` already did this; pinned so the trio stays a trio.

    Skipped without the driver, which is not an evasion: Django loads the
    backend module during setup, so "postgres is selectable" is only a claim
    that can be made where postgres is installable — which is a hosted install
    (`requirements-hosted.txt`), and exactly where it matters.
    """
    pytest.importorskip("psycopg",
                        reason="hosted driver; see requirements-hosted.txt")

    found = _settings_with(DATABASE_URL="postgres://u:p@db:5432/codesigner")

    assert "postgresql" in found["engine"]


def test_the_upload_store_is_selectable():
    found = _settings_with(
        DEFAULT_FILE_STORAGE="storages.backends.s3.S3Storage")

    assert found["storage"] == "storages.backends.s3.S3Storage"


# ── what a hosted instance refuses without ──────────────────────────────────

def test_trusted_origins_default_to_the_hosts_already_named():
    """Otherwise every form on a hosted instance fails CSRF verification.

    Derived from ALLOWED_HOSTS rather than being a second list of the same
    names, since disagreeing copies is the failure this would otherwise invite.
    """
    found = _settings_with(ALLOWED_HOSTS="codesigner.example.org",
                           CSRF_TRUSTED_ORIGINS=None)

    assert found["csrf"] == ["https://codesigner.example.org"]


def test_a_wildcard_host_contributes_no_origin():
    """`ALLOWED_HOSTS=*` is a real setting (the reference compose uses it), and
    `https://*` is not an origin — it would be carried into Django's matcher as
    a name nothing can match."""
    found = _settings_with(ALLOWED_HOSTS="*", CSRF_TRUSTED_ORIGINS=None)

    assert found["csrf"] == []


def test_trusted_origins_can_be_stated_outright():
    """A proxy may serve an origin that is not in ALLOWED_HOSTS verbatim."""
    found = _settings_with(ALLOWED_HOSTS="web",
                           CSRF_TRUSTED_ORIGINS="https://a.example,https://b.example")

    assert found["csrf"] == ["https://a.example", "https://b.example"]
