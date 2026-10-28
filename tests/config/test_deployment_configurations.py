"""The queue, the database and the upload store each default to a local file,
and each can point at a network service instead. Each probe loads settings in a
fresh subprocess because `config/settings.py` derives them at import time.
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
    """`None` unsets a variable, since an empty `HUEY_CLASS=` makes huey refuse
    to start.
    """
    env = {**os.environ, "DJANGO_SETTINGS_MODULE": "config.settings",
           "SECRET_KEY": "probe-only-not-a-real-secret",
           # The subprocess re-reads the developer's `.env`, and
           # `HUEY_IMMEDIATE` keeps djhuey from instantiating `RedisHuey`
           # without the driver.
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


# A: all-in-one

def test_the_defaults_are_one_machine():
    """Every development install and the whole test suite use the defaults.
    """
    found = _settings_with(HUEY_CLASS=None, HUEY_FILENAME=None, REDIS_URL=None,
                           DEFAULT_FILE_STORAGE=None)

    assert found["huey_class"] == "huey.SqliteHuey"
    assert found["huey_filename"].endswith("huey.sqlite3")
    assert found["huey_url"] is None, "a file broker has no URL to be reached at"
    assert found["storage"] == "django.core.files.storage.FileSystemStorage"
    assert "sqlite" in found["engine"]


# C: network services

def test_naming_redis_gives_the_broker_a_url_and_no_file():
    """A filename left beside a URL would enqueue into a local file the other
    machine never sees.
    """
    found = _settings_with(HUEY_CLASS="huey.RedisHuey",
                           REDIS_URL="redis://broker:6379/2")

    assert found["huey_class"] == "huey.RedisHuey"
    assert found["huey_url"] == "redis://broker:6379/2"
    assert found["huey_filename"] is None


def test_redis_without_a_url_still_does_not_fall_back_to_a_file():
    """Redis falls back to localhost and fails loudly."""
    found = _settings_with(HUEY_CLASS="huey.RedisHuey", REDIS_URL=None)

    assert found["huey_url"].startswith("redis://")
    assert found["huey_filename"] is None


def test_the_database_switch_needs_no_code():
    """The test needs psycopg because Django loads the database backend during
    setup.
    """
    pytest.importorskip("psycopg",
                        reason="hosted driver; see requirements-hosted.txt")

    found = _settings_with(DATABASE_URL="postgres://u:p@db:5432/codesigner")

    assert "postgresql" in found["engine"]


def test_the_upload_store_is_selectable():
    found = _settings_with(
        DEFAULT_FILE_STORAGE="storages.backends.s3.S3Storage")

    assert found["storage"] == "storages.backends.s3.S3Storage"


def test_trusted_origins_default_to_the_hosts_already_named():
    """Origins derive from ALLOWED_HOSTS to avoid two disagreeing lists; without
    them, every hosted form fails CSRF.
    """
    found = _settings_with(ALLOWED_HOSTS="codesigner.example.org",
                           CSRF_TRUSTED_ORIGINS=None)

    assert found["csrf"] == ["https://codesigner.example.org"]


def test_a_wildcard_host_contributes_no_origin():
    """The reference compose uses `ALLOWED_HOSTS=*`, and `https://*` is no valid
    origin."""
    found = _settings_with(ALLOWED_HOSTS="*", CSRF_TRUSTED_ORIGINS=None)

    assert found["csrf"] == []


def test_trusted_origins_can_be_stated_outright():
    """A proxy may serve an origin outside ALLOWED_HOSTS."""
    found = _settings_with(ALLOWED_HOSTS="web",
                           CSRF_TRUSTED_ORIGINS="https://a.example,https://b.example")

    assert found["csrf"] == ["https://a.example", "https://b.example"]
